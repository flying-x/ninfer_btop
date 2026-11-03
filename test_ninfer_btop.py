"""Unit tests for ninfer_btop (standard library only).

No network, no engine, no real terminal: urllib is faked to serve
deterministic /metrics bodies, and the curses screen is a recorder.

Run:  python3 -m unittest -v test_ninfer_btop
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest

import ninfer_btop as nb


# ------------------------------------------------------------- fakes / helpers

class _Resp:
    def __init__(self, body):
        self.body = body.encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_metrics(body):
    def _urlopen(url, timeout=5.0):
        return _Resp(body)
    nb.urllib.request.urlopen = _urlopen


class FakeScreen:
    """Records every curses write as (y, x, text, attr)."""

    def __init__(self, maxy=40, maxx=120):
        self.maxy, self.maxx = maxy, maxx
        self.writes = []

    def erase(self):
        self.writes = []

    def getmaxyx(self):
        return (self.maxy, self.maxx)

    def addnstr(self, y, x, text, n, attr=0):
        self.writes.append((y, x, text[:n], attr))

    def addstr(self, y, x, text, attr=0):
        self.writes.append((y, x, text, attr))

    def refresh(self):
        pass


class _CursesError(Exception):
    pass


def stub_curses():
    nb.curses.color_pair = lambda i: 0
    nb.curses.error = _CursesError
    nb.curses.A_REVERSE = 2


def rows_with(scr, needle):
    return sorted({y for (y, x, t, a) in scr.writes if needle in t})


def make_engine(ok):
    e = nb.Engine("http://x/metrics", 0.5)
    e.ok = ok
    e.last_err = "URLError: [Errno 111] Connection refused"
    e.last_good = 1700000000.0
    e.model = "m"
    e.start_time = None
    e.m = {}
    e.last_frame = None
    return e


CALIB_OK = {"ok": True, "concurrency": 8, "prefill_tps": 100.0,
            "decode_tps": 50.0, "duration": 10.0, "model": "m"}


class Base(unittest.TestCase):
    def setUp(self):
        self._real_urlopen = nb.urllib.request.urlopen
        self._real_calib_file = nb.CALIB_FILE
        nb.CALIB = None
        stub_curses()

    def tearDown(self):
        nb.urllib.request.urlopen = self._real_urlopen
        nb.CALIB_FILE = self._real_calib_file
        nb.CALIB = None

    def _sample(self, **over):
        """A plausible successful poll() frame; override any key via **over."""
        s = {"ready": True, "running": 4, "waiting": 0, "prefilling": 1,
             "decode_ready": 3, "paused": 0,
             "toks_per_s": 50.0, "prefill_tps": 100.0,
             "input_total": 1_000_000, "input_cached": 100_000,
             "output_total": 500_000, "prefill": 900_000,
             "dev_cap": 8e9, "dev_inuse": 5e9, "dev_cached": 2e9,
             "dev_peak": 7.5e9,
             "host_cap": 8e9, "host_cached": 7e9, "host_peak": 7.9e9,
             "util": 0.42, "cloud_cost": 1.955, "max_concurrency": 4,
             "tok_hit": 0.10, "req_hit": 0.25}
        s.update(over)
        return s


# --------------------------------------------- A. fetch_metrics label parsing

class TestFetchMetricsLabels(Base):
    def test_A1_escaped_label_values_are_unescaped(self):
        body = r'ninfer_model_info{model_name="custom\"model\\name\nline",speculative_backend="eagle"} 1'
        fake_metrics(body)
        m = nb.fetch_metrics("http://x/metrics")
        lbl = m["ninfer_model_info__labels"]
        self.assertEqual(lbl["model_name"], 'custom"model\\name\nline')
        self.assertEqual(lbl["speculative_backend"], "eagle")

    def test_A2_comma_inside_value_does_not_split(self):
        body = 'ninfer_model_info{model_name="my, model",speculative_backend="none"} 1'
        fake_metrics(body)
        m = nb.fetch_metrics("http://x/metrics")
        lbl = m["ninfer_model_info__labels"]
        self.assertEqual(lbl["model_name"], "my, model")
        self.assertEqual(lbl["speculative_backend"], "none")
        self.assertFalse(any(k.endswith('"') or k.startswith(' ') for k in lbl))

    def test_A3_first_series_wins_still_holds(self):
        body = (
            "ninfer_generation_tokens_total 123456789\n"
            "ninfer_engine_ready 1\n"
            'ninfer_requests_total{outcome="completed"} 400\n'
            'ninfer_requests_total{outcome="cancelled"} 2'
        )
        fake_metrics(body)
        m = nb.fetch_metrics("http://x/metrics")
        self.assertEqual(m["ninfer_generation_tokens_total"], 123456789.0)
        self.assertEqual(m["ninfer_engine_ready"], 1.0)
        self.assertEqual(m["ninfer_requests_total"], 400.0)
        self.assertEqual(m["ninfer_requests_total__labels"]["outcome"], "completed")

    def test_A4_comment_blank_and_nonnumeric_are_skipped(self):
        body = "# TYPE ninfer_weird counter\n\nninfer_good 5\nninfer_notanum abc"
        fake_metrics(body)
        m = nb.fetch_metrics("http://x/metrics")
        self.assertEqual(m["ninfer_good"], 5.0)
        self.assertNotIn("ninfer_notanum", m)
        self.assertNotIn("ninfer_weird", m)
        self.assertNotIn("TYPE", m)


# ----------------------------------------------- B. calibrate model-name path

class TestCalibrateModelName(Base):
    def test_B1_unescaped_model_name_is_sent_to_chat(self):
        body = (
            'ninfer_model_info{model_name="my, model",speculative_backend="none"} 1\n'
            "ninfer_prefill_tokens_total 100\n"
            "ninfer_generation_tokens_total 100\n"
            "ninfer_requests_running 0\n"
            "ninfer_max_concurrency 8"
        )
        fake_metrics(body)
        sent = []
        real_chat = nb._chat_first_chunk

        def fake_chat(api, model, prompt, max_tokens, timeout=120):
            sent.append(model)
            return (time.monotonic(), 0.1)

        nb._chat_first_chunk = fake_chat
        try:
            nb.calibrate("http://x/metrics", log=lambda *a: None, burst=1)
        finally:
            nb._chat_first_chunk = real_chat
        self.assertIn("my, model", sent)
        self.assertNotIn("my", sent)


# --------------------------------------------- C. Engine.poll inter-poll rates

class TestEnginePollRate(Base):
    def poll(self, eng, body):
        fake_metrics(body)
        return eng.poll()

    def test_C1_rate_produced_below_the_old_250ms_guard(self):
        eng = nb.Engine("http://x/metrics", interval=0.10)
        self.poll(eng, "ninfer_generation_tokens_total 1000\nninfer_engine_ready 1")
        time.sleep(0.10)
        s = self.poll(eng, "ninfer_generation_tokens_total 1100\nninfer_engine_ready 1")
        self.assertIsNotNone(s["toks_per_s"])
        self.assertGreater(s["toks_per_s"], 400)
        self.assertLess(s["toks_per_s"], 1600)

    def test_C2_guard_tracks_the_configured_interval(self):
        eng = nb.Engine("http://x/metrics", interval=0.05)
        self.poll(eng, "ninfer_generation_tokens_total 1000\nninfer_engine_ready 1")
        time.sleep(0.05)
        self.poll(eng, "ninfer_generation_tokens_total 1050\nninfer_engine_ready 1")
        time.sleep(0.05)
        s = self.poll(eng, "ninfer_generation_tokens_total 1100\nninfer_engine_ready 1")
        self.assertIsNotNone(s["toks_per_s"])

    def test_C3_flat_or_reset_window_is_never_negative(self):
        eng = nb.Engine("http://x/metrics", interval=0.05)
        self.poll(eng, "ninfer_generation_tokens_total 1000\nninfer_engine_ready 1")
        time.sleep(0.06)
        s = self.poll(eng, "ninfer_generation_tokens_total 1000\nninfer_engine_ready 1")
        self.assertTrue(s["toks_per_s"] is None or s["toks_per_s"] >= 0)


# --------------------------------------------- D. frame() down-state layout

class TestFrameDownState(Base):
    def frame(self, calib, ok):
        nb.CALIB = calib
        scr = FakeScreen(40, 120)
        graphs = {k: nb.Graph(60) for k in ("batch", "toks", "prefill")}
        nb.frame(scr, make_engine(ok), graphs, {})
        return scr

    def test_D1_down_calibrated_banner_and_panels_disjoint(self):
        scr = self.frame(CALIB_OK, ok=False)
        banner = rows_with(scr, "cannot reach engine")
        last = rows_with(scr, "last good sample")
        panel = rows_with(scr, "Batch size")
        self.assertTrue(banner and last and panel)
        self.assertLess(max(banner), min(panel))
        self.assertLess(max(last), min(panel))
        for (y, x, t, a) in scr.writes:
            is_banner = ("cannot reach engine" in t) or ("last good sample" in t)
            is_panel = any(p in t for p in ("Batch size", "Prefill t/s", "Decode t/s"))
            self.assertFalse(is_banner and is_panel, "row %d: %r" % (y, t))

    def test_D2_down_no_calib_banner_above_first_panel(self):
        scr = self.frame(None, ok=False)
        banner = rows_with(scr, "cannot reach engine")
        panel = rows_with(scr, "Batch size")
        self.assertLess(max(banner), min(panel))

    def test_D3_up_calibrated_header_block_consecutive(self):
        scr = self.frame(CALIB_OK, ok=True)
        calib = rows_with(scr, "calibrated in")
        panel = rows_with(scr, "Batch size")
        self.assertTrue(calib and panel)
        # the calibration line is followed directly by the first panel; the
        # redundant "util = ..." header line was removed (util is in the footer)
        self.assertEqual(calib[0] + 1, panel[0])
        self.assertFalse(rows_with(scr, "util = (prefill"))


# ------------------------------------- E. cheap pure-function / formatter guards

class TestPureFunctions(Base):
    def test_E1_draw_panel_keeps_the_last_title_char(self):
        scr = FakeScreen(40, 120)
        nb.draw_panel(scr, 0, "Token totals", "title", {}, width=40)
        row = [t for (y, x, t, a) in scr.writes if y == 0][0]
        self.assertTrue(row.startswith("─Token totals"), row)
        scr2 = FakeScreen(40, 120)
        nb.draw_panel(scr2, 0, "ABCDEFGHIJKLMNOP", "title", {}, width=18)
        row2 = [t for (y, x, t, a) in scr2.writes if y == 0][0]
        self.assertEqual(row2, "─ABCDEFGHIJKLMNOP")

    def test_E2_human_and_money_boundaries(self):
        for n, want in ((0, "0"), (999, "999"), (1000, "1.00K"),
                        (1000000, "1.00M"), (1200000000, "1.20G"),
                        (2500000000000, "2.50T")):
            self.assertEqual(nb.human(n), want, "human(%r)" % n)
        for v, want in ((0, "$0.00"), (0.42, "$0.42"), (9.99, "$9.99"),
                        (1000, "$1,000"), (1200000, "$1.2M")):
            self.assertEqual(nb.money(v), want, "money(%r)" % v)

    def test_E3_cloud_cost_math(self):
        prompt_total, cached, gen = 1000000, 100000, 500000
        body = (
            "ninfer_engine_ready 1\n"
            "ninfer_prompt_tokens_total %d\n"
            "ninfer_prompt_tokens_cached_total %d\n"
            "ninfer_generation_tokens_total %d" % (prompt_total, cached, gen)
        )
        fake_metrics(body)
        s = nb.Engine("http://x/metrics", 0.5).poll()
        expected = ((prompt_total - cached) / 1e6 * nb.CLOUD_IN_RATE
                    + cached / 1e6 * nb.CLOUD_CACHE_RATE
                    + gen / 1e6 * nb.CLOUD_OUT_RATE)
        self.assertAlmostEqual(s["cloud_cost"], expected, places=6)


# ------------------------------------- F. metric stats in the panel title banner

class TestMetricStatBanner(Base):
    def test_F1_segments_order_and_peak_flag(self):
        g = nb.Graph(10)
        for v in (2, 8, 4, 6, 0):
            g.add(v)
        segs = nb.metric_stat_segments(g, "4")
        self.assertEqual("".join(t for t, _ in segs), "(cur=4,max=8,avg=4,min=0)")
        flags = [flag for _, flag in segs]
        self.assertEqual(flags.count(True), 1)  # only the max token is highlighted
        self.assertTrue(next(t for t, flag in segs if flag).startswith(",max="))

    def test_F2_empty_window_has_no_banner_suffix(self):
        self.assertEqual(nb.metric_stat_segments(nb.Graph(10), "—"), [])

    def test_F3_draw_panel_paints_max_in_peak_color(self):
        g = nb.Graph(10)
        for v in (2, 8, 4):
            g.add(v)
        scr = FakeScreen(40, 120)
        nb.draw_panel(scr, 0, "Batch size  (ninfer_requests_running)", "batch", {}, 120,
                      stats=nb.metric_stat_segments(g, "4"))
        row = [w for w in scr.writes if w[0] == 0]
        joined = "".join(w[2] for w in row)
        for token in ("cur=4", "max=8", "avg=", "min="):
            self.assertIn(token, joined)
        # the max= token is painted with the peak attribute (A_REVERSE under the
        # colorless stub); the rest use the default attribute
        peak_attr = nb.curses.A_REVERSE
        self.assertTrue(any(w[2].startswith(",max=") and w[3] == peak_attr for w in row))
        self.assertTrue(any(w[2].startswith("(cur=") and w[3] == 0 for w in row))
        # the max token is never drawn in a plain (non-peak) attribute
        self.assertFalse(any(w[2].startswith(",max=") and w[3] != peak_attr for w in row))



class TestCacheHealth(Base):
    """G: cache reuse + device/host tier split (Cache panel) + status line."""

    def test_G2_poll_populates_cache_fields(self):
        # 10% of input cached, 90% recomputed; host pool pinned
        body = ("ninfer_engine_ready 1\n"
                "ninfer_max_concurrency 4\n"
                "ninfer_prompt_tokens_total 1000000\n"
                "ninfer_prompt_tokens_cached_total 100000\n"
                "ninfer_prefill_tokens_total 900000\n"
                "ninfer_generation_tokens_total 500000\n"
                "ninfer_root_selections_total 3\n"
                "ninfer_checkpoint_selections_total 1\n"
                "ninfer_context_pool_bytes{rank=\"0\",domain=\"host\",kind=\"capacity\"} 8000000000\n"
                "ninfer_context_pool_live_bytes{rank=\"0\",domain=\"host\",kind=\"in_use\"} 0\n"
                "ninfer_context_pool_live_bytes{rank=\"0\",domain=\"host\",kind=\"cached\"} 8000000000\n"
                "ninfer_context_pool_peak_bytes{rank=\"0\",domain=\"host\"} 8000000000\n"
                "ninfer_max_context_tokens 131072\n")
        nb._fetch = fake_metrics(body)
        s = nb.Engine("http://x/metrics", 0.5).poll()
        self.assertAlmostEqual(s["tok_hit"], 0.10)          # 100000/1000000
        self.assertAlmostEqual(s["req_hit"], 0.25)          # 1/(1+3)
        self.assertEqual(s["host_cached"], 8e9)             # pool pinned: all cached
        self.assertEqual(s["host_cap"], 8e9)
        self.assertEqual(s["ctx_window"], 131072)           # engine context window

    def test_G3_frame_renders_cache_panel_without_a_verdict(self):
        # the pool verdict row was dropped: a pinned pool is a normal steady
        # state, not an alarm, so nothing about it may be painted
        nb.curses.color_pair = lambda i: 1 + i              # distinct, inspectable pair ids
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        s = self._sample(host_cached=8e9, tok_hit=0.10, req_hit=0.05)
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = s
        eng.start_time = None
        graphs = {k: nb.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        nb.frame(scr, eng, graphs, c)
        self.assertTrue(rows_with(scr, "Device KV"))
        self.assertTrue(rows_with(scr, "Host KV"))
        self.assertTrue(rows_with(scr, "Cache (checkpoint reuse)"))
        joined = "".join(t for (_, _, t, _) in scr.writes)
        for gone in ("Pool    :", "TOO SMALL", "tight", "TIGHT"):
            self.assertNotIn(gone, joined)


    def test_G4_poll_splits_token_hits_by_serving_tier(self):
        # calibrated at 1000 B/tok; 8MB of lifetime h2d recall => 8000 of
        # the 100000 cached tokens were served from host RAM, 92000 from VRAM
        nb.CALIB = dict(CALIB_OK, bytes_per_tok=1000.0)
        body = ("ninfer_engine_ready 1\n"
                "ninfer_prompt_tokens_total 1000000\n"
                "ninfer_prompt_tokens_cached_total 100000\n"
                "ninfer_prefill_tokens_total 900000\n"
                "ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"0\",direction=\"h2d\"} 8000000\n")
        nb._fetch = fake_metrics(body)
        s = nb.Engine("http://x/metrics", 0.5).poll()
        self.assertAlmostEqual(s["tok_host"], 8000 / 1000000)
        self.assertAlmostEqual(s["tok_dev"], 92000 / 1000000)
        self.assertAlmostEqual(s["tok_dev"] + s["tok_host"], s["tok_hit"])

    def test_G5_host_estimate_clamps_to_the_cached_total(self):
        # 200MB of h2d recall at 1000 B/tok would imply 200000 host-served
        # tokens, more than the 100000 actually cached -> clamp to the total
        nb.CALIB = dict(CALIB_OK, bytes_per_tok=1000.0)
        body = ("ninfer_engine_ready 1\n"
                "ninfer_prompt_tokens_total 1000000\n"
                "ninfer_prompt_tokens_cached_total 100000\n"
                "ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"0\",direction=\"h2d\"} 200000000\n")
        nb._fetch = fake_metrics(body)
        s = nb.Engine("http://x/metrics", 0.5).poll()
        self.assertAlmostEqual(s["tok_host"], s["tok_hit"])
        self.assertAlmostEqual(s["tok_dev"], 0.0)

    def test_G6_frame_renders_the_tier_split_tok_hit_line(self):
        nb.curses.color_pair = lambda i: 1 + i
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        s = self._sample(tok_hit=0.80, tok_dev=0.46, tok_host=0.34)
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = s
        graphs = {k: nb.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        nb.frame(scr, eng, graphs, c)
        joined = "".join(t for (_, _, t, _) in scr.writes)
        # the line must fit the 38-char column at 120 cols: no space between
        # the two groups, both labels fully visible
        self.assertIn("46.0% (device)34.0% (host)", joined)

    def test_G7_without_calibration_the_plain_tok_hit_line_is_kept(self):
        # setUp leaves nb.CALIB = None -> no bpt -> no tier split
        body = ("ninfer_engine_ready 1\n"
                "ninfer_prompt_tokens_total 1000000\n"
                "ninfer_prompt_tokens_cached_total 100000\n")
        nb._fetch = fake_metrics(body)
        s = nb.Engine("http://x/metrics", 0.5).poll()
        self.assertIsNone(s["tok_dev"])
        self.assertIsNone(s["tok_host"])
        nb.curses.color_pair = lambda i: 1 + i
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = self._sample()
        graphs = {k: nb.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        nb.frame(scr, eng, graphs, c)
        self.assertFalse(rows_with(scr, "(device)"))
        self.assertTrue(rows_with(scr, "Tok hit"))

    def test_G9_lane_paused_figure_is_red_only_when_nonzero(self):
        nb.curses.color_pair = lambda i: 1 + i              # distinct, inspectable pair ids
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        graphs = {k: nb.Graph(60) for k in ("batch", "toks", "prefill")}
        # paused = 2 -> the "paused 2" segment carries the 'bad' (red) pair
        eng.last_frame = self._sample(paused=2)
        scr = FakeScreen(40, 120)
        nb.frame(scr, eng, graphs, c)
        bad_attr = 1 + c["bad"]
        self.assertTrue(any("paused 2" in t and a == bad_attr
                            for (_, _, t, a) in scr.writes))
        # paused = 0 -> the same segment is painted with the default color
        eng.last_frame = self._sample(paused=0)
        scr = FakeScreen(40, 120)
        nb.frame(scr, eng, graphs, c)
        self.assertTrue(any("paused 0" in t and a == 0
                            for (_, _, t, a) in scr.writes))

    def test_G10_status_line_shows_context_window_in_kitok(self):
        # 131072 tokens = 128.0 KiToks (1 KiToks = 1024 tokens)
        nb.curses.color_pair = lambda i: 1 + i
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        nb.CALIB = dict(CALIB_OK)
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = self._sample(ctx_window=131072)
        eng.start_time = None
        graphs = {k: nb.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        nb.frame(scr, eng, graphs, c)
        row1 = "".join(t for (y, x, t, a) in scr.writes if y == 1)
        self.assertIn("calibrated in", row1)
        self.assertIn("ctx 128.0 KiToks", row1)

    def test_G11_no_ctx_metric_no_ctx_segment(self):
        nb.curses.color_pair = lambda i: 1 + i
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        nb.CALIB = dict(CALIB_OK)
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = self._sample()  # no ctx_window key
        eng.start_time = None
        graphs = {k: nb.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        nb.frame(scr, eng, graphs, c)
        row1 = "".join(t for (y, x, t, a) in scr.writes if y == 1)
        self.assertIn("calibrated in", row1)
        self.assertNotIn("KiToks", row1)

    def test_G12_no_calib_status_line_is_the_ctx_window(self):
        nb.CALIB = None
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = self._sample(ctx_window=131072)
        eng.start_time = None
        graphs = {k: nb.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        nb.frame(scr, eng, graphs, {})
        row1 = "".join(t for (y, x, t, a) in scr.writes if y == 1)
        self.assertIn("128.0 KiToks", row1)

    def test_G8_explicit_bytes_per_tok_works_without_calibration(self):
        # --bytes-per-tok without calibration data: CALIB has no "ok" key,
        # the split still works and utilization stays n/a
        nb.CALIB = {"bytes_per_tok": 1000.0}
        body = ("ninfer_engine_ready 1\n"
                "ninfer_prompt_tokens_total 1000000\n"
                "ninfer_prompt_tokens_cached_total 100000\n"
                "ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"0\",direction=\"h2d\"} 30000000\n")
        nb._fetch = fake_metrics(body)
        s = nb.Engine("http://x/metrics", 0.5).poll()
        # 30MB / 1000 B/tok = 30000 host-served tokens
        self.assertAlmostEqual(s["tok_host"], 30000 / 1000000)
        self.assertIsNone(s["util"])


# ----------------------------------- H. spill (d2h) vs recall (h2d) bars

class TestXferSpillRecall(Base):
    """H: context-transfer bars split spill (device->host) from recall (host->device)."""

    def test_H1_fetch_keeps_every_direction_series(self):
        body = ("ninfer_context_transfer_bytes_total{resource=\"state\",direction=\"d2h\"} 100\n"
                "ninfer_context_transfer_bytes_total{resource=\"state\",direction=\"h2d\"} 40\n"
                "ninfer_context_transfer_bytes_total{resource=\"main_kv\",direction=\"d2h\"} 60\n"
                "ninfer_context_transfer_bytes_total{resource=\"main_kv\",direction=\"h2d\"} 20\n"
                "ninfer_context_transfer_bytes_total{resource=\"backend_kv\",direction=\"d2d\"} 7\n")
        fake_metrics(body)
        m = nb.fetch_metrics("http://x/metrics")
        ser = m["ninfer_context_transfer_bytes_total__series"]
        self.assertEqual(len(ser), 5)
        spill = sum(v for (kv, v) in ser if kv.get("direction") == "d2h")
        recall = sum(v for (kv, v) in ser if kv.get("direction") == "h2d")
        self.assertEqual(spill, 160.0)   # 100 + 60 summed across resources
        self.assertEqual(recall, 60.0)   # 40 + 20
        # first-series scalar and labels are unchanged (backward compatible)
        self.assertEqual(m["ninfer_context_transfer_bytes_total"], 100.0)
        self.assertEqual(m["ninfer_context_transfer_bytes_total__labels"]["direction"], "d2h")

    def test_H2_poll_computes_spill_and_recall_rates(self):
        t = [1000.0]
        real_mono = nb.time.monotonic
        nb.time.monotonic = lambda: t[0]
        try:
            b0 = ('ninfer_engine_ready 1\n'
                  'ninfer_context_transfer_bytes_total{resource="state",direction="d2h"} 100\n'
                  'ninfer_context_transfer_bytes_total{resource="state",direction="h2d"} 40\n')
            b1 = ('ninfer_engine_ready 1\n'
                  'ninfer_context_transfer_bytes_total{resource="state",direction="d2h"} 300\n'
                  'ninfer_context_transfer_bytes_total{resource="state",direction="h2d"} 70\n')
            e = nb.Engine("http://x/metrics", 0.5)
            fake_metrics(b0)
            s0 = e.poll()
            self.assertIsNone(s0["spill_rate"])    # first poll: no prior sample
            self.assertIsNone(s0["recall_rate"])
            t[0] = 1001.0                          # dt = 1.0 s
            fake_metrics(b1)
            s1 = e.poll()
            self.assertAlmostEqual(s1["spill_rate"], (300 - 100) / 1.0)   # 200 B/s
            self.assertAlmostEqual(s1["recall_rate"], (70 - 40) / 1.0)    # 30 B/s
        finally:
            nb.time.monotonic = real_mono

    def test_H3_frame_renders_spill_recall_sparklines(self):
        nb.curses.color_pair = lambda i: 1 + i
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        s = self._sample(spill_rate=200.0, recall_rate=50.0)
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = s
        eng.start_time = None
        d2h = nb.Graph(26)
        for v in [0.0] * 10 + [200.0] * 4 + [0.0] * 12:
            d2h.add(v)
        h2d = nb.Graph(26)
        for v in [0.0] * 24 + [30.0, 30.0]:
            h2d.add(v)
        graphs = {k: nb.Graph(60) for k in ("batch", "toks", "prefill")}
        graphs["xfer_d2h"] = d2h
        graphs["xfer_h2d"] = h2d
        scr = FakeScreen(40, 120)
        nb.frame(scr, eng, graphs, c)
        self.assertTrue(rows_with(scr, "Spill"))
        self.assertTrue(rows_with(scr, "Recall"))
        spill = next(t for (_, _, t, _) in scr.writes if "Spill" in t)
        recall = next(t for (_, _, t, _) in scr.writes if "Recall" in t)
        # each sparkline autoscales to its OWN window peak: a 200 B/s spill
        # burst and a 30 B/s recall burst both reach the full braille dot,
        # and the side label is that window peak (the scale of the dots)
        self.assertIn("⣿", spill)
        self.assertIn("⣿", recall)
        self.assertTrue(spill.endswith("200B/s"))
        self.assertTrue(recall.endswith("30B/s"))
        # the row must fit the ⅓-width cache column at 120 cols (38 chars)
        self.assertLessEqual(len(spill), 38)

    def test_H4_sparkline_autoscale_and_empty(self):
        # all-zero window: blank sparkline, 0B/s label
        g = nb.Graph(26)
        for _ in range(26):
            g.add(0.0)
        self.assertEqual(nb._xfer_row("  X  ", g),
                         "  X  " + " " * 13 + " " + "0B/s")
        # a burst autoscales against its own window peak, not a shared one
        g2 = nb.Graph(26)
        for v in [0.0] * 11 + [5.0] * 4 + [0.0] * 11:
            g2.add(v)
        row = nb._xfer_row("  X  ", g2)
        self.assertIn("⣿", row)            # 5 B/s burst = its own peak
        self.assertTrue(row.endswith("5B/s"))
        # no graph yet (first frames): dash placeholder
        self.assertEqual(nb._xfer_row("  X  ", None), "  X  " + " " * 14 + "—")
        # rate label formatting is unchanged
        self.assertEqual(nb._rate_val(None), "   —")
        self.assertEqual(nb._rate_val(12_400_000.0), "12.40MB/s")


# --------------------- J. batch panel height follows max concurrency

class TestBatchHeight(Base):
    """J: the batch panel is ceil(max_concurrency/4) chars high (1 braille
    character resolves 4 dot levels = 4 concurrent requests)."""

    def _panel_height(self, floor, calib, max_y=40):
        nb.CALIB = calib
        scr = FakeScreen(max_y, 120)
        eng = make_engine(True)
        eng.last_frame = self._sample()
        graphs = {"batch": nb.Graph(60, floor=floor),
                  "toks": nb.Graph(60), "prefill": nb.Graph(60)}
        nb.frame(scr, eng, graphs, {})
        b = min(rows_with(scr, "Batch size"))
        p = min(rows_with(scr, "Prefill t/s"))
        # title row + N graph rows + baseline row, then the next panel title
        return p - b - 2

    def test_J1_height_table(self):
        for floor, want in ((0, 6), (1, 1), (4, 1), (5, 2), (8, 2),
                            (9, 3), (12, 3), (24, 6), (28, 6)):
            got = self._panel_height(floor, None)
            self.assertEqual(got, want, "floor=%s" % floor)

    def test_J2_calibrated_header_same_rule(self):
        self.assertEqual(self._panel_height(1, CALIB_OK), 1)
        self.assertEqual(self._panel_height(8, CALIB_OK), 2)

    def test_J3_floor_preserved_on_resize_rebuild(self):
        g = nb.Graph(60, floor=8)
        g.add(3.0)
        g2 = nb.Graph(40, floor=g.floor, pinned=g.pinned)
        g2.hist.extend(list(g.hist)[-40:])
        self.assertEqual(g2.floor, 8.0)
        self.assertEqual(list(g2.hist), list(g.hist)[-40:])

    def test_J4_dots_pinned_to_plot_height(self):
        # pinned: max = the plot's total dot count (4*rows) - 1 dot = 1
        # request, a burst above the cap clips at the top
        self.assertEqual(nb.compute_dots([1.0, 2.0, 8.0], max=8, total_dots=8), [1, 2, 8])
        self.assertEqual(nb.compute_dots([1.0, 2.0, 11.0], max=8, total_dots=8), [1, 2, 8])
        g = nb.Graph(10, floor=4, pinned=True)
        for v in (1.0, 2.0, 8.0):
            g.add(v)
        self.assertEqual(g.dots(2), [1, 2, 8])
        # plain graphs autoscale to the window max
        g0 = nb.Graph(10)
        g0.add(1.0)
        g0.add(3.0)
        self.assertEqual(g0.dots(1), [2, 4])
        # floor no longer touches the scale of plain graphs (it only sizes
        # the batch panel); autoscale still takes the window max
        gp2 = nb.Graph(10, floor=4)
        gp2.add(1.0)
        gp2.add(3.0)
        self.assertEqual(gp2.dots(1), [2, 4])
        # ceil: any non-zero sample draws at least one dot; Nones stay blank
        self.assertEqual(nb.compute_dots([0.1, 1.0]), [1, 4])
        self.assertEqual(nb.compute_dots([None, 0.0, 2.0]), [0, 0, 4])


# ------------------------------------- I. pricing loader + row selection

class TestPricingLoader(Base):
    """The hand-rolled pricing.yaml loader, key selection, and the error path."""

    SAMPLE = textwrap.dedent(
        """\
        # a full-line comment that must be skipped
        qwen3.8-27b:
          input: 0.5
          cached_input: 0.05
          output: 3.0

        llama-3-8b:
          input: 0.2
          cached_input: 0.02
          output: 1.2
        """
    )

    def _write_yaml(self, text):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        return path

    def test_I1_loader_parses_sample_yaml(self):
        path = self._write_yaml(self.SAMPLE)
        try:
            p = nb.load_pricing(path)
        finally:
            os.unlink(path)
        self.assertEqual(p, {
            "qwen3.8-27b": {"input": 0.5, "cached_input": 0.05, "output": 3.0},
            "llama-3-8b": {"input": 0.2, "cached_input": 0.02, "output": 1.2},
        })
        for row in p.values():  # every parsed rate is a real float
            for v in row.values():
                self.assertIsInstance(v, float)

    def test_I2_missing_file_falls_back_to_default_row(self):
        p = nb.load_pricing("/no/such/pricing.yaml")
        self.assertEqual(set(p), {"qwen3.8-27b"})
        self.assertEqual(p["qwen3.8-27b"],
                         {"input": 0.5, "cached_input": 0.05, "output": 3.0})

    def test_I3_key_indexing_selects_the_right_row(self):
        path = self._write_yaml(self.SAMPLE)
        try:
            p = nb.load_pricing(path)
        finally:
            os.unlink(path)
        self.assertEqual(p["qwen3.8-27b"],
                         {"input": 0.5, "cached_input": 0.05, "output": 3.0})
        self.assertEqual(p["llama-3-8b"],
                         {"input": 0.2, "cached_input": 0.02, "output": 1.2})
        # selecting a row and feeding it to Engine actually changes cloud_cost
        body = ("ninfer_engine_ready 1\n"
                "ninfer_prompt_tokens_total 1000000\n"
                "ninfer_prompt_tokens_cached_total 100000\n"
                "ninfer_generation_tokens_total 500000")
        fake_metrics(body)
        s_def = nb.Engine("http://x/metrics", 0.5).poll()
        s_llm = nb.Engine("http://x/metrics", 0.5, p["llama-3-8b"]).poll()
        self.assertNotAlmostEqual(s_def["cloud_cost"], s_llm["cloud_cost"])
        want = ((1000000 - 100000) / 1e6 * 0.2
                + 100000 / 1e6 * 0.02
                + 500000 / 1e6 * 1.2)
        self.assertAlmostEqual(s_llm["cloud_cost"], want, places=6)

    def test_I4_missing_key_errors_and_lists_available(self):
        path = self._write_yaml(self.SAMPLE)
        sys.argv = ["ninfer_btop.py", "--url", "http://x",
                    "--text", "--pricing-file", path,
                    "--pricing-model", "does-not-exist"]
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    nb.main()
            self.assertEqual(ctx.exception.code, 2)
            msg = err.getvalue()
            self.assertIn("does-not-exist", msg)
            self.assertIn("qwen3.8-27b", msg)
            self.assertIn("llama-3-8b", msg)
        finally:
            os.unlink(path)

    def test_I5_bare_engine_uses_default_rates(self):
        eng = nb.Engine("http://x/metrics", 0.5)  # no rates supplied
        self.assertEqual(eng.rates,
                         {"input": 0.5, "cached_input": 0.05, "output": 3.0})
        self.assertEqual(eng.rates, dict(nb.DEFAULT_PRICING["qwen3.8-27b"]))
        # the module aliases still resolve to that default row's values
        row = nb.DEFAULT_PRICING["qwen3.8-27b"]
        self.assertEqual(nb.CLOUD_IN_RATE, row["input"])
        self.assertEqual(nb.CLOUD_CACHE_RATE, row["cached_input"])
        self.assertEqual(nb.CLOUD_OUT_RATE, row["output"])

    def test_I6_real_pricing_yaml_has_the_supported_models(self):
        # I1-I5 exercise the parser on a synthetic sample; this guards the file
        # actually shipped in the repo. Rates are the International-scope Alibaba
        # Cloud Model Studio list prices, verified 2026-10-15 - update this
        # alongside pricing.yaml if those change.
        self.assertTrue(os.path.exists(nb.pricing_file_path()),
                        "pricing.yaml is missing next to the script")
        p = nb.load_pricing()  # resolves pricing.yaml next to ninfer_btop.py
        # the real file was found and parsed (not the 1-key built-in fallback),
        # and it holds exactly the three models the ninfer engine ships
        self.assertEqual(set(p),
                         {"qwen3.8-27b", "qwen3.6-27b", "qwen3.6-35b-a3b"})
        self.assertIn("qwen3.8-27b", p)  # the default --pricing-model key exists
        # every row is a complete {input, cached_input, output} float triple
        for key, row in p.items():
            self.assertEqual(set(row), {"input", "cached_input", "output"}, key)
            for v in row.values():
                self.assertIsInstance(v, float)
        # lock the verified rates so a stray edit to the file is caught
        self.assertEqual(p["qwen3.8-27b"],
                         {"input": 0.5, "cached_input": 0.05, "output": 3.0})
        self.assertEqual(p["qwen3.6-27b"],
                         {"input": 0.6, "cached_input": 0.06, "output": 3.6})
        self.assertEqual(p["qwen3.6-35b-a3b"],
                         {"input": 0.375, "cached_input": 0.0375, "output": 2.25})


# ------------------------------- K. calibration cache (~/.ninfer_btop.json)

class TestCalibCache(Base):
    """K: the startup calibration is cached in ~/.ninfer_btop.json and
    reused only while the engine's URL and server start time are unchanged;
    --re-calib drops the cache and always benchmarks."""

    URL = "http://x/metrics"
    GOOD = {"ok": True, "model": "m", "concurrency": 4,
            "prefill_tps": 100.0, "decode_tps": 50.0, "duration": 10.0,
            "server_start_time": 1000.0}

    def _cache_file(self):
        nb.CALIB_FILE = tempfile.mktemp(suffix=".json")
        return nb.CALIB_FILE

    def _write_cache(self, url, start, path):
        with open(path, "w") as f:
            json.dump({"url": url, "server_start_time": start, "model": "m",
                       "concurrency": 4, "prefill_tps": 100.0,
                       "decode_tps": 50.0, "duration": 10.0}, f)

    def test_K1_no_file_benchmarks_and_caches(self):
        path = self._cache_file()
        nb.resolve_calib(self.URL, run=lambda: setattr(nb, "CALIB", dict(self.GOOD)),
                         probe=lambda u: 1000.0)
        self.assertEqual(nb.CALIB, self.GOOD)
        j = json.load(open(path))
        self.assertEqual(j["url"], self.URL)
        self.assertEqual(j["server_start_time"], 1000.0)
        self.assertEqual(j["concurrency"], 4)

    def test_K2_matching_start_time_reuses_cache_without_benchmark(self):
        path = self._cache_file()
        self._write_cache(self.URL, 1000.0, path)
        ran = []
        nb.resolve_calib(self.URL, run=lambda: ran.append(1), probe=lambda u: 1000.0)
        self.assertEqual(ran, [])
        self.assertTrue(nb.CALIB["ok"])
        self.assertEqual(nb.CALIB["concurrency"], 4)
        self.assertNotIn("server_start_time", nb.CALIB)  # cache dict shape

    def test_K3_restarted_engine_benchmarks_and_rewrites_cache(self):
        path = self._cache_file()
        self._write_cache(self.URL, 1000.0, path)
        def run():
            c = dict(self.GOOD)
            c["server_start_time"] = 2000.0
            nb.CALIB = c
        nb.resolve_calib(self.URL, run=run, probe=lambda u: 2000.0)
        self.assertEqual(nb.CALIB["server_start_time"], 2000.0)
        self.assertEqual(json.load(open(path))["server_start_time"], 2000.0)

    def test_K4_re_calib_drops_cache_and_always_benchmarks(self):
        path = self._cache_file()
        self._write_cache(self.URL, 1000.0, path)
        ran = []
        def run():
            ran.append(1)
            nb.CALIB = dict(self.GOOD)
        # start time still matches - only the flag forces the fresh run
        nb.resolve_calib(self.URL, re_calib=True, run=run, probe=lambda u: 1000.0)
        self.assertEqual(ran, [1])
        # the dropped cache is rewritten with the fresh result
        self.assertEqual(json.load(open(path))["url"], self.URL)

    def test_K5_different_url_benchmarks(self):
        path = self._cache_file()
        self._write_cache("http://y/metrics", 1000.0, path)
        ran = []
        nb.resolve_calib(self.URL, run=lambda: ran.append(1) or setattr(nb, "CALIB", {"ok": False}),
                         probe=lambda u: 1000.0)
        self.assertEqual(ran, [1])

    def test_K6_failed_calibration_is_not_cached(self):
        path = self._cache_file()
        nb.resolve_calib(self.URL,
                         run=lambda: setattr(nb, "CALIB", {"ok": False, "error": "boom"}),
                         probe=lambda u: None)
        self.assertFalse(os.path.exists(path))

    def test_K7_undeletable_cache_aborts(self):
        # a directory where the cache file should be: os.remove raises
        # IsADirectoryError - must abort, not leave the stale cache in place
        d = tempfile.mkdtemp()
        nb.CALIB_FILE = d
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                nb.resolve_calib(self.URL, re_calib=True,
                                 run=lambda: setattr(nb, "CALIB", dict(self.GOOD)),
                                 probe=lambda u: 1000.0)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("cannot remove", err.getvalue())

    def test_K8_unwritable_cache_aborts_after_benchmark(self):
        d = tempfile.mkdtemp()
        nb.CALIB_FILE = os.path.join(d, "no", "such", "cache.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                nb.resolve_calib(self.URL,
                                 run=lambda: setattr(nb, "CALIB", dict(self.GOOD)),
                                 probe=lambda u: 1000.0)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("cannot write", err.getvalue())

    def test_K9_unreadable_cache_aborts_before_benchmark(self):
        d = tempfile.mkdtemp()
        nb.CALIB_FILE = d  # open-for-read on a directory raises IsADirectoryError
        ran = []
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                nb.resolve_calib(self.URL, run=lambda: ran.append(1),
                                 probe=lambda u: 1000.0)
        self.assertEqual(ctx.exception.code, 1)
        self.assertEqual(ran, [])  # no benchmark spent on a doomed run
        self.assertIn("cannot read", err.getvalue())

    def test_K10_corrupt_cache_self_heals(self):
        path = self._cache_file()
        with open(path, "w") as f:
            f.write("not json")
        ran = []
        def run():
            ran.append(1)
            nb.CALIB = dict(self.GOOD)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            nb.resolve_calib(self.URL, run=run, probe=lambda u: 1000.0)
        self.assertEqual(ran, [1])
        self.assertIn("not valid JSON", err.getvalue())
        j = json.load(open(path))  # the fresh run rewrote the corrupt file
        self.assertEqual(j["url"], self.URL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
