"""Unit tests for ninfer_btop (standard library only).

No network, no engine, no real terminal: urllib is faked to serve
deterministic /metrics bodies, and the curses screen is a recorder.

Run:  python3 -m unittest -v test_ninfer_btop
"""
import array
import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap
import time
import types
import unittest

import bar_graphs as bg
import tui
import ninfer_engine_state as nes
import metrics_calib as mc
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
    mc.urllib.request.urlopen = _urlopen


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
    e = nes.NinferEngineState("http://x/metrics", 0.5)
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
        self._real_urlopen = mc.urllib.request.urlopen
        self._real_calib_file = mc.CALIB_FILE
        mc.set_calib(None)
        stub_curses()

    def tearDown(self):
        mc.urllib.request.urlopen = self._real_urlopen
        mc.CALIB_FILE = self._real_calib_file
        mc.set_calib(None)

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


# ------------------------------------- S. period/agg selector (phase 1)

class _LoopScreen(FakeScreen):
    """FakeScreen + scripted getch; snapshots each rendered frame's writes."""

    def __init__(self, keys, maxy=40, maxx=120):
        super().__init__(maxy, maxx)
        self.keys = list(keys)
        self.frames = []

    def curs_set(self, v):
        pass

    def nodelay(self, v):
        pass

    def keypad(self, v):
        pass

    def getch(self):
        return self.keys.pop(0) if self.keys else -1

    def erase(self):
        if self.writes:
            self.frames.append(self.writes)
        self.writes = []

    def rendered_frames(self):
        return self.frames + ([self.writes] if self.writes else [])


def _footer_row(scr, frame):
    """(x, text, attr) writes on the footer row (row maxy-1) of one frame."""
    y = scr.maxy - 1
    return [(x, t, a) for (yy, x, t, a) in frame if yy == y]


def _run_loop(keys, has_color=False):
    """Drive tui.main_loop to completion with scripted keys; (scr, frames)."""
    if has_color:
        nb.curses.curs_set = lambda v: None
        nb.curses.has_colors = lambda: True
        nb.curses.start_color = lambda: None
        nb.curses.use_default_colors = lambda: None
        nb.curses.init_pair = lambda i, fg, bg: None
        nb.curses.color_pair = lambda i: 1 + i   # distinct, inspectable pair ids
    else:
        nb.curses.curs_set = lambda v: None
        nb.curses.has_colors = lambda: False
        nb.curses.color_pair = lambda i: 0
    scr = _LoopScreen(keys)
    args = types.SimpleNamespace(interval=500, url="http://x/metrics",
                                 pricing_rates=None, pricing_model=None)
    rc = tui.main_loop(scr, args)
    assert rc == 0
    return scr, scr.rendered_frames()


class TestPeriodAggSelector(Base):
    """S: main_loop period/agg keys + footer selector rendering (phase 1).

    Phase 1 draws live data for every period value - the tests only
    assert the selector state/rendering, not the plotted data.
    """

    BODY = "ninfer_engine_ready 1\n"

    def _run(self, keys, has_color=False):
        fake_metrics(self.BODY)
        return _run_loop(keys, has_color)

    def test_S1_key_tables_cover_exactly_the_planned_keys(self):
        self.assertEqual(set(tui.PERIOD_KEYS.values()), set("nmhHdwl"))
        self.assertEqual(set(tui.AGG_KEYS.values()), set("AM"))
        quit_keys = {ord("q"), ord("Q"), tui.KEY_ESCAPE}
        self.assertFalse(set(tui.PERIOD_KEYS) & quit_keys)
        self.assertFalse(set(tui.AGG_KEYS) & quit_keys)
        self.assertEqual([k for k, _ in tui.PERIOD_LABELS], list("nmhHdwl"))
        self.assertEqual([k for k, _ in tui.AGG_LABELS], list("AM"))

    def test_S2_selector_segments_underline_only_the_active_entries(self):
        nb.curses.color_pair = lambda i: 1 + i
        c = {"dim": 1, "label": 8}
        segs = tui.selector_segments("h", "M", c)
        text = "".join(t for t, _ in segs)
        self.assertEqual(text, "  period=now 5m 30m 1h 1d 1w life  agg fn=Avg Max")
        active = [(t, a) for t, a in segs if a != nb.curses.color_pair(1)]
        # period "h" = 30 minutes; only the matching period entry + agg entry are active
        self.assertEqual([t.strip() for t, _ in active], ["30m", "Max"])
        want = nb.curses.color_pair(8) | nb.curses.A_UNDERLINE
        self.assertTrue(all(a == want for _, a in active))

    def test_S3_selector_falls_back_to_bold_without_colors(self):
        segs = tui.selector_segments("w", "A", {})
        active = [a for _, a in segs if a != nb.curses.color_pair(0)]
        # one active entry per dimension: period "w" + agg "A"
        self.assertEqual(active, [nb.curses.A_BOLD, nb.curses.A_BOLD])

    def test_S4_period_key_m_moves_the_active_marker(self):
        scr, frames = self._run([ord("m"), ord("q")])
        first, last = frames[0], frames[-1]

        def active_labels(frame):
            return {t.strip() for (x, t, a) in _footer_row(scr, frame)
                    if a == nb.curses.A_BOLD}
        self.assertIn("now", active_labels(first))       # default period
        self.assertNotIn("now", active_labels(last))
        self.assertIn("5m", active_labels(last))
        for lab in ("30m", "1h", "1d", "1w", "life"):
            self.assertNotIn(lab, active_labels(last))

    def test_S5_all_period_keys_switch_the_selection(self):
        labels = {"n": "now", "m": "5m", "h": "30m", "H": "1h",
                  "d": "1d", "w": "1w", "l": "life"}
        for k in "nmhHdwl":
            scr, frames = self._run([ord(k), ord("q")])
            last = frames[-1]
            act = [t.strip() for (x, t, a) in _footer_row(scr, last)
                   if a == nb.curses.A_BOLD and t.strip() in set(labels.values())]
            self.assertEqual(act, [labels[k]], f"key {k!r}")

    def test_S6_agg_keys_M_A_toggle_the_active_agg(self):
        scr, frames = self._run([ord("M"), ord("A"), ord("q")])

        def active_agg(frame):
            out = [t.strip() for (x, t, a) in _footer_row(scr, frame)
                   if a == nb.curses.A_BOLD and t.strip() in ("Avg", "Max")]
            self.assertEqual(len(out), 1)
            return out[0]
        self.assertEqual(active_agg(frames[0]), "Max")   # default agg
        self.assertEqual(active_agg(frames[1]), "Max")
        self.assertEqual(active_agg(frames[2]), "Avg")

    def test_S7_color_run_underlines_active_with_the_label_pair(self):
        scr, frames = self._run([ord("H"), ord("q")], has_color=True)
        last = frames[-1]
        # make_colors pair order: title=1 ok=2 bad=3 batch=4 toks=5
        # stat_batch=6 stat_toks=7 label=8 dim=9 bar_dim=10 val=11 kv=12 peak=13
        want = nb.curses.color_pair(8) | nb.curses.A_UNDERLINE
        self.assertEqual([a for (x, t, a) in _footer_row(scr, last) if t.strip() == "1h"], [want])
        self.assertNotIn(want, [a for (x, t, a) in _footer_row(scr, last) if t.strip() == "now"])

    def test_S8_frame_without_sel_keeps_the_old_footer(self):
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        s = self._sample()
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = s
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        tui.frame(scr, eng, graphs, c)
        joined = "".join(t for (_, _, t, _) in scr.writes)
        self.assertNotIn("period=", joined)
        self.assertNotIn("agg fn=", joined)

    def test_S9_narrow_terminal_truncates_the_selector_tail_first(self):
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        # no cloud cost: the footer stays short so the selector head reaches the right edge
        s = self._sample(cloud_cost=None)
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = s
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(30, 45)
        tui.frame(scr, eng, graphs, c, None, ("w", "M"))
        foot = [t for (y, x, t, a) in scr.writes if y == scr.maxy - 1]
        self.assertTrue(any(t.startswith(" q quit") for t in foot))
        self.assertTrue(any("UTIL" in t for t in foot))
        joined = "".join(foot)
        self.assertIn("now", joined)          # selector head survives
        self.assertNotIn("agg fn=", joined)     # selector tail truncates first


class TestPlotSamplesWiring(Base):
    """T: draw_bar_graph sources main-graph samples from Graph.plot_samples
    (phase 2): the three main graphs plot the selected period/agg, the xfer
    graphs keep the live 13-char window, compute_dots/dots_to_chars are
    untouched. Compressed non-pinned views scale against the raw window
    peak (graph.plot_peak) - one scale shared by both aggs, so the Avg
    dilution is visible on the plot; pinned views and the live view keep
    graph.dot_scale.
    """

    def _spiky_graph(self):
        """4600 samples @0.5s: 1000 zeros, then a 30-minute region holding
        one 10.0 spike every 50 samples - every 52/53-sample 'h' bucket
        contains a spike, so the window max is 10.0, the same as the
        lifetime peak (the scale choice cannot be seen from this graph;
        test_T4 drives that distinction).
        """
        g = bg.Graph(60, interval=0.5)
        for _ in range(1000):
            g.add(0.0)
        for i in range(3600):
            g.add(10.0 if i % 50 == 0 else 0.0)
        return g

    def _frame(self, agg):
        c = {"toks": 1, "stat_toks": 2, "dim": 3}
        scr = FakeScreen(10, 41)   # width 40 -> plot_len 34 -> 68 buckets
        y = tui.draw_bar_graph(scr, 0, self._spiky_graph(), 1, 40, "toks", c,
                               period="h", agg=agg)
        self.assertEqual(y, 2)     # 1 plot row + baseline, returned next row
        return scr

    def test_T1_compressed_frame_agg_max_draws_full_buckets(self):
        scr = self._frame("M")
        plot = [t for (yy, x, t, a) in scr.writes if yy == 0 and x <= 34]
        self.assertEqual(plot, ["⣿"] * 34)
        base = "".join(t for (yy, x, t, a) in scr.writes if yy == 1)
        self.assertIn("30 minutes", base)   # the real selected span
        self.assertIn("MhHDWL", base)       # period columns untouched

    def test_T2_agg_avg_measures_against_the_raw_window_peak(self):
        # the A-agg view scales against the RAW window peak
        # (plot_peak = 10.0, the spike) - the same scale T1's M view uses,
        # not the plotted max. The diluted bucket averages (~0.19) sit on
        # the 1-dot floor: that floor is the dilution, visible against
        # T1's full M bars (switching Avg<->Max changes the bar heights).
        scr = self._frame("A")
        plot = [t for (yy, x, t, a) in scr.writes if yy == 0 and x <= 34]
        self.assertEqual(plot, ["⣀"] * 34)         # 1 dot per bucket
        g = self._spiky_graph()
        self.assertEqual(g.plot_peak("h"), 10.0)    # the shared scale
        vals = g.plot_samples("h", "A", 68)
        self.assertEqual(set(bg.compute_dots(vals,
                                             max=g.plot_peak("h"),
                                             total_dots=4)),
                         {1})  # every diluted bucket lands on the floor

    def test_T4_burst_outside_the_window_is_excluded_from_the_scale(self):
        """6.1k samples @0.5s: a 3000.0 burst older than the 30-minute
        window, steady 100/300 traffic inside it. The scale is the raw
        WINDOW peak (plot_peak = 300): the burst outside the window is
        excluded from it, so the steady traffic keeps its full 2-4 dot
        range; scaled to the lifetime peak (3000) every steady bucket
        would land on the 1-dot floor (the frozen line a past burst left
        behind).
        """
        g = bg.Graph(60, interval=0.5)
        def chunk(n, off=0):
            for i in range(n):
                g.add(300.0 if ((i + off) // 500) % 2 else 100.0)
        chunk(2000)
        for _ in range(100):
            g.add(3000.0)
        chunk(4000, 250)
        c = {"toks": 1, "stat_toks": 2, "dim": 3}
        scr = FakeScreen(10, 41)
        tui.draw_bar_graph(scr, 0, g, 1, 40, "toks", c, period="h", agg="A")
        plot = [t for (yy, x, t, a) in scr.writes if yy == 0 and x <= 34]
        self.assertEqual(g.plot_peak("h"), 300.0)  # burst outside: excluded
        self.assertEqual(plot,
                         ["⣤"] * 2 + ["⣾"] + ["⣿"] * 4 + ["⣤"] * 4
                         + ["⣴"] + ["⣿"] * 4 + ["⣧"] + ["⣤"] * 4
                         + ["⣾"] + ["⣿"] * 4 + ["⣤"] * 4 + ["⣼"]
                         + ["⣿"] * 3)
        self.assertGreater(len(set(plot)), 1)      # not a flat line
        self.assertNotIn("⣀", plot)                # no 1-dot floor
        vals = g.plot_samples("h", "A", 68)
        self.assertEqual(bg.compute_dots(vals,
                                         max=g.dot_scale(1),
                                         total_dots=4),
                         [1] * len(vals))  # the lifetime-peak scale would
                                           # freeze the window at one dot

    def test_T5_pinned_graph_stays_pinned_under_aggregation(self):
        # pinned graphs map against the plot's total dot count (one dot =
        # one request), never against the plotted data: 2.0 steady traffic
        # on a 4-dot plot draws 2 dots - scaling to the window peak would
        # have drawn 4. The fixed axis also makes the young buffer's
        # history visible: only the buckets the 200 samples cover draw,
        # the rest stay empty instead of stretching across the width.
        g = bg.Graph(60, pinned=True, interval=0.5)
        for _ in range(200):
            g.add(2.0)
        c = {"toks": 1, "stat_toks": 2, "dim": 3}
        scr = FakeScreen(10, 41)
        tui.draw_bar_graph(scr, 0, g, 1, 40, "toks", c, period="h", agg="A")
        plot = [t for (yy, x, t, a) in scr.writes if yy == 0 and x <= 34]
        self.assertEqual(plot, ["⣤"] * 2)   # 2/4 dots, only covered buckets
        self.assertNotIn("⣿", plot)         # no full-height window-peak bars

    def test_T3_unselected_path_is_the_pre_phase2_draw(self):
        c = {"toks": 1, "stat_toks": 2, "dim": 3}
        g = bg.Graph(60, interval=0.5)
        for v in (0.0, 10.0, 0.0, 0.0, 20.0):
            g.add(v)
        scr = FakeScreen(10, 41)
        tui.draw_bar_graph(scr, 0, g, 1, 40, "toks", c)   # period=None
        plot = [t for (yy, x, t, a) in scr.writes if yy == 0 and x <= 34]
        # raw hist tail newest-first: [20, 0, 0, 10, 0] -> (4,0) (0,2) (0,0);
        # the newer sample sits in the left dot line of its character
        self.assertEqual(plot, ["⡇", "⢠"])
        base = "".join(t for (yy, x, t, a) in scr.writes if yy == 1)
        self.assertIn("34 seconds", base)                 # width-derived span
        self.assertNotIn("30 minutes", base)

    def test_T6_switching_agg_changes_bar_heights(self):
        # the user symptom: with the old per-agg rescale (scale = plotted
        # max) the Avg and Max plots looked identical, so switching the
        # agg changed nothing visible. Against the shared raw window peak
        # (plot_peak = 50.0, the spike) the young buffer's one spike draws
        # a full M bar and a 1-dot A bar (50/53 diluted) at the same spot.
        g = bg.Graph(60, interval=0.5)
        for _ in range(59):
            g.add(0.0)
        g.add(50.0)
        c = {"toks": 1, "stat_toks": 2, "dim": 3}
        scr_a = FakeScreen(10, 41)
        tui.draw_bar_graph(scr_a, 0, g, 1, 40, "toks", c, period="h", agg="A")
        scr_m = FakeScreen(10, 41)
        tui.draw_bar_graph(scr_m, 0, g, 1, 40, "toks", c, period="h", agg="M")
        plot_a = [t for (yy, x, t, a) in scr_a.writes if yy == 0 and x <= 34]
        plot_m = [t for (yy, x, t, a) in scr_m.writes if yy == 0 and x <= 34]
        self.assertNotEqual(plot_a, plot_m)   # the agg is visible
        self.assertEqual(plot_m, ["⡇"])      # M: 50.0 on the 50.0 peak = full
        self.assertEqual(plot_a, ["⡀"])      # A: 50/53 diluted = 1 dot
    def test_T7_life_period_draws_trimmed_history(self):
        # 'l' over the coarse lifetime hour table: hist_cap=100 trims the
        # raw ring fast, so hours 0/1 (constants 10/20, poison spike 5000
        # inside hour 0) have aged OUT of the ring into (sum, max) rows,
        # and the newest bars (hour 2, 30.0) sit in the buffer. The axis
        # is [0, seq) = 7400 samples -> 68 buckets (sizes 108/109); hour
        # rows are HOUR_SAMPLES = 3600 samples wide at interval 1.0, so
        # every bucket overlapping hour 0's row - buckets 0..33, incl. the
        # one straddling the hour 0/1 border - takes the row max 5000.0 in
        # M: full height at the oldest side. Every hour 1/2 bucket (20/30)
        # lands on the 1-dot floor against the shared peak plot_peak('l')
        # = 5000.0, which lives ONLY in the hour table (the ring holds
        # 30.0 only) - the recent 30-unit hour is proportionally shorter.
        # A on the same scale dilutes hour 0 to its row rate
        # hour_sum/HOUR_SAMPLES = 40990/3600 ~ 11.39: every bucket
        # (11.39..30) sits on the 1-dot floor, visibly shorter than M.
        # 'w' cannot see any of it: its source is the kept buffer, so the
        # same graph draws only the newest 100 samples (30.0, full on its
        # own plot_peak('w') = 30.0) - one character, no older traffic.
        g = bg.Graph(60, interval=1.0, hist_cap=100)
        for i in range(3600):        # hour 0: 10.0 + one poison spike
            g.add(5000.0 if i == 150 else 10.0)
        for _ in range(3600):        # hour 1: 20.0
            g.add(20.0)
        for _ in range(200):         # hour 2 (open): 30.0, newest in ring
            g.add(30.0)
        self.assertEqual((g.seq, len(g.samples),
                          g.seq - len(g.samples)),
                         (7400, 100, 7300))
        self.assertEqual(max(g.samples), 30.0)    # spike aged out of the ring
        self.assertEqual(g.plot_peak("l"), 5000.0)  # only the hour table has it
        c = {"toks": 1, "stat_toks": 2, "dim": 3}
        scr_m = FakeScreen(10, 41)
        tui.draw_bar_graph(scr_m, 0, g, 1, 40, "toks", c, period="l", agg="M")
        scr_a = FakeScreen(10, 41)
        tui.draw_bar_graph(scr_a, 0, g, 1, 40, "toks", c, period="l", agg="A")
        plot_m = [t for (yy, x, t, a) in scr_m.writes if yy == 0 and x <= 34]
        plot_a = [t for (yy, x, t, a) in scr_a.writes if yy == 0 and x <= 34]
        # M: oldest side full (row max = the 'l' peak), rest on the floor
        self.assertEqual(plot_m, ["⣀"] * 17 + ["⣿"] * 17)
        self.assertNotEqual(plot_a, plot_m)  # row-level dilution is visible
        self.assertEqual(plot_a, ["⣀"] * 34)
        v_a = g.plot_samples("l", "A", 68)
        self.assertAlmostEqual(v_a[0], 40990.0 / 3600, places=3)  # row rate
        # the axis is 'l'-wide: older-than-buffer traffic is visible in 'l'
        # only, not in 'w'
        scr_w = FakeScreen(10, 41)
        tui.draw_bar_graph(scr_w, 0, g, 1, 40, "toks", c, period="w", agg="M")
        plot_w = [t for (yy, x, t, a) in scr_w.writes if yy == 0 and x <= 34]
        self.assertNotEqual(plot_m, plot_w)
        self.assertEqual(plot_w, ["⡇"])   # 'w' draws the 100-sample ring only
    def test_T8_life_span_label_annotates_the_run_duration(self):
        # "l" is the one span label that depends on the graph's state: on
        # the axis rule it renders "this run (x D y H z M)" with duration =
        # life_count x interval (the L column's sample count), always all
        # three components, exact spacing. The static labels ("5 minutes"
        # .. "1 week") are unchanged, and the I7 centering math applies to
        # the longer label (at width 40 the 34-char plot region holds the
        # longest realistic label with room to spare).
        c = {"toks": 1, "stat_toks": 2, "dim": 3}

        def rule(scr):
            # the full axis-rule rendering in write order: the "─" fill,
            # the centered span label, the MhHDWL legend
            return "".join(t for (yy, x, t, a) in scr.writes if yy == 1)

        g = bg.Graph(60, interval=1.0)
        for _ in range(4500):        # 4500 s = 75 min at a 1 s poll
            g.add(5.0)
        self.assertEqual(g.life_count, 4500)
        scr = FakeScreen(10, 41)
        tui.draw_bar_graph(scr, 0, g, 1, 40, "toks", c, period="l", agg="M")
        self.assertEqual(rule(scr), "─" * 34 + "this run (0 D 1 H 15 M)"
                         + "MhHDWL")
        lab = [w for w in scr.writes if w[2] == "this run (0 D 1 H 15 M)"]
        self.assertEqual(len(lab), 1)   # centered like every other span
        self.assertEqual(lab[0][1], 1 + (34 - len(lab[0][2])) // 2)
        for _ in range(86880 - 4500):   # 86880 s = 1 d 0 h 8 m
            g.add(5.0)
        scr2 = FakeScreen(10, 41)
        tui.draw_bar_graph(scr2, 0, g, 1, 40, "toks", c, period="l", agg="M")
        self.assertEqual(rule(scr2), "─" * 34 + "this run (1 D 0 H 8 M)"
                         + "MhHDWL")
        # the static span labels are untouched by the annotation
        scr3 = FakeScreen(10, 41)
        tui.draw_bar_graph(scr3, 0, g, 1, 40, "toks", c, period="h", agg="M")
        self.assertIn("30 minutes", rule(scr3))




# --------------------------------------------- A. fetch_metrics label parsing

class TestFetchMetricsLabels(Base):
    def test_A1_escaped_label_values_are_unescaped(self):
        body = r'ninfer_model_info{model_name="custom\"model\\name\nline",speculative_backend="eagle"} 1'
        fake_metrics(body)
        m = mc.fetch_metrics("http://x/metrics")
        lbl = m["ninfer_model_info__labels"]
        self.assertEqual(lbl["model_name"], 'custom"model\\name\nline')
        self.assertEqual(lbl["speculative_backend"], "eagle")

    def test_A2_comma_inside_value_does_not_split(self):
        body = 'ninfer_model_info{model_name="my, model",speculative_backend="none"} 1'
        fake_metrics(body)
        m = mc.fetch_metrics("http://x/metrics")
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
        m = mc.fetch_metrics("http://x/metrics")
        self.assertEqual(m["ninfer_generation_tokens_total"], 123456789.0)
        self.assertEqual(m["ninfer_engine_ready"], 1.0)
        self.assertEqual(m["ninfer_requests_total"], 400.0)
        self.assertEqual(m["ninfer_requests_total__labels"]["outcome"], "completed")

    def test_A4_comment_blank_and_nonnumeric_are_skipped(self):
        body = "# TYPE ninfer_weird counter\n\nninfer_good 5\nninfer_notanum abc"
        fake_metrics(body)
        m = mc.fetch_metrics("http://x/metrics")
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
        real_chat = mc._chat_first_chunk

        def fake_chat(api, model, prompt, max_tokens, timeout=120):
            sent.append(model)
            return (time.monotonic(), 0.1)

        mc._chat_first_chunk = fake_chat
        try:
            mc.calibrate("http://x/metrics", log=lambda *a: None, burst=1)
        finally:
            mc._chat_first_chunk = real_chat
        self.assertIn("my, model", sent)
        self.assertNotIn("my", sent)


# --------------------------------------------- C. Engine.poll inter-poll rates

class TestEnginePollRate(Base):
    def poll(self, eng, body):
        fake_metrics(body)
        return eng.poll()

    def test_C1_rate_produced_below_the_old_250ms_guard(self):
        eng = nes.NinferEngineState("http://x/metrics", interval=0.10)
        self.poll(eng, "ninfer_generation_tokens_total 1000\nninfer_engine_ready 1")
        time.sleep(0.10)
        s = self.poll(eng, "ninfer_generation_tokens_total 1100\nninfer_engine_ready 1")
        self.assertIsNotNone(s["toks_per_s"])
        self.assertGreater(s["toks_per_s"], 400)
        self.assertLess(s["toks_per_s"], 1600)

    def test_C2_guard_tracks_the_configured_interval(self):
        eng = nes.NinferEngineState("http://x/metrics", interval=0.05)
        self.poll(eng, "ninfer_generation_tokens_total 1000\nninfer_engine_ready 1")
        time.sleep(0.05)
        self.poll(eng, "ninfer_generation_tokens_total 1050\nninfer_engine_ready 1")
        time.sleep(0.05)
        s = self.poll(eng, "ninfer_generation_tokens_total 1100\nninfer_engine_ready 1")
        self.assertIsNotNone(s["toks_per_s"])

    def test_C3_flat_or_reset_window_is_never_negative(self):
        eng = nes.NinferEngineState("http://x/metrics", interval=0.05)
        self.poll(eng, "ninfer_generation_tokens_total 1000\nninfer_engine_ready 1")
        time.sleep(0.06)
        s = self.poll(eng, "ninfer_generation_tokens_total 1000\nninfer_engine_ready 1")
        self.assertTrue(s["toks_per_s"] is None or s["toks_per_s"] >= 0)


# --------------------------------------------- D. frame() down-state layout

class TestFrameDownState(Base):
    def frame(self, calib, ok):
        mc.set_calib(calib)
        scr = FakeScreen(40, 120)
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        tui.frame(scr, make_engine(ok), graphs, {}, calib)
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
        tui.draw_panel(scr, 0, "Token totals", "title", {}, width=40)
        row = [t for (y, x, t, a) in scr.writes if y == 0][0]
        self.assertTrue(row.startswith("─Token totals"), row)
        scr2 = FakeScreen(40, 120)
        tui.draw_panel(scr2, 0, "ABCDEFGHIJKLMNOP", "title", {}, width=18)
        row2 = [t for (y, x, t, a) in scr2.writes if y == 0][0]
        self.assertEqual(row2, "─ABCDEFGHIJKLMNOP")

    def test_E2_human_and_money_boundaries(self):
        for n, want in ((0, "0"), (999, "999"), (1000, "1.00K"),
                        (1000000, "1.00M"), (1200000000, "1.20G"),
                        (2500000000000, "2.50T")):
            self.assertEqual(mc.human(n), want, "human(%r)" % n)
        for v, want in ((0, "$0.00"), (0.42, "$0.42"), (9.99, "$9.99"),
                        (1000, "$1,000"), (1200000, "$1.2M")):
            self.assertEqual(mc.money(v), want, "money(%r)" % v)

    def test_E3_cloud_cost_math(self):
        prompt_total, cached, gen = 1000000, 100000, 500000
        body = (
            "ninfer_engine_ready 1\n"
            "ninfer_prompt_tokens_total %d\n"
            "ninfer_prompt_tokens_cached_total %d\n"
            "ninfer_generation_tokens_total %d" % (prompt_total, cached, gen)
        )
        fake_metrics(body)
        s = nes.NinferEngineState("http://x/metrics", 0.5).poll()
        expected = ((prompt_total - cached) / 1e6 * mc.CLOUD_IN_RATE
                    + cached / 1e6 * mc.CLOUD_CACHE_RATE
                    + gen / 1e6 * mc.CLOUD_OUT_RATE)
        self.assertAlmostEqual(s["cloud_cost"], expected, places=6)


# ------------------------------------- F. metric stats in the panel title banner

class TestMetricStatBanner(Base):
    def test_F1_segments_order_and_peak_flag(self):
        g = bg.Graph(10)
        for v in (2, 8, 4, 6, 0):
            g.add(v)
        segs = tui.metric_stat_segments(g, "4")
        self.assertEqual("".join(t for t, _ in segs),
                           "(cur=4,max=8,peak=8,avg=4,min=0)")
        flags = [flag for _, flag in segs]
        self.assertEqual(flags.count(True), 1)  # only the max token is highlighted
        self.assertTrue(next(t for t, flag in segs if flag).startswith(",max="))

    def test_F2_empty_window_has_no_banner_suffix(self):
        self.assertEqual(tui.metric_stat_segments(bg.Graph(10), "—"), [])

    def test_F3_draw_panel_paints_max_in_peak_color(self):
        g = bg.Graph(10)
        for v in (2, 8, 4):
            g.add(v)
        scr = FakeScreen(40, 120)
        tui.draw_panel(scr, 0, "Batch size  (ninfer_requests_running)", "batch", {}, 120,
                      stats=tui.metric_stat_segments(g, "4"))
        row = [w for w in scr.writes if w[0] == 0]
        joined = "".join(w[2] for w in row)
        for token in ("cur=4", "max=8", "peak=8", "avg=", "min="):
            self.assertIn(token, joined)
        # the max= token is painted with the peak attribute (A_REVERSE under the
        # colorless stub); the rest use the default attribute
        peak_attr = nb.curses.A_REVERSE
        self.assertTrue(any(w[2].startswith(",max=") and w[3] == peak_attr for w in row))
        # the peak= token exists but stays plain, like the other stats
        self.assertTrue(any(w[2].startswith(",peak=") and w[3] == 0 for w in row))
        self.assertTrue(any(w[2].startswith("(cur=") and w[3] == 0 for w in row))
        # the max token is never drawn in a plain (non-peak) attribute
        self.assertFalse(any(w[2].startswith(",max=") and w[3] != peak_attr for w in row))

    def test_F4_graphs_paint_no_peak_highlight(self):
        g = bg.Graph(10)
        for v in (2, 8, 4, 6, 0):
            g.add(v)
        # the bar graph itself has no peak highlight: every glyph - the
        # window max included - is painted with the panel's base color
        # (0 under the colorless stub; a highlighted char would be A_REVERSE)
        scr = FakeScreen(40, 120)
        tui.draw_bar_graph(scr, 0, g, 2, 50, "toks", {})
        # the moving bars (x <= width-STAT_COLS) are painted in the base
        # color; the MhHDWL summary columns on the right are dimmed
        glyphs = [w for w in scr.writes
                  if w[1] <= 44 and not w[2].startswith("─")]
        self.assertTrue(glyphs)
        self.assertTrue(all(w[3] == 0 for w in glyphs))



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
        fake_metrics(body)
        s = nes.NinferEngineState("http://x/metrics", 0.5).poll()
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
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        tui.frame(scr, eng, graphs, c)
        self.assertTrue(rows_with(scr, "Device KV"))
        self.assertTrue(rows_with(scr, "Host KV"))
        self.assertTrue(rows_with(scr, "Cache (checkpoint reuse)"))
        joined = "".join(t for (_, _, t, _) in scr.writes)
        for gone in ("Pool    :", "TOO SMALL", "tight", "TIGHT"):
            self.assertNotIn(gone, joined)


    def test_G4_poll_splits_token_hits_by_serving_tier(self):
        # calibrated at 1000 B/tok; 8MB of lifetime h2d recall => 8000 of
        # the 100000 cached tokens were served from host RAM, 92000 from VRAM
        mc.set_calib(dict(CALIB_OK, bytes_per_tok=1000.0))
        body = ("ninfer_engine_ready 1\n"
                "ninfer_prompt_tokens_total 1000000\n"
                "ninfer_prompt_tokens_cached_total 100000\n"
                "ninfer_prefill_tokens_total 900000\n"
                "ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"0\",direction=\"h2d\"} 8000000\n")
        fake_metrics(body)
        s = nes.NinferEngineState("http://x/metrics", 0.5).poll()
        self.assertAlmostEqual(s["tok_host"], 8000 / 1000000)
        self.assertAlmostEqual(s["tok_dev"], 92000 / 1000000)
        self.assertAlmostEqual(s["tok_dev"] + s["tok_host"], s["tok_hit"])

    def test_G5_host_estimate_clamps_to_the_cached_total(self):
        # 200MB of h2d recall at 1000 B/tok would imply 200000 host-served
        # tokens, more than the 100000 actually cached -> clamp to the total
        mc.set_calib(dict(CALIB_OK, bytes_per_tok=1000.0))
        body = ("ninfer_engine_ready 1\n"
                "ninfer_prompt_tokens_total 1000000\n"
                "ninfer_prompt_tokens_cached_total 100000\n"
                "ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"0\",direction=\"h2d\"} 200000000\n")
        fake_metrics(body)
        s = nes.NinferEngineState("http://x/metrics", 0.5).poll()
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
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        tui.frame(scr, eng, graphs, c)
        joined = "".join(t for (_, _, t, _) in scr.writes)
        # the line must fit the 38-char column at 120 cols: no space between
        # the two groups, both labels fully visible
        self.assertIn("46.0% (device)34.0% (host)", joined)

    def test_G7_without_calibration_the_plain_tok_hit_line_is_kept(self):
        # setUp leaves the calib state as None (mc.set_calib(None)) -> no bpt -> no tier split
        body = ("ninfer_engine_ready 1\n"
                "ninfer_prompt_tokens_total 1000000\n"
                "ninfer_prompt_tokens_cached_total 100000\n")
        fake_metrics(body)
        s = nes.NinferEngineState("http://x/metrics", 0.5).poll()
        self.assertIsNone(s["tok_dev"])
        self.assertIsNone(s["tok_host"])
        nb.curses.color_pair = lambda i: 1 + i
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = self._sample()
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        tui.frame(scr, eng, graphs, c)
        self.assertFalse(rows_with(scr, "(device)"))
        self.assertTrue(rows_with(scr, "Tok hit"))

    def test_G9_lane_paused_figure_is_red_only_when_nonzero(self):
        nb.curses.color_pair = lambda i: 1 + i              # distinct, inspectable pair ids
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        # paused = 2 -> the "paused 2" segment carries the 'bad' (red) pair
        eng.last_frame = self._sample(paused=2)
        scr = FakeScreen(40, 120)
        tui.frame(scr, eng, graphs, c)
        bad_attr = 1 + c["bad"]
        self.assertTrue(any("paused 2" in t and a == bad_attr
                            for (_, _, t, a) in scr.writes))
        # paused = 0 -> the same segment is painted with the default color
        eng.last_frame = self._sample(paused=0)
        scr = FakeScreen(40, 120)
        tui.frame(scr, eng, graphs, c)
        self.assertTrue(any("paused 0" in t and a == 0
                            for (_, _, t, a) in scr.writes))

    def test_G10_status_line_shows_context_window_in_kitok(self):
        # 131072 tokens = 128.0 KiToks (1 KiToks = 1024 tokens)
        nb.curses.color_pair = lambda i: 1 + i
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        mc.set_calib(dict(CALIB_OK))
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = self._sample(ctx_window=131072)
        eng.start_time = None
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        tui.frame(scr, eng, graphs, c, mc.get_calib())
        row1 = "".join(t for (y, x, t, a) in scr.writes if y == 1)
        self.assertIn("calibrated in", row1)
        self.assertIn("ctx 128.0 KiToks", row1)

    def test_G11_no_ctx_metric_no_ctx_segment(self):
        nb.curses.color_pair = lambda i: 1 + i
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        mc.set_calib(dict(CALIB_OK))
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = self._sample()  # no ctx_window key
        eng.start_time = None
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        tui.frame(scr, eng, graphs, c, mc.get_calib())
        row1 = "".join(t for (y, x, t, a) in scr.writes if y == 1)
        self.assertIn("calibrated in", row1)
        self.assertNotIn("KiToks", row1)

    def test_G12_no_calib_status_line_is_the_ctx_window(self):
        mc.set_calib(None)
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = self._sample(ctx_window=131072)
        eng.start_time = None
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        scr = FakeScreen(40, 120)
        tui.frame(scr, eng, graphs, {})
        row1 = "".join(t for (y, x, t, a) in scr.writes if y == 1)
        self.assertIn("128.0 KiToks", row1)

    def test_G8_explicit_bytes_per_tok_works_without_calibration(self):
        # --bytes-per-tok without calibration data: get_calib() has no "ok" key,
        # the split still works and utilization stays n/a
        mc.set_calib({"bytes_per_tok": 1000.0})
        body = ("ninfer_engine_ready 1\n"
                "ninfer_prompt_tokens_total 1000000\n"
                "ninfer_prompt_tokens_cached_total 100000\n"
                "ninfer_context_transfer_bytes_total{resource=\"kv\",pool_id=\"0\",direction=\"h2d\"} 30000000\n")
        fake_metrics(body)
        s = nes.NinferEngineState("http://x/metrics", 0.5).poll()
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
        m = mc.fetch_metrics("http://x/metrics")
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
        real_mono = time.monotonic
        time.monotonic = lambda: t[0]
        try:
            b0 = ('ninfer_engine_ready 1\n'
                  'ninfer_context_transfer_bytes_total{resource="state",direction="d2h"} 100\n'
                  'ninfer_context_transfer_bytes_total{resource="state",direction="h2d"} 40\n')
            b1 = ('ninfer_engine_ready 1\n'
                  'ninfer_context_transfer_bytes_total{resource="state",direction="d2h"} 300\n'
                  'ninfer_context_transfer_bytes_total{resource="state",direction="h2d"} 70\n')
            e = nes.NinferEngineState("http://x/metrics", 0.5)
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
            time.monotonic = real_mono

    def test_H3_frame_renders_spill_recall_xfer_graphs(self):
        nb.curses.color_pair = lambda i: 1 + i
        c = {"ok": 1, "bad": 2, "toks": 3, "dim": 4, "kv": 5, "title": 6,
             "batch": 7, "label": 8, "val": 9, "peak": 10}
        s = self._sample(spill_rate=200.0, recall_rate=50.0)
        eng = make_engine(True)
        eng.m = {"ninfer_engine_ready": 1.0}
        eng.last_frame = s
        eng.start_time = None
        d2h = bg.Graph(26)
        for v in [0.0] * 10 + [200.0] * 4 + [0.0] * 12:
            d2h.add(v)
        h2d = bg.Graph(26)
        for v in [0.0] * 24 + [30.0, 30.0]:
            h2d.add(v)
        graphs = {k: bg.Graph(60) for k in ("batch", "toks", "prefill")}
        graphs["xfer_d2h"] = d2h
        graphs["xfer_h2d"] = h2d
        scr = FakeScreen(40, 120)
        tui.frame(scr, eng, graphs, c)
        self.assertTrue(rows_with(scr, "Spill"))
        self.assertTrue(rows_with(scr, "Recall"))
        spill = next(t for (_, _, t, _) in scr.writes if "Spill" in t)
        recall = next(t for (_, _, t, _) in scr.writes if "Recall" in t)
        # each xfer graph autoscales to its OWN window peak: a 200 B/s spill
        # burst and a 30 B/s recall burst both reach the full braille dot,
        # and the side label is that window peak (the scale of the dots)
        self.assertIn("⣿", spill)
        self.assertIn("⣿", recall)
        self.assertTrue(spill.endswith("200B/s"))
        self.assertTrue(recall.endswith("30B/s"))
        # the row must fit the ⅓-width cache column at 120 cols (38 chars)
        self.assertLessEqual(len(spill), 38)

    def test_H4_xfer_graph_autoscale_and_empty(self):
        # all-zero window: blank graph, 0B/s label
        g = bg.Graph(26)
        for _ in range(26):
            g.add(0.0)
        self.assertEqual(tui._xfer_row("  X  ", g),
                         "  X  " + " " * 13 + " " + "0B/s")
        # a burst autoscales against its own window peak, not a shared one
        g2 = bg.Graph(26)
        for v in [0.0] * 11 + [5.0] * 4 + [0.0] * 11:
            g2.add(v)
        row = tui._xfer_row("  X  ", g2)
        self.assertIn("⣿", row)            # 5 B/s burst = its own peak
        self.assertTrue(row.endswith("5B/s"))
        # no graph yet (first frames): dash placeholder
        self.assertEqual(tui._xfer_row("  X  ", None), "  X  " + " " * 14 + "—")
        # rate label formatting is unchanged
        self.assertEqual(tui._rate_val(None), "   —")
        self.assertEqual(tui._rate_val(12_400_000.0), "12.40MB/s")


# --------------------- J. batch panel height follows max concurrency

class TestBatchHeight(Base):
    """J: the batch panel is ceil(max_concurrency/4) chars high (1 braille
    character resolves 4 dot levels = 4 concurrent requests)."""

    def _panel_height(self, floor, calib, max_y=40):
        mc.set_calib(calib)
        scr = FakeScreen(max_y, 120)
        eng = make_engine(True)
        eng.last_frame = self._sample()
        graphs = {"batch": bg.Graph(60, floor=floor),
                  "toks": bg.Graph(60), "prefill": bg.Graph(60)}
        tui.frame(scr, eng, graphs, {}, calib)
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
        g = bg.Graph(60, floor=8)
        g.add(3.0)
        g2 = bg.Graph(40, floor=g.floor, pinned=g.pinned)
        g2.hist.extend(list(g.hist)[-40:])
        self.assertEqual(g2.floor, 8.0)
        self.assertEqual(list(g2.hist), list(g.hist)[-40:])

    def test_J4_dots_pinned_to_plot_height(self):
        # pinned: max = the plot's total dot count (4*rows) - 1 dot = 1
        # request, a burst above the cap clips at the top
        self.assertEqual(bg.compute_dots([1.0, 2.0, 8.0], max=8, total_dots=8), [1, 2, 8])
        self.assertEqual(bg.compute_dots([1.0, 2.0, 11.0], max=8, total_dots=8), [1, 2, 8])
        g = bg.Graph(10, floor=4, pinned=True)
        for v in (1.0, 2.0, 8.0):
            g.add(v)
        self.assertEqual(g.dots(2), [1, 2, 8])
        # plain graphs autoscale to the window max
        g0 = bg.Graph(10)
        g0.add(1.0)
        g0.add(3.0)
        self.assertEqual(g0.dots(1), [2, 4])
        # floor no longer touches the scale of plain graphs (it only sizes
        # the batch panel); autoscale still takes the window max
        gp2 = bg.Graph(10, floor=4)
        gp2.add(1.0)
        gp2.add(3.0)
        self.assertEqual(gp2.dots(1), [2, 4])
        # ceil: any non-zero sample draws at least one dot; Nones stay blank
        self.assertEqual(bg.compute_dots([0.1, 1.0]), [1, 4])
        self.assertEqual(bg.compute_dots([None, 0.0, 2.0]), [0, 0, 4])

    def test_J5_dots_scale_to_lifetime_peak(self):
        # the scale is the lifetime peak (the peak= banner token), not the
        # window max: after the peak sample scrolls out of the window the
        # bars stay proportionally small
        g = bg.Graph(4)
        g.add(8.0)
        for _ in range(4):
            g.add(1.0)
        self.assertEqual(g.dots(1), [1, 1, 1, 1])  # peak 8 -> 1/4 height
        # pinned graphs ignore the lifetime peak: 1 dot = 1 request, always
        gp = bg.Graph(10, pinned=True)
        gp.add(5.0)
        for _ in range(4):
            gp.add(1.0)
        self.assertEqual(gp.dots(2), [5, 1, 1, 1, 1])  # 1 dot = 1 request
        # before any non-zero sample plain graphs autoscale to the window max
        g1 = bg.Graph(4)
        g1.add_blank()
        self.assertIsNone(g1.dot_scale(1))


# ------------------------------------- I. pricing loader + row selection

class TestPeriodColumns(Base):
    """I: M/h/H/D/L period summary columns (avg left, max right)."""

    def test_I1_period_stats_windows(self):
        # 10 min of 0.5 s samples: first 5 min 2.0, second 5 min 10.0 -
        # the M (5 min) window sees exactly the second half; h/H/D/W/L all
        # clamp to the whole 10-minute buffer
        g = bg.Graph(50)
        for _ in range(600):
            g.add(2.0)
        for _ in range(600):
            g.add(10.0)
        stats = bg.period_stats(g)
        self.assertEqual(len(stats), 6)
        m_avg, m_max = stats[0]
        self.assertAlmostEqual(m_avg, 10.0)
        self.assertAlmostEqual(m_max, 10.0)
        for avg, mx in stats[1:]:
            self.assertAlmostEqual(avg, 6.0)
            self.assertAlmostEqual(mx, 10.0)

    def test_I2_short_history_clamps(self):
        g = bg.Graph(10)
        for _ in range(10):
            g.add(4.0)
        stats = bg.period_stats(g)
        self.assertEqual(len(stats), 6)
        for avg, mx in stats:
            self.assertAlmostEqual(avg, 4.0)
            self.assertAlmostEqual(mx, 4.0)

    def test_I3_labels_and_dim_summary(self):
        g = bg.Graph(20)
        for _ in range(30):
            g.add(5.0)
        scr = FakeScreen(40, 32)
        tui.draw_bar_graph(scr, 0, g, 1, 30, "toks", {})
        # the MhHDWL labels sit on the baseline row under the summary columns
        lab = [w for w in scr.writes if w[2] == "MhHDWL"]
        self.assertEqual(lab[0][:3], (1, 25, "MhHDWL"))
        self.assertEqual(lab[0][3], nb.curses.A_DIM)
        # all-5.0 samples: avg = max = full scale, so every summary column
        # is a full character, dimmed; six columns at x = width-5 .. width
        cols = sorted(x for (y, x, t, a) in scr.writes
                      if y == 0 and x >= 25 and a == nb.curses.A_DIM)
        self.assertEqual(cols, [25, 26, 27, 28, 29, 30])
        base = [w for w in scr.writes
                if w[0] == 1 and w[1] == 1 and w[2].startswith("─")]
        self.assertEqual(len(base[0][2]), 24)  # width-6, labels take 6

    def test_I4_max_leaves_window(self):
        # M window = 600 samples (5 min) at the 0.5 s default: the spike at
        # sample 0 must drop out of M's max once 600 newer samples
        # arrive, while L (whole life-time) still sees it
        g = bg.Graph(10)
        g.add(100.0)
        for _ in range(599):
            g.add(1.0)
        _, m_max = bg.period_stats(g)[0]
        self.assertAlmostEqual(m_max, 100.0)
        g.add(1.0)  # 601st sample: the spike slides out of the window
        _, m_max = bg.period_stats(g)[0]
        self.assertAlmostEqual(m_max, 1.0)
        _, l_max = bg.period_stats(g)[5]
        self.assertAlmostEqual(l_max, 100.0)

    def test_I6_all_zero_window_has_no_float_residue(self):
        # the running sum's float residue must not light a phantom avg dot
        # in an all-zero window: burst, wait it out, avg and max are both 0
        g = bg.Graph(240, interval=0.5)
        for _ in range(50):
            g.add(0.0)
        for _ in range(10):
            g.add(123.4567)
        for _ in range(610):  # the M window (600 samples) goes all zero
            g.add(0.0)
        avg, mx = bg.period_stats(g)[0]
        self.assertEqual(mx, 0.0)
        self.assertEqual(avg, 0.0)
        # and a new spike after the quiet stretch reports cleanly
        g.add(500.0)
        avg, mx = bg.period_stats(g)[0]
        self.assertEqual(mx, 500.0)
        self.assertGreater(avg, 0.0)

    def test_I7_baseline_time_span_label(self):
        # the baseline carries a centered label of the time span the plot
        # region covers: 2 samples per character * interval
        g = bg.Graph(20)
        for _ in range(30):
            g.add(1.0)
        scr = FakeScreen(40, 32)
        tui.draw_bar_graph(scr, 0, g, 1, 30, "toks", {})
        # width-STAT_COLS = 24 chars = 48 samples * 0.5 s = "24 seconds"
        span = "24 seconds"
        lab = [w for w in scr.writes if w[2] == span]
        self.assertEqual(len(lab), 1)
        self.assertEqual(lab[0][0], 1)  # the baseline row
        self.assertEqual(lab[0][1], 1 + (24 - len(span)) // 2)  # centered

    def test_I5_one_week_buffer_and_lifetime_totals(self):
        # interval 60 s: the raw history is trimmed to the last week
        # (10080 samples); after 10200 samples the spike at sample 0 is
        # gone from the history and from D/W, but the L column's running
        # total and max still cover the whole process life
        g = bg.Graph(10, interval=60)
        g.add(1000.0)
        for _ in range(10199):
            g.add(1.0)
        self.assertEqual(len(g.samples), 10080)
        stats = bg.period_stats(g)
        _, d_max = stats[3]
        self.assertAlmostEqual(d_max, 1.0)
        _, w_max = stats[4]  # the week window == the kept buffer
        self.assertAlmostEqual(w_max, 1.0)
        l_avg, l_max = stats[5]
        self.assertEqual(g.life_count, 10200)
        self.assertAlmostEqual(l_avg, (1000.0 + 10199 * 1.0) / 10200)
        self.assertAlmostEqual(l_max, 1000.0)


    def test_I8_full_week_buffer_refresh_timing(self):
        # a full week of 0.5 s samples (1,209,600) exactly fills the raw
        # buffer; the per-tick period_stats refresh over that full buffer
        # must stay inside one polling interval, or the UI would stutter
        # (measured on box hermes: about 26 ms per graph, about 126 ms for
        # all five graphs per tick)
        g = bg.Graph(10)
        n = int(bg.WEEK_SECS / g.interval)
        for _ in range(n):
            g.add(1.0)
        self.assertEqual(len(g.samples), n)  # the buffer holds exactly one week
        t0 = time.perf_counter()
        stats = bg.period_stats(g)
        dt = time.perf_counter() - t0
        print(f"I8 full-week refresh: {len(g.samples):,} samples, "
              f"period_stats {dt * 1000:.3f} ms "
              f"({dt / g.interval * 100:.2f}% of the {g.interval:.1f}s interval)")
        self.assertEqual(len(stats), 6)
        for avg, mx in stats:
            self.assertAlmostEqual(avg, 1.0)
            self.assertAlmostEqual(mx, 1.0)
        self.assertLess(dt, g.interval)

    def test_I9_two_week_numeric_correctness(self):
        # two weeks of 60 s samples (20,160); the buffer keeps the last
        # week (10,080), so M/h/H/D/W are exact functions of that tail
        # and L covers the whole life. Data is sparse and deterministic:
        # a 1-in-7 background pattern under 0.13 plus one spike planted
        # in each window's exclusive band (plus one in W's left-edge
        # partial block, to pin its edge-max), and a 2000.0 spike in the
        # first week that only the lifetime column may see.
        g = bg.Graph(10, interval=60)
        n = 2 * bg.WEEK_SECS // 60
        data = [((i * 7919) % 13) / 100.0 if i % 7 == 0 else 0.0 for i in range(n)]
        data[10] = 2000.0            # first week: trimmed from the buffer, L only
        # the windows are nested (M inside h inside H inside D inside W),
        # so each window's max comes from the largest spike inside it:
        # spikes grow with the window they are planted in
        end = n - 1
        data[end - 2] = 600.0        # in M (5 samples)
        data[end - 20] = 700.0       # in h (30)
        data[end - 45] = 800.0       # in H (60)
        data[end - 500] = 900.0      # in D (1440)
        data[end - 9000] = 1000.0    # in W (whole buffer) only
        data[10150] = 1100.0         # in W's left-edge partial block only
        stored = array.array("f", data)  # what the Graph actually keeps
        for v in data:
            g.add(v)
        self.assertEqual(len(g.samples), bg.WEEK_SECS // 60)  # trimmed to one week
        # independent one-pass reference over the stored values
        ref = []
        for w in (5, 30, 60, 1440, 10080):  # M/h/H/D/W in samples at 60 s
            seg = stored[len(stored) - w:]
            ref.append((sum(seg) / len(seg), max(seg)))
        ref.append((sum(stored) / len(stored), max(stored)))  # L
        stats = bg.period_stats(g)
        self.assertEqual(len(stats), 6)
        for (avg, mx), (r_avg, r_max) in zip(stats, ref):
            self.assertAlmostEqual(avg, r_avg, delta=1e-6)
            self.assertEqual(mx, r_max)  # max is bit-exact
        # the spike placement above must be what the columns report
        self.assertEqual([mx for _, mx in stats], [600.0, 700.0, 800.0, 900.0, 1100.0, 2000.0])


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
            p = mc.load_pricing(path)
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
        p = mc.load_pricing("/no/such/pricing.yaml")
        self.assertEqual(set(p), {"qwen3.8-27b"})
        self.assertEqual(p["qwen3.8-27b"],
                         {"input": 0.5, "cached_input": 0.05, "output": 3.0})

    def test_I3_key_indexing_selects_the_right_row(self):
        path = self._write_yaml(self.SAMPLE)
        try:
            p = mc.load_pricing(path)
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
        s_def = nes.NinferEngineState("http://x/metrics", 0.5).poll()
        s_llm = nes.NinferEngineState("http://x/metrics", 0.5, p["llama-3-8b"]).poll()
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
        eng = nes.NinferEngineState("http://x/metrics", 0.5)  # no rates supplied
        self.assertEqual(eng.rates,
                         {"input": 0.5, "cached_input": 0.05, "output": 3.0})
        self.assertEqual(eng.rates, dict(mc.DEFAULT_PRICING["qwen3.8-27b"]))
        # the module aliases still resolve to that default row's values
        row = mc.DEFAULT_PRICING["qwen3.8-27b"]
        self.assertEqual(mc.CLOUD_IN_RATE, row["input"])
        self.assertEqual(mc.CLOUD_CACHE_RATE, row["cached_input"])
        self.assertEqual(mc.CLOUD_OUT_RATE, row["output"])

    def test_I6_real_pricing_yaml_has_the_supported_models(self):
        # I1-I5 exercise the parser on a synthetic sample; this guards the file
        # actually shipped in the repo. Rates are the International-scope Alibaba
        # Cloud Model Studio list prices, verified 2026-10-15 - update this
        # alongside pricing.yaml if those change.
        self.assertTrue(os.path.exists(mc.pricing_file_path()),
                        "pricing.yaml is missing next to the script")
        p = mc.load_pricing()  # resolves pricing.yaml next to ninfer_btop.py
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
        mc.CALIB_FILE = tempfile.mktemp(suffix=".json")
        return mc.CALIB_FILE

    def _write_cache(self, url, start, path):
        with open(path, "w") as f:
            json.dump({"url": url, "server_start_time": start, "model": "m",
                       "concurrency": 4, "prefill_tps": 100.0,
                       "decode_tps": 50.0, "duration": 10.0}, f)

    def test_K1_no_file_benchmarks_and_caches(self):
        path = self._cache_file()
        mc.resolve_calib(self.URL, run=lambda: mc.set_calib(dict(self.GOOD)),
                         probe=lambda u: 1000.0)
        self.assertEqual(mc.get_calib(), self.GOOD)
        j = json.load(open(path))
        self.assertEqual(j["url"], self.URL)
        self.assertEqual(j["server_start_time"], 1000.0)
        self.assertEqual(j["concurrency"], 4)

    def test_K2_matching_start_time_reuses_cache_without_benchmark(self):
        path = self._cache_file()
        self._write_cache(self.URL, 1000.0, path)
        ran = []
        mc.resolve_calib(self.URL, run=lambda: ran.append(1), probe=lambda u: 1000.0)
        self.assertEqual(ran, [])
        self.assertTrue(mc.get_calib()["ok"])
        self.assertEqual(mc.get_calib()["concurrency"], 4)
        self.assertNotIn("server_start_time", mc.get_calib())  # cache dict shape

    def test_K3_restarted_engine_benchmarks_and_rewrites_cache(self):
        path = self._cache_file()
        self._write_cache(self.URL, 1000.0, path)
        def run():
            c = dict(self.GOOD)
            c["server_start_time"] = 2000.0
            mc.set_calib(c)
        mc.resolve_calib(self.URL, run=run, probe=lambda u: 2000.0)
        self.assertEqual(mc.get_calib()["server_start_time"], 2000.0)
        self.assertEqual(json.load(open(path))["server_start_time"], 2000.0)

    def test_K4_re_calib_drops_cache_and_always_benchmarks(self):
        path = self._cache_file()
        self._write_cache(self.URL, 1000.0, path)
        ran = []
        def run():
            ran.append(1)
            mc.set_calib(dict(self.GOOD))
        # start time still matches - only the flag forces the fresh run
        mc.resolve_calib(self.URL, re_calib=True, run=run, probe=lambda u: 1000.0)
        self.assertEqual(ran, [1])
        # the dropped cache is rewritten with the fresh result
        self.assertEqual(json.load(open(path))["url"], self.URL)

    def test_K5_different_url_benchmarks(self):
        path = self._cache_file()
        self._write_cache("http://y/metrics", 1000.0, path)
        ran = []
        mc.resolve_calib(self.URL, run=lambda: ran.append(1) or mc.set_calib({"ok": False}),
                         probe=lambda u: 1000.0)
        self.assertEqual(ran, [1])

    def test_K6_failed_calibration_is_not_cached(self):
        path = self._cache_file()
        mc.resolve_calib(self.URL,
                         run=lambda: mc.set_calib({"ok": False, "error": "boom"}),
                         probe=lambda u: None)
        self.assertFalse(os.path.exists(path))

    def test_K7_undeletable_cache_aborts(self):
        # a directory where the cache file should be: os.remove raises
        # IsADirectoryError - must abort, not leave the stale cache in place
        d = tempfile.mkdtemp()
        mc.CALIB_FILE = d
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                mc.resolve_calib(self.URL, re_calib=True,
                                 run=lambda: mc.set_calib(dict(self.GOOD)),
                                 probe=lambda u: 1000.0)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("cannot remove", err.getvalue())

    def test_K8_unwritable_cache_aborts_after_benchmark(self):
        d = tempfile.mkdtemp()
        mc.CALIB_FILE = os.path.join(d, "no", "such", "cache.json")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                mc.resolve_calib(self.URL,
                                 run=lambda: mc.set_calib(dict(self.GOOD)),
                                 probe=lambda u: 1000.0)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("cannot write", err.getvalue())

    def test_K9_unreadable_cache_aborts_before_benchmark(self):
        d = tempfile.mkdtemp()
        mc.CALIB_FILE = d  # open-for-read on a directory raises IsADirectoryError
        ran = []
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                mc.resolve_calib(self.URL, run=lambda: ran.append(1),
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
            mc.set_calib(dict(self.GOOD))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            mc.resolve_calib(self.URL, run=run, probe=lambda u: 1000.0)
        self.assertEqual(ran, [1])
        self.assertIn("not valid JSON", err.getvalue())
        j = json.load(open(path))  # the fresh run rewrote the corrupt file
        self.assertEqual(j["url"], self.URL)


class TestPlotSamples(unittest.TestCase):
    """Graph.plot_samples: the period/agg selection for the main plots
    (phase 2 core). n = raw hist tail; m..w = the fixed time axis -
    int(period/interval) samples cut into exactly `slots` divmod-sized
    buckets anchored at seq (a young buffer draws a few newest buckets,
    not a stretched plot); l = the kept buffer until the lifetime table
    lands in phase 4. Graph.plot_peak: the raw-sample max over the same
    window - the compressed views' scale reference."""

    def _graph(self, vals, width=40, interval=0.5):
        g = bg.Graph(width, interval=interval)
        for v in vals:
            if v is None:
                g.add_blank()
            else:
                g.add(v)
        return g

    def test_P1_n_draws_raw_hist_tail_blanks_as_zero(self):
        g = self._graph([1.0, 2.0, None, 3.0, None], width=10)
        self.assertEqual(g.plot_samples("n", "A", 100),
                         [1.0, 2.0, 0.0, 3.0, 0.0])
        # the agg is ignored for the live period
        self.assertEqual(g.plot_samples("n", "M", 100),
                         g.plot_samples("n", "A", 100))

    def test_P2_n_is_shorter_than_slots_without_padding(self):
        g = self._graph([1.0, 2.0], width=10)
        self.assertEqual(len(g.plot_samples("n", "A", 64)), 2)

    def test_P3_agg_avg_fixed_axis_full_width_dilution(self):
        # fixed axis: 'm' is 600 samples at 0.5s, cut into 10 buckets of
        # width 60 anchored at seq (divmod(600, 10): no remainder). The
        # 137-sample buffer covers only the newest three: [77,137) and
        # [17,77) fully, [-43,17) straddling the buffer start.
        g = self._graph([float(i) for i in range(1, 138)])  # 137 samples
        out = g.plot_samples("m", "A", 10)
        self.assertEqual(len(out), 10)
        self.assertEqual(out[:7], [0.0] * 7)  # older than the kept buffer
        # A divides by the FULL bucket width: the straddler's 43 implicit
        # zeros dilute it (153/60, not 153/17). All three buckets raw
        # scan (width < BLK_SAMPLES) and the values are small integers,
        # so the asserts are exact, no tolerance.
        self.assertEqual(out, [0.0] * 7
                         + [sum(range(1, 18)) / 60,
                            sum(range(18, 78)) / 60,
                            sum(range(78, 138)) / 60])

    def test_P4_agg_max_exact_and_newest_bucket_last(self):
        # same fixed axis as P3: M ignores the implicit zeros, so the
        # straddling bucket reports the max of its 17 existing samples;
        # the newest bucket (the 137th sample) is last
        g = self._graph([float(i) for i in range(1, 138)])
        out = g.plot_samples("m", "M", 10)
        self.assertEqual(out, [0.0] * 7 + [17.0, 77.0, 137.0])
    def test_P5_short_buffer_pads_the_oldest_side(self):
        # 3 samples against the 600-sample 'm' axis, 5 buckets of width
        # 120: only the newest bucket straddles the buffer start
        g = self._graph([10.0, 20.0, 30.0])
        self.assertEqual(g.plot_samples("m", "M", 5),
                         [0.0, 0.0, 0.0, 0.0, 30.0])
        # the same bucket's A divides by the full width 120 (117 implicit
        # zeros), so Avg sits low where Max is tall
        self.assertEqual(g.plot_samples("m", "A", 5),
                         [0.0, 0.0, 0.0, 0.0, 60.0 / 120])
        # slots above the period total: divmod(600, 700) = (0, 600) - the
        # newest 600 buckets are one sample wide, the oldest 100 empty
        for agg in "AM":
            self.assertEqual(g.plot_samples("m", agg, 700),
                             [0.0] * 697 + [10.0, 20.0, 30.0])
    def test_P6_period_span_mapping_m_and_H(self):
        g = self._graph([float(i) for i in range(1, 701)])  # 700 samples
        # 'm': 300 s / 0.5 s = 600 samples; divmod(600, 700) = (0, 600)
        # -> one-sample buckets for the kept tail, oldest 100 empty
        out_m = g.plot_samples("m", "M", 700)
        self.assertEqual(out_m, [0.0] * 100 + [float(i) for i in range(101, 701)])
        # 'H': 7200-sample axis, 700 buckets -> divmod(7200, 700): 500
        # buckets of width 10 + the newest 200 of width 11. The 700-
        # sample buffer fills the newest 63 buckets and bucket 636
        # straddles the buffer start (width 11, only 7 samples exist).
        self.assertEqual(g.plot_samples("H", "M", 700),
                         [0.0] * 636 + [7.0]
                         + [18.0 + 11.0 * i for i in range(63)])
        # A: the straddler divides its 28 by the full width 11; the full
        # buckets are exact means of their 11 samples
        self.assertEqual(g.plot_samples("H", "A", 700),
                         [0.0] * 636 + [28.0 / 11.0]
                         + [13.0 + 11.0 * i for i in range(63)])
    def test_P7_l_selects_the_whole_run(self):
        # 137 samples, no trims: the lifetime table is empty and 'l' is
        # the w-style aggregation of the WHOLE RUN (divmod(seq, slots))
        # - divmod(137, 24) = (5, 17) - not the 'w' window, which
        # stretches to the one-week span (one huge bucket + empty ones)
        g = self._graph([float(i) for i in range(1, 138)])
        self.assertEqual(g.plot_samples("l", "A", 24),
                         [3.0, 8.0, 13.0, 18.0, 23.0, 28.0, 33.0]
                         + [38.5 + 6.0 * i for i in range(17)])
        self.assertEqual(g.plot_samples("l", "M", 24),
                         [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
                         + [41.0 + 6.0 * i for i in range(17)])

    def test_P8_empty_buffer(self):
        g = bg.Graph(10)
        self.assertEqual(g.plot_samples("n", "A", 16), [])
        self.assertEqual(g.plot_samples("m", "A", 16), [0.0] * 16)
        self.assertEqual(g.plot_samples("w", "M", 16), [0.0] * 16)

    def test_P9_pure_no_state_mutation(self):
        g = self._graph([float(i % 7) for i in range(200)] + [None] * 5, width=64)
        snap = lambda: (list(g.hist), g.samples.tobytes(), list(g.blk_s),
                        list(g.blk_m), g.blk_first, g.seq, g.life_sum,
                        g.life_count, g.life_max)
        before = snap()
        for w in "nmhHdwl":
            for p in "AM":
                g.plot_samples(w, p, 33)
            g.plot_peak(w)
        self.assertEqual(before, snap())
        # the returned list is a fresh object: mutating it leaves the graph alone
        r = g.plot_samples("w", "A", 33)
        r[0] = 999.0
        self.assertEqual(before, snap())

    def test_P10_full_block_rows_flow_through_the_tables(self):
        # 3072 samples = exactly three 1024-sample block rows. The 'h'
        # axis is 3600 samples (> the buffer), so the 3 buckets are each
        # 1200 wide, anchored at seq: bucket 2 [1872,3072) = full block
        # row 2 [2048,3072) + left edge [1872,2048) - the table-driven
        # path; bucket 1 [672,1872) spans two partial rows (raw scan);
        # bucket 0 [-528,672) straddles the buffer start and dilutes A
        # over its full width 1200 (528 implicit zeros).
        vals = [float((r + 1) * 1000 + (i % 37))
                for r in range(3) for i in range(1024)]
        g = self._graph(vals, width=16)
        # A: every value is a small integer, so the float64 table sums
        # are exact and equal the direct-scan sums bit-for-bit
        self.assertEqual(g.plot_samples("h", "A", 3),
                         [sum(vals[0:672]) / 1200,
                          sum(vals[672:1872]) / 1200,
                          sum(vals[1872:3072]) / 1200])
        # M: bucket 2's max comes from the exact float32 row-max table
        self.assertEqual(g.plot_samples("h", "M", 3),
                         [1036.0, 2036.0, 3036.0])
    def test_P11_live_path_ignores_slots(self):
        # the live period returns the entire hist; slots smaller than the
        # hist (even 0) do not truncate here - truncation to the last
        # slots samples and the plot-width bound happen in draw_bar_graph
        g = self._graph([1.0, 2.0, 3.0, 4.0, 5.0], width=10)
        self.assertEqual(g.plot_samples("n", "A", 3),
                         [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(g.plot_samples("n", "A", 0),
                         [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_P12_live_path_reflects_the_hist_width_cap(self):
        # hist is a deque capped at width: the live path sees exactly the
        # last width samples, whatever slots says
        g = self._graph([float(i) for i in range(1, 16)], width=10)
        self.assertEqual(g.plot_samples("n", "A", 100),
                         [float(i) for i in range(6, 16)])

    def test_P13_blanks_fill_hist_and_samples_neither_none(self):
        # add_blank: the hist gets None, the samples buffer gets 0.0 -
        # the live path draws the blank; a compressed bucket aggregates
        # it as 0.0 and divides by its full width (never a None)
        g = self._graph([1.0, None, 3.0], width=10)
        self.assertEqual(list(g.samples), [1.0, 0.0, 3.0])
        self.assertEqual(g.plot_samples("n", "A", 10), [1.0, 0.0, 3.0])
        self.assertEqual(g.plot_samples("m", "M", 3), [0.0, 0.0, 3.0])
        self.assertEqual(g.plot_samples("m", "A", 3),
                         [0.0, 0.0, 4.0 / 200])  # (1+0+3) over width 200
        # add(None): the hist only - the samples buffer never sees it,
        # and the straddling bucket aggregates the 2 existing samples
        g2 = bg.Graph(10)
        g2.add(1.0)
        g2.add(None)
        g2.add(3.0)
        self.assertEqual(len(g2.samples), 2)
        self.assertEqual(g2.plot_samples("m", "M", 3), [0.0, 0.0, 3.0])
        self.assertEqual(g2.plot_samples("m", "A", 3),
                         [0.0, 0.0, 4.0 / 200])
    def test_P14_compressed_slots_zero_draws_nothing(self):
        g = self._graph([1.0, 2.0])
        self.assertEqual(g.plot_samples("m", "A", 0), [])
        self.assertEqual(g.plot_samples("w", "M", 0), [])

    def test_P15_unknown_period_key_raises(self):
        g = self._graph([1.0])
        for key in ("x", "M"):  # periods are lowercase; M is an agg key
            with self.assertRaises(KeyError):
                g.plot_samples(key, "A", 10)

    def test_P16_full_block_row_plus_right_edge(self):
        # 1500 samples @1s: the 'h' window is 1800 samples - the single
        # bucket is [-300,1500): straddling the buffer start, full block
        # row 0 [0,1024) + right edge [1024,1500) - the aligned-start
        # boundary (no left edge). A divides by the full width 1800, so
        # the 300 implicit zeros dilute it (624.58, not 749.5)
        g = self._graph([float(i) for i in range(1500)], interval=1.0)
        self.assertEqual(g.plot_samples("h", "A", 1),
                         [sum(range(1500)) / 1800])
        self.assertEqual(g.plot_samples("h", "M", 1), [1499.0])
    def test_P17_full_block_row_plus_left_edge(self):
        # 2048 samples @1s: the 'h' tail [248,2048) is full block row 1
        # [1024,2048) + left edge [248,1024) - the aligned-end boundary
        # (a1 == block boundary: no right edge)
        g = self._graph([float(i) for i in range(2048)], interval=1.0)
        self.assertEqual(g.plot_samples("h", "A", 1),
                         [sum(range(248, 2048)) / 1800])  # 1147.5
        self.assertEqual(g.plot_samples("h", "M", 1), [2047.0])

    def test_P18_full_block_row_plus_both_edges(self):
        # 4200 samples @1s: the 'h' tail [2400,4200) is full block row 3
        # [3072,4096) + both partial edges - the fully table-driven path
        g = self._graph([float(i) for i in range(4200)], interval=1.0)
        self.assertEqual(g.plot_samples("h", "A", 1),
                         [sum(range(2400, 4200)) / 1800])  # 3299.5
        self.assertEqual(g.plot_samples("h", "M", 1), [4199.0])

    def test_P19_bucket_straddling_two_partial_blocks_raw_scans(self):
        # 3048 samples @1s, slots=2: bucket 0 [1248,2148) covers partial
        # block 1 + partial block 2 - no fully covered row, so it raw
        # scans even though it spans more than one block (the documented
        # rare two-partial case); bucket 1 [2148,3048) is one partial
        g = self._graph([float(i) for i in range(3048)], interval=1.0)
        self.assertEqual(g.plot_samples("h", "A", 2),
                         [sum(range(1248, 2148)) / 900,
                          sum(range(2148, 3048)) / 900])   # 1697.5/2597.5
        self.assertEqual(g.plot_samples("h", "M", 2), [2147.0, 3047.0])

    def test_P20_unaligned_trim_keeps_partial_head_row_out_of_the_tables(self):
        # 5032 samples @150s: the one-week cap (4032) trims mid-block,
        # keeping the stale partial head row 0 in the tables. Block 0
        # holds 1e7 poison values that left the buffer: any read of row 0
        # (a r0 below the trim) would inflate A ~1000x and set M to 1e7.
        vals = [1e7] * 1000 + [float(i) for i in range(1000, 5032)]
        g = self._graph(vals, interval=150.0)
        self.assertEqual(g.seq, 5032)
        self.assertEqual(len(g.samples), 4032)   # trimmed to the week cap
        self.assertEqual(g.blk_first, 0)         # unaligned trim keeps row 0
        self.assertEqual(g.plot_samples("w", "A", 1),
                         [sum(range(1000, 5032)) / 4032])  # 3015.5
        self.assertEqual(g.plot_samples("w", "M", 1), [5031.0])

    def test_P21_aligned_trim_drops_the_head_row(self):
        # 5056 samples @150s: the trim lands exactly on a block boundary,
        # deleting row 0 from the tables (blk_first advances) - the bucket
        # then starts its full rows right at the trim
        vals = [1e7] * 1024 + [float(i) for i in range(1024, 5056)]
        g = self._graph(vals, interval=150.0)
        self.assertEqual(len(g.samples), 4032)
        self.assertEqual(g.blk_first, 1)         # aligned trim dropped row 0
        self.assertEqual(g.plot_samples("w", "A", 1),
                         [sum(range(1024, 5056)) / 4032])  # 3039.5
        self.assertEqual(g.plot_samples("w", "M", 1), [5055.0])

    def test_P22_young_buffer_draws_a_single_newest_bucket(self):
        # one minute of data (120 samples @0.5s) against the 'd' axis
        # (172800 samples): divmod(172800, 228) = (757, 204) - the
        # 228-bucket view holds 204 buckets of width 758 + 24 of width
        # 757. Only the newest bucket straddles the buffer start, so
        # exactly one bar is nonzero - a young buffer no longer draws a
        # stretched full-width plot
        g = self._graph([1500.0] * 100 + [3000.0] * 20, width=120)
        self.assertEqual(g.plot_samples("d", "M", 228),
                         [0.0] * 227 + [3000.0])
        # the straddling bucket's A divides by its FULL width 758:
        # (100x1500 + 20x3000) / 758
        self.assertEqual(g.plot_samples("d", "A", 228),
                         [0.0] * 227 + [210000.0 / 758])

    def test_P23_straddling_bucket_A_dilutes_M_does_not(self):
        # 250 samples of 10.0 against the 600-sample 'm' axis, 2 buckets
        # of width 300: the newest bucket is [-50,250) - A divides 2500
        # by 300 (50 implicit zeros), M takes the max of the 250
        # existing samples
        g = self._graph([10.0] * 250)
        self.assertEqual(g.plot_samples("m", "M", 2), [0.0, 10.0])
        self.assertEqual(g.plot_samples("m", "A", 2),
                         [0.0, 2500.0 / 300])

    def test_P24_bucket_widths_are_buffer_length_independent(self):
        # the same (period, interval, slots) gives the same bucket width
        # whatever the buffer length: 100 vs 3000 samples of 5.0 @0.5s,
        # 'd' axis (172800), 100 buckets of width 1728. The long buffer
        # fills the newest bucket [1272,3000) completely; its bucket 98
        # straddles with 1272 of its 1728 width, the short buffer's
        # newest bucket straddles with 100
        gs = self._graph([5.0] * 100)
        gl = self._graph([5.0] * 3000)
        self.assertEqual(gl.plot_samples("d", "A", 100),
                         [0.0] * 98 + [5.0 * 1272 / 1728, 5.0])
        self.assertEqual(gs.plot_samples("d", "A", 100),
                         [0.0] * 99 + [5.0 * 100 / 1728])
        # same 1728 width: the long buffer's newest bucket is full (5.0)
        # while the short buffer's newest bucket is diluted over it
        self.assertEqual(gl.plot_samples("d", "M", 100),
                         [0.0] * 98 + [5.0, 5.0])
        self.assertEqual(gs.plot_samples("d", "M", 100),
                         [0.0] * 99 + [5.0])

    def test_P25_samples_older_than_the_window_leave_every_bucket(self):
        # 5000 samples @1s with a 9999 spike in the oldest 100: the 'h'
        # window is [3200,5000) (18 buckets of width 100) - the spike
        # sits in the buffer but older than the window, so no bucket may
        # see it
        vals = [9999.0] * 100 + [float(i) for i in range(100, 5000)]
        g = self._graph(vals, interval=1.0)
        out_m = g.plot_samples("h", "M", 18)
        self.assertEqual(out_m, [3299.0 + 100.0 * k for k in range(18)])
        self.assertNotIn(9999.0, out_m)
        self.assertEqual(g.plot_samples("h", "A", 18),
                         [3249.5 + 100.0 * k for k in range(18)])

    def test_P26_plot_peak_exact_maxes_and_edges(self):
        # plot_peak = the raw sample max over the same fixed window: the
        # compressed views' scale reference (shared by both aggs)
        g = self._graph([1.0, None, 3.0], width=10)
        self.assertEqual(g.plot_peak("n"), 3.0)  # blanks count as 0.0
        self.assertEqual(self._graph([None, None], width=10).plot_peak("n"),
                         0.0)  # all-blank hist
        empty = bg.Graph(10)
        self.assertEqual(empty.plot_peak("n"), 0.0)
        self.assertEqual(empty.plot_peak("w"), 0.0)  # empty buffer
        # 'h' window [2400,4200): the 9999 spike in the oldest 100
        # samples is outside the window
        g = self._graph([9999.0] * 100 + [float(i) for i in range(100, 4200)],
                        interval=1.0)
        self.assertEqual(g.plot_peak("h"), 4199.0)
        # buffer shorter than the window: the whole buffer is the window
        g = self._graph([float(i) for i in range(300)])
        self.assertEqual(g.plot_peak("d"), 299.0)
        # l falls back to w until the lifetime table lands in phase 4
        self.assertEqual(g.plot_peak("l"), g.plot_peak("w"))
        # unaligned week trim (P20 mirror): the 1e7 poison head row stays
        # in the tables but is excluded from the window
        vals = [1e7] * 1000 + [float(i) for i in range(1000, 5032)]
        self.assertEqual(self._graph(vals, interval=150.0).plot_peak("w"),
                         5031.0)


    def test_P27_trim_folds_dropped_samples_into_hour_rows(self):
        # phase-4 test knob: a 1000-sample ring at 1s keeps ~1.7 minutes
        # while the lifetime hour is 3600 samples. Unique per-position
        # values make any double or lost fold visible; the rows must
        # exactly equal aggregating the dropped samples per hour.
        g = bg.Graph(16, interval=1.0, hist_cap=1000)
        for i in range(8500):
            g.add(float(i))
        base = g.seq - len(g.samples)  # 7500
        H = g.HOUR_SAMPLES            # 3600
        self.assertEqual(len(g.hour_s), 2)  # hours [0,3600), [3600,7200)
        for h in range(2):
            self.assertEqual(g.hour_s[h],
                             sum(float(i) for i in range(h * H, (h + 1) * H)))
            self.assertEqual(g.hour_m[h], float((h + 1) * H - 1))
        # the open head hour: every aged-out sample of [7200, 7500)
        head = base // H
        self.assertEqual(g._hour_acc_c, base - head * H)  # 300 samples
        self.assertEqual(g._hour_acc_s,
                         sum(float(i) for i in range(head * H, base)))
        self.assertEqual(g._hour_acc_m, float(base - 1))

    def test_P28_l_plot_exact_over_trims(self):
        # 3 hours of 1s samples (values constant per hour, one poison
        # spike per hour) against a 1000-sample ring: rows 0 and 1 are
        # closed, the head partial [7200, 9800) and the buffer
        # [9800, 10800) hold hour 2. slots = 3: divmod(10800, 3) cuts
        # exactly along the hour boundaries, so the assembly is
        # bit-exact against the reference over ALL samples ever added.
        g = bg.Graph(16, interval=1.0, hist_cap=1000)
        H = 3600
        n = 3 * H
        all_v = []
        for i in range(n):
            v = (9999.0 if i == 100
                 else 12345.0 if i == n - 1
                 else 7777.0 if i == 7250
                 else float(i // H + 1) * 10)
            g.add(v)
            all_v.append(v)
        self.assertEqual(divmod(n, 3), (H, 0))
        out_a = g.plot_samples("l", "A", 3)
        out_m = g.plot_samples("l", "M", 3)
        # reference: sum/max over every sample of each hour
        self.assertEqual(out_a,
                         [sum(all_v[h * H:(h + 1) * H]) / H for h in range(3)])
        self.assertEqual(out_m,
                         [max(all_v[h * H:(h + 1) * H]) for h in range(3)])
        # pinned exactly: the pre-buffer spikes (idx 100 in the closed
        # row 0, 7250 in the head partial) are included exactly once,
        # via their lifetime cells, and never from the buffer
        self.assertEqual(out_a, [45989 / 3600, 20.0, 128062 / 3600])
        self.assertEqual(out_m, [9999.0, 20.0, 12345.0])

    def test_P29_l_peak_reaches_trimmed_history(self):
        # plot_peak('l') = the max over the visible 'l' axis (slots=None
        # -> the graph's bar width; these runs are shorter than width
        # hours, so the whole run is visible): a peak that lives only in
        # trimmed history is still visible via the hour rows / the open
        # accumulator, while the kept-buffer views miss it
        g = bg.Graph(16, interval=1.0, hist_cap=500)
        for i in range(4500):
            g.add(99999.0 if i == 100 else float(i % 50))
        self.assertEqual(g.plot_peak("l"), 99999.0)  # via closed row 0
        self.assertLess(g.plot_peak("w"), 99999.0)   # the buffer never saw it
        # peak in the open head partial (aged out of the ring, not yet a row)
        g = bg.Graph(16, interval=1.0, hist_cap=1000)
        for i in range(7600):
            g.add(77777.0 if i == 4000 else float(i % 50))
        self.assertEqual(g.plot_peak("l"), 77777.0)
        # empty and all-blank runs
        self.assertEqual(bg.Graph(10).plot_peak("l"), 0.0)
        e2 = bg.Graph(16, interval=1.0, hist_cap=100)
        for _ in range(8000):
            e2.add_blank()
        self.assertEqual(e2.plot_peak("l"), 0.0)

    def test_P30_l_plots_are_pure(self):
        g = bg.Graph(16, interval=1.0, hist_cap=1000)
        for i in range(8000):
            g.add(float(i % 11) + (1.0 if i % 3 == 0 else 0.0))
        snap = lambda: (list(g.hist), g.samples.tobytes(), list(g.blk_s),
                        list(g.blk_m), g.blk_first, g.seq, g.life_sum,
                        g.life_count, g.life_max, list(g.hour_s),
                        list(g.hour_m), g._hour_acc_c, g._hour_acc_s,
                        g._hour_acc_m, g._hour_acc_h)
        before = snap()
        for slots in (1, 3, 50, 3000, 20000):
            g.plot_samples("l", "A", slots)
            g.plot_samples("l", "M", slots)
        g.plot_peak("l")
        self.assertEqual(before, snap())
        # plotting twice gives identical lists
        self.assertEqual(g.plot_samples("l", "A", 50),
                         g.plot_samples("l", "A", 50))
        self.assertEqual(g.plot_samples("l", "M", 50),
                         g.plot_samples("l", "M", 50))

    def test_P31_l_equals_w_style_whole_run_without_trims(self):
        # no trims: the table is empty and 'l' is the w-style
        # aggregation of the raw buffer over the WHOLE RUN
        # (divmod(seq, slots) - the w window would stretch to a week)
        g = self._graph([float(i % 13) for i in range(2000)], interval=1.0)
        n = g.seq
        for slots in (7, 24):
            per, rem = divmod(n, slots)
            sizes = [per] * slots
            for b in range(slots - rem, slots):
                sizes[b] = per + 1
            seg = list(g.samples)  # == every sample ever added (no trims)
            ref_a, ref_m = [], []
            off = 0
            for b in range(slots - 1, -1, -1):
                a1 = n - off
                off += sizes[b]
                a0 = a1 - sizes[b]
                ref_a.insert(0, sum(seg[a0:a1]) / sizes[b])
                ref_m.insert(0, max(seg[a0:a1]))
            self.assertEqual(g.plot_samples("l", "A", slots), ref_a)
            self.assertEqual(g.plot_samples("l", "M", slots), ref_m)
        # a trimmed graph must NOT equal 'w': the older-than-buffer
        # buckets become non-zero via the hour table
        g = bg.Graph(16, interval=1.0, hist_cap=1000)
        for i in range(10800):
            g.add(float(i % 7) + 1.0)
        self.assertNotEqual(g.plot_samples("l", "A", 3), g.plot_samples("w", "A", 3))
        self.assertNotEqual(g.plot_samples("l", "M", 3), g.plot_samples("w", "M", 3))

    def test_P32_l_buckets_cutting_hours_use_row_level_values(self):
        # bucket boundaries that cut through lifetime hours: a cut row
        # contributes width x (row_sum / H) to A and the row max to M
        # - with values constant per hour (0, 1, 2) the rule is exact
        # against the per-sample reference
        g = bg.Graph(16, interval=1.0, hist_cap=100)
        H = 3600
        for i in range(8100):
            g.add(float(i // H))
        all_v = [float(i // H) for i in range(8100)]
        out_a = g.plot_samples("l", "A", 10)
        out_m = g.plot_samples("l", "M", 10)
        # slots = 10: ten 810-wide buckets, every boundary hour-misaligned
        per, rem = divmod(8100, 10)
        self.assertEqual((per, rem), (810, 0))
        exp_a, exp_m = [], []
        off = 0
        for b in range(9, -1, -1):
            a1 = 8100 - off
            off += per
            a0 = a1 - per
            exp_a.insert(0, sum(all_v[a0:a1]) / per)
            exp_m.insert(0, max(all_v[a0:a1]))
        self.assertEqual(out_a, exp_a)
        self.assertEqual(out_m, exp_m)
        # pinned: b4 [3240,4050) mixes 360 x 0.0 (row 0) and 450 x 1.0
        # (row 1); b8 [6480,7290) mixes 720 x 1.0 and 90 x 2.0 (open head)
        self.assertEqual(out_a[4], 450 / 810)
        self.assertEqual(out_a[8], 900 / 810)
        self.assertEqual(out_a, [0.0] * 4 + [450 / 810, 1.0, 1.0, 1.0,
                                             900 / 810, 2.0])
        self.assertEqual(out_m, [0.0] * 4 + [1.0, 1.0, 1.0, 1.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
