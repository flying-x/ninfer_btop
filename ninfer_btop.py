#!/usr/bin/env python3
"""ninfer_btop - a btop-style live dashboard for a ninfer inference engine.

Polls the engine's Prometheus metrics endpoint and renders:
  * scrolling bar graph: batch size  (ninfer_requests_running)
    (height follows the engine's max concurrency: one braille character
    resolves 4 dot levels = 4 concurrent requests, so the panel is
    ceil(max_concurrency/4) chars high. The scale is the panel's total
    dot count (4 or 8, from the height) - one dot level is always one
    request, and overload clips at the top; for ninfer's cap of 8 the
    panel is 1 or 2 chars high)
  * scrolling bar graph: prefill t/s (rate of ninfer_prefill_tokens_total)
  * scrolling bar graph: decode t/s  (rate of ninfer_generation_tokens_total)
    (btop's braille glyphs: two samples per character - each sample is a
    single vertical dot line, the newer sample in the left dot column, so
    a character may be a rising/falling glyph - and four vertical levels
    per text line; the graphs scroll left to right: the newest sample
    enters at the left edge, and each poll shifts every dot line by
    exactly half a character, so the motion can never jump)
    - the rightmost 6 braille columns of each graph are period summaries,
      labeled MhHDWL on the baseline row: M last 5 minutes, h last
      half-hour, H last hour, D last day, W last week, L this monitor
      process's life-time. Each
      column's left dot line is the period average, its right dot line the
      period max, in the graph's own dot scale (a period max above the
      current display scale clips at the top)
  * totals: input (prompt) / output (generation) tokens
  * Device KV / Host KV / Cache (checkpoint reuse): three side-by-side
    panels sharing one title row (btop-style). The KV columns show the
    engine's unified context pool per domain (engine >= 2026-10-23 state/KV
    unification) — capacity, in use (device only), cached and lifetime peak
    bytes, read from ninfer_context_pool_{bytes,live_bytes,peak_bytes}; the
    cache panel holds "Tok hit" (token-level input hit) and "Req hit"
    (request-level hit) ratios and two context-transfer bars: "Spill (d2h)"
    (device→host) vs "Recall (h2d)" (host→device), read per-direction from
    ninfer_context_transfer_bytes_total
  * request lanes: running / waiting / prefilling / decode-ready / paused
    (the paused figure is painted red whenever non-zero)
  * utilization at the bottom: (prefill time + decode time) / uptime,
    estimated from a startup benchmark of the engine

On startup the script benchmarks the engine: one large uncached prompt
(max prefill t/s) and an 8-way parallel burst (max decode t/s and the
effective max concurrency). The result is cached in ~/.ninfer_btop.json and
reused on the next start while the engine's start time is unchanged (a
restart re-benchmarks automatically); --re-calib drops the cache and forces
a fresh benchmark. The device/host tier split of the token hit needs
--bytes-per-tok (exact context bytes per token, see docs/bytes_per_token.md).

Standard library only. No packages to install.

The cloud-API pricing baseline ($ per million tokens) is read from pricing.yaml
next to this script; select which model's row to use with --pricing-model
(default qwen3.8-27b). If pricing.yaml is missing, the built-in qwen3.8-27b
row is used instead.

Usage:
  python3 ninfer_btop.py --url URL [-i INTERVAL_MS] [--text] [--re-calib]
      [--burst N] [--bytes-per-tok N] [--pricing-model KEY] [--pricing-file PATH]

Keys in TUI mode:
  q / Esc  quit
"""

import argparse
import curses
import signal
import sys
import time

from bar_graphs import compute_dots, dots_to_chars
from metrics_calib import (
    CLOUD_LABEL,
    DEFAULT_PRICING,
    fetch_metrics,
    get_calib,
    get_metric,
    human,
    load_pricing,
    pool,
    resolve_calib,
    run_calibration,
    set_calib,
)


# ---------------------------------------------------------------- state

class Engine:
    def __init__(self, url, interval=0.5, rates=None, label=None):
        self.url = url
        self.interval = interval
        # display name for the "saved vs …" figure: the selected pricing
        # model key plus "Cloud API" (e.g. "qwen3.8-27b Cloud API"), or the
        # generic "cloud API" when no pricing model is supplied
        self.cloud_label = f"{label} Cloud API" if label else CLOUD_LABEL
        self.ok = True
        self.last_err = ""
        self.last_good = None
        self.model = ""
        self.start_time = None
        self.m = {}
        # cloud-API pricing row ($/M: input, cached_input, output); fall back
        # to the built-in qwen3.8-27b row when none is supplied. Accepts either
        # a {input, cached_input, output} dict or three floats in that order.
        if rates is None:
            rates = DEFAULT_PRICING["qwen3.8-27b"]
        elif not isinstance(rates, dict) and len(rates) == 3:
            rates = {"input": rates[0], "cached_input": rates[1],
                     "output": rates[2]}
        self.rates = {
            "input": float(rates.get("input", 0.0)),
            "cached_input": float(rates.get("cached_input", 0.0)),
            "output": float(rates.get("output", 0.0)),
        }
        # previous sample for inter-poll rates
        self._last_t = None
        self._last_gen = None
        self._last_prefill = None
        self._last_spill = None
        self._last_recall = None

    def poll(self):
        try:
            m = fetch_metrics(self.url)
            self.ok = True
            self.last_err = ""
            self.last_good = time.time()
            lbl = m.get("ninfer_model_info__labels", {})
            self.model = lbl.get("model_name", lbl.get("model", ""))
            backend = lbl.get("speculative_backend", "")
            if backend and backend != "none":
                self.model = f"{self.model} (spec: {backend})"
            st = get_metric(m, "ninfer_server_start_time_seconds")
            self.start_time = st
            now = time.monotonic()
            gen = get_metric(m, "ninfer_generation_tokens_total", 0.0)
            pre = get_metric(m, "ninfer_prefill_tokens_total", 0.0)
            # inter-poll rates: the engine commits decode tokens per decode step
            # and prefill tokens per prefill step, so both track the live phase
            # of each request (decode t/s stays 0 during the prefill phase)
            toks = None
            ptok = None
            spill_rate = None
            recall_rate = None
            # per-direction context-transfer bytes: d2h = device->host spill,
            # h2d = host->device recall (summed across state / main_kv /
            # backend_kv resources)
            ser = m.get("ninfer_context_transfer_bytes_total__series") or []
            spill_tot = sum(v for (kv, v) in ser if kv.get("direction") == "d2h")
            recall_tot = sum(v for (kv, v) in ser if kv.get("direction") == "h2d")
            if self._last_gen is not None:
                dt = now - self._last_t
                if dt > self.interval * 0.5:
                    if gen >= self._last_gen:
                        toks = (gen - self._last_gen) / dt
                    if pre >= self._last_prefill:
                        ptok = (pre - self._last_prefill) / dt
                    if self._last_spill is not None and spill_tot >= self._last_spill:
                        spill_rate = (spill_tot - self._last_spill) / dt
                    if self._last_recall is not None and recall_tot >= self._last_recall:
                        recall_rate = (recall_tot - self._last_recall) / dt
            self._last_t, self._last_gen, self._last_prefill = now, gen, pre
            self._last_spill, self._last_recall = spill_tot, recall_tot
            # lifetime utilization: estimated busy time (tokens / calibrated
            # max rates) over engine uptime. An underestimate by design:
            # cached prompt tokens and time below max rate are not counted.
            util = None
            calib = get_calib()
            if calib and calib.get("ok") and st:
                up = time.time() - st
                if up > 1.0:
                    busy = (pre / max(calib["prefill_tps"], 1.0)
                            + gen / max(calib["decode_tps"], 1.0))
                    util = min(1.0, max(0.0, busy / up))
            # lifetime money saved vs the cheapest cloud API (self-host ~ $0)
            in_tot = get_metric(m, "ninfer_prompt_tokens_total", 0.0)
            in_cache = get_metric(m, "ninfer_prompt_tokens_cached_total", 0.0)
            out_tot = get_metric(m, "ninfer_generation_tokens_total", 0.0)
            cloud_cost = ((in_tot - in_cache) / 1e6 * self.rates["input"]
                          + in_cache / 1e6 * self.rates["cached_input"]
                          + out_tot / 1e6 * self.rates["output"])
            # cache reuse
            root = get_metric(m, "ninfer_root_selections_total", 0.0)
            chkp = get_metric(m, "ninfer_checkpoint_selections_total", 0.0)
            tok_hit = (in_cache / in_tot) if in_tot > 0 else None
            req_hit = (chkp / (chkp + root)) if (chkp + root) > 0 else None
            # tier split of the token hits: lifetime host->device recall
            # bytes / bytes-per-token (--bytes-per-tok) ~ how many cached
            # tokens were served from host RAM (clamped to the cached
            # total — it also absorbs non-hit restores); the remainder was
            # served in place from VRAM. None without the flag.
            tok_dev = tok_host = None
            bpt = calib.get("bytes_per_tok") if calib else None
            if bpt and in_tot > 0:
                host_tok = min(in_cache, recall_tot / bpt)
                tok_host = host_tok / in_tot
                tok_dev = (in_cache - host_tok) / in_tot
            # unified context pool, per domain (device = KV, host = state),
            # summed across ranks; 0.0 on pre-unification engine builds
            dev_cap = pool(m, "ninfer_context_pool_bytes", "device", "capacity")
            dev_inuse = pool(m, "ninfer_context_pool_live_bytes", "device", "in_use")
            dev_cached = pool(m, "ninfer_context_pool_live_bytes", "device", "cached")
            dev_peak = pool(m, "ninfer_context_pool_peak_bytes", "device")
            host_cap = pool(m, "ninfer_context_pool_bytes", "host", "capacity")
            host_inuse = pool(m, "ninfer_context_pool_live_bytes", "host", "in_use")
            host_cached = pool(m, "ninfer_context_pool_live_bytes", "host", "cached")
            host_peak = pool(m, "ninfer_context_pool_peak_bytes", "host")
            self.m = m
            return {
                "batch": get_metric(m, "ninfer_requests_running", 0.0),
                "toks_per_s": toks,
                "prefill_tps": ptok,
                "input_total": in_tot,
                "output_total": out_tot,
                "input_cached": in_cache,
                "prefill": get_metric(m, "ninfer_prefill_tokens_total", 0.0),
                "cloud_cost": cloud_cost,
                "dev_cap": dev_cap,
                "dev_inuse": dev_inuse,
                "dev_cached": dev_cached,
                "dev_peak": dev_peak,
                "host_cap": host_cap,
                "host_cached": host_cached,
                "host_peak": host_peak,
                "running": get_metric(m, "ninfer_requests_running", 0.0),
                "waiting": get_metric(m, "ninfer_requests_waiting", 0.0),
                "paused": get_metric(m, "ninfer_requests_paused", 0.0),
                "prefilling": get_metric(m, "ninfer_requests_prefilling", 0.0),
                "decode_ready": get_metric(m, "ninfer_requests_decode_ready", 0.0),
                "ready": get_metric(m, "ninfer_engine_ready", 0.0),
                "max_concurrency": get_metric(m, "ninfer_max_concurrency", 0.0),
                "ctx_window": get_metric(m, "ninfer_max_context_tokens", 0.0),
                "util": util,
                "tok_hit": tok_hit,
                "req_hit": req_hit,
                "tok_dev": tok_dev,
                "tok_host": tok_host,
                "spill_rate": spill_rate,
                "recall_rate": recall_rate,
            }
        except Exception as e:
            self.ok = False
            self.last_err = f"{type(e).__name__}: {e}"
            return None


# width of the one-row spill/recall graphs in characters; fixed and
# independent of the terminal width (buffer = 2 samples per character)
XFER_COLS = 13


def _xfer_row(label, graph, width=XFER_COLS):
    """One cache-panel transfer row: a one-row braille graph over the
    graph's window (the same dots and double-wide characters as the main
    bar graphs - each character holds two adjacent samples, the newer in
    the left dot line), autoscaled to its own window peak, with that peak
    as the side label for scale. Unlike a one-poll bar, a sub-frame transfer burst (a few-MS
    recall DMA) stays visible for the whole graph window. The row is label
    + width + 1 + the peak label, which at 120 cols fits the ⅓-width cache
    column with the longest label (…MB/s, 10 chars). Like the main graphs
    the newest sample is on the left, and a partial window grows from that
    edge."""
    if graph is None:
        return label + " " * (width + 1) + "—"
    hist = list(graph.hist)
    n = min(2 * width, len(hist))
    hist = hist[-n:]  # the newest n samples (the row reads newest first)
    dots = compute_dots(hist, total_dots=4)  # autoscale to its own window peak
    vals = [v for v in hist if v is not None]
    mx = max(vals) if vals else 0.0
    if n % 2:
        dots.append(0)
    chars = "".join(dots_to_chars(dots[::-1], 1)[0]).ljust(width)  # partial window grows from the left
    return label + chars + " " + _rate_val(mx)


def _rate_val(rate):
    """Human rate label for a transfer rate; an em dash when there is no data."""
    if rate is None:
        return "   —"
    return human(rate, "B/s")


def _kv_row(label, val, cap):
    """Labeled context-pool row for the Device KV / Host KV panels: value plus
    the share of pool capacity; 'n/a' when the engine build does not report a
    pool capacity for that domain."""
    if not cap:
        return f"  {label:<9} : n/a"
    pct = f"  ({val / cap * 100:4.1f}%)" if val else ""
    return f"  {label:<9} : {human(val)}{pct}"


def main():
    ap = argparse.ArgumentParser(description="btop-style dashboard for a ninfer engine")
    ap.add_argument("--url", required=True, help="metrics endpoint (required, e.g. http://host:port/metrics)")
    ap.add_argument("-i", "--interval", type=int, default=500,
                help="poll interval in ms (default 500); each poll shifts the "
                     "bar graphs by half a character (one dot line)")
    ap.add_argument("--text", action="store_true", help="one-shot plain-text stats, no TUI")
    ap.add_argument("--re-calib", action="store_true",
                    help="drop the ~/.ninfer_btop.json calibration cache and "
                         "re-run the startup benchmark (default: reuse the "
                         "cache while the engine's start time is unchanged)")
    ap.add_argument("--bytes-per-tok", type=float, default=None, metavar="N",
                    help="context bytes per token for the device/host hit split "
                         "(exact value from the model architecture: full-attention "
                         "layers x 2 x kv_heads x (head_dim x dtype_bytes + scale "
                         "overhead; e.g. INT8 group-64 = 256+8 B per 256-dim head, "
                         "plus the MTP draft layer if pooled). Required for the "
                         "device/host tier split; without it the Tok hit line "
                         "shows the combined figure")
    ap.add_argument("--burst", type=int, default=8,
                    help="parallel requests for the calibration burst (default 8)")
    ap.add_argument("--pricing-model", default="qwen3.8-27b",
                    help="model key in pricing.yaml selecting the cloud-API pricing "
                         "row (default qwen3.8-27b)")
    ap.add_argument("--pricing-file", default=None,
                    help="path to a pricing YAML file "
                         "(default: pricing.yaml next to the script)")
    args = ap.parse_args()

    pricing = load_pricing(args.pricing_file)
    if args.pricing_model not in pricing:
        print(f"error: unknown --pricing-model {args.pricing_model!r}; "
              f"available keys: {', '.join(sorted(pricing))}", file=sys.stderr)
        sys.exit(2)
    args.pricing_rates = pricing[args.pricing_model]

    resolve_calib(args.url, re_calib=args.re_calib,
                  run=lambda: run_calibration(args))
    if args.bytes_per_tok:
        set_calib({**(get_calib() or {}), "bytes_per_tok": args.bytes_per_tok})
    if args.text:
        import frame_chrome
        sys.exit(frame_chrome.run_text(args.url, args.pricing_rates, args.pricing_model))
    signal.signal(signal.SIGTTOU, signal.SIG_IGN)  # like top: safe even as a background job
    import frame_chrome  # lazy: keeps `import ninfer_btop` free of screen code
    try:
        curses.wrapper(frame_chrome.main_loop, args, get_calib())
    except curses.error as e:
        print(f"curses error (run inside a real terminal?): {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # frame_chrome imports ninfer_btop by name; running this file makes the
    # module __main__, so alias it: frame_chrome then reuses this module
    # (one Engine, one xfer-row code path) instead of importing a second copy
    sys.modules["ninfer_btop"] = sys.modules[__name__]
    main()
