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

Import tree (acyclic; the two leaves are stdlib-only):

    ninfer_btop          CLI entry: argparse, pricing, calibration, dispatch
    ├── metrics_calib    /metrics client, pricing, calibration cache  [stdlib-only]
    └── tui              render layer, event loop, row/column geometry
        ├── ninfer_engine_state    engine's live state per poll
        │   └── metrics_calib
        ├── bar_graphs             braille graph engine               [stdlib-only]
        └── metrics_calib

    tui imports nothing from ninfer_btop (no cycle); the from-imports at the top
    of this file bind main_loop/run_text (from tui) plus the five metrics_calib
    names eagerly.

This file is the CLI entry point only: argument parsing, pricing/calibration
resolution, and dispatch to the TUI (tui.main_loop) or --text mode
(tui.run_text). The engine's live state (NinferEngineState) lives in
ninfer_engine_state.py; the screen render layer (including the KV/transfer
row builders) in tui.py; the dot/braille pipeline in bar_graphs.py;
the /metrics client and calibration in metrics_calib.py.

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

from metrics_calib import (
    get_calib,
    load_pricing,
    resolve_calib,
    run_calibration,
    set_calib,
)

from tui import main_loop, run_text


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
        sys.exit(run_text(args.url, args.pricing_rates, args.pricing_model))
    signal.signal(signal.SIGTTOU, signal.SIG_IGN)  # like top: safe even as a background job
    try:
        curses.wrapper(main_loop, args, get_calib())
    except curses.error as e:
        print(f"curses error (run inside a real terminal?): {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
