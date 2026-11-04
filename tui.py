#!/usr/bin/env python3
"""ninfer_btop tui: the curses screen layer of the dashboard.

Screen composition and display code: frame layout, status line and footer,
colors/attributes, the MhHDWL period-summary column geometry (STAT_LABEL /
STAT_COLS), the cache panel's row builders (_xfer_row, _rate_val, _kv_row,
XFER_COLS), the --text output, and the main_loop event loop.

Imports ninfer_engine_state (NinferEngineState, the engine's live state)
and the two stdlib-only leaves metrics_calib (formatting helpers) and
bar_graphs (Graph, dot/braille rendering, period stats); ninfer_btop
imports THIS module eagerly at top level (main_loop, run_text) and nothing
else in the tree is above it, so the import graph stays acyclic.
frame() takes the resolved calibration dict as a value instead of reading
metrics_calib's calibration state, so the render layer never binds it.
"""
import curses
import time

from bar_graphs import Graph, compute_dots, dots_to_chars, period_stats
from metrics_calib import human, money
from ninfer_engine_state import NinferEngineState

# ---------------------------------------------------------------- text mode

def _pool_body(cap, inuse, cached, peak):
    """One line of context-pool figures for --text output; a note when the
    engine build does not report the pool at all."""
    if not cap:
        return "n/a (pool not reported by this engine build)"
    body = f"cap {human(cap)}"
    if inuse is not None:
        body += f"  in-use {human(inuse)} ({inuse / cap * 100:.1f}%)"
    body += f"  cached {human(cached)} ({cached / cap * 100:.1f}%)"
    body += f"  peak {human(peak)}"
    return body


def run_text(url, rates=None, label=None):
    eng = NinferEngineState(url, rates=rates, label=label)
    s = eng.poll()
    if s is None:
        print(f"error: cannot reach {url}: {eng.last_err}")
        return 1
    print(f"engine : {url}  ({'ready' if s['ready'] else 'not ready'})")
    if eng.model:
        print(f"model  : {eng.model}")
    if s.get("ctx_window"):
        cw = s["ctx_window"]
        print(f"context: {cw:,.0f} tok  ({cw / 1024:.1f} KiToks)")
    print(f"batch  : running={s['running']:.0f} waiting={s['waiting']:.0f} "
          f"prefilling={s['prefilling']:.0f} decode_ready={s['decode_ready']:.0f}")
    print(f"input  : {human(s['input_total'])} tokens total "
          f"(cached {human(s['input_cached'])}, prefill {human(s['prefill'])})")
    print(f"output : {human(s['output_total'])} tokens total")
    u = s.get("util")
    if u is not None:
        print(f"util   : {u * 100:5.1f}%  (est. (prefill time + decode time) / uptime, from startup calibration)")
    else:
        print("util   : n/a  (no calibration data)")
    cc = s.get("cloud_cost")
    if cc is not None:
        print(f"saved  : ≈{money(cc)} vs {eng.cloud_label}  "
              f"({eng.rates['input']} $/M in, {eng.rates['cached_input']} $/M cached in, "
              f"{eng.rates['output']} $/M out)")
    print(f"kv     : device  {_pool_body(s['dev_cap'], s['dev_inuse'], s['dev_cached'], s['dev_peak'])}")
    print(f"         host    {_pool_body(s['host_cap'], None, s['host_cached'], s['host_peak'])}")
    _th, _rh = s.get("tok_hit"), s.get("req_hit")
    _bits = []
    if _th is not None:
        if s.get("tok_dev") is not None and s.get("tok_host") is not None:
            _bits.append(f"input hit {_th * 100:.1f}% "
                         f"({s['tok_dev'] * 100:.1f}% device, "
                         f"{s['tok_host'] * 100:.1f}% host)")
        else:
            _bits.append(f"input hit {_th * 100:.1f}%")
    if _rh is not None:
        _bits.append(f"request hit {_rh * 100:.1f}%")
    print(f"cache  : {', '.join(_bits) if _bits else 'n/a'}")
    return 0


# ---------------------------------------------------------------- curses UI

def make_colors(scr):
    if not curses.has_colors():
        return {}
    curses.start_color()
    curses.use_default_colors()
    c = {}
    # 256-palette entries for the MhHDWL summary columns (terminals that
    # drop SGR-2 dim): 22 = #005f00, 100 = #878700
    DARK_GREEN, DARK_OLIVE = 22, 100
    pairs = {
        "title": (curses.COLOR_CYAN, -1),
        "ok": (curses.COLOR_GREEN, -1),
        "bad": (curses.COLOR_RED, -1),
        "batch": (curses.COLOR_GREEN, -1),
        "toks": (curses.COLOR_YELLOW, -1),
        # 16-color terminals fail to init the 256-palette pair and fall back to A_DIM
        "stat_batch": (DARK_GREEN, -1),
        "stat_toks": (DARK_OLIVE, -1),
        "label": (curses.COLOR_CYAN, -1),
        "dim": (curses.COLOR_WHITE, -1),
        "bar_dim": (curses.COLOR_BLACK, -1),
        "val": (curses.COLOR_WHITE, -1),
        "kv": (curses.COLOR_MAGENTA, -1),
        "peak": (curses.COLOR_GREEN, -1),
    }
    for i, (k, (fg, bg)) in enumerate(pairs.items(), start=1):
        try:
            curses.init_pair(i, fg, bg)
            c[k] = i
        except curses.error:
            pass
    return c


def cstr(scr, c, key, s):
    attr = curses.color_pair(c[key]) if key in c else 0
    return s, attr


def metric_stat_segments(graph, cur_str):
    """Banner suffix for a graph panel title:
    (cur=..,max=..,peak=..,avg=..,min=..).

    Returns a list of (text, is_peak) segments, empty when there are no
    samples yet. The max= token is flagged is_peak so draw_panel paints it in
    the highlighted peak color; peak= is the lifetime max (which also sets
    the graph's scale) and stays plain. min/avg/max cover the whole plotted
    window; cur is the latest sample, passed in as cur_str.
    """
    vals = graph.values
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    av = sum(vals) / len(vals)
    return [
        (f"(cur={cur_str}", False),
        (f",max={mx:,.0f}", True),
        (f",peak={graph.life_max:,.0f}", False),
        (f",avg={av:,.0f}", False),
        (f",min={mn:,.0f}", False),
        (")", False),
    ]


def _stat_attr(c, is_peak):
    """Attribute for a title-banner stat segment: peak value highlighted, rest dim."""
    if is_peak:
        return (curses.color_pair(c["peak"]) | curses.A_BOLD) if "peak" in c else curses.A_REVERSE
    return curses.color_pair(c.get("dim", 0))


def draw_panel(scr, y, title, key, c, width, stats=None):
    """Draw a panel title row.

    When ``stats`` (segments from ``metric_stat_segments``) is given, the
    suffix "─  (cur=..,max=..,avg=..,min=..)" is appended right after the
    title with the peak value highlighted; otherwise the title bar is dashed
    out to the full width as before.
    """
    w = width - 1  # leave the last column empty (see note in frame())
    if stats:
        suffix_len = len("─  ") + sum(len(t) for t, _ in stats)
        avail = max(0, w - 1 - suffix_len)  # title room after the leading "─"
        lead = "─" + title[:avail] + "─  "
        try:
            scr.addnstr(y, 0, lead[:w], w)
        except curses.error:
            pass
        x = len(lead[:w])
        for text, is_peak in stats:
            if x >= w:
                break
            n = min(len(text), w - x)
            if n <= 0:
                break
            try:
                scr.addnstr(y, x, text[:n], n, _stat_attr(c, is_peak))
            except curses.error:
                break
            x += n
    else:
        avail = w - 1  # columns after the leading "─"
        body = title[:avail]  # truncate the title only if it overflows
        scr.addnstr(y, 0, "─" + body + "─" * (avail - len(body)), w)
    return y + 1


def draw_triple_panel(scr, y, left_title, mid_title, right_title, c, width):
    """One title row holding three labeled columns, each a third of the width.

    The columns occupy [0, s1), [s1, s2) and [s2, width-1). Lets three
    panels (Device KV / Host KV / Cache) run side by side in a quarter of
    the vertical space of three stacked panels. Returns (y + 1, s1, s2) so
    callers write their content rows into the three columns one row below
    the titles.
    """
    w = width - 1
    s1 = w // 3
    s2 = w - 1 - w // 3
    for x0, x1, title in ((0, s1, left_title), (s1, s2, mid_title),
                          (s2, w, right_title)):
        span = x1 - x0
        body = "─" + title
        txt = (body + "─" * max(0, span - len(body)))[:span]
        try:
            scr.addnstr(y, x0, txt, span)
        except curses.error:
            pass
    return y + 1, s1, s2


# rightmost columns of each bar graph: one period summary per label
# character (avg/max over the last 5 min / half-hour / hour / day /
# week / process life); period_stats emits exactly len(STAT_LABEL)
# figures
STAT_LABEL = "MhHDWL"
STAT_COLS = len(STAT_LABEL)

# ESC: classic value 27; some curses builds (e.g. the box's system
# python3.14) omit the KEY_ESCAPE constant entirely
KEY_ESCAPE = getattr(curses, "KEY_ESCAPE", 27)


def draw_bar_graph(scr, y, graph: Graph, rows, width, key, c):
    hist_vals = list(graph.hist)
    dots = graph.dots(rows)  # one dot height per sample
    # two samples per braille character: the plot is half the buffer wide;
    # the buffer holds oldest -> newest, but the newest sample is on the
    # left, so the window is read newest first
    # the rightmost STAT_COLS columns are reserved for the MhHDWL summaries
    n = min(len(hist_vals), 2 * (width - STAT_COLS))
    grid = dots_to_chars(dots[-n:][::-1], rows)
    plot_w = min(width - STAT_COLS, (n + 1) // 2)
    base_attr = curses.color_pair(c.get(key, 0))
    # the newest sample is on the left: the leftmost character holds it and
    # the second-newest sample, and the graph scrolls left to right (bars
    # enter left, exit right); the newer sample always sits in the left dot
    # line of its character, so each poll shifts every dot line by exactly
    # half a character - the pairing never flips, it cannot jump; a partial
    # (not-yet-full) window grows from the left edge
    for col in range(plot_w):
        for r in range(rows):
            ch = grid[r][col]
            if ch == " ":
                continue
            try:
                scr.addstr(y + r, 1 + col, ch, base_attr)
            except curses.error:
                pass
    # rightmost STAT_COLS columns: M/h/H/D/W/L summaries (left dot line = avg, right
    # = max), grayed out against the moving bars (darker color pair on
    # 256-color terminals, A_DIM fallback otherwise), in the plot's dot
    # scale
    stats = period_stats(graph)
    stat_key = "stat_" + key
    stat_attr = curses.color_pair(c[stat_key]) if stat_key in c \
        else base_attr | curses.A_DIM
    if stats:
        for col, (avg, mx) in enumerate(stats):
            g5 = dots_to_chars(compute_dots([avg, mx],
                                            max=graph.dot_scale(rows),
                                            total_dots=4 * rows), rows)
            for r in range(rows):
                ch = g5[r][0]
                if ch == " ":
                    continue
                try:
                    scr.addstr(y + r, width - STAT_COLS + 1 + col, ch, stat_attr)
                except curses.error:
                    pass
    # baseline; the period labels sit under their summary columns and a
    # time-window label sits in the middle of the plot: two samples per
    # character, so the region spans 2*(width-STAT_COLS) samples
    plot_len = width - STAT_COLS
    try:
        scr.addnstr(y + rows, 1, "─" * plot_len, plot_len)
    except curses.error:
        pass
    span = f"{2 * plot_len * graph.interval:g} seconds"
    x = max(1, 1 + (plot_len - len(span)) // 2)
    try:
        scr.addnstr(y + rows, x, span, plot_len - (x - 1),
                    curses.color_pair(c.get("dim", 0)))
    except curses.error:
        pass
    if stats:
        try:
            scr.addnstr(y + rows, width - STAT_COLS + 1, STAT_LABEL, STAT_COLS, stat_attr)
        except curses.error:
            pass
    # the cur/min/max/avg figures live in the panel title banner now
    return y + rows + 1

# --- layout constants

# tallest a bar-graph plot may grow (braille rows)
MAX_PLOT_ROWS = 6
# minimum drawn terminal width (columns) for the layout
MIN_WIDTH = 20
# footer UTIL colour thresholds: "ok" below warn, "toks" below bad, "bad" at/above
UTIL_WARN, UTIL_BAD = 0.7, 0.9
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


def frame(scr, eng, graphs, c, calib=None):
    scr.erase()
    try:
        max_y, max_x = scr.getmaxyx()
    except Exception:
        return
    width = max_x - 1

    s = eng.m and eng.last_frame
    model = eng.model or "?"
    if eng.start_time:
        up = time.time() - eng.start_time
        h_, rem = divmod(int(up), 3600)
        m_, sec = divmod(rem, 60)
        uptime = f"{h_:02d}:{m_:02d}:{sec:02d}"
    else:
        uptime = "?"
    status = "READY" if (eng.ok and s and s["ready"]) else ("DOWN" if not eng.ok else "BUSY")
    status_key = "ok" if status == "READY" else "bad"
    head = f"ninfer btop ─ {model} ─ engine: {status} ─ up {uptime} ─ {eng.url}"
    try:
        # never fill the last column: a cursor parked at EOL puts ncurses in
        # pending-wrap state and its relative row moves land 1-2 rows low
        scr.addnstr(0, 0, head[:width - 1], width - 1, curses.color_pair(c.get("title", 0)))
    except curses.error:
        pass

    # status line (row 1): calibration figures, plus the engine's context
    # window in KiToks (1 KiToks = 1024 tokens) when the engine reports it
    ctx = (s or {}).get("ctx_window")
    ctx_txt = f"{ctx / 1024:.1f} KiToks" if ctx else ""
    if calib and calib.get("ok"):
        cd = calib
        peak_attr = (curses.color_pair(c["peak"]) | curses.A_BOLD) if "peak" in c else curses.A_REVERSE
        lbl = curses.color_pair(c.get("label", 0))
        segs = [(f"calibrated in {cd['duration']:.1f}s:  concurrency ", lbl),
                (f"{cd['concurrency']}", peak_attr),
                ("  ·  prefill ", lbl),
                (f"{cd['prefill_tps']:,.0f} t/s", peak_attr),
                ("  ·  decode ", lbl),
                (f"{cd['decode_tps']:,.0f} t/s", peak_attr)]
        if ctx_txt:
            segs += [("  ·  ctx ", lbl), (ctx_txt, peak_attr)]
        x = 0
        for txt, a in segs:
            n = min(len(txt), width - x)
            if n <= 0:
                break
            try:
                scr.addnstr(1, x, txt[:n], n, a)
            except curses.error:
                break
            x += n
    elif calib and calib.get("error"):
        prefix = "calibration failed: "  # 5 = margin before the trailing note
        msg = f"{prefix}{calib['error'][:width - len(prefix) - 5]}  (util n/a)"
        if ctx_txt:
            msg += f"  ·  ctx {ctx_txt}"
        try:
            scr.addnstr(1, 0, msg[:width - 1], width - 1, curses.color_pair(c.get("dim", 0)))
        except curses.error:
            pass
    elif ctx_txt:
        # no calibration data: the status line still carries the context window
        try:
            scr.addnstr(1, 0, ctx_txt, width - 1, curses.color_pair(c.get("dim", 0)))
        except curses.error:
            pass

    # row 0 header, row 1 status line; the panels start on row 2
    y = 2
    if not eng.ok:
        # the banner sits below the header/calibration block and pushes the
        # panels down instead of sharing their rows
        try:
            msg = "cannot reach engine: " + eng.last_err
            scr.addnstr(y, 0, msg, min(len(msg) - 1, width - 1),
                        curses.color_pair(c.get("bad", 0)))
            scr.addnstr(y + 1, 0, f"last good sample: {eng.last_good}", width - 1)
        except curses.error:
            pass
        y += 2
    g_batch = graphs["batch"]
    g_toks = graphs["toks"]
    # fixed rows: header + status line + 3 titles + 3 baselines
    # + 1 lane + 1 totals title + 2 totals + 1 shared Device KV/Host KV/Cache
    # title + 4 shared rows (incl. the spill/recall bars) + 1 footer; the
    # cur/min/max/avg figures sit in the panel titles, plots take the rest
    fixed = 18
    # the batch panel's height follows the engine's max concurrency (kept in
    # Graph.floor, updated on every poll): one braille character resolves 4
    # dot levels, so N concurrent requests need ceil(N/4) characters
    # (<=4 -> 1 char, 5-8 -> 2, 9-12 -> 3, ...). The panel's scale is its
    # own total dot count (4*rows), which makes each dot level exactly one
    # request. ninfer caps concurrency at 8, so its batch graph is 1 or 2
    # chars high. Before the first successful poll (/metrics is unreachable
    # until after the engine's warmup, and then max_concurrency is in every
    # scrape) the panel height falls back to the even three-way split; the
    # scale stays pinned to the height's dot count.
    if g_batch.floor > 0:
        rows_batch = max(1, min(MAX_PLOT_ROWS, -(-int(g_batch.floor) // 4)))
    else:
        rows_batch = max(1, min(MAX_PLOT_ROWS, (max_y - fixed) // 3))
    # prefill/decode share the rows the batch panel does not use
    plot_rows = max(1, min(MAX_PLOT_ROWS, (max_y - fixed - rows_batch) // 2))

    # --- batch panel
    b = (s or {}).get("running")
    cur_b = f"{b:.0f}" if b is not None else "—"
    y = draw_panel(scr, y, "Batch size  (ninfer_requests_running)", "batch", c, width,
                   metric_stat_segments(g_batch, cur_b))
    y = draw_bar_graph(scr, y, g_batch, rows_batch, width, "batch", c)
    if s:
        # paused is the one lane number that signals a problem: a live
        # request stalled on a capacity event, so paint it red when non-zero
        lane_base = (f"running {s['running']:.0f}  ·  waiting {s['waiting']:.0f}  ·  "
                     f"prefilling {s['prefilling']:.0f}  ·  "
                     f"decode-ready {s['decode_ready']:.0f}  ·  ")
        lane_end = f"paused {s['paused']:.0f}"
        end_attr = curses.color_pair(c.get("bad", 0)) if s["paused"] else 0
        x = 1
        for txt, attr in ((lane_base, 0), (lane_end, end_attr)):
            n = min(len(txt), width - 1 - x)
            if n <= 0:
                break
            try:
                scr.addnstr(y, x, txt[:n], n, attr)
            except curses.error:
                break
            x += n
        y += 1

    # --- prefill tokens/s panel (prompt processing; decode t/s is flat
    #     during the prefill phase, this shows what the engine is doing then)
    g_pre = graphs.get("prefill")
    if g_pre is not None:
        p = (s or {}).get("prefill_tps")
        cur_p = f"{p:,.0f}" if p is not None else "—"
        y = draw_panel(scr, y, "Prefill t/s  (prompt processing)", "toks", c, width,
                       metric_stat_segments(g_pre, cur_p))
        y = draw_bar_graph(scr, y, g_pre, plot_rows, width, "toks", c)

    # --- decode tokens/s panel
    t = (s or {}).get("toks_per_s")
    cur_t = f"{t:,.0f}" if t is not None else "—"
    y = draw_panel(scr, y, "Decode t/s  (generation)", "toks", c, width,
                   metric_stat_segments(g_toks, cur_t))
    y = draw_bar_graph(scr, y, g_toks, plot_rows, width, "toks", c)

    # --- totals panel
    y = draw_panel(scr, y, "Token totals", "title", c, width)
    if s:
        lines = [
            f"  Input  tokens : {s['input_total']:>14,.0f}   (prefill {human(s['prefill'])}, cached from checkpoints {human(s['input_cached'])})",
            f"  Output tokens : {s['output_total']:>14,.0f}",
        ]
        for i, ln in enumerate(lines):
            try:
                scr.addnstr(y + i, 0, ln[:width - 1], width - 1, curses.color_pair(c.get("val", 0)))
            except curses.error:
                pass
        y += len(lines)
    else:
        try:
            scr.addnstr(y, 1, "waiting for first sample…", width - 2)
        except curses.error:
            pass
        y += 1

    # --- device KV + host KV + cache, three side-by-side panels sharing one
    #     title row (btop-style): a quarter of the space three stacked
    #     panels would need
    if y + 5 <= max_y - 1:
        if s:
            left = [f"  Capacity  : " + (human(s['dev_cap']) if s['dev_cap'] else "n/a"),
                    _kv_row("In use", s["dev_inuse"], s["dev_cap"]),
                    _kv_row("Cached", s["dev_cached"], s["dev_cap"]),
                    _kv_row("Peak", s["dev_peak"], s["dev_cap"])]
            mid = [f"  Capacity  : " + (human(s['host_cap']) if s['host_cap'] else "n/a"),
                   _kv_row("Cached", s["host_cached"], s["host_cap"]),
                   _kv_row("Peak", s["host_peak"], s["host_cap"]),
                   ""]
            th, rh = s.get("tok_hit"), s.get("req_hit")
            td, ths = s.get("tok_dev"), s.get("tok_host")
            if td is not None and ths is not None:
                # tier split (approximate, see poll()); no space between
                # the groups: at 120 cols this column is 38 chars wide and
                # the line must fit, 100.0% included
                in_line = (f"  Tok hit : {td * 100:4.1f}% (device)"
                           f"{ths * 100:4.1f}% (host)")
            elif th is not None:
                in_line = f"  Tok hit : {th * 100:5.1f}%"
            else:
                in_line = "  Tok hit :   n/a  (no input yet)"
            req_line = (f"  Req hit : {rh * 100:5.1f}%" if rh is not None
                        else "  Req hit :   n/a  (no bindings yet)")
            right = [in_line, req_line,
                     _xfer_row("  Spill  (d2h)  ", graphs.get("xfer_d2h")),
                     _xfer_row("  Recall (h2d)  ", graphs.get("xfer_h2d"))]
        else:
            left = ["  Capacity  : —", "  In use    : —", "  Cached    : —",
                    "  Peak      : —"]
            mid = ["  Capacity  : —", "  Cached    : —", "  Peak      : —", ""]
            right = ["  Tok hit : —", "  Req hit : —",
                     _xfer_row("  Spill  (d2h)  ", graphs.get("xfer_d2h")),
                     _xfer_row("  Recall (h2d)  ", graphs.get("xfer_h2d"))]
        y, s1, s2 = draw_triple_panel(scr, y, "Device KV", "Host KV",
                                      "Cache (checkpoint reuse)", c, width)
        w0 = s1 - 1
        w1 = s2 - 1 - s1
        w2 = width - 1 - s2
        for i in range(4):
            try:
                scr.addnstr(y + i, 0, left[i][:w0], w0)
                scr.addnstr(y + i, s1, mid[i][:w1], w1)
                scr.addnstr(y + i, s2, right[i][:w2], w2)
            except curses.error:
                pass
        y += 4

    # --- footer (live utilization at the bottom)
    y = max_y - 1
    util = (s or {}).get("util")
    if util is not None:
        ukey = "ok" if util < UTIL_WARN else ("toks" if util < UTIL_BAD else "bad")
        utext = f" UTIL {util * 100:5.1f}%"
    else:
        ukey, utext = "dim", "UTIL n/a "
    _cc = (s or {}).get("cloud_cost")
    if _cc is not None:
        stext, skey = f" ≈{money(_cc)} saved vs {eng.cloud_label}", "ok"
    else:
        stext, skey = "", "dim"
    x = 0
    for txt, key in ((" q quit", "dim"),
                     (utext, ukey),
                     (stext, skey)):
        if x >= width:
            break
        n = min(len(txt), width - x - 1)
        if n <= 0:
            break
        try:
            scr.addnstr(y, x, txt[:n], n, curses.color_pair(c.get(key, 0)))
        except curses.error:
            break
        x += n
    scr.refresh()



def main_loop(stdscr, args, calib=None):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    c = make_colors(stdscr)

    max_y, max_x = stdscr.getmaxyx()
    gwidth = max(MIN_WIDTH, max_x - 1)  # drawn columns (x=1 .. max_x-1)
    interval = args.interval / 1000.0
    # two samples per drawn character: the buffer holds exactly what the
    # scroll region draws (width-STAT_COLS chars = 2*(width-STAT_COLS)
    # samples) - the rightmost STAT_COLS columns are the MhHDWL summaries,
    # so a wider buffer would
    # hide scroll data under them and the banner max/avg/min (computed over
    # the buffer) would lag the visible plot
    graphs = {"batch": Graph(2 * (gwidth - STAT_COLS), pinned=True, interval=interval),  # pinned: 1 dot = 1 request, always
              "toks": Graph(2 * (gwidth - STAT_COLS), interval=interval),
              "prefill": Graph(2 * (gwidth - STAT_COLS), interval=interval),
              # transfer graphs: fixed XFER_COLS-char width (2 samples per char,
              # ~13 s at the default 500 ms poll), independent of terminal width
              "xfer_d2h": Graph(2 * XFER_COLS), "xfer_h2d": Graph(2 * XFER_COLS)}
    eng = NinferEngineState(args.url, interval, args.pricing_rates, label=args.pricing_model)
    last_poll = 0.0
    running = True

    while running:
        now = time.time()
        # each poll appends exactly one sample (half a character); the newer
        # sample always occupies the left dot line of its character, so a
        # redraw after every poll shifts every dot line by exactly half a
        # character - the pairing never flips, the motion cannot jump
        if now - last_poll >= interval:
            last_poll = now
            s = eng.poll()
            eng.last_frame = s
            if s is not None:
                if s["max_concurrency"] > 0:
                    graphs["batch"].floor = s["max_concurrency"]
                graphs["batch"].add(s["running"])
                graphs["toks"].add(s["toks_per_s"])
                graphs["prefill"].add(s["prefill_tps"])
                graphs["xfer_d2h"].add(s.get("spill_rate"))
                graphs["xfer_h2d"].add(s.get("recall_rate"))
            else:
                for k in graphs:
                    graphs[k].add_blank()
            frame(stdscr, eng, graphs, c, calib)

        # resync graph width on terminal resize
        ny, nx = stdscr.getmaxyx()
        if abs(nx - 1 - gwidth) > 2:
            gwidth = max(MIN_WIDTH, nx - 1)
            for k in ("batch", "toks", "prefill"):  # xfer graphs stay 13 chars
                old = list(graphs[k].hist)
                g2 = Graph(2 * (gwidth - STAT_COLS), floor=graphs[k].floor, pinned=graphs[k].pinned,
                           interval=graphs[k].interval)
                g2.hist.extend(old[-2 * gwidth:])
                g2.adopt_history(graphs[k])  # raw history + L aggregates
                graphs[k] = g2
            frame(stdscr, eng, graphs, c, calib)

        ch = stdscr.getch()
        if ch == -1:
            # small sleep so we don't spin when no key
            time.sleep(0.05)
            continue
        if ch in (ord("q"), ord("Q"), KEY_ESCAPE):
            running = False

    return 0


