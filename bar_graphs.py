#!/usr/bin/env python3
"""ninfer_btop bar graphs: sample history and dot/braille rendering.

Graph (the raw sample history with its period-stat block tables),
compute_dots (dot scaling), dots_to_chars (paired braille rendering), and
period_stats (the M/h/H/D/W/L summary figures).

Stdlib-only leaf: imports nothing from the rest of the project; tui
imports from here, so every `import bar_graphs` is cycle-free.
Invariants: compute_dots gives every non-zero sample at least one dot and
clips at the plot's total dot count; one braille character holds a pair of
adjacent samples (the newer in the left dot line, odd counts padded with a
trailing zero), so each poll shifts the graph exactly half a character;
the M/h/H/D/W summaries are exact functions of the one-week block-
partitioned history, while L is an append-only running total over the
process life.
"""
import array
import builtins
import math
from collections import deque


# seconds per day: the D window, and the building block of the week
# buffer span
DAY_SECS = 86400
# one week: the raw sample buffer keeps one week, and the W summary
# window spans the whole kept buffer
WEEK_SECS = 7 * DAY_SECS
# samples per block in the Graph period-stat tables: a float64 partial sum
# and an exact max per block, so a summary window is a few table rows plus
# at most two partial block edges instead of a rescan of the whole buffer
BLK_SAMPLES = 1024


class Graph:
    """Scrolling bar history." The newest sample sits at the left edge: bars
    enter on the left and scroll out on the right. Two samples per drawn
    character (the newer in the left dot line), so a redraw after each poll
    shifts every dot line by exactly half a character - it can never jump,
    and the buffer holds twice as many samples as the drawn column count.

    The prefill/decode graphs scale against their lifetime peak
    (dot_scale, the peak= banner token), so the scale is stable for the
    whole monitor run. `pinned` graphs (the batch graph) map against the
    plot's total dot count instead: one dot = one request, overload
    clips at the top. `floor` is the engine's max_concurrency, which
    sizes the panel height.

    `samples` is the raw history, trimmed to the last week (blanks as
    0.0). The M/h/H/D/W summary figures are recomputed from that buffer
    on every poll via small block tables partitioning it (a float64
    partial sum and an exact max per 1024-sample block - a fraction of a
    millisecond per graph); the figures stay exact functions of the
    stored samples and cannot drift, and the week window spans the whole
    kept buffer. The L column is a separate running total and max over
    the whole process life. Memory stays bounded (about 5 MB per graph at
    the default 0.5 s poll) no matter how long the process runs. The
    figures cover this monitor run, not the engine's life."""

    # M 5 min, h 30 min, H 1 h, D 1 day, W 1 week (== the kept raw history)
    _PERIOD_SECS = (300, 1800, 3600, DAY_SECS, WEEK_SECS)

    def __init__(self, width, floor=0.0, pinned=False, interval=0.5):
        self.interval = interval if interval and interval > 0 else 0.5
        self.hist = deque(maxlen=width)
        self.samples = array.array("f")  # raw history, trimmed to the last week
        self.hist_cap = int(WEEK_SECS / self.interval)  # one week of samples
        self.seq = 0  # absolute count of samples ever recorded (block rows key on it)
        self.blk_s = []  # float64 partial sum per BLK_SAMPLES sample block of `samples`
        self.blk_m = []  # exact max per block (values as stored, float32)
        self.blk_first = 0  # absolute block of row 0 (head rows leave with the buffer)
        self.window_ns = tuple(
            max(1, int(secs / self.interval)) for secs in self._PERIOD_SECS
        )  # M/h/H/D/W window sizes in samples (W == the kept history)
        self.life_sum = 0.0   # whole-process running total (L column)
        self.life_count = 0
        self.life_max = 0.0   # whole-process max (never shrinks: scalar)
        self.floor = floor  # engine max_concurrency (sizes the batch panel)
        self.pinned = pinned  # True: scale is the plot's total dot count

    def _record(self, v):
        """Fold one sample (blanks as 0.0) into the raw history, its block
        table row (O(1)), and the whole-process L aggregates. The M/h/H/D/W
        figures are recomputed from the block tables in period_stats."""
        self.samples.append(v)
        self.seq += 1
        v32 = self.samples[-1]  # the stored value: the tables mirror the buffer
        b = (self.seq - 1) // BLK_SAMPLES
        i = b - self.blk_first  # row of that absolute block
        if i >= len(self.blk_s):
            self.blk_s.append(v32)
            self.blk_m.append(v32)
        else:
            self.blk_s[i] += v32
            if v32 > self.blk_m[i]:
                self.blk_m[i] = v32
        self.life_sum += v
        self.life_count += 1
        if v > self.life_max:
            self.life_max = v
        if len(self.samples) > self.hist_cap:
            del self.samples[:len(self.samples) - self.hist_cap]
            base = self.seq - self.hist_cap
            if base > 0 and base % BLK_SAMPLES == 0:
                # the new buffer start is block-aligned: the head block has
                # left the buffer; mid-block starts keep their (partial) row
                del self.blk_s[:1]
                del self.blk_m[:1]
                self.blk_first += 1

    def adopt_history(self, old):
        """Take over another graph's raw history, block tables, and L
        aggregates (terminal resize); window_ns is identical for the same
        interval."""
        self.samples = old.samples
        self.seq = old.seq
        self.blk_s = old.blk_s
        self.blk_m = old.blk_m
        self.blk_first = old.blk_first
        self.life_sum = old.life_sum
        self.life_count = old.life_count
        self.life_max = old.life_max

    def add(self, v):
        v = None if v is None else max(0.0, float(v))
        self.hist.append(v)
        if v is not None:
            self._record(v)

    def add_blank(self):
        self.hist.append(None)
        self._record(0.0)

    @property
    def values(self):
        return [v for v in self.hist if v is not None]

    def dot_scale(self, rows=1):
        """Dot scale for a `rows`-char plot: pinned graphs map against the
        plot's total dot count (one dot = one request, overload clips at
        the top); the rest map against their lifetime peak (the peak=
        banner token) so the scale is stable for the whole monitor run,
        falling back to the window max (None) before any non-zero sample.
        The M/h/H/D/L summary columns use the same scale as the plot
        next to them."""
        if self.pinned:
            return 4 * rows
        return self.life_max if self.life_max > 0 else None

    def dots(self, rows=1):
        """Dot heights (0..4*rows) for the current window of a `rows`-char
        plot, scaled by dot_scale (the lifetime peak)."""
        return compute_dots(list(self.hist),
                            max=self.dot_scale(rows),
                            total_dots=4 * rows)


# Braille glyph table for the bar graphs, taken from btop (aristocratos/btop,
# Symbols::graph_symbols["braille_up"] in src/btop_draw.cpp) - btop's default
# graph symbol. As in btop, one character holds two adjacent samples
# (btop: older = left dot column, newer = right; we mirror it: newer in the
# left dot column, older in the right), so the full table is used and a
# character can be any of the 25 glyphs - rising and falling edges render
# as diagonals instead of flat tops. Each text row packs four dot rows, so
# a `rows`-row plot resolves 4*rows vertical levels.
# Index: [left level][right level], levels 0..4 (0 = empty cell).
BRAILLE = (
    (" ", "⢀", "⢠", "⢰", "⢸"),
    ("⡀", "⣀", "⣠", "⣰", "⣸"),
    ("⡄", "⣄", "⣤", "⣴", "⣼"),
    ("⡆", "⣆", "⣦", "⣶", "⣾"),
    ("⡇", "⣇", "⣧", "⣷", "⣿"),
)


def compute_dots(samples, max=None, total_dots=4):
    """Dot heights (0..total_dots) for the samples of a bar graph.

    `max` is the value that fills the plot: the graphs pass their scale
    from Graph.dot_scale (pinned graphs the plot's total dot count, the
    rest the lifetime peak with a window-max fallback before traffic),
    or None to autoscale to the window max (the spill/recall transfer
    rows).
    no_dots = ceil(v / (max / total_dots)): every non-zero sample draws at
    least one dot, and overshoots clip at total_dots."""
    if max is None:  # the parameter shadows the builtin max
        vals = [v for v in samples if v is not None]
        max = builtins.max(vals) if vals else 0.0
    if max <= 0:
        return [0] * len(samples)
    step = max / total_dots
    return [0 if v is None else min(total_dots, math.ceil(v / step))
            for v in samples]


def dots_to_chars(dots, rows):
    """The braille characters for a `rows`-row plot from a dots array.

    One sample per single dot line: a braille character is double-wide and
    holds the dot lines of two adjacent samples - the newer sample in the
    left dot column, the older in the right (so a character may be a
    rising/falling glyph). An odd-length array is padded with a trailing
    zero. Returns a rows x len//2 grid of 5x5 glyphs (indexed
    [left level][right level]); grid row 0 is the top of the plot, so a
    sample fills up from the last row. A sample with d dots occupies dots
    [(rows-1-r)*4, (rows-r)*4) of row r's band."""
    if len(dots) % 2:
        dots = [*dots, 0]
    grid = []
    for r in range(rows):
        band = (rows - 1 - r) * 4  # first dot of this row's band
        line = []
        for c in range(len(dots) // 2):
            l = min(4, max(0, dots[2 * c] - band))
            h = min(4, max(0, dots[2 * c + 1] - band))
            line.append(BRAILLE[l][h])
        grid.append(line)
    return grid


def period_stats(graph):
    """(average, max) for the M/h/H/D/W/L summary columns. M/h/H/D/W are
    recomputed on every poll from the block tables partitioning the
    one-week raw history (the week window spans the whole kept buffer):
    each window is full block rows plus at most two raw partial block
    edges, so the figures are exact functions of the stored samples; L
    is the running total and max over the whole process life. The
    figures cover this monitor run, not the engine's life."""
    L = len(graph.samples)
    if not L:
        return []
    seq = graph.seq
    base = seq - L
    first = graph.blk_first  # rows below it no longer mirror the buffer
    out = []
    for n in graph.window_ns:
        m = min(n, L)  # clamps to the whole buffer if shorter
        g0 = seq - m
        fb = (g0 + BLK_SAMPLES - 1) // BLK_SAMPLES  # first fully-covered block
        fe = seq // BLK_SAMPLES  # first block the window does not fully cover
        if fe > fb:
            ts = float(sum(graph.blk_s[fb - first:fe - first]))
            mx = float(max(graph.blk_m[fb - first:fe - first]))
            seg = graph.samples[L - m:fb * BLK_SAMPLES - base]
            if len(seg):
                ts += sum(seg)
                mx = max(mx, max(seg))
            seg = graph.samples[fe * BLK_SAMPLES - base:L]
            if len(seg):
                ts += sum(seg)
                mx = max(mx, max(seg))
        else:
            # the window lies inside at most one block: plain raw scan
            seg = graph.samples[L - m:]
            ts, mx = sum(seg), max(seg)
        out.append((ts / m, mx))
    out.append((graph.life_sum / graph.life_count, graph.life_max))
    return out
