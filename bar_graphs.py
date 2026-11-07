#!/usr/bin/env python3
"""ninfer_btop bar graphs: sample history and dot/braille rendering.

Graph class
  - the raw sample history with its period-stat block tables
  - the plot_samples period/agg selection for the main plots
  - over the coarse lifetime hour table to support the "l" view
  - plot_peak, the compressed views' scale reference
  - compute_dots (dot scaling)
  - dots_to_chars (paired braille rendering)
  - period_stats (the M/h/H/D/W/L summary figures).

Stdlib-only leaf: imports nothing from the rest of the project; tui
imports from here, so every `import bar_graphs` is cycle-free.

Invariants:
  - compute_dots gives every non-zero sample at least one dot and
    clips at the plot's total dot count
  - one braille character holds a pair of adjacent samples (the
    newer in the left dot line, odd counts padded with a
    trailing zero), so each poll shifts the graph exactly half a
    character
  - the M/h/H/D/W summaries are exact functions of the one-week block-
    partitioned history, while L is coarse grain hourly stats that
    grow unbound but stays small in any reasonable life time length.
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
# seconds per lifetime-table row: the coarse hour resolution of the 'l'
# (whole-run) plots; a Graph's row width is HOUR_SECS/interval samples
HOUR_SECS = 3600


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
    kept buffer. plot_samples reuses the same tables when aggregating
    plot periods (full block rows plus at most two raw partial edges per
    bucket). The 'l' (whole-run) plots read a coarse lifetime table
    instead: one (sum, max) row per hour that has fully aged out of the
    ring - folded in amortized as samples trim, plus the open head-hour
    partial - so a week of process life costs a few hundred bytes. The
    L column is a separate running total and max over
    the whole process life. Memory stays bounded (about 5 MB per graph at
    the default 0.5 s poll) no matter how long the process runs. The
    figures cover this monitor run, not the engine's life."""

    # M 5 min, h 30 min, H 1 h, D 1 day, W 1 week (== the kept raw history)
    _PERIOD_SECS = (300, 1800, HOUR_SECS, DAY_SECS, WEEK_SECS)
    # seconds per plot period key for plot_samples (the same spans as
    # _PERIOD_SECS; n = the live hist tail, l = the whole run over the
    # lifetime hour table)
    _PLOT_PERIOD_SECS = {"m": 300, "h": 1800, "H": HOUR_SECS,
                         "d": DAY_SECS, "w": WEEK_SECS}

    def __init__(self, width, floor=0.0, pinned=False, interval=0.5,
                 hist_cap=None):
        self.interval = interval if interval and interval > 0 else 0.5
        self.hist = deque(maxlen=width)
        self.samples = array.array("f")  # raw history, trimmed to the last week
        # one week of samples by default; a test-visible override shrinks
        # the ring so lifetime-table folds are exercisable in a unit test
        self.hist_cap = (hist_cap if hist_cap is not None and hist_cap > 0
                         else int(WEEK_SECS / self.interval))
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
        self.HOUR_SAMPLES = max(1, int(HOUR_SECS / self.interval))  # 'l' row width
        self.hour_s = []  # float64 sum per fully aged-out lifetime hour
        self.hour_m = []  # exact max per lifetime hour (values as stored, float32)
        # samples of the buffer's head hour that have already aged out of
        # the ring: they wait in this accumulator until the hour closes
        self._hour_acc_c = 0
        self._hour_acc_s = 0.0
        self._hour_acc_m = 0.0
        self._hour_acc_h = -1  # hour of the accumulator (-1: none)
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
            old_base = self.seq - len(self.samples)
            dropped = self.samples[:len(self.samples) - self.hist_cap]
            del self.samples[:len(self.samples) - self.hist_cap]
            base = self.seq - self.hist_cap
            if base > 0 and base % BLK_SAMPLES == 0:
                # the new buffer start is block-aligned: the head block has
                # left the buffer; mid-block starts keep their (partial) row
                del self.blk_s[:1]
                del self.blk_m[:1]
                self.blk_first += 1
            self._fold_hours(dropped, old_base, base)

    def _fold_hours(self, dropped, old_base, new_base):
        """Fold the samples that just aged out of the ring into the
        lifetime hour table: one (sum, max) row per fully aged-out
        HOUR_SAMPLES-sample hour. Samples of the buffer's head hour that
        have aged out wait in the open accumulator until the hour
        closes. Amortized O(len(dropped)): each sample is folded exactly
        once, on the trim that drops it - no rescans on the hot add()
        path."""
        H = self.HOUR_SAMPLES
        tmp = []  # (hour, sum, max) of brand-new closing hours, in order
        for i, v in enumerate(dropped, start=old_base):
            h = i // H
            if (h + 1) * H > new_base:  # the head hour: stays open
                self._hour_acc_h = h
                self._hour_acc_c += 1
                self._hour_acc_s += v
                if v > self._hour_acc_m:
                    self._hour_acc_m = v
            elif h == self._hour_acc_h:  # the open hour closes with this trim
                self._hour_acc_c += 1
                self._hour_acc_s += v
                if v > self._hour_acc_m:
                    self._hour_acc_m = v
            elif not tmp or tmp[-1][0] != h:  # a brand-new closing hour
                tmp.append([h, 0.0, 0.0])
            if h != self._hour_acc_h and (h + 1) * H <= new_base:
                tmp[-1][1] += v
                if v > tmp[-1][2]:
                    tmp[-1][2] = v
        if self._hour_acc_h >= 0 and (self._hour_acc_h + 1) * H <= new_base:
            # the open hour has fully aged out: it becomes the next row
            self.hour_s.append(self._hour_acc_s)
            self.hour_m.append(self._hour_acc_m)
            self._hour_acc_c = 0
            self._hour_acc_s = 0.0
            self._hour_acc_m = 0.0
            self._hour_acc_h = -1
        for h, s, m in tmp:
            self.hour_s.append(s)
            self.hour_m.append(m)

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
        self.hour_s = old.hour_s
        self.hour_m = old.hour_m
        self._hour_acc_c = old._hour_acc_c
        self._hour_acc_s = old._hour_acc_s
        self._hour_acc_m = old._hour_acc_m
        self._hour_acc_h = old._hour_acc_h

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

    def plot_samples(self, period: str, agg: str, slots: int) -> list[float]:
        """The sample values a plot with `slots` sample positions (two
        per drawn character) draws. `period`: n = the live hist tail
        (blanks as 0.0 - today's draw; it may be shorter than slots, the
        plot grows a partial period from the left). The compressed
        periods use a FIXED TIME AXIS: the window is the last
        total = int(period_secs/interval) samples, cut into exactly
        `slots` buckets anchored to the absolute sample index (the
        newest bucket ends at seq), sizes from divmod(total, slots) -
        the newest `rem` buckets get one extra sample. A bucket width
        depends only on (period, interval, slots), never on how much
        data is buffered, so a young buffer draws a few newest buckets
        plus empty older ones instead of a stretched full-width plot,
        and switching periods rescales the axis visibly. `agg`: A =
        average, M = max per bucket (ignored for n). Buckets entirely
        older than the kept buffer are 0.0 (skipped without scanning); a
        bucket straddling the buffer start aggregates only the
        available samples, but A divides by the FULL bucket width -
        unmonitored time counts as 0.0, exactly like stored blanks, so a
        sparse window's Avg bar sits low while the same bucket's M bar
        is tall; M ignores the implicit zeros (samples are >= 0).
        Bucket interiors aggregate through the per-1024-sample block
        tables (full rows plus at most two raw partial edges, the
        period_stats pattern; the straddling bucket's left edge clamps
        to the buffer start): the A average is a float64-approximate
        function of the stored samples, M the exact float32 max. "l"
        (life) is the WHOLE RUN - window [0, seq): its buckets aggregate
        the lifetime hour rows (one per fully aged-out HOUR_SAMPLES
        sample hour) plus the open head-hour accumulator plus the kept
        buffer; a bucket edge cutting a lifetime hour takes
        width x (row_sum / row_width) for A and the row max for M
        (exact for values constant within the hour). The run's samples
        are never implicit zeros - every pre-buffer sample was folded
        into a row - so 'l' averages dilute only where the run itself
        has gaps. Pure: it touches neither the hist, the block tables,
        the lifetime table, nor the L aggregates, so the MhHDWL columns
        are unaffected. The result feeds compute_dots/dots_to_chars
        unchanged."""
        if period == "n":
            return [0.0 if v is None else float(v) for v in self.hist]
        if slots <= 0:
            return []
        base = self.seq - len(self.samples)  # absolute of samples[0]
        out = [0.0] * slots
        total = (self.seq if period == "l"  # the whole run, phase 4
                 else int(self._PLOT_PERIOD_SECS[period] / self.interval))
        per, rem = divmod(total, slots)
        sizes = [per] * slots
        for b in range(slots - rem, slots):  # the newest `rem` buckets
            sizes[b] = per + 1
        first = self.blk_first  # rows below it no longer mirror the buffer
        is_a = agg == "A"
        is_l = period == "l"
        off = 0  # samples newer than the current bucket's end
        for b in range(slots - 1, -1, -1):  # newest bucket ends at seq
            a1 = self.seq - off
            off += sizes[b]
            size = sizes[b]
            if is_l:
                # window [0, seq): nothing is implicit zero; only empty
                # buckets (seq shorter than the axis) stay 0.0
                if size == 0 or a1 <= 0:
                    continue
                out[b] = self._l_bucket(a1 - size, a1, size, is_a, base)
                continue
            if size == 0 or a1 <= base:
                continue  # zero pad, or entirely older than the kept buffer
            a0 = a1 - size
            if a0 < base:
                a0 = base  # straddling bucket: left edge clamps to the buffer
            r0 = (a0 + BLK_SAMPLES - 1) // BLK_SAMPLES  # first full row
            r1 = a1 // BLK_SAMPLES - 1  # last full row
            if r1 < r0:
                # the bucket lies inside at most one block row: raw scan
                seg = self.samples[a0 - base:a1 - base]
                out[b] = sum(seg) / size if is_a else max(seg)
                continue
            if is_a:
                v = sum(self.blk_s[r0 - first:r1 - first + 1])
                e = r0 * BLK_SAMPLES - base  # left partial edge [a0, e)
                if e > a0 - base:
                    v += sum(self.samples[a0 - base:e])
                e = (r1 + 1) * BLK_SAMPLES - base  # right partial edge
                if e < a1 - base:
                    v += sum(self.samples[e:a1 - base])
                out[b] = v / size
            else:
                v = max(self.blk_m[r0 - first:r1 - first + 1])
                e = r0 * BLK_SAMPLES - base
                if e > a0 - base:
                    v = max(v, max(self.samples[a0 - base:e]))
                e = (r1 + 1) * BLK_SAMPLES - base
                if e < a1 - base:
                    v = max(v, max(self.samples[e:a1 - base]))
                out[b] = v
        return out

    def _l_bucket(self, a0, a1, size, is_a, base):
        """One whole-run bucket [a0, a1) of plot_samples('l'). A = the
        sum over the samples in the bucket divided by the FULL bucket
        width; M = the max over the available samples. Sources, in
        order: closed lifetime hour rows, the open head-hour
        accumulator, then the kept buffer (full block rows plus raw
        edges, the period_stats pattern). A bucket edge cutting a
        lifetime hour takes width x (row_sum / row_width) for A and the
        row max for M - exact for values constant within the hour;
        buffer edges are raw, so they are always exact."""
        H = self.HOUR_SAMPLES
        s = 0.0
        mx = 0.0
        # table part [a0, min(a1, base)): closed rows + the open head hour
        t1 = a1 if a1 < base else base
        if t1 > a0:
            x = (a0 + H - 1) // H * H  # end of a head-cut partial hour
            if x > t1:
                x = t1
            if x > a0:
                w = x - a0
                h = a0 // H
                if base and h == base // H:  # inside the open head hour
                    acc_w = base - h * H
                    if is_a:
                        s += w * self._hour_acc_s / acc_w
                    mx = max(mx, self._hour_acc_m)
                else:  # a cut closed row: row-level contribution
                    if is_a:
                        s += w * self.hour_s[h] / H
                    mx = max(mx, self.hour_m[h])
            y = (t1 // H) * H  # start of a tail-cut partial hour
            if y < x:
                y = x
            if y < t1:
                w = t1 - y
                h = t1 // H
                if base and h == base // H:  # inside the open head hour
                    acc_w = base - h * H
                    if is_a:
                        s += w * self._hour_acc_s / acc_w
                    mx = max(mx, self._hour_acc_m)
                else:
                    if is_a:
                        s += w * self.hour_s[h] / H
                    mx = max(mx, self.hour_m[h])
            if x < y:  # fully covered closed rows
                if is_a:
                    s += sum(self.hour_s[x // H:y // H])
                else:
                    mx = max(mx, max(self.hour_m[x // H:y // H]))
        # buffer part [max(a0, base), a1)
        b0 = a0 if a0 > base else base
        if b0 < a1:
            first = self.blk_first
            r0 = (b0 + BLK_SAMPLES - 1) // BLK_SAMPLES  # first full row
            r1 = a1 // BLK_SAMPLES - 1  # last full row
            if r1 < r0:
                seg = self.samples[b0 - base:a1 - base]
                if is_a:
                    s += sum(seg)
                else:
                    mx = max(mx, max(seg))
            elif is_a:
                v = sum(self.blk_s[r0 - first:r1 - first + 1])
                e = r0 * BLK_SAMPLES - base
                if e > b0 - base:
                    v += sum(self.samples[b0 - base:e])
                e = (r1 + 1) * BLK_SAMPLES - base
                if e < a1 - base:
                    v += sum(self.samples[e:a1 - base])
                s += v
            else:
                v = max(self.blk_m[r0 - first:r1 - first + 1])
                e = r0 * BLK_SAMPLES - base
                if e > b0 - base:
                    v = max(v, max(self.samples[b0 - base:e]))
                e = (r1 + 1) * BLK_SAMPLES - base
                if e < a1 - base:
                    v = max(v, max(self.samples[e:a1 - base]))
                mx = max(mx, v)
        return s / size if is_a else mx

    def plot_peak(self, period: str) -> float:
        """The max of the raw samples inside the period window - the
        scale reference for the compressed views (shared by both aggs,
        so the aggregated function is visible on the plot instead of
        freezing against the lifetime peak or hiding behind the
        aggregated max). n = the max over the live hist tail (blanks as
        0.0); m..w = the last int(period_secs/interval) samples of the
        kept buffer (the whole buffer when it is shorter); l = the WHOLE
        RUN (phase 4): the max of the closed lifetime hour rows, the
        open head-hour accumulator, and the kept buffer. The max flows
        through the block tables' exact float32 maxima (full rows plus
        at most two raw partial edges, the period_stats pattern).
        Pure: reads only the hist, the buffer/tables, and the lifetime
        table."""
        if period == "n":
            return builtins.max((v for v in self.hist if v is not None),
                                default=0.0)
        L = len(self.samples)
        if period == "l":
            v = 0.0
            if self.hour_m:
                v = max(v, max(self.hour_m))
            if self._hour_acc_c:
                v = max(v, self._hour_acc_m)
            if L:
                base = self.seq - L
                first = self.blk_first
                r0 = (base + BLK_SAMPLES - 1) // BLK_SAMPLES  # first full row
                r1 = self.seq // BLK_SAMPLES - 1  # last full row
                if r1 < r0:
                    v = max(v, max(self.samples))
                else:
                    v = max(v, max(self.blk_m[r0 - first:r1 - first + 1]))
                    e = r0 * BLK_SAMPLES - base
                    if e > 0:
                        v = max(v, max(self.samples[:e]))
                    e = (r1 + 1) * BLK_SAMPLES - base
                    if e < L:
                        v = max(v, max(self.samples[e:]))
            return v
        if not L:
            return 0.0
        base = self.seq - L
        g0 = max(self.seq - int(self._PLOT_PERIOD_SECS[period] / self.interval),
                 base)  # absolute of the window head, clamped to the buffer
        g1 = self.seq
        first = self.blk_first
        r0 = (g0 + BLK_SAMPLES - 1) // BLK_SAMPLES
        r1 = g1 // BLK_SAMPLES - 1
        if r1 < r0:
            return max(self.samples[g0 - base:g1 - base])
        v = max(self.blk_m[r0 - first:r1 - first + 1])
        e = r0 * BLK_SAMPLES - base
        if e > g0 - base:
            v = max(v, max(self.samples[g0 - base:e]))
        e = (r1 + 1) * BLK_SAMPLES - base
        if e < g1 - base:
            v = max(v, max(self.samples[e:g1 - base]))
        return v
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
