# Graph rendering: compute_dots and dots_to_chars

From the [ninfer_btop](https://github.com/flying-x/ninfer_btop) project — a btop-style TUI dashboard for the NInfer inference engine.

How the braille bar graphs are built from raw metric samples. The goal is a
btop-style graph: dense, smooth-looking, and readable in a small area —
without a finer vertical resolution level than braille gives us.

Pipeline: raw samples -> `compute_dots()` (one dot height per sample)
-> `dots_to_chars()` (two dot lines per braille character)
-> `draw_bar_graph()` / `_xfer_row()`.

## compute_dots(samples, max=None, total_dots=4)

Converts raw values to per-sample dot heights in 0..total_dots.

- The graphs pass an explicit scale from `Graph.dot_scale`: pinned
  graphs (the batch graph) use `max = 4*rows`, so one dot line is
  exactly one running request; the prefill/decode graphs use the
  lifetime peak (the `peak=` banner token) once there has been any
  non-zero sample - stable for the whole monitor run - and pass
  `max=None` (autoscale to the window maximum) before that. The
  spill/recall transfer rows always autoscale to their own window peak.
- The mapping is `dots = ceil(v / (max / total_dots))`, clipped at
  `total_dots`. Ceil rounding is deliberate: any non-zero sample draws at
  least one dot, so a small value is never erased by the scale, and the
  step edges of the graph stay sharp instead of bleeding into fractional
  heights.
- Note `max` shadows the builtin; the function uses `builtins.max`.

## dots_to_chars(dots, rows)

Resolves dot heights into braille characters, the way btop does:

- A character is **double-wide**: it holds two adjacent samples, the
  **newer** sample in the left dot line, the older in the right. Doubling
  the sample density in the same area is what makes the graph look smooth
  and keeps ~13 s of history in the 13-character transfer rows.
- Each (left, right) level pair indexes the full 5x5 btop glyph table, so
  a character can be any of the 25 glyphs — rising and falling edges
  render as diagonal glyphs instead of a staircase of flat tops.
- `rows` > 1 stacks the glyphs vertically: row `r` shows the band
  `[(rows-1-r)*4, (rows-r)*4]` of dot heights, so a tall spike is one
  column of stacked glyphs.
- An odd-length dot array is padded with a trailing zero before pairing.

## Motion and orientation

- The buffer holds `2*gwidth` samples (the xfer rows use `Graph(26)`);
  the newest sample is drawn on the left, so the graph scrolls left to
  right.
- Because a character holds two samples and the newer one is always the
  left dot line, each poll shifts every dot line by exactly **half a
  character** — the pairing never flips and the graph never jumps.
- A partial (not-yet-full) window grows from the left edge.
- There is no peak highlighting inside the graphs: every glyph is painted
  in the panel's base color. The window max is highlighted as the `max=`
  token in the panel title banner; the lifetime peak sits beside it as a
  plain `peak=` token and is what scales the graph.
 - The baseline row under the plot is labeled in its middle with the time
   span the plot region covers: two samples per character, so the span is
   `2 * (width - STAT_COLS) * interval` seconds ("113 seconds" at the
   default 119-column width and 0.5 s poll).

## Period summary columns (MhHDWL)

The rightmost 6 braille columns of each scrolling graph are reserved for
period summaries, labeled `MhHDWL` on the baseline row where the dashes
would run (the dashes fill the scrolling area only):

- **M** last 5 minutes, **h** last half-hour, **H** last hour, **D** last
  day, **W** last week, **L** this monitor process's life-time. The
  windows are counted in poll intervals (e.g. at the default 0.5 s poll:
  M = 600 samples); a period longer than the process life clamps to all
  available samples; W spans the kept one-week buffer.
  The figures cover **this monitor run**, not the engine's life.
- Each column is one double-wide braille character: the **left dot line
  is the period average, the right dot line the period max**. The M/h/H/D/W
  figures are recomputed from the one-week raw history buffer (trimmed to
  the last week; W spans the whole kept buffer) on every poll - via small
  per-graph block tables (a partial sum and an exact max per 1024-sample
  block, so a window is a few table rows plus at most two partial block
  edges rather than a full rescan; a fraction of a millisecond per graph) -
  so the displayed figures are exact functions of the stored samples and
  cannot drift from accumulated float error. L is a separate running total and
  max over the whole process life (plain scalars - its window never
  shrinks). Memory is bounded (about 5 MB per graph at the default
  0.5 s poll), no matter how long the process runs. Blanks count as 0,
  like the scrolling plot.
- The summaries are drawn in the graph's own dot scale
  (`Graph.dot_scale` — the plot height for the batch graph, the lifetime
  peak for the rest), so they read against the plot next to them.
- They are grayed out against the moving bars, including the `MhHDWL`
  labels on the baseline row: darker color pairs on 256-color terminals
  (dark green / dark yellow), an `A_DIM` attribute on terminals without
  those pairs.

## Period/agg selector (compressed views)

The footer carries one global selector for the three main graphs (batch,
prefill, decode): `period=now 5m 30m 1h 1d 1w life  agg fn=Avg Max`.
Keys: `n m h H d w l` set the period (default `n`, the live scrolling view
of the previous section); `A`/`M` set the aggregation function (default
`M`). The two spill/recall transfer rows keep their fixed ~13 s live view
regardless of the selection.

The compressed periods (everything except `now`) are **fixed bins,
anchored at the monitor's lifetime start** — deliberately not a sliding
window:

- The bin width is `W = max(1, period_samples // slots)`, a constant per
  (period, poll interval): `period_samples` is the period in poll
  intervals, `slots` the number of sample positions the plot draws (two
  per character). Bin `j` covers absolute samples `[j*W, (j+1)*W)` and
  keeps doing so forever — **a closed bin never changes**.
- The view is the newest `slots` bins. Only the newest, still-filling bin
  moves, so the graph **freezes while that bin fills, then steps right
  exactly one column when the sample counter crosses a bin edge** (every
  `W` polls) — literal freeze-then-jump. A young buffer draws the few
  newest bins it has plus empty older ones, not a stretched full-width
  plot.
- `Avg` = the bin's sum over its *available* samples divided by the full
  width `W` — samples older than the kept buffer (or before the run
  started) aggregate nothing, so unmonitored time dilutes the average
  exactly like stored blanks. `Max` = the max over the available samples.
- The compressed views scale to **the max of the raw samples inside the
  shown bins**, shared by both aggregation functions: the `Avg` view's
  diluted averages sit visibly below that scale, so switching the
  aggregation changes the bar heights instead of hiding behind a rescale
  (a lifetime-peak scale would instead freeze ordinary traffic at the
  1-dot floor after a single old burst). The live `now` view and the
  pinned batch graph keep their usual scales (lifetime peak / dot count).
- **life** plots the whole monitor run on its own lattice: `W =
  ceil(seq/slots)` recomputed per draw, still anchored at sample 0. Bins
  freeze between width changes; a width change re-cuts the whole lattice
  (the one expected compression jump). Raw samples cover the last week;
  everything older was folded into coarse per-hour rows (count / sum /
  max) as it aged out of the buffer, so `life` stays exact and cheap for
  runs of any length.

## Best practices (for agent readers)

Reusable rules distilled from this implementation:

- Prefer braille over block characters for TUI metric graphs: the 5x5 dot
  grid is the only character set with 25 combinations (any level on each
  side), which is what makes slopes and spikes read as texture.
- Encode two adjacent samples per character (newer in the left dot line).
  That doubles sample density in the same area and shifts every dot line by
  half a character per poll, so the scroll is continuous - it never jumps
  or flips the pairing.
- Map values to dot heights with `ceil(v / step)` and guarantee at least
  one dot for every non-zero sample. Fractional heights and zero-clipped
  small values both make a graph look broken.
- Keep each graph's axis stable for the whole run: pin a metric with
  an exact unit (1 dot = 1 concurrent request) to the plot's dot
  count, and scale the rest to the lifetime peak (a plain `peak=`
  banner token next to the highlighted window `max=`) so the whole
  history fits on screen; autoscale to the window max only for
  pre-traffic frames and the transfer rows.
- Do not highlight per-sample "peaks" inside double-wide glyphs: a
  character spans two samples, so a peak highlight smears. Show the window
  max as a plain text token in the panel title instead.
- Keep long-window context visible in a fixed corner: reserve a few
  columns for period summaries (avg + max per window, labeled on the
  baseline row, dimmed so they read as context rather than live data),
  fed by a small raw-sample buffer alongside the sliding-window
  history.
- For compressed long-period views, use fixed bins anchored at the run
  start, not a sliding window: a window re-anchored at the newest sample
  moves every bucket's span by one sample on every poll, so all bars
  morph in place; a lifetime-anchored lattice keeps closed bins
  immutable and the view only moves when a bin closes (freeze-then-jump).
- Recompute period summaries (avg + max) from a time-capped raw-sample
  buffer on every poll instead of maintaining running aggregates
  incrementally: repeated add/subtract on floats leaves residue
  (a phantom ~1e-5 sum in an all-zero window lights a phantom average
  dot while the max shows none), and the recomputation over a bounded
  buffer - a few cached per-block rows plus at most two partial block
  edges, not a full rescan - costs a fraction of a millisecond. L
  (whole life) exceeds the buffer, so it stays append-only scalars - a
  running total that is only ever added to, and a scalar max.
- Size buffers to 2x the character width and grow partial windows from the
  newest edge.
- Verify scroll motion mechanically: drive scripted traffic, capture the
  terminal (`tmux capture-pane -e -p`) at several timestamps, and inspect
  the SGR codes. This repo ships `hero_render.py` (terminal capture -> PNG).

## Testing

- `test_ninfer_btop.py` covers the unit behavior: J tests pin the batch
  graph's `compute_dots` scaling, H tests the spill/recall transfer rows,
  and I tests the MhHDWL period summary columns; the paired braille
  rendering of `dots_to_chars` rides along through every drawn row.
