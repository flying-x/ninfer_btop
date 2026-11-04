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
  the SGR codes. This repo ships `bar_graph_test.py` (`--text` for a
  non-TUI check) and `hero_render.py` (terminal capture -> PNG).

## Testing

- `test_ninfer_btop.py` covers the unit behavior (J tests for
  `compute_dots`/`dots_to_chars`, H tests for the frame integration,
  I tests for the MhHDWL period summary columns).
- `bar_graph_test.py` is a visual tester: run it against a live endpoint
  (or with `--text`) to watch the scroll motion and glyph texture.
