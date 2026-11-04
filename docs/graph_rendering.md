# Graph rendering: compute_dots and dots_to_chars

From the [ninfer_btop](https://github.com/flying-x/ninfer_btop) project — a btop-style TUI dashboard for the NInfer inference engine.

How the braille bar graphs are built from raw metric samples. The goal is a
btop-style graph: dense, smooth-looking, and readable in a small area —
without a second vertical resolution level than braille gives us.
 from raw metric samples. The goal is a
btop-style graph: dense, smooth-looking, and readable in a small area —
without a second vertical resolution level than braille gives us.

Pipeline: raw samples -> `compute_dots()` (one dot height per sample)
-> `dots_to_chars()` (two dot lines per braille character)
-> `draw_bar_graph()` / `_xfer_row()`.

## compute_dots(samples, max=None, total_dots=4)

Converts raw values to per-sample dot heights in 0..total_dots.

- `max=None` autoscales to the window maximum (prefill, decode, spill,
  recall); a pinned graph passes an explicit scale instead — the batch
  graph pins `max = 4*rows`, so one dot line equals exactly one running
  request and the scale never moves under load.
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
  in the panel's base color. The window max is highlighted only as the
  `max=` token in the panel title banner.

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
- When a metric's unit is exact (1 dot = 1 concurrent request), pin the
  scale instead of autoscaling to the window max, so the axis does not
  move under load. Autoscale only when there is no natural unit.
- Do not highlight per-sample "peaks" inside double-wide glyphs: a
  character spans two samples, so a peak highlight smears. Show the window
  max as a plain text token in the panel title instead.
- Size buffers to 2x the character width and grow partial windows from the
  newest edge.
- Verify scroll motion mechanically: drive scripted traffic, capture the
  terminal (`tmux capture-pane -e -p`) at several timestamps, and inspect
  the SGR codes. This repo ships `bar_graph_test.py` (`--text` for a
  non-TUI check) and `hero_render.py` (terminal capture -> PNG).

## Testing

- `test_ninfer_btop.py` covers the unit behavior (J tests for
  `compute_dots`/`dots_to_chars`, H tests for the frame integration).
- `bar_graph_test.py` is a visual tester: run it against a live endpoint
  (or with `--text`) to watch the scroll motion and glyph texture.
