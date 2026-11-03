#!/usr/bin/env python3
"""Standalone visual test for the ninfer_btop braille bar graph.

Draws a staircase of bars rising from the minimum height (one step above
the baseline) to the maximum height (full scale), using the real Graph,
compute_dots and draw_bar_graph code from ninfer_btop.py, so what you
see is exactly what the TUI renders.

Expect:
  * every dot line fills UP from the baseline (never down from the top)
  * the staircase ascends left -> right (min on the left = newest, max on
    the right = oldest); every character is painted in the panel's base
    color - the graph has no peak highlight
  * each braille character holds two adjacent samples (the newer in the
    left dot line) - one single dot line per sample - so the slope is
    drawn with rising/falling glyphs: four dot levels per text row

Run:   python3 bar_graph_test.py            (real terminal; any key quits)
       python3 bar_graph_test.py --text     (plain stdout preview, piped)

Note the convention: the graph scrolls left to right - the newest sample
enters at the left edge. This test renders the full window at once, so the
staircase runs min (left, newest) -> max (right, oldest).
"""
import argparse
import curses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ninfer_btop as nb


def make_staircase(rows):
    """A graph whose samples descend max -> min in time order (oldest = max);
    with the graph's newest-sample-on-the-left convention the rendered
    staircase ascends left -> right from one level above the baseline to
    full scale, and the rightmost (oldest) column carries the peak. One
    integer sample per dot level, so compute_dots' ceil mapping is exact."""
    levels = rows * 4
    g = nb.Graph(levels)
    for j in range(levels):
        g.add(levels - j)  # levels (oldest) ... 1 (newest)
    return g


def run_curses():
    def main(scr):
        scr.erase()
        max_y, max_x = scr.getmaxyx()
        if max_y < 6 or max_x < 30:
            try:
                scr.addnstr(0, 0, "terminal too small (need at least 30x6)", max_x)
            except curses.error:
                pass
            scr.refresh()
            scr.getch()
            return
        width = max_x - 1
        rows = max(2, max_y - 4)
        g = make_staircase(rows)
        try:
            scr.addnstr(0, 0, "bar graph test: staircase min (left) -> max (right)", width)
        except curses.error:
            pass
        nb.draw_bar_graph(scr, 1, g, rows, width, "toks", {})
        try:
            scr.addnstr(max_y - 1, 0, "press any key to quit", width)
        except curses.error:
            pass
        scr.refresh()
        scr.getch()

    curses.wrapper(main)


def run_text():
    """Plain stdout preview using the same character math as the curses
    path."""
    rows = 12
    g = make_staircase(rows)
    plot_w = (len(list(g.hist)) + 1) // 2
    grid = nb.dots_to_chars(g.dots(rows)[::-1], rows)  # newest sample on the left
    out = ["".join(line) for line in grid]
    out.append("─" * plot_w)
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description="visual test for the ninfer_btop bar graph")
    ap.add_argument("--text", action="store_true",
                    help="plain stdout preview instead of the curses screen")
    args = ap.parse_args()
    if args.text or not sys.stdout.isatty():
        run_text()
    else:
        run_curses()


if __name__ == "__main__":
    main()
