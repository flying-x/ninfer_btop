#!/usr/bin/env python3
"""Render a tmux `capture-pane -p -e` SGR capture to the hero screenshot
(images/ninfer_btop.png).

Recipe (see the project agent doc, section 7): run the TUI in a
120x32 tmux pane against the live engine, drive traffic at the engine's
/v1/chat/completions endpoint, then capture the pane once graphs show
activity:

    tmux new-session -d -s hero -x 120 -y 32 \
        "python3 ninfer_btop.py --url http://ENGINE:PORT/metrics"
    tmux capture-pane -t hero -p  > /tmp/hero_plain.txt   # for redaction
    tmux capture-pane -t hero -p -e > /tmp/hero_ansi.txt  # SGR-colored
    python3 hero_render.py /tmp/hero_ansi.txt /tmp/hero_plain.txt \
        images/ninfer_btop.png "MODEL" "FOOTER_LABEL"

Rendering: PIL + DejaVuSansMono (15px, cell 9x20), BG (13,17,23),
FG (200,208,216), redaction fill (43,52,66). DejaVu Sans Mono has no
braille block, so braille graph cells fall back to DejaVu Sans (drawn at
fixed cell x-positions, so its proportional width is harmless).
Redaction spans (header model, header URL, footer pricing label) are
computed from the plain capture. MODEL and FOOTER_LABEL are required, so
the script itself holds no model name to leak.
"""
import re
import sys
from PIL import Image, ImageDraw, ImageFont

ROWS, COLS, FSIZE = 32, 120, 15
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
# DejaVu Sans Mono has no braille block (renders .notdef boxes); DejaVu Sans
# (proportional) covers U+2800-28FF. Use it only for braille cells; those are
# drawn at fixed cell x-positions, so proportional width is harmless.
FONT_BRAILLE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BRAILLE_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TERM_BG = (13, 17, 23)
TERM_FG = (200, 208, 216)
REDACT = (43, 52, 66)
ESC = re.compile(r"\x1b\[([0-9;]*)m")


def _xterm256():
    pal = []
    r = [0, 95, 135, 175, 215, 255]
    for i in range(216):
        pal.append((r[i // 36], r[(i // 6) % 6], r[i % 6]))
    v = 8
    while v < 238:
        pal.append((v, v, v))
        v += 10
    return tuple(pal)


PAL256 = _xterm256()
BASE16 = {
    30: (0, 0, 0), 31: (170, 0, 0), 32: (0, 135, 0), 33: (170, 85, 0),
    34: (0, 0, 170), 35: (170, 0, 170), 36: (0, 135, 170), 37: (170, 170, 170),
    90: (128, 128, 128), 91: (255, 0, 0), 92: (0, 255, 0), 93: (255, 255, 0),
    94: (0, 0, 255), 95: (255, 0, 255), 96: (0, 255, 255), 97: (255, 255, 255),
    40: (0, 0, 0), 41: (170, 0, 0), 42: (0, 135, 0), 43: (170, 85, 0),
    44: (0, 0, 170), 45: (170, 0, 170), 46: (0, 135, 170), 47: (170, 170, 170),
    100: (128, 128, 128), 101: (255, 0, 0), 102: (0, 255, 0), 103: (255, 255, 0),
    104: (0, 0, 255), 105: (255, 0, 255), 106: (0, 255, 255), 107: (255, 255, 255),
}


def _rgb(v, default):
    if v is None:
        return default
    if isinstance(v, tuple):
        return v
    return PAL256[v]


def parse_sgr(params, st):
    codes = (params.split(";") if params else ["0"])
    i = 0
    while i < len(codes):
        n = int(codes[i] or "0")
        if n == 0:
            st.update(fg=None, bg=None, bold=False, rev=False)
        elif n == 1:
            st["bold"] = True
        elif n == 7:
            st["rev"] = not st["rev"]
        elif 30 <= n <= 37 or 90 <= n <= 97:
            st["fg"] = BASE16[n]
        elif 40 <= n <= 47 or 100 <= n <= 107:
            st["bg"] = BASE16[n]
        elif n in (38, 48):
            key = "fg" if n == 38 else "bg"
            if i + 2 < len(codes) and codes[i + 1] == "5":
                st[key] = PAL256[int(codes[i + 2])]
                i += 2
            elif i + 4 < len(codes) and codes[i + 1] == "2":
                st[key] = (int(codes[i + 2]), int(codes[i + 3]), int(codes[i + 4]))
                i += 4
        i += 1


def parse_grid(ansi, rows, cols):
    lines = ansi.split("\n")
    if len(lines) < rows:
        lines += [""] * (rows - len(lines))
    lines = lines[:rows]
    grid = [[[ " ", None, None, False, False] for _ in range(cols)] for _ in range(rows)]
    for r, line in enumerate(lines):
        st = {"fg": None, "bg": None, "bold": False, "rev": False}
        col = 0
        pos = 0
        for m in ESC.finditer(line):
            for ch in line[pos:m.start()]:
                if col < cols:
                    grid[r][col] = [ch, st["fg"], st["bg"], st["bold"], st["rev"]]
                    col += 1
            parse_sgr(m.group(1), st)
            pos = m.end()
        for ch in line[pos:]:
            if col < cols:
                grid[r][col] = [ch, st["fg"], st["bg"], st["bold"], st["rev"]]
                col += 1
    return grid


def render(grid, rows, cols, redact=()):
    font = ImageFont.truetype(FONT_REG, FSIZE)
    fontb = ImageFont.truetype(FONT_BOLD, FSIZE)
    font_br = ImageFont.truetype(FONT_BRAILLE, FSIZE)
    font_br_b = ImageFont.truetype(FONT_BRAILLE_BOLD, FSIZE)
    cw = int(round(font.getlength("0")))
    asc, desc = font.getmetrics()
    lh = max(cw + 6, int(round(FSIZE * 1.35)))
    mx = 16
    my = 16
    W = 2 * mx + cols * cw
    H = 2 * my + rows * lh
    img = Image.new("RGB", (W, H), TERM_BG)
    d = ImageDraw.Draw(img)
    rowpad = (lh - (asc + desc)) // 2
    asc_br, desc_br = font_br.getmetrics()
    rowpad_br = (lh - (asc_br + desc_br)) // 2
    for r in range(rows):
        for c in range(cols):
            ch, fg, bg, bold, rev = grid[r][c]
            f = _rgb(fg, TERM_FG)
            b = _rgb(bg, TERM_BG)
            if rev:
                f, b = b, f
            x = mx + c * cw
            y = my + r * lh
            d.rectangle([x, y, x + cw - 1, y + lh - 1], fill=b)
            if ch != " " and ch != "\x00":
                o = ord(ch)
                if 0x2800 <= o <= 0x28FF:
                    use = font_br_b if bold else font_br
                else:
                    use = fontb if bold else font
                pad = rowpad_br if (0x2800 <= o <= 0x28FF) else rowpad
                d.text((x, y + pad), ch, font=use, fill=f)
    for (rr, c0, c1) in redact:
        x0 = mx + c0 * cw
        y0 = my + rr * lh
        x1 = mx + (c1 + 1) * cw - 1
        y1 = my + (rr + 1) * lh - 1
        d.rectangle([x0, y0, x1, y1], fill=REDACT)
    return img


def redact_spans(plain, model, label):
    lines = plain.split("\n")
    head = lines[0] if lines else ""
    spans = []
    a = head.find("ninfer btop \u2500 ")
    b = head.find(" \u2500 engine")
    if a != -1 and b > a:
        m0 = a + len("ninfer btop \u2500 ")
        m1 = b - 1
        if model and head[m0:m1 + 1].startswith(model):
            spans.append((0, m0, m1))
    u = head.rfind("http")
    if u != -1:
        spans.append((0, u, len(head.rstrip()) - 1))
    foot = lines[ROWS - 1] if len(lines) >= ROWS else ""
    k = foot.find(label)
    if k != -1:
        spans.append((ROWS - 1, k, k + len(label) - 1))
    return spans


def main():
    ansi = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    plain = open(sys.argv[2], encoding="utf-8", errors="replace").read()
    out = sys.argv[3]
    model = sys.argv[4] if len(sys.argv) > 4 else None
    label = sys.argv[5] if len(sys.argv) > 5 else None
    if model is None or label is None:
        raise SystemExit("error: both MODEL and FOOTER_LABEL must be specified")
    spans = redact_spans(plain, model, label)
    grid = parse_grid(ansi, ROWS, COLS)
    img = render(grid, ROWS, COLS, spans)
    img.save(out)
    print("saved", out, img.size, "redact spans:", spans)


if __name__ == "__main__":
    main()
