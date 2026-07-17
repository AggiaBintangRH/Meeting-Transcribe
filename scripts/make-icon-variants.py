#!/usr/bin/env python3
"""Generate dark + tinted macOS app-icon appearance variants from 1024.png.

Input:  Assets.xcassets/AppIcon.appiconset/1024.png (1024x1024 RGBA, transparent
        canvas, white-stroked squircle, blue->purple gradient bg, near-white mic).
Output (written next to the input):
  1024-dark.png    full-color, darkened background, glyph/border/canvas untouched.
  1024-tinted.png  pure grayscale luminance (system colorizes it), alpha preserved.

Run with the project venv: .venv/bin/python3 scripts/make-icon-variants.py
"""
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
ICONSET = (
    ROOT
    / "MeetingTranscriber"
    / "Sources"
    / "MeetingTranscriber"
    / "Assets.xcassets"
    / "AppIcon.appiconset"
)
SRC = ICONSET / "1024.png"

# Saturation threshold: pixels with S above this are treated as "colored
# background" and get darkened. White glyph, white border ring, and the
# transparent canvas all have low saturation and are left alone.
SAT_THRESHOLD = 80
VALUE_SCALE = 0.55


def make_dark(src: Image.Image) -> Image.Image:
    src = src.convert("RGBA")
    r, g, b, a = src.split()
    rgb = Image.merge("RGB", (r, g, b))
    hsv = rgb.convert("HSV")
    h, s, v = hsv.split()

    s_px = s.load()
    v_px = v.load()
    w, ht = src.size
    for y in range(ht):
        for x in range(w):
            if s_px[x, y] > SAT_THRESHOLD:
                v_px[x, y] = int(v_px[x, y] * VALUE_SCALE)

    dark_rgb = Image.merge("HSV", (h, s, v)).convert("RGB")
    dr, dg, db = dark_rgb.split()
    return Image.merge("RGBA", (dr, dg, db, a))


def make_tinted(src: Image.Image) -> Image.Image:
    src = src.convert("RGBA")
    r, g, b, a = src.split()
    rgb = Image.merge("RGB", (r, g, b))
    lum = rgb.convert("L")
    lum = ImageOps.autocontrast(lum, cutoff=1)
    gray_rgb = Image.merge("RGB", (lum, lum, lum))
    gr, gg, gb = gray_rgb.split()
    return Image.merge("RGBA", (gr, gg, gb, a))


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source icon not found: {SRC}", file=sys.stderr)
        return 1
    src = Image.open(SRC)
    if src.size != (1024, 1024):
        print(f"ERROR: expected 1024x1024, got {src.size}", file=sys.stderr)
        return 1

    dark = make_dark(src)
    tinted = make_tinted(src)

    dark_path = ICONSET / "1024-dark.png"
    tinted_path = ICONSET / "1024-tinted.png"
    dark.save(dark_path)
    tinted.save(tinted_path)
    print(f"wrote {dark_path} ({dark.size} {dark.mode})")
    print(f"wrote {tinted_path} ({tinted.size} {tinted.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
