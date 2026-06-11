"""Shareable WANTED-poster card: the social/share artifact of a finished case.

Renders 'what you said vs who it was' as a vintage poster PNG the player
downloads and dares friends with. Pure PIL — zero GPU, zero model.
"""
from __future__ import annotations

import io

import cairosvg
from PIL import Image, ImageDraw, ImageFont

from .face import FaceSpec, render_face_svg

PAPER = (244, 239, 228)
INK = (43, 42, 40)
RED = (163, 51, 39)
GREEN = (29, 107, 47)
TAPE = (201, 162, 39)


def _face_img(spec: FaceSpec, width: int) -> Image.Image:
    png = cairosvg.svg2png(bytestring=render_face_svg(spec, width=width).encode(),
                           output_width=width)
    return Image.open(io.BytesIO(png)).convert("RGB")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def make_wanted_poster(
    sketch: FaceSpec,
    culprit: FaceSpec,
    correct: bool,
    accuracy_pct: int,
    case_name: str,
    rank: str,
) -> Image.Image:
    W, H = 820, 1000
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)

    # border + header
    d.rectangle([14, 14, W - 14, H - 14], outline=INK, width=4)
    d.rectangle([22, 22, W - 22, H - 22], outline=INK, width=1)
    d.text((W // 2, 64), "W A N T E D", font=_font(58, bold=True), fill=INK, anchor="mm")
    d.text((W // 2, 112), f"CASE: {case_name.upper()}  ·  RANK: {rank}",
           font=_font(20), fill=INK, anchor="mm")
    d.line([60, 140, W - 60, 140], fill=INK, width=2)

    # faces
    f1 = _face_img(sketch, 330)
    f2 = _face_img(culprit, 330)
    img.paste(f1, (70, 180))
    img.paste(f2, (W - 70 - 330, 180))
    d.rectangle([70, 180, 70 + 330, 180 + 396], outline=INK, width=3)
    d.rectangle([W - 70 - 330, 180, W - 70, 180 + 396], outline=INK, width=3)
    d.text((70 + 165, 610), "THE WITNESS SAID", font=_font(19, bold=True), fill=INK, anchor="mm")
    d.text((W - 70 - 165, 610), "WHO IT ACTUALLY WAS", font=_font(19, bold=True), fill=INK, anchor="mm")

    # accuracy + verdict stamp
    d.text((W // 2, 680), f"MEMORY ACCURACY: {accuracy_pct}%",
           font=_font(30, bold=True), fill=INK, anchor="mm")
    stamp_text = "ARREST CONFIRMED" if correct else "STILL AT LARGE"
    stamp_color = GREEN if correct else RED
    stamp_font = _font(44, bold=True)
    # rotated stamp
    stamp = Image.new("RGBA", (620, 110), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    sd.rectangle([4, 4, 616, 106], outline=stamp_color, width=6)
    sd.text((310, 55), stamp_text, font=stamp_font, fill=stamp_color, anchor="mm")
    stamp = stamp.rotate(-5, expand=True, resample=Image.BICUBIC)
    img.paste(stamp, (W // 2 - stamp.width // 2, 716), stamp)

    # footer
    d.text((W // 2, 900), "Could YOUR memory do better?", font=_font(26), fill=INK, anchor="mm")
    d.text((W // 2, 944), "EYEWITNESS · build-small-hackathon · 3s glimpse, then you're on your own",
           font=_font(16), fill=(93, 86, 74), anchor="mm")
    return img
