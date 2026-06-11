"""Shared SVG→PIL rasterization for components that need real images."""
from __future__ import annotations

import io

import cairosvg
from PIL import Image

from .face import FaceSpec, render_face_svg


def face_image(spec: FaceSpec, width: int = 300) -> Image.Image:
    png = cairosvg.svg2png(bytestring=render_face_svg(spec, width=width).encode(),
                           output_width=width)
    return Image.open(io.BytesIO(png)).convert("RGB")
