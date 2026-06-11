"""Parametric police-composite-sketch face renderer (pure SVG).

Every face is a FaceSpec of closed-vocabulary attributes. The renderer is
deterministic: same spec -> same sketch. This is the "author the ground
truth" core — the game always knows exactly what the culprit looks like.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict, fields

# ---------------------------------------------------------------- vocabulary
VOCAB: dict[str, list[str]] = {
    "face_shape": ["oval", "round", "square", "long"],
    "skin": ["light", "medium", "dark"],
    "hair_style": ["bald", "buzz", "short_messy", "slick_back", "curly", "long", "ponytail", "mohawk"],
    "hair_color": ["black", "brown", "blond", "red", "gray"],
    "brows": ["thin", "thick", "bushy", "unibrow"],
    "eyes": ["normal", "narrow", "big", "droopy"],
    "glasses": ["none", "round", "square", "sunglasses"],
    "nose": ["small", "big", "hooked", "wide"],
    "mouth": ["neutral", "smirk", "frown", "open"],
    "facial_hair": ["none", "stubble", "mustache", "goatee", "full_beard"],
    "hat": ["none", "beanie", "cap", "fedora"],
    "extra": ["none", "scar_cheek", "earring", "neck_tattoo", "mole"],
}

# attributes a witness is MOST likely to notice — used to weight scoring/lineups
SALIENCE = {
    "hat": 3.0, "hair_style": 2.5, "glasses": 2.5, "facial_hair": 2.5,
    "hair_color": 2.0, "extra": 2.0, "face_shape": 1.5, "mouth": 1.0,
    "brows": 1.0, "nose": 1.0, "eyes": 1.0, "skin": 1.5,
}

HAIR_HEX = {"black": "#23211e", "brown": "#5b4226", "blond": "#b89a4e", "red": "#8c4a2a", "gray": "#8d8d89"}
SKIN_HEX = {"light": "#efe6d8", "medium": "#dcc4a2", "dark": "#a9805c"}
INK = "#2b2a28"
PAPER = "#f4efe4"


@dataclass(frozen=True)
class FaceSpec:
    face_shape: str
    skin: str
    hair_style: str
    hair_color: str
    brows: str
    eyes: str
    glasses: str
    nose: str
    mouth: str
    facial_hair: str
    hat: str
    extra: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @staticmethod
    def random(rng: random.Random | None = None) -> "FaceSpec":
        rng = rng or random
        return FaceSpec(**{k: rng.choice(v) for k, v in VOCAB.items()})

    def with_changes(self, **kw: str) -> "FaceSpec":
        d = self.to_dict()
        d.update(kw)
        return FaceSpec(**d)

    def diff(self, other: "FaceSpec") -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) != getattr(other, f.name)]


# ---------------------------------------------------------------- renderer
def _head_path(shape: str) -> str:
    """Head outline path for each face shape (300x360 canvas, center x=150)."""
    return {
        "oval":   "M150,68 C108,68 84,104 84,160 C84,216 112,262 150,262 C188,262 216,216 216,160 C216,104 192,68 150,68 Z",
        "round":  "M150,72 C100,72 78,116 78,168 C78,222 108,260 150,260 C192,260 222,222 222,168 C222,116 200,72 150,72 Z",
        "square": "M150,70 C112,70 92,92 90,128 L88,210 C88,242 116,262 150,262 C184,262 212,242 212,210 L210,128 C208,92 188,70 150,70 Z",
        "long":   "M150,62 C114,62 92,96 92,150 C92,216 116,272 150,272 C184,272 208,216 208,150 C208,96 186,62 150,62 Z",
    }[shape]


def _hair(spec: FaceSpec) -> tuple[str, str]:
    """(behind_head_svg, front_svg) for hair. Front layer draws over forehead."""
    c = HAIR_HEX[spec.hair_color]
    s = spec.hair_style
    behind, front = "", ""
    if s == "bald":
        return "", ""
    if s == "buzz":
        front = f'<path d="M96,128 C96,84 122,64 150,64 C178,64 204,84 204,128 C196,100 176,86 150,86 C124,86 104,100 96,128 Z" fill="{c}" opacity="0.85"/>'
    elif s == "short_messy":
        front = (f'<path d="M92,134 C88,84 118,56 150,58 C184,60 214,86 208,134 '
                 f'C204,108 196,104 188,112 C184,96 172,90 162,98 C156,84 142,84 136,96 '
                 f'C126,86 112,92 110,106 C100,102 94,114 92,134 Z" fill="{c}"/>')
    elif s == "slick_back":
        front = f'<path d="M94,120 C92,76 120,58 150,58 C180,58 208,76 206,120 C200,92 178,78 150,78 C122,78 100,92 94,120 Z" fill="{c}"/>'
    elif s == "curly":
        circles = "".join(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>'
            for x, y, r in [(105,110,18),(125,92,19),(150,84,20),(175,92,19),(195,110,18),(98,134,14),(202,134,14)]
        )
        front = circles
    elif s == "long":
        behind = f'<path d="M92,120 C84,180 86,236 96,258 L116,250 C108,210 106,160 112,124 Z M208,120 C216,180 214,236 204,258 L184,250 C192,210 194,160 188,124 Z" fill="{c}"/>'
        front = f'<path d="M92,130 C88,80 118,56 150,56 C182,56 212,80 208,130 C198,96 178,84 150,84 C122,84 102,96 92,130 Z" fill="{c}"/>'
    elif s == "ponytail":
        behind = f'<path d="M196,108 C232,116 240,160 228,204 C222,180 214,150 196,132 Z" fill="{c}"/>'
        front = f'<path d="M94,124 C92,78 120,58 150,58 C180,58 208,78 206,124 C198,94 176,80 150,80 C124,80 102,94 94,124 Z" fill="{c}"/>'
    elif s == "mohawk":
        front = f'<path d="M138,30 L162,30 L168,86 C162,78 138,78 132,86 Z" fill="{c}"/><path d="M132,86 C140,76 160,76 168,86 C162,82 138,82 132,86 Z" fill="{c}"/>'
    return behind, front


def _brows(spec: FaceSpec) -> str:
    y = 142
    if spec.brows == "thin":
        return (f'<path d="M112,{y} C122,{y-4} 134,{y-4} 142,{y-1}" stroke="{INK}" stroke-width="2.6" fill="none"/>'
                f'<path d="M158,{y-1} C166,{y-4} 178,{y-4} 188,{y}" stroke="{INK}" stroke-width="2.6" fill="none"/>')
    if spec.brows == "thick":
        return (f'<path d="M110,{y} C122,{y-7} 136,{y-7} 143,{y-2}" stroke="{INK}" stroke-width="6" fill="none" stroke-linecap="round"/>'
                f'<path d="M157,{y-2} C164,{y-7} 178,{y-7} 190,{y}" stroke="{INK}" stroke-width="6" fill="none" stroke-linecap="round"/>')
    if spec.brows == "bushy":
        return (f'<path d="M108,{y+1} C120,{y-10} 138,{y-9} 144,{y-2}" stroke="{INK}" stroke-width="9" fill="none" stroke-linecap="round"/>'
                f'<path d="M156,{y-2} C162,{y-9} 180,{y-10} 192,{y+1}" stroke="{INK}" stroke-width="9" fill="none" stroke-linecap="round"/>')
    # unibrow
    return f'<path d="M110,{y} C126,{y-8} 174,{y-8} 190,{y}" stroke="{INK}" stroke-width="7" fill="none" stroke-linecap="round"/>'


def _eyes(spec: FaceSpec) -> str:
    y = 158
    lx, rx = 127, 173
    if spec.eyes == "narrow":
        return (f'<path d="M{lx-14},{y} L{lx+12},{y}" stroke="{INK}" stroke-width="3.4" stroke-linecap="round"/>'
                f'<path d="M{rx-12},{y} L{rx+14},{y}" stroke="{INK}" stroke-width="3.4" stroke-linecap="round"/>')
    if spec.eyes == "big":
        return (f'<ellipse cx="{lx}" cy="{y}" rx="12" ry="9" fill="white" stroke="{INK}" stroke-width="2.4"/>'
                f'<ellipse cx="{rx}" cy="{y}" rx="12" ry="9" fill="white" stroke="{INK}" stroke-width="2.4"/>'
                f'<circle cx="{lx}" cy="{y}" r="4.4" fill="{INK}"/><circle cx="{rx}" cy="{y}" r="4.4" fill="{INK}"/>')
    if spec.eyes == "droopy":
        return (f'<path d="M{lx-12},{y-3} C{lx-4},{y+6} {lx+6},{y+7} {lx+11},{y+3}" stroke="{INK}" stroke-width="2.8" fill="none"/>'
                f'<circle cx="{lx}" cy="{y+2}" r="3.2" fill="{INK}"/>'
                f'<path d="M{rx-11},{y+3} C{rx-6},{y+7} {rx+4},{y+6} {rx+12},{y-3}" stroke="{INK}" stroke-width="2.8" fill="none"/>'
                f'<circle cx="{rx}" cy="{y+2}" r="3.2" fill="{INK}"/>')
    # normal
    return (f'<ellipse cx="{lx}" cy="{y}" rx="9" ry="6" fill="white" stroke="{INK}" stroke-width="2.2"/>'
            f'<ellipse cx="{rx}" cy="{y}" rx="9" ry="6" fill="white" stroke="{INK}" stroke-width="2.2"/>'
            f'<circle cx="{lx}" cy="{y}" r="3.4" fill="{INK}"/><circle cx="{rx}" cy="{y}" r="3.4" fill="{INK}"/>')


def _glasses(spec: FaceSpec) -> str:
    if spec.glasses == "none":
        return ""
    y = 158
    if spec.glasses == "round":
        return (f'<circle cx="127" cy="{y}" r="17" fill="none" stroke="{INK}" stroke-width="3"/>'
                f'<circle cx="173" cy="{y}" r="17" fill="none" stroke="{INK}" stroke-width="3"/>'
                f'<path d="M144,{y} L156,{y} M110,{y} L96,{y-6} M190,{y} L204,{y-6}" stroke="{INK}" stroke-width="3"/>')
    if spec.glasses == "square":
        return (f'<rect x="110" y="{y-14}" width="34" height="27" rx="4" fill="none" stroke="{INK}" stroke-width="3"/>'
                f'<rect x="156" y="{y-14}" width="34" height="27" rx="4" fill="none" stroke="{INK}" stroke-width="3"/>'
                f'<path d="M144,{y} L156,{y} M110,{y} L96,{y-6} M190,{y} L204,{y-6}" stroke="{INK}" stroke-width="3"/>')
    # sunglasses
    return (f'<rect x="108" y="{y-13}" width="38" height="26" rx="6" fill="{INK}"/>'
            f'<rect x="154" y="{y-13}" width="38" height="26" rx="6" fill="{INK}"/>'
            f'<path d="M146,{y-4} L154,{y-4} M108,{y-4} L94,{y-9} M192,{y-4} L206,{y-9}" stroke="{INK}" stroke-width="3.4"/>')


def _nose(spec: FaceSpec) -> str:
    if spec.nose == "small":
        return f'<path d="M150,168 C148,178 146,184 144,188 C148,192 154,192 157,188" stroke="{INK}" stroke-width="2.6" fill="none" stroke-linecap="round"/>'
    if spec.nose == "big":
        return f'<path d="M152,162 C152,176 158,188 160,194 C154,201 142,200 138,193 C142,189 146,180 147,166" stroke="{INK}" stroke-width="3" fill="none" stroke-linecap="round"/>'
    if spec.nose == "hooked":
        return f'<path d="M151,162 C158,170 162,182 156,192 C150,197 142,195 140,191" stroke="{INK}" stroke-width="3" fill="none" stroke-linecap="round"/>'
    # wide
    return (f'<path d="M149,166 C148,178 146,184 143,189" stroke="{INK}" stroke-width="2.6" fill="none"/>'
            f'<path d="M133,192 C138,198 162,198 167,192" stroke="{INK}" stroke-width="3.2" fill="none" stroke-linecap="round"/>'
            f'<circle cx="139" cy="191" r="1.8" fill="{INK}"/><circle cx="161" cy="191" r="1.8" fill="{INK}"/>')


def _mouth(spec: FaceSpec) -> str:
    y = 218
    if spec.mouth == "smirk":
        return f'<path d="M126,{y} C140,{y+8} 158,{y+6} 176,{y-7}" stroke="{INK}" stroke-width="3.4" fill="none" stroke-linecap="round"/>'
    if spec.mouth == "frown":
        return f'<path d="M128,{y+6} C140,{y-4} 160,{y-4} 172,{y+6}" stroke="{INK}" stroke-width="3.4" fill="none" stroke-linecap="round"/>'
    if spec.mouth == "open":
        return f'<ellipse cx="150" cy="{y+2}" rx="16" ry="9" fill="{INK}" opacity="0.85"/><path d="M138,{y-1} C146,{y-5} 154,{y-5} 162,{y-1}" stroke="{PAPER}" stroke-width="2" fill="none"/>'
    return f'<path d="M130,{y} C142,{y+5} 158,{y+5} 170,{y}" stroke="{INK}" stroke-width="3.2" fill="none" stroke-linecap="round"/>'


def _facial_hair(spec: FaceSpec) -> str:
    c = HAIR_HEX[spec.hair_color]
    fh = spec.facial_hair
    if fh == "none":
        return ""
    if fh == "stubble":
        dots = "".join(
            f'<circle cx="{110 + (i * 37) % 82}" cy="{206 + (i * 17) % 46}" r="1.1" fill="{INK}" opacity="0.5"/>'
            for i in range(46)
        )
        return dots
    if fh == "mustache":
        return f'<path d="M124,208 C136,198 164,198 176,208 C164,204 136,204 124,208 Z" fill="{c}" stroke="{INK}" stroke-width="1.4"/>'
    if fh == "goatee":
        return (f'<path d="M128,209 C140,201 160,201 172,209 C162,206 138,206 128,209 Z" fill="{c}"/>'
                f'<path d="M136,226 C140,248 160,248 164,226 C162,240 138,240 136,226 Z" fill="{c}" stroke="{INK}" stroke-width="1.4"/>')
    # full beard
    return (f'<path d="M100,180 C100,238 118,262 150,262 C182,262 200,238 200,180 '
            f'C198,224 192,244 150,244 C108,244 102,224 100,180 Z" fill="{c}"/>'
            f'<path d="M124,208 C136,200 164,200 176,208 C164,205 136,205 124,208 Z" fill="{c}"/>')


def _hat(spec: FaceSpec) -> str:
    if spec.hat == "none":
        return ""
    if spec.hat == "beanie":
        return (f'<path d="M92,118 C92,70 122,52 150,52 C178,52 208,70 208,118 L208,108 C208,118 92,118 92,108 Z" fill="{INK}" opacity="0.88"/>'
                f'<rect x="90" y="104" width="120" height="16" rx="8" fill="{INK}"/>')
    if spec.hat == "cap":
        return (f'<path d="M94,112 C94,68 122,52 150,52 C178,52 206,68 206,112 Z" fill="{INK}" opacity="0.9"/>'
                f'<path d="M94,112 L210,112 C236,112 240,124 232,128 L96,122 Z" fill="{INK}"/>')
    # fedora
    return (f'<ellipse cx="150" cy="106" rx="86" ry="14" fill="{INK}"/>'
            f'<path d="M104,106 C104,62 130,46 150,46 C170,46 196,62 196,106 Z" fill="{INK}"/>'
            f'<rect x="104" y="88" width="92" height="10" fill="#4a4540"/>')


def _extra(spec: FaceSpec) -> str:
    if spec.extra == "scar_cheek":
        return (f'<path d="M186,184 L196,210" stroke="{INK}" stroke-width="2.6"/>'
                + "".join(f'<path d="M{183+i*3},{190+i*6} L{191+i*3},{188+i*6}" stroke="{INK}" stroke-width="1.8"/>' for i in range(4)))
    if spec.extra == "earring":
        return f'<circle cx="217" cy="186" r="5" fill="none" stroke="{INK}" stroke-width="2.6"/>'
    if spec.extra == "neck_tattoo":
        return f'<path d="M132,272 L140,284 L148,272 L156,284 L164,272" stroke="{INK}" stroke-width="2.4" fill="none"/>'
    if spec.extra == "mole":
        return f'<circle cx="178" cy="206" r="3.4" fill="{INK}"/>'
    return ""


def render_face_svg(spec: FaceSpec, width: int = 300, paper: bool = True, seed_jitter: int = 0) -> str:
    """Render a FaceSpec to a self-contained SVG string (sketch style)."""
    skin = SKIN_HEX[spec.skin]
    hair_behind, hair_front = _hair(spec)
    ear_y = 178
    bg = (
        f'<rect width="300" height="360" fill="{PAPER}"/>'
        f'<rect width="300" height="360" fill="url(#grain)" opacity="0.5"/>'
        if paper else ""
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 360" width="{width}">
<defs>
  <pattern id="grain" width="5" height="5" patternUnits="userSpaceOnUse">
    <circle cx="1" cy="1" r="0.45" fill="#cfc6b2"/>
  </pattern>
  <filter id="rough"><feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="1" seed="{7 + seed_jitter}" result="n"/>
    <feDisplacementMap in="SourceGraphic" in2="n" scale="2.2"/></filter>
</defs>
{bg}
<g filter="url(#rough)">
  <path d="M118,250 L114,300 C112,318 100,322 84,330 L216,330 C200,322 188,318 186,300 L182,250"
        fill="{skin}" stroke="{INK}" stroke-width="3"/>
  {hair_behind}
  <ellipse cx="93" cy="{ear_y}" rx="11" ry="16" fill="{skin}" stroke="{INK}" stroke-width="2.6"/>
  <ellipse cx="207" cy="{ear_y}" rx="11" ry="16" fill="{skin}" stroke="{INK}" stroke-width="2.6"/>
  <path d="{_head_path(spec.face_shape)}" fill="{skin}" stroke="{INK}" stroke-width="3.4"/>
  {_facial_hair(spec)}
  {_brows(spec)}
  {_eyes(spec)}
  {_nose(spec)}
  {_mouth(spec)}
  {hair_front}
  {_hat(spec)}
  {_glasses(spec)}
  {_extra(spec)}
</g>
</svg>'''
    return svg


def render_unknown_svg(width: int = 300) -> str:
    """Placeholder silhouette for un-described attributes / mystery faces."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 360" width="{width}">
<rect width="300" height="360" fill="{PAPER}"/>
<path d="M118,250 L114,300 C112,318 100,322 84,330 L216,330 C200,322 188,318 186,300 L182,250 M150,68 C108,68 84,104 84,160 C84,216 112,262 150,262 C188,262 216,216 216,160 C216,104 192,68 150,68"
 fill="#d8d0bd" stroke="#9a9284" stroke-width="3" stroke-dasharray="7 6"/>
<text x="150" y="180" text-anchor="middle" font-family="monospace" font-size="64" fill="#9a9284">?</text>
</svg>'''
