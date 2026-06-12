"""Testimony parser: free-text witness description -> attribute slots.

Tier A (this file, always available): deterministic synonym matcher, EN+ES.
Tier B (model.py, when deployed): MiniCPM5-1B slot-filling — handles messy
real language ("kind of a roundish face, hadn't slept in days, big caterpillar
eyebrows"). Tier A doubles as validation fallback when the model output fails.

Output contract for BOTH tiers: dict[attr -> value|None], values from VOCAB.
None = the witness never mentioned it (that silence powers the lineup).
"""
from __future__ import annotations

import re
import unicodedata

from .face import VOCAB

# attr -> value -> list of trigger phrases (EN + ES), checked longest-first
SYNONYMS: dict[str, dict[str, list[str]]] = {
    "sex": {
        "male": ["a man", "a guy", "a dude", "male", "he was", "un hombre", "un tio", "un tipo", "un chico", "varon"],
        "female": ["a woman", "a lady", "a girl", "female", "she was", "una mujer", "una tia", "una chica", "una senora"],
    },
    "age": {
        "young": ["young", "a kid", "teenager", "in his twenties", "in her twenties", "joven", "un chaval", "una chavala", "veintipocos"],
        "adult": ["middle aged", "middle-aged", "adult", "de mediana edad", "adulto", "cuarenton", "treintanero"],
        "old": ["old", "elderly", "a senior", "retiree", "in his seventies", "mayor", "anciano", "anciana", "un abuelo", "una abuela", "viejo", "vieja", "jubilado"],
    },
    "face_shape": {
        "oval": ["oval face", "cara ovalada", "oval"],
        "round": ["round face", "roundish", "chubby face", "cara redonda", "regordeta", "mofletudo"],
        "square": ["square face", "square jaw", "jawline", "cara cuadrada", "mandibula marcada"],
        "long": ["long face", "narrow face", "thin face", "cara alargada", "cara larga", "cara fina"],
    },
    "skin": {
        "light": ["pale", "light skin", "fair skin", "palido", "piel clara", "blanquito"],
        "medium": ["tan", "medium skin", "olive skin", "moreno claro", "piel media", "tostado"],
        "dark": ["dark skin", "dark-skinned", "piel oscura", "moreno oscuro", "negro"],
    },
    "hair_style": {
        "bald": ["bald", "no hair", "shaved head", "calvo", "sin pelo", "rapado al cero"],
        "buzz": ["buzz cut", "buzzcut", "crew cut", "very short hair", "rapado", "pelo al uno"],
        "short_messy": ["messy hair", "short hair", "bedhead", "pelo corto", "despeinado", "pelo revuelto"],
        "slick_back": ["slicked back", "slick back", "slicked his hair back", "hair back like a banker", "gelled", "combed back", "engominado", "peinado hacia atras", "repeinado", "pelo hacia atras"],
        "curly": ["curly", "curls", "afro", "rizado", "rizos", "pelo chino"],
        "long": ["long hair", "hair down", "melena", "pelo largo"],
        "ponytail": ["ponytail", "pony tail", "coleta"],
        "mohawk": ["mohawk", "mohican", "cresta"],
    },
    "hair_color": {
        "black": ["black hair", "pelo negro", "moreno de pelo"],
        "brown": ["brown hair", "brunette", "pelo castano", "castano", "pelo marron"],
        "blond": ["blond", "blonde", "rubio", "pelo amarillo"],
        "red": ["red hair", "redhead", "ginger", "pelirrojo", "pelo rojo"],
        "gray": ["gray hair", "grey hair", "white hair", "silver hair", "canoso", "pelo gris", "pelo blanco", "canas"],
    },
    "brows": {
        "thin": ["thin eyebrows", "thin brows", "cejas finas", "cejas depiladas"],
        "thick": ["thick eyebrows", "thick brows", "strong brows", "cejas gruesas", "cejas marcadas"],
        "bushy": ["bushy eyebrows", "bushy brows", "caterpillar", "cejas pobladas", "cejas de oruga", "cejudo"],
        "unibrow": ["unibrow", "monobrow", "uniceja", "cejijunto", "una sola ceja"],
    },
    "eyes": {
        "narrow": ["narrow eyes", "squinty", "squinting", "ojos pequenos", "ojos entrecerrados", "ojos rasgados"],
        "big": ["big eyes", "wide eyes", "bulging", "ojos grandes", "ojos saltones", "ojazos"],
        "droopy": ["droopy eyes", "tired eyes", "sleepy eyes", "sad eyes", "hadn't slept", "ojos caidos", "ojos tristes", "ojos cansados", "ojeras"],
        "normal": ["normal eyes", "ojos normales"],
    },
    "glasses": {
        "round": ["round glasses", "circular glasses", "grandpa glasses", "gafas redondas", "lentes redondos", "gafas de abuelo", "gafas de esas redondas"],
        "square": ["square glasses", "rectangular glasses", "gafas cuadradas", "gafas de pasta"],
        "sunglasses": ["sunglasses", "shades", "dark glasses", "gafas de sol", "gafas oscuras", "lentes oscuros"],
        "none": ["no glasses", "sin gafas"],
    },
    "nose": {
        "small": ["small nose", "little nose", "button nose", "nariz pequena", "naricilla", "nariz chata"],
        "big": ["big nose", "large nose", "huge nose", "narizon", "nariz grande", "napia"],
        "hooked": ["hooked nose", "hook nose", "roman nose", "beak", "nariz aguilena", "nariz de gancho", "nariz curva"],
        "wide": ["wide nose", "broad nose", "flat nose", "nariz ancha"],
    },
    "mouth": {
        "smirk": ["smirk", "smirking", "smug smile", "half smile", "sonrisilla", "sonrisa de lado", "media sonrisa", "sonrisa chulesca"],
        "frown": ["frown", "frowning", "scowl", "angry mouth", "ceno", "boca enfadada", "mueca"],
        "open": ["mouth open", "open mouth", "gasping", "boca abierta"],
        "neutral": ["neutral mouth", "boca normal"],
    },
    "facial_hair": {
        "none": ["clean shaven", "clean-shaven", "no beard", "afeitado", "sin barba", "bien afeitado"],
        "stubble": ["stubble", "five o'clock shadow", "unshaven", "scruffy", "barba de tres dias", "sin afeitar", "barba incipiente"],
        "mustache": ["mustache", "moustache", "bigote", "mostacho"],
        "goatee": ["goatee", "perilla", "chivo"],
        "full_beard": ["full beard", "big beard", "huge beard", "thick beard", "bearded", "barba", "barbudo", "barba cerrada", "barbaza"],
    },
    "hat": {
        "beanie": ["beanie", "wool hat", "knit hat", "gorro", "gorro de lana"],
        "cap": ["baseball cap", "cap", "gorra", "visera"],
        "fedora": ["fedora", "brimmed hat", "trilby", "sombrero"],
        "none": ["no hat", "sin gorro", "sin sombrero", "sin gorra"],
    },
    "extra": {
        "scar_cheek": ["scar", "cicatriz", "marca en la cara"],
        "earring": ["earring", "ear ring", "pendiente", "arete", "aro en la oreja"],
        "neck_tattoo": ["neck tattoo", "tattoo", "tatuaje", "tattoo en el cuello"],
        "mole": ["mole", "beauty mark", "lunar"],
        "none": [],
    },
}

# precedence quirks: "barba" matches full_beard but "barba de tres dias" is stubble —
# the longest-first matching below handles it.


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text)


_MATCHERS: list[tuple[str, str, str]] = sorted(
    ((phrase, attr, value)
     for attr, values in SYNONYMS.items()
     for value, phrases in values.items()
     for phrase in phrases),
    key=lambda t: len(t[0]), reverse=True,
)


def parse_testimony(text: str) -> dict[str, str | None]:
    """Tier A: longest-phrase-first deterministic matching, EN+ES."""
    norm = _normalize(text)
    out: dict[str, str | None] = {attr: None for attr in VOCAB}
    consumed: list[tuple[int, int]] = []
    for phrase, attr, value in _MATCHERS:
        if out[attr] is not None:
            continue
        i = norm.find(phrase)
        while i != -1:
            span = (i, i + len(phrase))
            if not any(s < span[1] and span[0] < e for s, e in consumed):
                out[attr] = value
                consumed.append(span)
                break
            i = norm.find(phrase, i + 1)
    # bare-color heuristic: "rubio"/"blond" w/o the word hair still means hair color
    return out
