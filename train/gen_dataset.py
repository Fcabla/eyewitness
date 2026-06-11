"""Synthetic testimony dataset — author-the-ground-truth, training edition.

The engine knows every culprit's exact FaceSpec. This script runs the REVERSE
direction programmatically: spec -> noisy natural-language testimony (EN/ES/mixed,
partial mentions, fillers, hedges, indirect phrasings). The (testimony -> spec JSON)
pairs are perfect supervision BY CONSTRUCTION — no labeling, no big-model distillation.

Usage: python train/gen_dataset.py [n_samples] > train/dataset.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.face import FaceSpec, VOCAB  # noqa: E402

# phrasings per attribute value: list of (lang, template) — {x} never used, plain strings
PHRASES: dict[str, dict[str, list[str]]] = {
    "face_shape": {
        "oval": ["oval face", "normal face shape", "cara normal, ovalada"],
        "round": ["round face", "roundish face", "chubby cheeks", "cara redonda", "carirredondo", "cara de pan"],
        "square": ["square face", "strong jaw", "square jaw, like a brick", "cara cuadrada", "mandibula cuadrada"],
        "long": ["long face", "narrow face", "horse-ish face honestly", "cara alargada", "cara larga y fina"],
    },
    "skin": {
        "light": ["pale", "very pale skin", "light-skinned", "palido", "blanquisimo", "piel clara"],
        "medium": ["tanned", "olive skin", "medium skin tone", "morenito", "piel tostada"],
        "dark": ["dark skin", "dark-skinned", "piel oscura", "bastante moreno de piel"],
    },
    "hair_style": {
        "bald": ["bald", "completely bald", "no hair at all", "calvo", "calvo total", "ni un pelo"],
        "buzz": ["buzz cut", "army haircut", "super short hair", "rapado", "pelo al uno", "corte militar"],
        "short_messy": ["short messy hair", "bedhead", "hair all over the place but short", "pelo corto despeinado", "despeinado"],
        "slick_back": ["slicked back hair", "hair combed back like a banker", "gelled back", "engominado", "pelo repeinado hacia atras", "peinado de mafioso"],
        "curly": ["curly hair", "a big afro", "all curls", "pelo rizado", "un afro", "rizos por todas partes"],
        "long": ["long hair", "hair down to the shoulders", "melena", "pelo largo", "greñas"],
        "ponytail": ["a ponytail", "hair tied back in a ponytail", "coleta", "el pelo recogido en coleta"],
        "mohawk": ["a mohawk", "one of those punk crests", "cresta", "cresta punki"],
    },
    "hair_color": {
        "black": ["black hair", "jet black hair", "pelo negro", "pelo negro azabache"],
        "brown": ["brown hair", "brownish hair", "castano", "pelo marron"],
        "blond": ["blond", "blonde hair", "rubio", "rubio de bote diria"],
        "red": ["red hair", "a ginger", "pelirrojo", "pelo zanahoria"],
        "gray": ["gray hair", "going white", "silver hair", "canoso", "lleno de canas", "pelo gris"],
    },
    "brows": {
        "thin": ["thin eyebrows", "barely-there eyebrows", "cejas finas", "cejas depiladisimas"],
        "thick": ["thick eyebrows", "strong eyebrows", "cejas gruesas", "cejas marcadas"],
        "bushy": ["bushy eyebrows", "caterpillar eyebrows", "wild eyebrows", "cejas pobladas", "cejas como orugas", "cejas de matorral"],
        "unibrow": ["a unibrow", "eyebrows that meet in the middle", "uniceja", "cejijunto", "las cejas juntas"],
    },
    "eyes": {
        "normal": ["normal eyes", "nothing special about the eyes", "ojos normales"],
        "narrow": ["narrow eyes", "squinty eyes", "always squinting", "ojos pequenos", "ojos entrecerrados"],
        "big": ["big eyes", "huge bulging eyes", "wide eyes", "ojos enormes", "ojos saltones", "ojazos"],
        "droopy": ["droopy tired eyes", "looked like he hadn't slept in days", "sleepy sad eyes", "ojos caidos", "cara de no haber dormido", "ojeras tremendas"],
    },
    "glasses": {
        "none": ["no glasses", "wasn't wearing glasses", "sin gafas"],
        "round": ["round glasses", "little round glasses, grandpa style", "circular spectacles", "gafas redondas", "gafas de abuelo", "gafitas redondas"],
        "square": ["square glasses", "rectangular thick-rimmed glasses", "gafas cuadradas", "gafas de pasta"],
        "sunglasses": ["sunglasses", "dark shades indoors, the nerve", "gafas de sol", "gafas oscuras"],
    },
    "nose": {
        "small": ["a small nose", "tiny button nose", "nariz pequena", "nariz chata"],
        "big": ["a big nose", "a huge honker", "enormous nose", "narizon", "una napia tremenda", "nariz grande"],
        "hooked": ["a hooked nose", "hawk nose", "beak of a nose", "nariz aguilena", "nariz de gancho"],
        "wide": ["a wide nose", "broad flat nose", "nariz ancha", "nariz aplastada"],
    },
    "mouth": {
        "neutral": ["normal mouth", "nothing odd about the mouth", "boca normal"],
        "smirk": ["smirking", "this smug half-smile", "a cocky grin", "una sonrisilla", "sonrisa de listillo", "sonriendo de lado"],
        "frown": ["frowning", "angry mouth, scowling", "ceno fruncido", "cara de enfado", "boca de amargado"],
        "open": ["mouth hanging open", "gawping", "boca abierta", "con la boca abierta como un pasmado"],
    },
    "facial_hair": {
        "none": ["clean shaven", "no beard at all", "freshly shaved", "bien afeitado", "sin barba", "cara lavada"],
        "stubble": ["stubble", "a few days unshaven", "five o'clock shadow", "barba de tres dias", "sin afeitar", "barba incipiente"],
        "mustache": ["a mustache", "a proper mustache", "bigote", "un bigoton", "mostacho"],
        "goatee": ["a goatee", "one of those chin beards", "perilla", "barba de chivo"],
        "full_beard": ["a full beard", "a huge thick beard", "proper lumberjack beard", "barba cerrada", "un barbon", "barba de lenador"],
    },
    "hat": {
        "none": ["no hat", "nothing on his head", "sin gorro ni nada"],
        "beanie": ["a beanie", "one of those wool hats", "a knit cap", "un gorro de lana", "gorro de esos de invierno"],
        "cap": ["a baseball cap", "a cap, maybe backwards", "una gorra", "gorra de beisbol"],
        "fedora": ["a fedora", "an old-style brimmed hat", "un sombrero", "sombrero de detective de pelicula"],
    },
    "extra": {
        "none": [],
        "scar_cheek": ["a scar on his cheek", "a nasty scar across the cheek", "una cicatriz en la mejilla", "una marca en la cara, como un corte"],
        "earring": ["an earring", "a hoop earring", "un pendiente", "un aro en la oreja"],
        "neck_tattoo": ["a tattoo on his neck", "neck tattoo, zigzag thing", "un tatuaje en el cuello", "tatuaje cutre en el cuello"],
        "mole": ["a mole on his cheek", "a beauty mark", "un lunar", "un lunar bien gordo"],
    },
}

FILLERS_EN = ["I think", "maybe", "pretty sure", "like,", "honestly", "if I remember right", "I swear"]
FILLERS_ES = ["creo", "me parece", "juraria que", "no se,", "o sea", "si no recuerdo mal", "te lo juro"]
CONNECT = [", ", ", and ", ". ", ", y ", " — ", "... "]


def spec_to_testimony(spec: FaceSpec, rng: random.Random) -> tuple[str, dict]:
    """Pick 3-7 attributes, phrase them noisily, return (text, partial truth dict)."""
    attrs = [a for a in VOCAB if PHRASES[a].get(getattr(spec, a))]
    rng.shuffle(attrs)
    chosen = attrs[: rng.randint(3, min(7, len(attrs)))]
    parts, truth = [], {a: None for a in VOCAB}
    for a in chosen:
        v = getattr(spec, a)
        phrase = rng.choice(PHRASES[a][v])
        if rng.random() < 0.3:
            filler = rng.choice(FILLERS_ES if any(c in phrase for c in "ñáéíóú") or rng.random() < 0.4 else FILLERS_EN)
            phrase = f"{filler} {phrase}"
        parts.append(phrase)
        truth[a] = v
    text = ""
    for i, p in enumerate(parts):
        text += p + (rng.choice(CONNECT) if i < len(parts) - 1 else "")
    return text.strip(), truth


def main(n: int = 4000, seed: int = 7) -> None:
    rng = random.Random(seed)
    for _ in range(n):
        spec = FaceSpec.random(rng)
        text, truth = spec_to_testimony(spec, rng)
        print(json.dumps({"testimony": text, "labels": truth}, ensure_ascii=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
