"""Case generation: a culprit, a crime, and difficulty-scaled parameters.

Ground truth is authored here — the game always knows the culprit's exact
FaceSpec. Templates make cases playable offline; MiniCPM5-1B (when live)
rewrites the flavor text, never the facts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .face import FaceSpec, VOCAB

CRIMES = [
    ("the Museum Job", "lifted a 300-year-old pocket watch from the city museum — walked out whistling"),
    ("the Bakery Heist", "made off with the entire prize-winning sourdough starter, 'Greg'"),
    ("the Pigeon Racket", "was caught training pigeons to steal sandwiches from the plaza"),
    ("the Karaoke Incident", "stole the golden microphone mid-song and finished the chorus outside"),
    ("the Ferry Fiasco", "boarded the ferry with 47 'borrowed' garden gnomes"),
    ("the Library Caper", "returned 312 books — none of which belonged to this library"),
    ("the Cheese Affair", "rolled a 40kg wheel of manchego out of the night market"),
    ("the Lighthouse Prank", "repainted the lighthouse beam... disco"),
]

RANKS = [
    # (rank name, glimpse seconds, lineup size, culprit_changes_between_glimpse_and_lineup)
    ("ROOKIE", 3.0, 4, 0),
    ("DETECTIVE", 2.0, 6, 0),
    ("INSPECTOR", 1.5, 6, 1),   # culprit alters ONE salient attribute ("he shaved...")
    ("CHIEF", 1.0, 8, 1),
]

DISGUISE_LINES = {
    "hat": "Word on the street: he ditched the headwear.",
    "glasses": "Informant says he changed his eyewear.",
    "facial_hair": "Heads up — he's been to a barber since.",
    "hair_style": "He's changed his hair. Of course he has.",
    "hair_color": "Reports say the hair color was... temporary.",
}


@dataclass
class Case:
    case_no: int
    rank: str
    crime_name: str
    crime_blurb: str
    culprit: FaceSpec          # what the witness SAW (glimpse)
    lineup_culprit: FaceSpec   # what they look like AT THE LINEUP (may be disguised)
    disguise_attr: str | None
    glimpse_seconds: float
    lineup_size: int
    seed: int

    @property
    def disguise_line(self) -> str:
        if not self.disguise_attr:
            return ""
        return DISGUISE_LINES.get(self.disguise_attr, "They say he looks a little different now.")


def make_case(case_no: int, seed: int | None = None) -> Case:
    """Deterministic from seed -> reproducible cases (and a shareable case bank)."""
    seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(seed)
    rank_idx = min(case_no - 1, len(RANKS) - 1)
    rank, glimpse_s, lineup_n, n_changes = RANKS[rank_idx]

    culprit = FaceSpec.random(rng)
    # ROOKIE kindness: guarantee at least two salient, easy-to-describe features
    if rank == "ROOKIE":
        if culprit.hat == "none" and culprit.glasses == "none" and culprit.facial_hair == "none":
            culprit = culprit.with_changes(hat=rng.choice(["beanie", "cap", "fedora"]))
        if culprit.hair_style == "bald":
            culprit = culprit.with_changes(hair_style=rng.choice(["curly", "long", "mohawk"]))

    lineup_culprit, disguise_attr = culprit, None
    if n_changes:
        # change one SALIENT attribute the witness probably anchored on — cruel, fair, funny
        candidates = [a for a in ("hat", "glasses", "facial_hair", "hair_style", "hair_color")
                      if getattr(culprit, a) != "none" or a in ("hair_style", "hair_color")]
        disguise_attr = rng.choice(candidates)
        options = [v for v in VOCAB[disguise_attr] if v != getattr(culprit, disguise_attr)]
        lineup_culprit = culprit.with_changes(**{disguise_attr: rng.choice(options)})

    crime_name, crime_blurb = CRIMES[rng.randrange(len(CRIMES))]
    return Case(
        case_no=case_no, rank=rank, crime_name=crime_name, crime_blurb=crime_blurb,
        culprit=culprit, lineup_culprit=lineup_culprit, disguise_attr=disguise_attr,
        glimpse_seconds=glimpse_s, lineup_size=lineup_n, seed=seed,
    )
