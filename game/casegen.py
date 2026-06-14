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


# Authored verdict punchlines per crime (caught / escaped). Same philosophy as
# the cases: author the comedy, don't ask a 1B to improvise it (taunt_lab: ~3%
# hit rate on free-form jokes). The live model still adds the personalized jab.
CRIME_TAUNTS = {
    "the Museum Job": {
        "caught": [
            "Three hundred years that watch waited for a wrist with TASTE. We had eleven beautiful minutes.",
            "You heard the whistling and STILL needed a witness? The tune was a full confession, detective.",
            "Three centuries of tick-tock and the museum never ONCE took it dancing. I did.",
            "I gave that watch eleven minutes of freedom. Time well stolen.",
        ],
        "escaped": [
            "Somewhere a pocket watch is ticking just for me. You'll hear it tonight, detective. At bedtime.",
            "I'd return the watch, but it finally belongs to someone punctual.",
            "The museum got a lovely empty case out of it. Minimalism, detective. You're welcome.",
            "Tick-tock, detective. That's not the watch — that's your career.",
        ],
    },
    "the Bakery Heist": {
        "caught": [
            "Greg was a hostage of MEDIOCRE croissants. I gave him a better life, and you gave me handcuffs.",
            "Arrest me, fine — but Greg rises at dawn, and so will my appeal.",
            "Greg doubled in size under MY roof. The bakery starved him. Who's the real criminal here?",
            "You found me by following the smell of fresh bread. I respect it. Greg doesn't.",
        ],
        "escaped": [
            "Greg and I are very happy together. He's bubbling. I'm free. Bon appétit, detective.",
            "You lost the case AND the starter. Greg sends his regards from an undisclosed warm spot.",
            "Greg's sourdough children are in every bakery in this city now. You can't arrest a bloodline.",
            "The bakery wants Greg back? Greg votes no. We took a vote. It was unanimous.",
        ],
    },
    "the Pigeon Racket": {
        "caught": [
            "You caught ME, but the pigeons unionized. Good luck negotiating with THEM.",
            "Fine, it was my racket. But the birds kept the tips — check THEIR coop, not mine.",
            "Three months of training and they sing for breadcrumbs at the first interrogation. Pigeons, detective. Never again.",
            "The sandwiches were an inside job — the pigeons were already in the plaza. I just gave them PURPOSE.",
        ],
        "escaped": [
            "My pigeons watched you arrest the wrong man. They remember faces better than your witness.",
            "Every sandwich in that plaza is still mine, by air mail. Coo, detective. Coo.",
            "Forty birds saw everything and your witness still picked the wrong man. Even the pigeons are laughing.",
            "My pigeons are still on duty, detective. Guard your lunch.",
        ],
    },
    "the Karaoke Incident": {
        "caught": [
            "You arrested me before the encore. The audience deserves to know how it ends.",
            "I hit the high note AND the exit. One arrest doesn't change the scoreboard, detective.",
            "The golden mic, the high note, the standing ovation outside — and THIS is my encore? Handcuffs?",
            "I didn't steal the microphone. The microphone and I eloped. There were witnesses. Apparently.",
        ],
        "escaped": [
            "The golden mic and I are booked every Friday now. First round's on the innocent guy.",
            "You can't catch a man mid-chorus. Legally, I was still performing.",
            "Tell the bar I said thanks for the acoustics. The getaway had REVERB.",
            "Somewhere an innocent man is being booked, and I'm still holding the last note.",
        ],
    },
    "the Ferry Fiasco": {
        "caught": [
            "BORROWED, detective. Gnome number forty-eight stayed home — that's restraint, and restraint is innocence.",
            "Forty-seven gnomes and you only caught ONE of us. They scattered. I'm so proud.",
            "Forty-seven gnomes wanted to see the ocean ONCE before retirement. I regret nothing.",
            "The gnomes formed a tiny pyramid to board, detective. You missed it. Everyone missed it. Tragic.",
        ],
        "escaped": [
            "The gnomes and I saw the sea together. Some dreams are worth an innocent man's afternoon.",
            "Forty-seven tiny accomplices and your witness described NONE of them. Amateurs, everywhere.",
            "Customs checked every bag and missed forty-seven pointed hats. I'll never trust this country's borders again.",
            "The gnomes send postcards now, detective. Tiny ones. From the sea.",
        ],
    },
    "the Library Caper": {
        "caught": [
            "I ENRICHED your library by 312 volumes and THIS is the thanks. Culture is dead, detective.",
            "Show me the law against returning books. Take your time. I'll wait — I read fast.",
            "None of these books are mine. I didn't take them — I DELIVERED them, you ingrate.",
            "Three hundred twelve books, alphabetized, on time. Arrest me when this city does literacy HALF as well.",
        ],
        "escaped": [
            "Those 312 books came from somewhere, detective. That's tomorrow's case. See you then.",
            "An innocent man is doing time for DONATING literature. The late fees are on your conscience.",
            "Check the inside covers, detective. Every name in this town but mine.",
            "I'd lie low, but I just got 312 recommendations and the reading list is BRUTAL.",
        ],
    },
    "the Cheese Affair": {
        "caught": [
            "Forty kilos of manchego rolls DOWNHILL, detective. Physics did the crime. I merely steered.",
            "You smelled the manchego before you saw me, admit it. The cheese confessed. I never will.",
            "Forty kilos, downhill, midnight, zero casualties. That's not theft — that's LOGISTICS.",
            "The manchego chose me, detective. You saw the wheel roll. Did it ever once roll BACK?",
        ],
        "escaped": [
            "The wheel and I retired to the countryside. It's aging beautifully. So is your case file.",
            "Forty kilos rolled right past your witness and they still described ME wrong? The CHEESE was right there.",
            "Aged eighteen months, gone in ninety seconds. Tell the night market to update their security and their recipes.",
            "You'll find me when the manchego runs out. So — never. We've done the math, the wheel and I.",
        ],
    },
    "the Lighthouse Prank": {
        "caught": [
            "Ships were BORED, detective. Now the harbor has rhythm and I have handcuffs. Art is sacrifice.",
            "You arrested a visionary. Tonight that beam spins gold and violet, and the sea DANCES.",
            "Crime? I did it with a DISCO, detective. That beam was beige. BEIGE. On the SEA.",
            "Three ships honked along last night. You call it vandalism; the harbor calls it Friday.",
        ],
        "escaped": [
            "Every ship that docks tonight docks in STYLE. The man you grabbed can't even moonwalk.",
            "The beam still spins disco, detective. Some crimes you can see from twenty miles away. Catch up.",
            "The paint is waterproof and so is my alibi. Enjoy the light show, detective.",
            "An innocent man, a disco beam, and you dancing between them. Beautiful case. No notes.",
        ],
    },
}

# Modal-bank crimes (no authored lines for them) fall back to these.
GENERIC_TAUNTS = {
    "caught": [
        "Caught at last. It took your whole department and one lucky witness — frame the achievement.",
        "Yes, it was me. It was ALWAYS me. The execution was flawless; my exit needed work.",
    ],
    "escaped": [
        "Some poor soul is signing MY paperwork right now. Give him my coffee order.",
        "You had me. You HAD me. And then you trusted a witness — adorable.",
    ],
}


def _load_modal_bank() -> list[tuple[str, str]]:
    """Crimes batch-generated by MiniCPM5-1B-GGUF via llama.cpp on Modal
    (modal_factory.py). Extends the hand-written set when present."""
    import json
    from pathlib import Path

    bank_path = Path(__file__).resolve().parent.parent / "assets" / "case_bank.json"
    if not bank_path.exists():
        return []
    try:
        items = json.loads(bank_path.read_text())
        return [(it["name"], it["blurb"]) for it in items
                if isinstance(it, dict) and it.get("name") and it.get("blurb")]
    except (json.JSONDecodeError, OSError):
        return []


CRIMES = CRIMES + _load_modal_bank()

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
