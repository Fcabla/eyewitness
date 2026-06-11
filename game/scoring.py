"""Transparent scoring — fairness by showing the work.

The reveal shows three columns: what you SAID, what the artist DREW from it,
and the TRUTH. Score is objective: per-attribute accuracy (salience-weighted)
+ the lineup verdict. No fuzzy similarity, no rigged feel.
"""
from __future__ import annotations

from dataclasses import dataclass

from .face import FaceSpec, VOCAB, SALIENCE

LABELS_EN = {
    "face_shape": "Face", "skin": "Skin", "hair_style": "Hair", "hair_color": "Hair color",
    "brows": "Brows", "eyes": "Eyes", "glasses": "Glasses", "nose": "Nose",
    "mouth": "Mouth", "facial_hair": "Facial hair", "hat": "Headwear", "extra": "Marks",
}


@dataclass
class TestimonyReport:
    rows: list[tuple[str, str, str, str]]  # (label, said, truth, verdict) verdict: hit|miss|silent
    hits: int
    misses: int
    silents: int
    accuracy_pct: int
    weighted_pct: int


def grade_testimony(described: dict[str, str | None], truth: FaceSpec) -> TestimonyReport:
    rows, hits, misses, silents = [], 0, 0, 0
    w_total = w_earned = 0.0
    for attr in VOCAB:
        label = LABELS_EN[attr]
        truth_v = getattr(truth, attr)
        said_v = described.get(attr)
        w = SALIENCE.get(attr, 1.0)
        w_total += w
        if said_v is None:
            silents += 1
            verdict = "silent"
            said_txt = "—"
        elif said_v == truth_v:
            hits += 1
            w_earned += w
            verdict = "hit"
            said_txt = said_v.replace("_", " ")
        else:
            misses += 1
            verdict = "miss"
            said_txt = said_v.replace("_", " ")
        rows.append((label, said_txt, truth_v.replace("_", " "), verdict))
    n_said = hits + misses
    return TestimonyReport(
        rows=rows, hits=hits, misses=misses, silents=silents,
        accuracy_pct=round(100 * hits / n_said) if n_said else 0,
        weighted_pct=round(100 * w_earned / w_total),
    )


def detective_rating(report: TestimonyReport, picked_culprit: bool, glimpse_s: float) -> tuple[str, str]:
    """(badge, one-liner) for the verdict screen."""
    if picked_culprit and report.weighted_pct >= 60:
        return "★ STAR WITNESS", "The sketch artist wants to shake your hand."
    if picked_culprit:
        return "LUCKY BADGE", "Terrible description. Correct arrest. We'll take it."
    if report.weighted_pct >= 55:
        return "ALMOST", "Great memory — wrong arrest. He walked right past you."
    return "GOLDFISH CLEARANCE", f"{glimpse_s:.0f} seconds was apparently not enough. The pigeons remember more."
