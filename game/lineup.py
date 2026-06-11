"""Lineup builder — the signature mechanic.

Distractors are derived from the PLAYER'S OWN testimony errors:
- Attributes you got WRONG become shared between culprit and distractors
  exactly as YOU described them (so your own sketch misleads you).
- Attributes you NAILED are varied across distractors (they don't help
  you less than they should — the truth stays discriminative).
- Attributes you NEVER MENTIONED become the confusion axes.

Result: the lineup is hardest exactly where your memory failed. Fair,
transparent, and personal.
"""
from __future__ import annotations

import random

from .face import FaceSpec, VOCAB, SALIENCE


def build_lineup(
    truth: FaceSpec,
    described: dict[str, str | None],
    size: int,
    rng: random.Random,
) -> tuple[list[FaceSpec], int]:
    """Returns (faces, culprit_index)."""
    wrong = {a: v for a, v in described.items() if v and v != getattr(truth, a)}
    correct = {a for a, v in described.items() if v and v == getattr(truth, a)}
    missed = [a for a in VOCAB if not described.get(a)]
    # sort missed by salience: confuse on what they SHOULD have noticed
    missed.sort(key=lambda a: SALIENCE.get(a, 1.0), reverse=True)

    distractors: list[FaceSpec] = []
    seen = {truth.to_dict().__str__()}
    attempts = 0
    while len(distractors) < size - 1 and attempts < 200:
        attempts += 1
        d = truth.to_dict()
        # 1) plant the player's WRONG beliefs in ~60% of distractors
        for attr, believed in wrong.items():
            if rng.random() < 0.6:
                d[attr] = believed
        # 2) vary 2-3 of the MISSED salient attributes
        for attr in rng.sample(missed, k=min(len(missed), rng.randint(2, 3))):
            d[attr] = rng.choice([v for v in VOCAB[attr] if v != d[attr]])
        # 3) sometimes vary a CORRECT one too, so right answers still need looking
        if correct and rng.random() < 0.35:
            attr = rng.choice(sorted(correct))
            d[attr] = rng.choice([v for v in VOCAB[attr] if v != d[attr]])
        # never identical to the truth
        spec = FaceSpec(**d)
        if spec.diff(truth) and str(d) not in seen:
            seen.add(str(d))
            distractors.append(spec)

    # fallback fill (degenerate cases): random-but-near faces
    while len(distractors) < size - 1:
        d = truth.to_dict()
        for attr in rng.sample(list(VOCAB), k=3):
            d[attr] = rng.choice([v for v in VOCAB[attr] if v != d[attr]])
        spec = FaceSpec(**d)
        if spec.diff(truth):
            distractors.append(spec)

    culprit_idx = rng.randrange(size)
    faces = distractors[:]
    faces.insert(culprit_idx, truth)
    return faces, culprit_idx
