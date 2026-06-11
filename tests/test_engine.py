"""Engine QA: edge cases + lineup fairness simulation."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.casegen import make_case
from game.face import FaceSpec, VOCAB, render_face_svg
from game.lineup import build_lineup
from game.parser import parse_testimony
from game.scoring import grade_testimony
from app import sketch_from_testimony  # noqa: E402


def test_edge_testimonies():
    cases = {
        "empty": "",
        "gibberish": "asdf qwerty blorp 12345 !!!",
        "contradiction": "bald with a long ponytail and also curly",
        "very_long": "well I mean " * 200 + " he had a mustache",
        "emoji_mixed": "🤔 tenia barba 🧔 y gafas de sol 😎 creo",
        "negations": "no hat, clean shaven, no glasses",
    }
    for name, text in cases.items():
        d = parse_testimony(text)
        assert set(d) == set(VOCAB), f"{name}: keys mismatch"
        for attr, v in d.items():
            assert v is None or v in VOCAB[attr], f"{name}: invalid {attr}={v}"
        # sketch must always render
        svg = render_face_svg(sketch_from_testimony(d))
        assert svg.startswith("<svg"), f"{name}: sketch failed"
    d = parse_testimony(cases["negations"])
    assert d["hat"] == "none" and d["facial_hair"] == "none" and d["glasses"] == "none"
    d = parse_testimony(cases["very_long"])
    assert d["facial_hair"] == "mustache"
    print("edge testimonies: OK")


def test_lineup_fairness(n_sims: int = 400):
    """The culprit must always be present, identifiable-in-principle, and the
    lineup must never leak (no distractor identical to culprit)."""
    rng = random.Random(123)
    trivial = 0
    for i in range(n_sims):
        case = make_case(1 + i % 4, seed=i)
        # simulate a mediocre witness: notices 4 random attrs, gets 1 wrong
        attrs = rng.sample(list(VOCAB), 4)
        described = {a: None for a in VOCAB}
        for a in attrs[:3]:
            described[a] = getattr(case.culprit, a)
        wrong_attr = attrs[3]
        described[wrong_attr] = rng.choice(
            [v for v in VOCAB[wrong_attr] if v != getattr(case.culprit, wrong_attr)])
        faces, idx = build_lineup(case.lineup_culprit, described, case.lineup_size, rng)
        assert faces[idx] == case.lineup_culprit
        assert len(faces) == case.lineup_size
        for j, f in enumerate(faces):
            if j != idx:
                assert f.diff(case.lineup_culprit), f"sim {i}: distractor identical to culprit"
        # trivial = culprit uniquely matches ALL correctly-described attrs
        said_right = {a: v for a, v in described.items()
                      if v and v == getattr(case.lineup_culprit, a)}
        if said_right:
            matches = [f for f in faces
                       if all(getattr(f, a) == v for a, v in said_right.items())]
            if len(matches) == 1:
                trivial += 1
    pct = 100 * trivial / n_sims
    print(f"lineup fairness: OK ({n_sims} sims; uniquely-identifiable-from-correct-info: {pct:.0f}%)")
    # should be sometimes-but-not-always: winning is possible, not free
    assert 10 <= pct <= 90, f"difficulty out of band: {pct:.0f}%"


def test_scoring_bounds():
    rng = random.Random(7)
    for i in range(100):
        truth = FaceSpec.random(rng)
        described = {a: (getattr(truth, a) if rng.random() < 0.5 else None) for a in VOCAB}
        rep = grade_testimony(described, truth)
        assert 0 <= rep.weighted_pct <= 100 and rep.misses == 0
    print("scoring bounds: OK")


if __name__ == "__main__":
    test_edge_testimonies()
    test_lineup_fairness()
    test_scoring_bounds()
    print("ALL QA TESTS PASS")
