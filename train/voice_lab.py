"""Voice anchor lab (RTX 4090): real timbre variety, measured not hoped.

VoxCPM2 ignores textual style descriptions (verified: it reads them aloud).
Without a reference it samples a RANDOM voice per run — so: generate many
candidates of the SAME neutral line, measure fundamental frequency (f0) and
loudness, pick three genuinely distinct timbres (low/mid/high pitch), peak-
normalize, and emit anchors + a report. Human ears make the final call.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parent / "voice_candidates"
ANCHOR_TEXT = "Okay, okay. It was me. Take me in. But write this down properly, detective."
N_CANDIDATES = 14


def f0_median(wav: np.ndarray, sr: int) -> float:
    import librosa
    f0 = librosa.yin(wav.astype(np.float32), fmin=60, fmax=400, sr=sr)
    voiced = f0[(f0 > 65) & (f0 < 390)]
    return float(np.median(voiced)) if voiced.size else 0.0


def main():
    from voxcpm import VoxCPM

    OUT.mkdir(exist_ok=True)
    tts = VoxCPM.from_pretrained("openbmb/VoxCPM2")
    sr = tts.tts_model.sample_rate
    print(f"sample rate: {sr}")

    rows = []
    for i in range(N_CANDIDATES):
        wav = np.asarray(tts.generate(text=ANCHOR_TEXT), dtype=np.float32)
        dur = len(wav) / sr
        if not 2.0 < dur < 12.0:
            print(f"cand {i:02d}: DEGENERATE ({dur:.1f}s) — skipped")
            continue
        peak = float(np.abs(wav).max()) or 1.0
        wav = wav * (0.70 / peak)  # uniform headroom: fixes the "suena super alto"
        pitch = f0_median(wav, sr)
        path = OUT / f"cand_{i:02d}.wav"
        sf.write(path, wav, sr)
        rows.append((i, pitch, dur, path))
        print(f"cand {i:02d}: f0={pitch:5.1f} Hz  dur={dur:4.1f}s")

    rows.sort(key=lambda r: r[1])
    picks = {"low": rows[0], "mid": rows[len(rows) // 2], "high": rows[-1]}
    print("\n=== SELECTED ANCHORS ===")
    for name, (i, pitch, dur, path) in picks.items():
        dest = OUT / f"anchor_{name}.wav"
        dest.write_bytes(path.read_bytes())
        print(f"{name:4s}: cand_{i:02d}  f0={pitch:.0f} Hz  dur={dur:.1f}s -> {dest.name}")
    print(f"\nspread: {picks['high'][1] - picks['low'][1]:.0f} Hz between low and high")
    print(f"listen: mpv {OUT}/anchor_low.wav etc. — Fernando decides.")


if __name__ == "__main__":
    main()
