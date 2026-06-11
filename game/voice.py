"""Live verdict voice: VoxCPM2 clones a designed anchor and speaks the 1B's line.

Anchor wavs (assets/anchor_*.wav, built on Modal) give each suspect a stable,
face-plausible voice; the case seed picks one deterministically. Every failure
path returns None — callers degrade to the pre-rendered bank, then to text.
"""
from __future__ import annotations

import threading
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# transcript of every anchor recording — cloning requires the exact prompt text
ANCHOR_TEXTS = {
    "gravel": "Deep, gravelly male voice, slow and self-satisfied. Okay, okay. It was me. Take me in.",
    "sharp": "Sharp, fast female voice, mocking and theatrical. Okay, okay. It was me. Take me in.",
    "nasal": "Thin, nasal male voice, whiny and indignant. Okay, okay. It was me. Take me in.",
}

try:
    import spaces
    _gpu = spaces.GPU(duration=90)  # VoxCPM2 cold load is the long pole
except Exception:
    def _gpu(fn):
        return fn

_lock = threading.Lock()
_tts = None


def _available_anchors() -> list[str]:
    return sorted(n for n in ANCHOR_TEXTS if (ASSETS / f"anchor_{n}.wav").exists())


def voice_enabled() -> bool:
    import os
    if not _available_anchors():
        return False
    if os.environ.get("EYEWITNESS_FORCE_VOICE") == "1":
        return True
    if os.environ.get("SPACES_ZERO_GPU"):
        try:
            import spaces  # noqa: F401
            return True
        except Exception:
            return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _load():
    global _tts
    if _tts is None:
        with _lock:
            if _tts is None:
                from voxcpm import VoxCPM
                _tts = VoxCPM.from_pretrained("openbmb/VoxCPM2")
    return _tts


def anchor_for_seed(seed: int) -> str:
    names = _available_anchors()
    return names[seed % len(names)]


@_gpu
def _render(line: str, anchor: str):
    tts = _load()
    wav = tts.generate(
        text=line,
        prompt_wav_path=str(ASSETS / f"anchor_{anchor}.wav"),
        prompt_text=ANCHOR_TEXTS[anchor],
    )
    return tts.tts_model.sample_rate, wav


def speak(line: str, seed: int) -> tuple[int, "np.ndarray"] | None:
    """(sample_rate, int16 samples) for gr.Audio, or None on any failure."""
    import numpy as np

    if not voice_enabled() or not line:
        return None
    try:
        sr, wav = _render(line, anchor_for_seed(seed))
        wav = np.asarray(wav)
        if wav.size < sr // 4 or wav.size > sr * 20:  # degenerate render guard
            print(f"[voice] degenerate render: {wav.size} samples", flush=True)
            return None
        if wav.dtype != np.int16:
            wav = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
        print(f"[voice] ok: {wav.size / sr:.1f}s via anchor", flush=True)
        return int(sr), wav
    except Exception as e:  # visible in Space logs; caller falls back to bank
        print(f"[voice] render failed: {type(e).__name__}: {e}", flush=True)
        return None
