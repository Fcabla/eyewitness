"""Live verdict voice: VoxCPM2 clones a designed anchor and speaks the 1B's line.

Anchor wavs (assets/anchor_*.wav, built on Modal) give each suspect a stable,
face-plausible voice; the case seed picks one deterministically. Every failure
path returns None — callers degrade to the pre-rendered bank, then to text.
"""
from __future__ import annotations

import threading
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# transcript of every anchor recording — cloning requires the exact prompt text.
# Anchors are pitch-selected on real renders (117/173/249 Hz median f0, see
# train/voice_lab.py): VoxCPM2 ignores textual style descriptions, so timbre
# variety comes from measured selection, not wishful prompting.
_ANCHOR_LINE = "Okay, okay. It was me. Take me in. But write this down properly, detective."
ANCHOR_TEXTS = {"low": _ANCHOR_LINE, "mid": _ANCHOR_LINE, "high": _ANCHOR_LINE}

try:
    import spaces
    _gpu = spaces.GPU(duration=30)  # warm render only; model pre-loaded on CPU
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


def preload():
    """Called at Space startup (CPU): pay the VoxCPM2 download/load up front."""
    if _available_anchors():
        _load()


def anchor_for_case(seed: int, culprit=None) -> str:
    """Voice casting: match timbre to visible attributes so a bearded suspect
    doesn't speak in a high register. Imperfect by design — the sketch is
    deliberately gender-neutral — but kills the jarring mismatches."""
    names = _available_anchors()
    if culprit is not None and "low" in names:
        if getattr(culprit, "facial_hair", "none") != "none":
            return "low"
        if getattr(culprit, "hair_style", "") in ("long", "ponytail") and "high" in names:
            return ("high", "mid")[seed % 2] if "mid" in names else "high"
    return names[seed % len(names)]


def anchor_for_seed(seed: int) -> str:  # backwards-compatible alias
    return anchor_for_case(seed)


@_gpu
def _render(line: str, anchor: str):
    tts = _load()
    wav = tts.generate(
        text=line,
        prompt_wav_path=str(ASSETS / f"anchor_{anchor}.wav"),
        prompt_text=ANCHOR_TEXTS[anchor],
    )
    return tts.tts_model.sample_rate, wav


def trim_to_speech(wav: "np.ndarray", sr: int, pad_ms: int = 140) -> "np.ndarray":
    """Cut leading non-speech (VoxCPM renders carry up to ~1.7s of it):
    onset = first sustained RMS above an adaptive threshold."""
    import numpy as np

    win = max(1, int(sr * 0.03))
    rms = np.sqrt(np.convolve(wav.astype(np.float32) ** 2,
                              np.ones(win) / win, mode="same"))
    thresh = max(0.04, 0.15 * float(rms.max()))
    onset = int(np.argmax(rms > thresh))
    if onset <= 0:
        return wav
    return wav[max(0, onset - int(sr * pad_ms / 1000)):]


def speak(line: str, seed: int, culprit=None) -> tuple[int, "np.ndarray"] | None:
    """(sample_rate, int16 samples) for gr.Audio, or None on any failure."""
    import numpy as np

    if not voice_enabled() or not line:
        return None
    try:
        sr, wav = _render(line, anchor_for_case(seed, culprit))
        wav = trim_to_speech(np.asarray(wav), sr)
        if wav.size < sr // 4 or wav.size > sr * 20:  # degenerate render guard
            print(f"[voice] degenerate render: {wav.size} samples", flush=True)
            return None
        if wav.dtype != np.int16:
            peak = float(np.abs(wav).max()) or 1.0
            wav = wav * (0.70 / peak)  # uniform headroom — live clones too
            wav = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
        print(f"[voice] ok: {wav.size / sr:.1f}s via anchor", flush=True)
        return int(sr), wav
    except Exception as e:  # visible in Space logs; caller falls back to bank
        print(f"[voice] render failed: {type(e).__name__}: {e}", flush=True)
        return None
