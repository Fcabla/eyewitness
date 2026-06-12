"""Spoken testimony: witness talks into the mic, ASR fills the statement.

Primary backend: Cohere Transcribe 2B (sponsor model; verbatim by design —
transcribes what was SAID, not what it thinks you meant; EN + ES native).
Fallback backend: faster-whisper small (already cached locally) so the feature
works even while the primary downloads or if it fails to load.
"""
from __future__ import annotations

import threading

COHERE_ASR_ID = "CohereLabs/cohere-transcribe-03-2026"

try:
    import spaces
    _gpu = spaces.GPU(duration=20)
except Exception:
    def _gpu(fn):
        return fn

_lock = threading.Lock()
_backend = None  # ("cohere", processor, model) | ("whisper", model)


def _load():
    global _backend
    if _backend is None:
        with _lock:
            if _backend is None:
                try:
                    import torch
                    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                    proc = AutoProcessor.from_pretrained(COHERE_ASR_ID)
                    mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
                        COHERE_ASR_ID, torch_dtype=torch.float16)
                    _backend = ("cohere", proc, mdl)
                    print("[asr] backend: cohere-transcribe", flush=True)
                except Exception as e:
                    print(f"[asr] cohere unavailable ({type(e).__name__}) -> whisper", flush=True)
                    from faster_whisper import WhisperModel
                    _backend = ("whisper", WhisperModel("small", device="auto",
                                                        compute_type="int8"))
    return _backend


def preload():
    _load()


@_gpu
def transcribe(audio: tuple[int, "np.ndarray"] | None) -> str:
    """gr.Audio mic tuple -> witness text. Empty string on any failure."""
    import numpy as np

    if audio is None:
        return ""
    sr, wav = audio
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if np.abs(wav).max() > 1.5:  # int16-ranged input
        wav = wav / 32768.0
    if len(wav) < sr // 2:
        return ""
    try:
        backend = _load()
        if backend[0] == "cohere":
            import torch
            _, proc, mdl = backend
            if sr != 16000:
                import math
                idx = np.linspace(0, len(wav) - 1, int(len(wav) * 16000 / sr))
                wav = wav[np.floor(idx).astype(int)]
            inputs = proc(wav, sampling_rate=16000, return_tensors="pt")
            inputs = {k: (v.to(mdl.device, dtype=mdl.dtype) if v.dtype.is_floating_point
                          else v.to(mdl.device)) for k, v in inputs.items()
                      if hasattr(v, "to")}
            with torch.no_grad():
                ids = mdl.generate(**inputs, max_new_tokens=120)
            text = proc.batch_decode(ids, skip_special_tokens=True)[0]
        else:
            import soundfile as sf
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wav, sr)
                segments, _info = backend[1].transcribe(f.name)
                text = " ".join(seg.text for seg in segments)
        text = text.strip()
        print(f"[asr] ok: {text[:90]}", flush=True)
        return text
    except Exception as e:
        print(f"[asr] failed: {type(e).__name__}: {e}", flush=True)
        return ""
