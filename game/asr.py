"""Spoken testimony: witness talks into the mic, ASR fills the statement.

Backend: Cohere Transcribe 2B, exclusively (sponsor model; verbatim by design —
transcribes what was SAID, not what it thinks you meant; EN + ES native).
The witness picks the language in the UI; the decoder prompt is language-specific,
so we decode in that language and only retry the other one if the first is empty.

Gated model: the Space authenticates via the HF_TOKEN secret. On any failure the
function returns an empty string and logs the exact error — it NEVER fabricates text.
"""
from __future__ import annotations

import threading

COHERE_ASR_ID = "CohereLabs/cohere-transcribe-03-2026"

try:
    import spaces
    _gpu = spaces.GPU(duration=30)  # cold 2B host->device transfer + short generate
except Exception:
    def _gpu(fn):
        return fn

_lock = threading.Lock()
_backend = None  # (processor, model)


def _load():
    """ZeroGPU-canonical (mirrors game/model.py): weights load once on CPU; the
    @spaces.GPU call moves them to CUDA. Keeps GPU calls short and admissible."""
    global _backend
    if _backend is None:
        with _lock:
            if _backend is None:
                import torch
                from transformers import AutoProcessor, CohereAsrForConditionalGeneration
                proc = AutoProcessor.from_pretrained(COHERE_ASR_ID)
                mdl = CohereAsrForConditionalGeneration.from_pretrained(
                    COHERE_ASR_ID, torch_dtype=torch.float16)
                _backend = (proc, mdl)
                print("[asr] backend: cohere-transcribe", flush=True)
    proc, mdl = _backend
    import torch
    if torch.cuda.is_available() and mdl.device.type != "cuda":
        mdl.to("cuda")
    return proc, mdl


def preload():
    _load()
    _selftest()


@_gpu
def transcribe(audio: tuple[int, "np.ndarray"] | None, language: str = "en") -> str:
    """gr.Audio mic tuple -> witness text in the chosen language ('en'/'es').
    Decodes in the witness's language; only retries the other language if that
    pass is empty. Empty string on any failure (never fabricated text)."""
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
    primary = language if language in ("en", "es") else "en"
    order = [primary, "es" if primary == "en" else "en"]  # other lang as fallback
    try:
        import torch
        proc, mdl = _load()
        if sr != 16000:
            try:  # proper anti-aliased resample; never hard-fail on a missing dep
                import librosa
                wav = librosa.resample(np.ascontiguousarray(wav), orig_sr=sr, target_sr=16000)
            except Exception:
                idx = np.linspace(0, len(wav) - 1, int(len(wav) * 16000 / sr))
                wav = wav[np.floor(idx).astype(int)]
        for lang in order:
            # `language` is a REQUIRED processor arg: it builds the decoder prompt.
            inputs = proc(wav, language=lang, sampling_rate=16000, return_tensors="pt")
            inputs = {k: (v.to(mdl.device, dtype=mdl.dtype) if v.dtype.is_floating_point
                          else v.to(mdl.device)) for k, v in inputs.items()
                      if hasattr(v, "to")}
            with torch.no_grad():
                ids = mdl.generate(**inputs, max_new_tokens=120)
            text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if text:  # chosen language won; the other is only tried if this is empty
                print(f"[asr] ok ({lang}): {text[:90]}", flush=True)
                return text
            print(f"[asr] empty ({lang}), trying fallback", flush=True)
        print("[asr] empty in both languages", flush=True)
        return ""
    except Exception as e:
        print(f"[asr] failed: {type(e).__name__}: {e}", flush=True)
        return ""


def _selftest():
    """Deploy-time smoke test (set ASR_SELFTEST=1): transcribe the model's own
    demo clip so the Space logs prove Cohere works end-to-end, no mic needed."""
    import os
    if os.getenv("ASR_SELFTEST") != "1":
        return
    try:
        import numpy as np
        import soundfile as sf
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=COHERE_ASR_ID,
                               filename="demo/voxpopuli_test_en_demo.wav")
        wav, sr = sf.read(path, dtype="float32")
        text = transcribe((int(sr), np.asarray(wav)), "en")
        print(f"[asr-selftest] OK: {text!r}", flush=True)
    except Exception as e:
        import traceback
        print(f"[asr-selftest] FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()}",
              flush=True)
