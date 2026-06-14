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
def transcribe(audio: "str | tuple[int, np.ndarray] | None", language: str = "en") -> str:
    """gr.Audio filepath (or legacy (sr, array) tuple) -> witness text in the
    selected language ('en'/'es'). Empty string on any failure, and a silent clip
    returns "" rather than a hallucination — never fabricated text."""
    import numpy as np

    if audio is None:
        return ""
    # Read as float [-1, 1]. soundfile handles wav/flac/ogg; librosa (ffmpeg) covers
    # m4a/mp3/opus uploads. (gr.Audio has format="wav", so recordings arrive as wav.)
    # A legacy (sample_rate, numpy) tuple is still accepted.
    if isinstance(audio, str):
        try:
            import soundfile as sf
            wav, sr = sf.read(audio, dtype="float32", always_2d=False)
        except Exception as e_sf:  # libsndfile can't do m4a/aac/mp3/opus -> ffmpeg via librosa
            import librosa
            wav, sr = librosa.load(audio, sr=None, mono=True)
            wav = np.asarray(wav, dtype=np.float32)
            print(f"[asr] soundfile failed ({type(e_sf).__name__}: {e_sf}); "
                  f"used librosa/ffmpeg", flush=True)
    else:
        sr, wav = audio
        wav = np.asarray(wav, dtype=np.float32)
        if np.abs(wav).max() > 1.5:  # legacy int16-ranged numpy input
            wav = wav / 32768.0
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # Silence guard: a silent/near-silent clip must NOT be fed to the model — it
    # answers confident hallucinations. Return "" instead.
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if len(wav) < sr // 2 or peak < 1e-3:
        print(f"[asr] no usable audio (samples={len(wav)}, peak={peak:.5f}) -> empty", flush=True)
        return ""
    wav = (wav / peak) * 0.9  # level-match the model's expected input (demo peaks ~0.9)
    lang = language if language in ("en", "es") else "en"  # explicit UI selector; no guessing
    try:
        import torch
        proc, mdl = _load()
        if sr != 16000:  # fast polyphase resample (librosa's default was ~13s on mic clips)
            try:
                from scipy.signal import resample_poly
                wav = resample_poly(wav, 16000, sr).astype(np.float32)
            except Exception:
                try:
                    import librosa
                    wav = librosa.resample(np.ascontiguousarray(wav), orig_sr=sr, target_sr=16000)
                except Exception:
                    idx = np.linspace(0, len(wav) - 1, int(len(wav) * 16000 / sr))
                    wav = wav[np.floor(idx).astype(int)]
        # `language` is a REQUIRED processor arg: it builds the decoder prompt.
        inputs = proc(wav, language=lang, sampling_rate=16000, return_tensors="pt")
        inputs = {k: (v.to(mdl.device, dtype=mdl.dtype) if v.dtype.is_floating_point
                      else v.to(mdl.device)) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.no_grad():
            ids = mdl.generate(**inputs, max_new_tokens=120)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        print(f"[asr] {lang}: {text[:90]!r}", flush=True)
        return text
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
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=COHERE_ASR_ID,
                               filename="demo/voxpopuli_test_en_demo.wav")
        text = transcribe(path, "en")  # exercise the exact filepath path the mic uses
        print(f"[asr-selftest] OK: {text!r}", flush=True)
    except Exception as e:
        import traceback
        print(f"[asr-selftest] FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()}",
              flush=True)
