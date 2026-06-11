"""Tier B testimony parser: MiniCPM5-1B slot-filling (the AI-load-bearing core).

The 1B model translates MESSY natural witness language ("kind of a roundish
face, caterpillar eyebrows, looked like he hadn't slept") into the strict
attribute JSON the engine needs. This is exactly what small models are good
at: constrained structured extraction over a closed vocabulary.

Runs on ZeroGPU when deployed (@spaces.GPU); CPU locally for dev (a 1B parse
is seconds-cheap). Every output is validated against VOCAB; invalid or missing
slots fall back to the deterministic Tier A matcher — the game NEVER breaks.
"""
from __future__ import annotations

import json
import re
import threading

from .face import VOCAB
from .parser import parse_testimony as _tier_a

import os

# Overridable so the Space can point at the published fine-tune.
MODEL_ID = os.environ.get("EYEWITNESS_MODEL_ID", "openbmb/MiniCPM5-1B")

try:  # ZeroGPU decorator when running in a HF Space
    import spaces
    _gpu = spaces.GPU(duration=15)
except Exception:  # local dev
    def _gpu(fn):
        return fn

_lock = threading.Lock()
_model = None
_tokenizer = None


def model_enabled() -> bool:
    """Tier B only where it's fast: a ZeroGPU Space, real CUDA, or override.
    On CPU a 1B parse takes ~30-50s — Tier A handles local dev instantly."""
    import os
    if os.environ.get("EYEWITNESS_FORCE_MODEL") == "1":
        return True
    if os.environ.get("SPACES_ZERO_GPU"):  # ZeroGPU Space (CUDA is lazy there)
        try:
            import spaces  # noqa: F401
            return True
        except Exception:
            pass
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _load():
    global _model, _tokenizer
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                )
    return _model, _tokenizer


def _schema_block() -> str:
    return "\n".join(f'  "{attr}": one of {opts} or null' for attr, opts in VOCAB.items())


PROMPT = """You are a police sketch-artist assistant. A witness describes a suspect.
Extract ONLY what the witness actually said into this exact JSON schema (null when not mentioned):
{{
{schema}
}}
Rules: output ONLY the JSON object. Use null for anything the witness did not mention.
Map loose language to the closest allowed value, even when words are split apart
(e.g. "caterpillar eyebrows"->brows "bushy"; "hadn't slept"->eyes "droopy";
"gorro de lana"->hat "beanie"; "slicked his hair back"->hair_style "slick_back";
"gafas de esas de abuelo, redondas"->glasses "round"). English or Spanish.

Example testimony: "uno con sombrero, la nariz enorme, y el pelo todo engominado hacia atras"
Example JSON: {{"face_shape": null, "skin": null, "hair_style": "slick_back", "hair_color": null, "brows": null, "eyes": null, "glasses": null, "nose": "big", "mouth": null, "facial_hair": null, "hat": "fedora", "extra": null}}

Witness testimony: "{testimony}"

JSON:"""


@_gpu
def _generate(testimony: str) -> str:
    model, tok = _load()
    prompt = PROMPT.format(schema=_schema_block(), testimony=testimony.replace('"', "'"))
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=220, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_testimony_model(testimony: str) -> dict[str, str | None]:
    """Model first, Tier A as per-slot backstop. Validated against VOCAB."""
    base = _tier_a(testimony)
    if not testimony.strip():
        return base
    raw = _generate(testimony)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return base
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return base
    out: dict[str, str | None] = {}
    for attr in VOCAB:
        v = data.get(attr)
        if isinstance(v, str):
            v = v.strip().lower().replace(" ", "_")
        # Tier A is literal-match (high precision): where it spoke, it wins.
        # The model fills the silence (messy/indirect phrasings) — this also
        # caps model hallucinations to attributes Tier A had no opinion on.
        out[attr] = base.get(attr) or (v if v in VOCAB[attr] else None)
    return out
