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

# the LoRA fine-tune only knows slot-filling (catastrophic forgetting — it
# babbles dataset phrasings on open prompts); creative lines use the base model
TAUNT_MODEL_ID = os.environ.get("EYEWITNESS_TAUNT_MODEL_ID", "openbmb/MiniCPM5-1B")

_lock = threading.Lock()
_models: dict[str, tuple] = {}


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


def _load(model_id: str = MODEL_ID):
    if model_id not in _models:
        with _lock:
            if model_id not in _models:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                )
                _models[model_id] = (mdl, tok)
    return _models[model_id]


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
    # json.dumps gives fully-escaped quoting; bare replace() left injectable quotes
    safe = json.dumps(testimony, ensure_ascii=False)[1:-1]
    prompt = PROMPT.format(schema=_schema_block(), testimony=safe)
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=220, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


TAUNT_PROMPT = """You are the culprit in a comedy detective game. Outcome: you were {outcome}.
The witness just described you to a sketch artist. Their wrong claims: {wrong}.
What they failed to notice: {missed}. What they got right: {right}.
Speak ONE smug in-character line (under 25 words) mocking their SPECIFIC mistakes.
Plain text only, no quotes, no emoji, no hashtags, no explanations, exactly one sentence.

Example (wrong claim "said my hat was beanie (it was fedora)"): A beanie? Detective, this fedora has more class than your entire memory.

Line:"""


@_gpu
def _generate_taunt(outcome: str, wrong: str, missed: str, right: str) -> str:
    model, tok = _load(TAUNT_MODEL_ID)
    prompt = TAUNT_PROMPT.format(outcome=outcome, wrong=wrong, missed=missed, right=right)
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.8,
                         top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def culprit_taunt(report_rows: list, correct: bool) -> str | None:
    """Personalized verdict line from the witness's actual mistakes. None on any
    failure or low-quality output — callers fall back to the canned quote."""
    wrong = [f"said my {label.lower()} was {said} (it was {truth})"
             for label, said, truth, v in report_rows if v == "miss"][:3]
    missed = [label.lower() for label, _s, _t, v in report_rows if v == "silent"][:3]
    right = [label.lower() for label, _s, _t, v in report_rows if v == "hit"][:2]
    try:
        raw = _generate_taunt(
            "caught red-handed" if correct else "wrongly let go — they arrested someone else",
            "; ".join(wrong) or "none",
            ", ".join(missed) or "nothing",
            ", ".join(right) or "nothing",
        ).strip().strip('"').split("\n")[0].strip()
    except Exception:
        return None
    # quality gate: reject JSON leakage (slot-filling fine-tune habit) and degenerate output
    if not raw or raw.startswith("{") or len(raw) < 15 or len(raw) > 220:
        return None
    return raw


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
