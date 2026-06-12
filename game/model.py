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
    # request small: ZeroGPU ADMITS calls by requested duration vs the visitor's
    # remaining quota — oversized requests get rejected outright (seen in logs).
    # Models are pre-loaded on CPU at startup, so calls only pay transfer+generate.
    _gpu = spaces.GPU(duration=20)
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
    """ZeroGPU-canonical: weights live on CPU (loaded once per process); the
    @spaces.GPU call moves them to CUDA. Keeps GPU calls short and admissible."""
    if model_id not in _models:
        with _lock:
            if model_id not in _models:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True, torch_dtype=torch.float16)
                _models[model_id] = (mdl, tok)
    mdl, tok = _models[model_id]
    import torch
    if torch.cuda.is_available() and mdl.device.type != "cuda":
        mdl.to("cuda")
    return mdl, tok


def preload():
    """Called at Space startup (CPU): pay downloads/loads before any user."""
    _load(MODEL_ID)
    if TAUNT_MODEL_ID != MODEL_ID:
        _load(TAUNT_MODEL_ID)


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
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # chat template without thinking support
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=220, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


TAUNT_PROMPT = """Roleplay: you are a smug petty criminal taunting the detective who questioned a witness about you. You were {outcome}.
The witness's mistakes about your appearance:
{wrong_lines}
Mock the detective in ONE short sentence (max 22 words), first person, naming one specific mistake. Gloat — never describe yourself neutrally.

Examples of your style:
- A BEANIE? I wear a fedora, sweetheart. Ask the mirror how your memory feels.
- Curly hair? I spend twenty minutes slicking it back and THIS is my reward?

Your line:"""

_BANNED = ("witness", "detective game", "criminal's", "roleplay", "example",
           "scenario", "one-liner", "the suspect", "i am a smug")


@_gpu
def _generate_taunt_batch(outcome: str, wrong_lines: str, n: int = 5) -> list[str]:
    """One GPU call, n sampled candidates (best-of-N beats a 1B's ~20% hit rate)."""
    model, tok = _load(TAUNT_MODEL_ID)
    prompt = TAUNT_PROMPT.format(outcome=outcome, wrong_lines=wrong_lines)
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # chat template without thinking support
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=55, do_sample=True, temperature=0.75,
                         top_p=0.9, num_return_sequences=n,
                         pad_token_id=tok.eos_token_id)
    return [tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            .strip().strip('"').split("\n")[0].strip() for seq in out]


def _valid_taunt(line: str, anchors: list[str]) -> bool:
    """Must reference a real detail of THIS case, read first-person, stay clean."""
    words = line.split()
    if not (6 <= len(words) <= 28):
        return False
    low = line.lower()
    if any(b in low for b in _BANNED):
        return False
    if not any(a in low for a in anchors):  # must cite a concrete case detail
        return False
    # role confusion: case attributes claimed of the DETECTIVE ("your beard...",
    # "you're wearing curly hair", "I said your glasses") — the 1B's main failure
    for a in anchors:
        if f"your {a}" in low or f"you're wearing {a}" in low or f"you are wearing {a}" in low:
            return False
    if "i said your" in low or "you said my" in low and "you said my" != low[:11]:
        pass  # "you said my X" is FINE (culprit quoting the detective's claim)
    return ("i" in low.split() or "i'm" in low or "my" in low.split() or low.startswith(("a ", "an ")))


def _template_taunt(wrongs: list, misseds: list, correct: bool, seed: int) -> str:
    """Deterministic personalized fallback — even the worst case stays dynamic."""
    import random as _r
    rng = _r.Random(seed)
    if wrongs:
        label, said, truth = rng.choice(wrongs)
        if correct:
            line = rng.choice([
                f"You got me — but '{said}'? It's a {truth}, detective. It was ALWAYS a {truth}.",
                f"Fine, cuff me. But tell the sketch artist my {label.lower()} is a {truth}, not '{said}'.",
            ])
        else:
            line = rng.choice([
                f"'{said}'? It was a {truth}. The wrong man you arrested sends his regards.",
                f"They wrote down '{said}'. A {truth}, people. I walked free on YOUR {label.lower()} mistake.",
            ])
    else:
        line = ("Lucky badge, goldfish memory. You couldn't name ONE thing about me." if correct
                else "You remembered NOTHING about me. Honestly? I'm almost offended.")
    if misseds and rng.random() < 0.6:
        line += f" And you never even noticed my {rng.choice(misseds).lower()}."
    return line


def _claims_wrong_value(line: str, wrongs: list) -> bool:
    """Reject lines asserting the WITNESS'S wrong value as the culprit's truth
    (e.g. truth=goatee but line says 'I have a full beard')."""
    low = line.lower()
    for _label, said, _truth in wrongs:
        if re.search(rf"i\s+(?:have|wear|am wearing|'m wearing|got)\b[^.;!?]*\b{re.escape(said.lower())}", low):
            return True
    return False


def culprit_taunt(report_rows: list, correct: bool, seed: int = 0) -> str:
    """Personalized verdict line. Pipeline: best-of-5 from the base 1B, strict
    validation, deterministic personalized template as the floor. ALWAYS returns
    a line that references this round's actual mistakes — never canned."""
    wrongs = [(label, said, truth) for label, said, truth, v in report_rows if v == "miss"][:3]
    misseds = [label for label, _s, _t, v in report_rows if v == "silent"][:3]
    wrong_lines = "\n".join(f"- they said your {l.lower()} was {s}; it is actually {t}"
                            for l, s, t in wrongs) \
        or "- they remembered nothing specific about you at all"
    anchors = [w.lower() for _l, s, t in wrongs for w in (s.split() + t.split())] \
        or [m.lower() for m in misseds] or ["nothing", "remember"]
    try:
        cands = _generate_taunt_batch(
            "caught red-handed" if correct else "wrongly let go (they arrested an innocent man)",
            wrong_lines)
        good = [c for c in cands
                if _valid_taunt(c, anchors) and not _claims_wrong_value(c, wrongs)]
        if good:
            best = max(good, key=len)
            print(f"[taunt] model {len(good)}/{len(cands)} valid: {best[:110]}", flush=True)
            return best
        print(f"[taunt] 0/{len(cands)} valid -> template", flush=True)
    except Exception as e:  # visible in Space logs
        print(f"[taunt] generation failed -> template: {type(e).__name__}: {e}", flush=True)
    return _template_taunt(wrongs, misseds, correct, seed)


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
