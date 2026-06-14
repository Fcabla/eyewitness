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
Example JSON: {{"sex": "male", "age": null, "face_shape": null, "skin": null, "hair_style": "slick_back", "hair_color": null, "brows": null, "eyes": null, "glasses": null, "nose": "big", "mouth": null, "facial_hair": null, "hat": "fedora", "extra": null}}

Witness testimony: "{testimony}"

JSON:"""

# The fine-tune was trained on THIS exact chat shape (train_modal.py) — feeding
# it the few-shot PROMPT above poisons it: it parrots the example's values
# (fedora, bushy, big nose) into every parse. Schema-in-prompt is only for base.
FINETUNE_SYSTEM = ("You are a police sketch-artist assistant. Extract ONLY what the witness "
                   "said into the attribute JSON. Use null for anything not mentioned. "
                   "Output only the JSON object.")


@_gpu
def _generate(testimony: str) -> str:
    model, tok = _load()
    # json.dumps gives fully-escaped quoting; bare replace() left injectable quotes
    safe = json.dumps(testimony, ensure_ascii=False)[1:-1]
    if MODEL_ID != "openbmb/MiniCPM5-1B":  # fine-tune: match its training shape
        messages = [{"role": "system", "content": FINETUNE_SYSTEM},
                    {"role": "user", "content": f'Witness testimony: "{safe}"'}]
    else:
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


TAUNT_PROMPT = """Roleplay: you are a petty criminal with theatrical flair, just {outcome}. Your crime — {crime_name}: you {blurb}.
Say ONE short, funny line (max 20 words), first person, about the crime or your situation right now. Wit over information. No explanations, no quotes.

Examples of the ENERGY (different crimes — steal the attitude, never the words):
- The watch kept perfect time for 300 years. With me it finally had somewhere to BE.
- You can't arrest a man for finishing a chorus. Artistically, I was contractually obligated.
- Some poor soul is signing MY paperwork right now. Give him my coffee order.

Your line:"""

_BANNED = ("witness", "roleplay", "example", "scenario", "one-liner",
           "i'm sorry", "as an ai", "i am an ai", "language model", "assistant",
           "the criminal", "the culprit", "the suspect")


@_gpu
def _generate_taunt_batch(crime_name: str, blurb: str, outcome: str, n: int = 6) -> list[str]:
    """One GPU call, n sampled candidates (best-of-N beats a 1B's low hit rate)."""
    model, tok = _load(TAUNT_MODEL_ID)
    prompt = TAUNT_PROMPT.format(crime_name=crime_name, blurb=blurb, outcome=outcome)
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # chat template without thinking support
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=45, do_sample=True, temperature=0.85,
                         top_p=0.92, num_return_sequences=n,
                         pad_token_id=tok.eos_token_id)
    return [tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            .strip().strip('"').split("\n")[0].strip() for seq in out]


def _has_attitude(line: str) -> bool:
    """Flat lines lose to the authored floor; only lines with comic bite
    (incredulity, emphasis) earn the spotlight."""
    return ("?" in line or "!" in line
            or any(w.isupper() and len(w) >= 3 for w in line.split()))


_KW_STOP = {"with", "from", "that", "this", "they", "them", "were", "just", "made",
            "over", "into", "walked", "year", "years", "which", "belonged", "entire",
            "caught", "training", "stole", "stolen", "steal", "boarded", "returned",
            "rolled", "night", "city", "out", "off", "the", "and", "his", "her"}


def _crime_keywords(crime_name: str, blurb: str) -> list[str]:
    """The concrete nouns of THIS crime (watch, gnomes, manchego...) — a valid
    taunt must name at least one, so it's about this case, not generic mush."""
    text = f"{crime_name} {blurb}".lower()
    return [w for w in re.findall(r"[a-z]{4,}", text) if w not in _KW_STOP]


def _valid_situational(line: str, keywords: list[str]) -> bool:
    """About THIS crime, first-person, has comic bite, clean, right length."""
    words = line.split()
    if not (5 <= len(words) <= 24):
        return False
    low = line.lower()
    if any(b in low for b in _BANNED):
        return False
    if not any(k in low for k in keywords):  # must be about this crime
        return False
    first_person = ("i" in low.split() or "i'm" in low
                    or "my" in low.split() or low.startswith(("the ", "a ")))
    return first_person and _has_attitude(line)


def culprit_taunt(crime_name: str, crime_blurb: str, correct: bool, seed: int = 0,
                  use_model: bool = True) -> tuple[str, str]:
    """Verdict line -> (line, source) where source is 'model' or 'template'.
    The culprit jokes about the CRIME and the situation (caught/escaped), never
    the witness's appearance errors. An authored pool per crime is the floor;
    the live base 1B replaces it only when a candidate is about this crime AND
    has comic attitude (taunt_lab: free-form 1B comedy lands ~3% of the time)."""
    import random as _r
    from .casegen import CRIME_TAUNTS, GENERIC_TAUNTS

    rng = _r.Random(seed)
    outcome = "caught" if correct else "escaped"
    authored = rng.choice(CRIME_TAUNTS.get(crime_name, GENERIC_TAUNTS)[outcome])
    if use_model:
        try:
            cands = _generate_taunt_batch(
                crime_name, crime_blurb,
                "caught red-handed" if correct else "let go — they pinned it on an innocent man")
            kw = _crime_keywords(crime_name, crime_blurb)
            good = [c for c in cands if _valid_situational(c, kw)]
            if good:
                best = max(good, key=lambda c: ("?" in c or "!" in c, len(c)))
                print(f"[taunt] model {len(good)}/{len(cands)}: {best[:110]}", flush=True)
                return best, "model"
            print(f"[taunt] 0/{len(cands)} usable -> authored", flush=True)
        except Exception as e:  # visible in Space logs
            print(f"[taunt] generation failed -> authored: {type(e).__name__}: {e}", flush=True)
    return authored, "template"


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
