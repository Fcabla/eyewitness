"""EYEWITNESS factory on Modal — the build-time pipeline (Modal award + Llama Champion).

Three jobs, all OFF the user's interaction path:
  1. case-bank   : batch-generate crime flavor texts + case seeds with
                   MiniCPM5-1B-GGUF through llama.cpp (CPU is fine for 1B batch).
  2. voice-bank  : pre-render the culprit's verdict lines with VoxCPM2 (GPU).
  3. export      : write banks as JSON/wav into the repo's assets/ dir.

Run (after `modal token new`):
    modal run modal_factory.py::build_case_bank
    modal run modal_factory.py::build_voice_bank
"""
from __future__ import annotations

import json

import modal

app = modal.App("eyewitness-factory")

llama_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("llama-cpp-python", "huggingface_hub")
)

voice_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("voxcpm", "torch>=2.5", "huggingface_hub", "soundfile")
)

VERDICT_LINES = {
    "caught": [
        "Okay, okay. It was me. Take me in.",
        "The hat was a mistake. I admit it.",
        "Fine! FINE. But Greg the sourdough deserved freedom.",
    ],
    "escaped": [
        "Wrong guy. I walked RIGHT past you. Twice.",
        "Your sketch artist deserves a raise. You don't.",
        "I'd say see you around, but you clearly won't notice.",
    ],
}


@app.function(image=llama_image, timeout=1800, cpu=8)
def build_case_bank(n_cases: int = 48) -> list[dict]:
    """Batch-generate crime blurbs with MiniCPM5-1B GGUF via llama.cpp (Llama Champion)."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    gguf = hf_hub_download("openbmb/MiniCPM5-1B-GGUF",
                           filename="MiniCPM5-1B-Q4_K_M.gguf")
    llm = Llama(model_path=gguf, n_ctx=2048, verbose=False)

    prompt_head = (
        "Write ONE short, funny, family-friendly petty-crime headline and a one-sentence "
        "description for a comedy detective game. Format strictly as JSON: "
        '{"name": "the <Something> <Job/Heist/Affair/Caper>", "blurb": "<one sentence, third person>"}'
        "\nTheme hint: "
    )
    hints = ["food", "animals", "music", "transport", "sports", "art", "weather", "technology"]
    bank = []
    for i in range(n_cases):
        out = llm(prompt_head + hints[i % len(hints)] + "\nJSON:",
                  max_tokens=120, temperature=0.9, stop=["\n\n"])
        text = out["choices"][0]["text"]
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            item = json.loads(text[start:end])
            if {"name", "blurb"} <= set(item):
                bank.append({"name": item["name"][:60], "blurb": item["blurb"][:160], "seed": 1000 + i})
        except (ValueError, json.JSONDecodeError):
            continue
    print(f"case bank: {len(bank)}/{n_cases} valid")
    return bank


@app.function(image=voice_image, gpu="A10G", timeout=1800)
def build_voice_bank() -> dict[str, list[bytes]]:
    """Pre-render the culprit's verdict lines with VoxCPM2.

    Voice consistency trick: render an anchor line with the default voice once,
    then self-clone it (prompt_wav + its transcript) for every other line so the
    culprit keeps ONE voice across the whole bank."""
    import io

    import soundfile as sf
    from voxcpm import VoxCPM

    tts = VoxCPM.from_pretrained("openbmb/VoxCPM2")

    anchor_text = "Okay, okay. It was me. Take me in."
    anchor = tts.generate(text=anchor_text)
    anchor_path = "/tmp/anchor.wav"
    sf.write(anchor_path, anchor, 16000)

    def render(line: str):
        if line == anchor_text:
            return anchor
        return tts.generate(text=line, prompt_wav_path=anchor_path, prompt_text=anchor_text)

    rendered: dict[str, list[bytes]] = {"caught": [], "escaped": []}
    for kind, lines in VERDICT_LINES.items():
        for line in lines:
            wav = render(line)
            buf = io.BytesIO()
            sf.write(buf, wav, 16000, format="WAV")
            rendered[kind].append(buf.getvalue())
    return rendered


@app.local_entrypoint()
def voices_only():
    voices = build_voice_bank.remote()
    for kind, blobs in voices.items():
        for i, blob in enumerate(blobs):
            with open(f"assets/voice_{kind}_{i}.wav", "wb") as f:
                f.write(blob)
    print("voice bank complete -> assets/")


@app.local_entrypoint()
def main():
    bank = build_case_bank.remote(48)
    with open("assets/case_bank.json", "w") as f:
        json.dump(bank, f, indent=1)
    voices = build_voice_bank.remote()
    for kind, blobs in voices.items():
        for i, blob in enumerate(blobs):
            with open(f"assets/voice_{kind}_{i}.wav", "wb") as f:
                f.write(blob)
    print("factory complete -> assets/")
