---
title: EYEWITNESS
emoji: 👁️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - build-small-hackathon
  - thousand-token-wood
  - openbmb
  - minicpm
  - modal
  - tiny-titan
  - off-brand
  - off-the-grid
  - llama-champion
  - game
short_description: You saw the thief for 3s. Can your memory convict him?
---

# 👁️ EYEWITNESS

**You saw the thief for 3 seconds. Your memory is the only witness.**

A street camera catches a 3-second glimpse of a thief. You describe them from memory
to a police sketch artist — then you must pick the real culprit out of a lineup
**built from your own mistakes**. The worse your memory, the harder the lineup.

Four escalating ranks: shorter glimpses, bigger lineups, and from INSPECTOR up,
the culprit *changes their look* between the crime and the lineup ("he's been to
a barber since").

## How the AI is load-bearing

| Stage | Who does it |
|---|---|
| Spoken testimony → verbatim text (EN/ES) | **Cohere Transcribe 2B** — a court reporter, not an interpreter: it writes down what you SAID |
| Your messy testimony → strict attribute JSON | **[MiniCPM5-1B fine-tune](https://huggingface.co/Fcabla/MiniCPM5-1B-eyewitness)** (LoRA, slot-filling over a closed vocabulary — exactly what a 1B is great at) |
| The composite sketch, the lineup, the scoring | A deterministic engine. Ground truth is **authored, never detected** — the game always knows the culprit's exact face, so difficulty is honest and the scoring is transparent |
| The culprit's verdict taunt — a joke about the CRIME and his fate | **MiniCPM5-1B** (base; improvises live, best-of-6 + validators, with an authored per-crime floor) |
| The culprit's voice, cloned live | **VoxCPM2** (anchor voices cast by sex/age, selected by measured pitch) |
| Crime flavor + case bank | MiniCPM5-1B-GGUF batch-generated **via llama.cpp on Modal** |

**Runtime stack ≈ 5.7B parameters, all inside this Space. No cloud APIs.** Every
reveal shows *what you said → what the artist drew → the truth*, side by side —
fairness by transparency.

## Team

- HF: [`Fcabla`](https://huggingface.co/Fcabla)

## Links

- Demo video: https://youtu.be/BKOjBMKcb_E
- Social post: https://x.com/i/status/2065901377465819231
- Source code: [github.com/fcabla/eyewitness](https://github.com/fcabla/eyewitness)
- Fine-tuned parser: [Fcabla/MiniCPM5-1B-eyewitness](https://huggingface.co/Fcabla/MiniCPM5-1B-eyewitness)

*Built for the Build Small Hackathon (Thousand Token Wood track) with Claude Code
doing the engineering. Lineage: every eyewitness-memory study ever published ×
police procedural fantasy — the game is honest about how bad your memory is,
because we authored the ground truth.*