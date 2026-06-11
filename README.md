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
short_description: You saw the thief for 3 seconds. Your memory is the only witness.
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
| Your messy testimony → strict attribute JSON | **MiniCPM5-1B** (slot-filling over a closed vocabulary — exactly what a 1B is great at) |
| The composite sketch, the lineup, the scoring | A deterministic engine. Ground truth is **authored, never detected** — the game always knows the culprit's exact face, so difficulty is honest and the scoring is transparent |
| Crime flavor + case bank | MiniCPM5-1B-GGUF batch-generated **via llama.cpp on Modal** |
| The culprit's voice | **VoxCPM2** (pre-rendered lines — zero GPU in the cursor path) |

**Runtime stack ≈ 3.4B parameters. No cloud APIs at runtime.** Every reveal shows
*what you said → what the artist drew → the truth*, side by side — fairness by
transparency.

## Team

- HF: `<TODO-FERNANDO-HF-USERNAME>`

## Links

- Demo video: `<TODO>`
- Social post: `<TODO>`
- Source: this repo (`app.py`, `game/`, `modal_factory.py`)

*Built for the Build Small Hackathon (Thousand Token Wood track) with Claude Code
doing the engineering. Lineage: every eyewitness-memory study ever published ×
police procedural fantasy — the game is honest about how bad your memory is,
because we authored the ground truth.*
