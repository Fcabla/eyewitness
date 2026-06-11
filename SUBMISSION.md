# EYEWITNESS — Submission Pack

Deadline: **June 15, 2026** · Track: **Thousand Token Wood** · Space under `build-small-hackathon` org.

## Submission checklist

- [ ] Space deployed in org, golden path tested ON the Space (not just locally)
- [ ] README: Fernando's HF username filled in, video + social links added
- [ ] Demo video (≤60s, script below) uploaded
- [ ] Social post published (draft below)
- [ ] Field Notes blog post on the org (draft below)
- [ ] Modal factory run at least once (case bank + voice bank committed to assets/)
- [ ] Datasets pushed to org namespace (case bank → Sharing is Caring)

## Fernando's 10 minutes (the ONLY things I can't do)

1. `! hf auth login` → paste HF token (write scope).
2. Create the Space inside the org: `hf repo create build-small-hackathon/eyewitness --repo-type space --space-sdk gradio` (or via web UI).
3. `modal token new` (browser login) → then `modal run modal_factory.py`.
4. Fill `<TODO-FERNANDO-HF-USERNAME>` in README.md.
5. Record the demo video following the script (or give me screen recordings and I cut it).
6. Publish the social post.

## Demo video script (≤60 s)

| t | Shot | Audio/Caption |
|---|---|---|
| 0–4s | Black. Type-on text: "You saw the thief for 3 seconds." | Typewriter SFX |
| 4–8s | Screen: case intro "the Bakery Heist… sourdough starter, 'Greg'". Click ROLL THE FOOTAGE | — |
| 8–12s | The suspect face appears — timer bar drains — SIGNAL LOST static | Heartbeat |
| 12–20s | Typing testimony fast: "round face… bushy eyebrows… a cap? maybe a beanie…" | Caption: "describe him from MEMORY" |
| 20–28s | The sketch reveal — it looks confidently WRONG. Caption: "the artist draws what YOU said" | — |
| 28–38s | The lineup appears. Caption: "the lineup is built from YOUR mistakes". Cursor hovers, hesitates, clicks | Tense sting |
| 38–46s | WRONG ARREST stamp + culprit quote: "I walked RIGHT past you. Twice." + the said/truth table with red ✘s | — |
| 46–54s | Fast montage: rank-up cards (DETECTIVE → INSPECTOR "he's been to a barber since" → CHIEF 1s glimpse) | Music up |
| 54–60s | Final card: "EYEWITNESS · MiniCPM5-1B turns your words into evidence · 3.4B params · no cloud APIs" + Space URL | — |

## Social post draft

> I saw the thief for 3 seconds. I described him to the sketch artist. The artist drew
> my words EXACTLY. The sketch was a stranger. 👁️
>
> EYEWITNESS — a memory game for the @Gradio × @huggingface Build Small Hackathon
> where a 1B MiniCPM turns your testimony into evidence and the police lineup is
> built from your own mistakes. My memory scored 9%. Beat me:
>
> ▶️ [Space link]  #BuildSmallHackathon #MiniCPM

## Field Notes blog draft (org post)

**Title: "Your memory is the worst witness: building EYEWITNESS with a 1B model"**

Outline:
1. The psychology hook: eyewitness testimony is famously unreliable — we made that a game.
2. "Author the ground truth, don't detect it": the engine composes every face, so it always
   knows the truth — the model never has to *find* anything, only *translate* your messy
   words into a closed vocabulary. Honest 1B work.
3. The signature mechanic: lineups derived from YOUR errors (wrong beliefs get planted on
   distractors; what you never mentioned becomes the confusion axis).
4. Gradio 6 war story: column-visibility updates desync after demo.load — bisected with
   three minimal repros, rebuilt the UI on @gr.render. (Genuinely useful for other builders.)
5. The Modal factory: case bank via llama.cpp + GGUF, voice bank via VoxCPM2 — all build-time,
   zero GPU in the cursor path.
6. Scores from playtesting: humans average ~30% memory accuracy on ROOKIE. The pigeons remember more.

## Prize mapping (what this submission claims)

| Prize | Claim |
|---|---|
| Thousand Token Wood podium | The game itself: delight + honest small-model fit + polish |
| OpenBMB pool | MiniCPM5-1B core parser + GGUF case bank + VoxCPM2 voices; structured output everywhere |
| Modal | Build pipeline: case bank (llama.cpp) + voice bank (VoxCPM2) on Modal |
| Tiny Titan | ~3.4B runtime (1B parser + 2.3B voice, pre-rendered); engine is pure Python |
| Off-Brand | Case-file UI, zero default-Gradio look |
| Off the Grid | No cloud APIs at runtime |
| Llama Champion | MiniCPM5-1B-GGUF via llama.cpp on Modal |
| Field Notes | Org blog post (draft above) |
| Sharing is Caring | Case-bank dataset pushed to org |
| Best Demo | The 60s script above — the game records itself |
