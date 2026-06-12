"""EYEWITNESS — you saw the thief for 3 seconds. Your memory is the only witness.

Game loop: case intro -> timed glimpse -> testimony -> sketch reveal -> lineup
(built from YOUR errors) -> verdict. Escalating ranks shorten the glimpse and
grow the lineup; from INSPECTOR the culprit changes one feature before the
lineup ("he's been to a barber since").

UI architecture: a single state-driven @gr.render stage. Column-visibility
toggling desyncs in Gradio 6 once demo.load touches it (see FIELD_NOTES.md).
"""
from __future__ import annotations

import base64
import os
import random

import gradio as gr

from pathlib import Path

from game.casegen import make_case, Case, RANKS
from game.face import render_face_svg, FaceSpec
from game.lineup import build_lineup
from game.parser import parse_testimony
from game.poster import make_wanted_poster
from game.scoring import grade_testimony, detective_rating

ASSETS = Path(__file__).resolve().parent / "assets"


def verdict_voice(correct: bool) -> tuple[int, "np.ndarray"] | None:
    """Pre-rendered VoxCPM2 line (Modal voice bank) if present — zero live GPU.

    Returned in-memory as (sample_rate, samples): gr.Audio file paths outside
    Gradio's allowed dirs raise InvalidPathError and kill the whole render."""
    import random as _r
    import wave

    import numpy as np

    kind = "caught" if correct else "escaped"
    files = sorted(ASSETS.glob(f"voice_{kind}_*.wav")) if ASSETS.exists() else []
    if not files:
        return None
    with wave.open(str(_r.choice(files))) as w:
        frames = w.readframes(w.getnframes())
        return w.getframerate(), np.frombuffer(frames, dtype=np.int16)

try:  # Tier B (deployed): MiniCPM5-1B slot-filler. Falls back to Tier A locally.
    from game.model import parse_testimony_model, model_enabled, culprit_taunt
    HAS_MODEL = model_enabled()
except Exception:
    HAS_MODEL = False

try:  # live VoxCPM2 verdict voice (anchored per suspect); bank is the fallback
    from game.voice import speak as live_speak
except Exception:
    def live_speak(line, seed, culprit=None):
        return None

try:  # spoken testimony (Cohere Transcribe primary, whisper fallback)
    from game.asr import transcribe as asr_transcribe
    HAS_ASR = True
except Exception:
    HAS_ASR = False


def transcribe_testimony(s: dict, audio, current_text: str):
    text = asr_transcribe(audio)
    if not text:
        return s, current_text
    merged = (current_text.strip() + " " + text).strip() if current_text.strip() else text
    return s, merged


# ------------------------------------------------------------------ helpers
def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


from game.render import face_image as face_png  # gr.Gallery needs PIL, not data-URIs


def glimpse_html(case: Case) -> str:
    """Timed reveal in pure CSS (gr.HTML strips <script>): the face blurs out and
    the SIGNAL LOST static fades in after exactly N seconds via animation-delay."""
    s = case.glimpse_seconds
    face = svg_uri(render_face_svg(case.culprit, width=320))
    return f"""
<div class="ew-glimpse">
  <img src="{face}" alt="suspect" style="animation: ew-blurout 0.25s linear {s}s forwards"/>
  <div class="ew-static" style="animation: ew-appear 0.2s linear {s}s forwards"><span>SIGNAL LOST</span></div>
  <div class="ew-timerbar"><div class="ew-timerfill" style="animation-duration:{s}s"></div></div>
</div>"""


def report_table_html(report) -> str:
    rows = ""
    icon = {"hit": "✔", "miss": "✘", "silent": "·"}
    cls = {"hit": "ew-hit", "miss": "ew-miss", "silent": "ew-silent"}
    for label, said, truth, verdict in report.rows:
        rows += (f'<tr class="{cls[verdict]}"><td>{icon[verdict]}</td><td>{label}</td>'
                 f'<td>{said}</td><td>{truth}</td></tr>')
    return f"""
<table class="ew-report">
  <thead><tr><th></th><th></th><th>YOU SAID</th><th>THE TRUTH</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def sketch_from_testimony(described: dict[str, str | None]) -> FaceSpec:
    """The artist draws exactly what you said; unsaid attrs get neutral defaults."""
    spec = {a: (v if v else "none") for a, v in described.items()}
    neutral = {"face_shape": "oval", "skin": "medium", "hair_style": "short_messy",
               "hair_color": "brown", "brows": "thin", "eyes": "normal", "nose": "small",
               "mouth": "neutral"}
    for a, v in neutral.items():
        if not described.get(a):
            spec[a] = v
    return FaceSpec(**spec)


# ------------------------------------------------------------------ state transitions
def new_session() -> dict:
    s = {"screen": "intro", "case_no": 1, "history": []}
    s["case"] = make_case(1)
    return s


def go_glimpse(s: dict) -> dict:
    s = dict(s)
    s["screen"] = "glimpse"
    return s


def go_testimony(s: dict) -> dict:
    s = dict(s)
    s["screen"] = "testimony"
    return s


def submit_testimony(s: dict, text: str) -> dict:
    s = dict(s)
    case: Case = s["case"]
    text = (text or "").strip()
    if len(text) < 8:  # the artist refuses to draw from nothing
        s["testimony_warn"] = ("The sketch artist looks at you. \"...That's it? "
                               "Give me SOMETHING, detective. Face, hair, hat — anything.\"")
        s["screen"] = "testimony"
        return s
    s.pop("testimony_warn", None)
    if HAS_MODEL:
        try:
            described = parse_testimony_model(text)
        except Exception:
            described = parse_testimony(text)
    else:
        described = parse_testimony(text)
    s["described"] = described
    rng = random.Random(case.seed + 1)
    faces, culprit_idx = build_lineup(case.lineup_culprit, described, case.lineup_size, rng)
    s["lineup"] = faces
    s["culprit_idx"] = culprit_idx
    s["screen"] = "lineup"
    return s


def pick_suspect(s: dict, picked: int) -> dict:
    s = dict(s)
    case: Case = s["case"]
    s["picked"] = picked
    s["correct"] = picked == s["culprit_idx"]
    s["history"] = s["history"] + [{"case": case.case_no, "correct": s["correct"]}]
    s["screen"] = "verdict"
    return s


def next_case(s: dict) -> dict:
    s = dict(s)
    s["case_no"] = min(s["case_no"] + 1, len(RANKS))
    s["case"] = make_case(s["case_no"])
    s["screen"] = "intro"
    for stale in ("described", "lineup", "culprit_idx", "picked", "correct"):
        s.pop(stale, None)
    return s


# ------------------------------------------------------------------ UI
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Special+Elite&display=swap');
:root { --paper:#f4efe4; --ink:#2b2a28; --red:#a33327; --tape:#c9a227; }
.gradio-container, body, .dark, .light {
  /* longhand wins the theme war: gradio light theme resets background-color */
  background-color:#191713 !important;
  font-family:'Courier New',monospace !important;
  background-image: radial-gradient(ellipse at 50% -10%, rgba(201,162,39,0.07) 0%, transparent 55%) !important; }
#ew-root { max-width: 900px; margin: 0 auto; }
.ew-header { text-align:center; color:var(--paper); letter-spacing:7px; font-size:34px; padding:12px 0 0;
  font-family:'Special Elite','Courier New',monospace; text-shadow: 0 2px 0 rgba(0,0,0,.6); }
.ew-sub { text-align:center; color:#8d8678; font-size:12px; letter-spacing:2px; margin-bottom:8px; }
.ew-rank { font-family:monospace; color:var(--tape); text-align:center; letter-spacing:2px; font-size:13px; }
.ew-card { background:var(--paper) !important; border-radius:4px; padding:22px 26px !important; color:var(--ink);
           box-shadow: 0 10px 40px rgba(0,0,0,.5); }
.ew-card * { color: var(--ink); }
/* inputs: gradio's dark theme paints them near-black — force paper on paper */
.ew-card textarea, .ew-card input[type="text"], .ew-card input {
  background: #fffdf6 !important; color: var(--ink) !important;
  border: 1px solid #b9b09a !important; }
.ew-card textarea::placeholder, .ew-card input::placeholder { color: #9a8f7c !important; }
.ew-card label span { color: var(--ink) !important; }
.ew-model-badge { font-family: monospace; font-size: 10.5px; letter-spacing: 1px;
  color: #6d6354; background: #ece5d4; border: 1px dashed #b9b09a; border-radius: 4px;
  padding: 4px 10px; margin: 2px 0 10px; display: inline-block; }
.ew-warn { color: var(--red) !important; font-weight: bold; font-size: 13px; }
.ew-glimpse { position:relative; width:320px; margin:0 auto; }
.ew-glimpse::before { content:'● REC'; position:absolute; top:8px; left:10px; z-index:3;
  color:#e03b2f; font-size:12px; letter-spacing:2px; animation: ew-blink 1.1s step-end infinite; }
.ew-glimpse::after { content:''; position:absolute; inset:0 0 13px 0; z-index:2; pointer-events:none;
  box-shadow: inset 0 0 60px rgba(0,0,0,.55);
  background: repeating-linear-gradient(0deg, transparent 0 3px, rgba(0,0,0,0.06) 3px 4px); }
@keyframes ew-blink { 50% { opacity: 0; } }
.ew-glimpse img { width:100%; filter: blur(0); }
@keyframes ew-blurout { to { filter: blur(30px) contrast(0.4); } }
.ew-static { position:absolute; inset:0 0 13px 0; display:flex; align-items:center; justify-content:center;
  background: repeating-linear-gradient(0deg,#111 0 2px,#2c2c2c 2px 4px); color:#9b958a;
  font-size:20px; letter-spacing:5px; opacity:0; pointer-events:none; }
@keyframes ew-appear { to { opacity: 1; } }
.ew-timerbar { height:7px; background:#d8d0bd; margin-top:6px; }
.ew-timerfill { height:100%; background:var(--red); width:100%; transform-origin:left;
  animation: ewshrink linear forwards; }
@keyframes ewshrink { from { transform:scaleX(1);} to { transform:scaleX(0);} }
.ew-sketch { display:block; margin:0 auto; border:1px solid #b9b09a; }
.ew-report { width:100%; border-collapse:collapse; font-size:13px; margin-top:14px; }
.ew-report th, .ew-report td { text-align:left; padding:4px 10px; border-bottom:1px solid #d8d0bd; }
.ew-hit td { color:#1d6b2f !important; } .ew-miss td { color:var(--red) !important; } .ew-silent td { color:#8d8678 !important; }
.ew-verdict { text-align:center; }
.ew-stamp { display:inline-block; border:4px solid var(--red); color:var(--red); padding:6px 22px;
  font-size:26px; letter-spacing:4px; transform:rotate(-6deg) scale(1); margin:6px 0 10px;
  font-family:'Special Elite','Courier New',monospace;
  animation: ew-stamp-in .28s cubic-bezier(.2,2.2,.5,1) both; }
@keyframes ew-stamp-in { from { opacity:0; transform: rotate(-6deg) scale(2.4); } }
.ew-good .ew-stamp { border-color:#1d6b2f; color:#1d6b2f; }
.ew-quote { font-style:italic; margin-bottom:12px; }
.ew-pair { display:flex; gap:18px; justify-content:center; }
.ew-pair figure { margin:0; } .ew-pair figcaption { font-size:11px; letter-spacing:2px; text-align:center; }
.ew-badge { font-size:20px; letter-spacing:3px; margin-top:12px; }
.ew-line { color:#5d564a !important; font-size:13px; margin-top:4px; }
"""

HEADER = ('<div class="ew-header">EYEWITNESS</div>'
          '<div class="ew-sub">YOU SAW THE THIEF FOR 3 SECONDS · YOUR MEMORY IS THE ONLY WITNESS</div>')


with gr.Blocks(title="EYEWITNESS") as demo:
    state = gr.State(new_session())  # gr.State deep-copies per session

    with gr.Column(elem_id="ew-root"):
        gr.HTML(HEADER)

        @gr.render(inputs=state)
        def stage(s: dict):
            case: Case = s["case"]
            gr.HTML(f'<div class="ew-rank">CASE #{case.case_no:03d} · RANK: {case.rank} · '
                    f'GLIMPSE: {case.glimpse_seconds:g}s · LINEUP: {case.lineup_size}</div>')
            scr = s["screen"]

            if scr == "intro":
                with gr.Column(elem_classes=["ew-card"]):
                    gr.Markdown(
                        f"## {case.crime_name}\n\nThe suspect {case.crime_blurb}.\n\n"
                        f"A street camera caught **{case.glimpse_seconds:g} seconds** of footage "
                        f"before the feed died.\nWatch closely, detective. Then tell the sketch "
                        f"artist everything you remember.")
                    b = gr.Button("▶ ROLL THE FOOTAGE", variant="primary")
                    b.click(go_glimpse, state, state)

            elif scr == "glimpse":
                with gr.Column(elem_classes=["ew-card"]):
                    gr.HTML(glimpse_html(case))
                    b = gr.Button("I SAW HIM →", variant="primary")
                    b.click(go_testimony, state, state)

            elif scr == "testimony":
                with gr.Column(elem_classes=["ew-card"]):
                    gr.Markdown("### Tell the sketch artist everything.\n"
                                "*Face, hair, glasses, beard, hat, marks… anything you remember. "
                                "English o castellano.*")
                    gr.HTML('<span class="ew-model-badge">🧠 NEXT: MiniCPM5-1B (fine-tuned) '
                            'translates your words into the official attribute sheet</span>')
                    if s.get("testimony_warn"):
                        gr.HTML(f'<p class="ew-warn">{s["testimony_warn"]}</p>')
                    tb = gr.Textbox(lines=4, label="Your testimony",
                                    placeholder="e.g. round face, bushy eyebrows, beanie, sunglasses, big nose, looked smug...")
                    with gr.Row():
                        mic = gr.Audio(sources=["microphone"], type="numpy",
                                       label="…or SPEAK your testimony", scale=2)
                        if HAS_ASR:
                            gr.HTML('<span class="ew-model-badge">🎤 Cohere Transcribe 2B '
                                    'writes down EXACTLY what you say — verbatim, like a real '
                                    'court reporter</span>')
                    if HAS_ASR:
                        mic.stop_recording(transcribe_testimony, [state, mic, tb], [state, tb])
                    b = gr.Button("SEND TO SKETCH ARTIST", variant="primary")
                    b.click(submit_testimony, [state, tb], state)

            elif scr == "lineup":
                described = s["described"]
                n_said = sum(1 for v in described.values() if v)
                sketch = sketch_from_testimony(described)
                with gr.Column(elem_classes=["ew-card"]):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### The artist's sketch (from YOUR words)")
                            gr.HTML(f'<img src="{svg_uri(render_face_svg(sketch, width=280, seed_jitter=3))}" class="ew-sketch"/>')
                            gr.Markdown(f"*The artist drew what you gave them ({n_said} details). "
                                        + ("Bold of you to call that a description.*" if n_said < 4 else "Not bad, detective.*"))
                            if case.disguise_attr:
                                gr.Markdown(f"**⚠ {case.disguise_line}**")
                        with gr.Column(scale=2):
                            gr.Markdown("### THE LINEUP — click the culprit")
                            gr.HTML('<span class="ew-model-badge">⚙️ NO model here: the engine '
                                    'builds the lineup FROM YOUR ERRORS — wrong claims get planted '
                                    'on innocents; what you never mentioned becomes the disguise</span>')
                            gr.Markdown("*Your sketch helps exactly as much as your memory was "
                                        "right: trust it blindly and you'll arrest the innocent "
                                        "wearing YOUR mistakes.*")
                            n = len(s["lineup"])
                            rows = -(-n // 4)  # ceil
                            gal = gr.Gallery(
                                value=[(face_png(f, width=240), f"Nº {i + 1}")
                                       for i, f in enumerate(s["lineup"])],
                                columns=4, rows=rows, height=rows * 330,
                                allow_preview=False, label="")

                            def _pick(st: dict, evt: gr.SelectData):
                                return pick_suspect(st, evt.index)

                            gal.select(_pick, state, state)

            elif scr == "verdict":
                correct = s["correct"]
                report = grade_testimony(s["described"], case.culprit)
                badge, line = detective_rating(report, correct, case.glimpse_seconds)
                truth_img = svg_uri(render_face_svg(case.culprit, width=240))
                picked_img = svg_uri(render_face_svg(s["lineup"][s["picked"]], width=240))
                head = "ARREST CONFIRMED" if correct else "WRONG ARREST"
                # personalized taunt: best-of-5 from the 1B, validated, with a
                # deterministic personalized template as the floor — never canned
                if HAS_MODEL:
                    taunt = culprit_taunt(report.rows, correct, seed=case.seed)
                else:
                    from game.model import _template_taunt
                    wrongs = [(l, s, t) for l, s, t, v in report.rows if v == "miss"][:3]
                    misseds = [l for l, _s, _t, v in report.rows if v == "silent"][:3]
                    taunt = _template_taunt(wrongs, misseds, correct, case.seed)
                quote = f"“{taunt}”"
                with gr.Column(elem_classes=["ew-card"]):
                    gr.HTML(f"""
<div class="ew-verdict {'ew-good' if correct else 'ew-bad'}">
  <div class="ew-stamp">{head}</div>
  <div class="ew-quote">{quote}</div>
  <div class="ew-pair">
    <figure><img src="{picked_img}"/><figcaption>YOUR PICK</figcaption></figure>
    <figure><img src="{truth_img}"/><figcaption>THE CULPRIT{' (at the time)' if case.disguise_attr else ''}</figcaption></figure>
  </div>
  <div class="ew-badge">{badge}</div>
  <div class="ew-line">{line} · Memory accuracy: <b>{report.weighted_pct}%</b></div>
</div>
{report_table_html(report)}""")
                    live_voice = live_speak(taunt, case.seed, case.culprit) if taunt else None
                    voice = live_voice or verdict_voice(correct)
                    badge = ('🧠 LIVE: MiniCPM5-1B (base) wrote that line about YOUR mistakes · '
                             '🔊 VoxCPM2 cloned the suspect\'s voice just now'
                             if live_voice else
                             '🧠 LIVE: MiniCPM5-1B (base) wrote that line about YOUR mistakes')
                    gr.HTML(f'<span class="ew-model-badge">{badge}</span>')
                    if voice:
                        # minimal kwargs: the Space's gradio build rejects newer
                        # Audio options like show_download_button
                        gr.Audio(value=voice, autoplay=True, show_label=False,
                                 container=False)
                    poster = make_wanted_poster(
                        sketch_from_testimony(s["described"]), case.culprit, correct,
                        report.weighted_pct, case.crime_name, case.rank)
                    with gr.Accordion("📜 YOUR WANTED POSTER — download & dare a friend", open=False):
                        gr.Image(value=poster, show_label=False, height=420)
                    if case.case_no < len(RANKS):
                        nxt = RANKS[case.case_no][0]
                        b = gr.Button(f"NEXT CASE → RANK: {nxt}", variant="primary")
                        b.click(next_case, state, state)
                    else:
                        solved = sum(1 for h in s["history"] if h["correct"])
                        gr.Markdown(f"## CAREER COMPLETE\n**{solved}/{len(RANKS)} arrests confirmed.** "
                                    "The precinct thanks you. The pigeons remain at large.")
                        b = gr.Button("NEW CAREER", variant="primary")
                        b.click(lambda: new_session(), None, state)


if __name__ == "__main__":
    if os.environ.get("SPACES_ZERO_GPU"):  # pay model loads at startup, not per user
        try:
            from game import model as _m, voice as _v
            _m.preload()
            _v.preload()
            print("[startup] models preloaded on CPU", flush=True)
        except Exception as e:
            print(f"[startup] preload failed (cascade covers it): {e}", flush=True)
    demo.launch(css=CSS)
