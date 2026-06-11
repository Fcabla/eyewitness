"""EYEWITNESS — you saw the thief for 3 seconds. Your memory is the only witness.

Game loop: case intro -> timed glimpse -> testimony -> sketch reveal -> lineup
(built from YOUR errors) -> verdict. Escalating ranks shorten the glimpse and
grow the lineup; from INSPECTOR the culprit changes one feature before the
lineup ("he's been to a barber since").

Architecture: deterministic engine owns all ground truth (specs, coordinates,
scoring). Models only translate: MiniCPM5-1B parses messy human testimony into
attribute JSON (Tier B, ZeroGPU); a synonym matcher is the offline Tier A.
"""
from __future__ import annotations

import base64
import random

import gradio as gr

from game.casegen import make_case, Case, RANKS
from game.face import render_face_svg, render_unknown_svg, FaceSpec
from game.lineup import build_lineup
from game.parser import parse_testimony
from game.scoring import grade_testimony, detective_rating, LABELS_EN

try:  # Tier B (deployed): MiniCPM5-1B slot-filler. Falls back to Tier A locally.
    from game.model import parse_testimony_model  # noqa: F401
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False


# ------------------------------------------------------------------ helpers
def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def glimpse_html(case: Case) -> str:
    """Client-side timed reveal: face shows for N seconds, then static fuzz."""
    ms = int(case.glimpse_seconds * 1000)
    face = svg_uri(render_face_svg(case.culprit, width=320))
    return f"""
<div class="ew-glimpse" id="ew-glimpse">
  <img src="{face}" alt="suspect" id="ew-suspect"/>
  <div class="ew-static" id="ew-static"><span>SIGNAL LOST</span></div>
  <div class="ew-timerbar"><div class="ew-timerfill" id="ew-timerfill" style="animation-duration:{ms}ms"></div></div>
</div>
<script>
(function() {{
  const img = document.getElementById('ew-suspect');
  const st = document.getElementById('ew-static');
  st.style.display = 'none';
  setTimeout(() => {{ img.style.filter = 'blur(30px) contrast(0.4)'; st.style.display = 'flex'; }}, {ms});
}})();
</script>"""


def lineup_gallery(faces: list[FaceSpec]) -> list[tuple[str, str]]:
    return [(svg_uri(render_face_svg(f, width=240)), f"Nº {i + 1}") for i, f in enumerate(faces)]


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


# ------------------------------------------------------------------ state
def new_session() -> dict:
    return {"case_no": 1, "case": None, "described": None, "lineup": None,
            "culprit_idx": None, "history": []}


def start_case(state: dict):
    case = make_case(state["case_no"])
    state["case"] = case
    rank_line = f"CASE #{case.case_no:03d} · RANK: {case.rank} · GLIMPSE: {case.glimpse_seconds:g}s · LINEUP: {case.lineup_size}"
    intro = (f"## {case.crime_name}\n\nThe suspect {case.crime_blurb}.\n\n"
             f"A street camera caught **{case.glimpse_seconds:g} seconds** of footage before the feed died.\n"
             f"Watch closely, detective. Then tell the sketch artist everything you remember.")
    return (state, rank_line, intro,
            gr.update(visible=True),   # intro screen
            gr.update(visible=False),  # glimpse
            gr.update(visible=False),  # testimony
            gr.update(visible=False),  # lineup
            gr.update(visible=False))  # verdict


def show_glimpse(state: dict):
    return (glimpse_html(state["case"]),
            gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))


def to_testimony(state: dict):
    return (gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))


def submit_testimony(state: dict, text: str):
    case: Case = state["case"]
    text = (text or "").strip()
    if HAS_MODEL:
        try:
            described = parse_testimony_model(text)
        except Exception:
            described = parse_testimony(text)
    else:
        described = parse_testimony(text)
    state["described"] = described

    # the sketch from YOUR words: described attrs as said, the rest stays unknown-neutral
    sketch_spec_dict = {a: (v if v else "none") for a, v in described.items()}
    # un-described non-optional attrs get neutral defaults so the sketch renders
    neutral = {"face_shape": "oval", "skin": "medium", "hair_style": "short_messy",
               "hair_color": "brown", "brows": "thin", "eyes": "normal", "nose": "small",
               "mouth": "neutral"}
    for a, v in neutral.items():
        if not described.get(a):
            sketch_spec_dict[a] = v
    sketch = FaceSpec(**sketch_spec_dict)

    n_said = sum(1 for v in described.values() if v)
    artist_line = (f"The artist drew what you gave them ({n_said} details). "
                   + ("Bold of you to call that a description." if n_said < 4 else "Not bad, detective."))

    rng = random.Random(case.seed + 1)
    faces, culprit_idx = build_lineup(case.lineup_culprit, described, case.lineup_size, rng)
    state["lineup"] = faces
    state["culprit_idx"] = culprit_idx

    disguise_note = f"**⚠ {case.disguise_line}**" if case.disguise_attr else ""
    return (state,
            f'<img src="{svg_uri(render_face_svg(sketch, width=300, seed_jitter=3))}" class="ew-sketch"/>',
            artist_line, disguise_note,
            lineup_gallery(faces),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=True), gr.update(visible=False))


def pick_suspect(state: dict, evt: gr.SelectData):
    case: Case = state["case"]
    picked = evt.index
    correct = picked == state["culprit_idx"]
    report = grade_testimony(state["described"], case.culprit)
    badge, line = detective_rating(report, correct, case.glimpse_seconds)
    state["history"].append({"case": case.case_no, "correct": correct, "acc": report.weighted_pct})

    truth_img = svg_uri(render_face_svg(case.culprit, width=260))
    picked_img = svg_uri(render_face_svg(state["lineup"][picked], width=260))
    verdict_head = "ARREST CONFIRMED" if correct else "WRONG ARREST"
    culprit_quote = ("“Okay, okay. It was the gnomes. Take me in.”" if correct
                     else "“Wrong guy. I walked RIGHT past you. Twice.”")
    summary = f"""
<div class="ew-verdict {'ew-good' if correct else 'ew-bad'}">
  <div class="ew-stamp">{verdict_head}</div>
  <div class="ew-quote">{culprit_quote}</div>
  <div class="ew-pair">
    <figure><img src="{picked_img}"/><figcaption>YOUR PICK</figcaption></figure>
    <figure><img src="{truth_img}"/><figcaption>THE CULPRIT{(' (at the time)') if case.disguise_attr else ''}</figcaption></figure>
  </div>
  <div class="ew-badge">{badge}</div>
  <div class="ew-line">{line} · Memory accuracy: <b>{report.weighted_pct}%</b></div>
</div>
{report_table_html(report)}"""
    state["case_no"] = min(case.case_no + 1, len(RANKS))
    next_label = f"NEXT CASE → {RANKS[min(state['case_no'] - 1, len(RANKS) - 1)][0]}"
    return (state, summary, gr.update(value=next_label),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))


# ------------------------------------------------------------------ UI
CSS = """
:root { --paper:#f4efe4; --ink:#2b2a28; --red:#a33327; --tape:#c9a227; }
.gradio-container { background:#191713 !important; font-family:'Courier New',monospace !important; }
#ew-root { max-width: 900px; margin: 0 auto; }
.ew-header { text-align:center; color:var(--paper); letter-spacing:6px; font-size:30px; padding:10px 0 0; }
.ew-sub { text-align:center; color:#8d8678; font-size:12px; letter-spacing:2px; margin-bottom:8px; }
.ew-rank { font-family:monospace; color:var(--tape); text-align:center; letter-spacing:2px; font-size:13px; }
.ew-card { background:var(--paper); border-radius:4px; padding:22px 26px; color:var(--ink);
           box-shadow: 0 10px 40px rgba(0,0,0,.5); }
.ew-glimpse { position:relative; width:320px; margin:0 auto; }
.ew-glimpse img { width:100%; transition: filter .18s; }
.ew-static { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  background: repeating-linear-gradient(0deg,#111 0 2px,#2c2c2c 2px 4px); color:#9b958a;
  font-size:20px; letter-spacing:5px; }
.ew-timerbar { height:7px; background:#d8d0bd; margin-top:6px; }
.ew-timerfill { height:100%; background:var(--red); width:100%; transform-origin:left;
  animation: ewshrink linear forwards; }
@keyframes ewshrink { from { transform:scaleX(1);} to { transform:scaleX(0);} }
.ew-sketch { display:block; margin:0 auto; border:1px solid #b9b09a; }
.ew-report { width:100%; border-collapse:collapse; font-size:13px; margin-top:14px; }
.ew-report th, .ew-report td { text-align:left; padding:4px 10px; border-bottom:1px solid #d8d0bd; }
.ew-hit td { color:#1d6b2f; } .ew-miss td { color:var(--red); } .ew-silent td { color:#8d8678; }
.ew-verdict { text-align:center; }
.ew-stamp { display:inline-block; border:4px solid var(--red); color:var(--red); padding:6px 22px;
  font-size:26px; letter-spacing:4px; transform:rotate(-6deg); margin:6px 0 10px; }
.ew-good .ew-stamp { border-color:#1d6b2f; color:#1d6b2f; }
.ew-quote { font-style:italic; margin-bottom:12px; }
.ew-pair { display:flex; gap:18px; justify-content:center; }
.ew-pair figure { margin:0; } .ew-pair figcaption { font-size:11px; letter-spacing:2px; text-align:center; }
.ew-badge { font-size:20px; letter-spacing:3px; margin-top:12px; color:var(--ink); }
.ew-line { color:#5d564a; font-size:13px; margin-top:4px; }
button.primary { background:var(--red) !important; border:none !important; letter-spacing:2px !important; }
"""

with gr.Blocks(css=CSS, title="EYEWITNESS") as demo:
    state = gr.State(new_session())

    with gr.Column(elem_id="ew-root"):
        gr.HTML('<div class="ew-header">EYEWITNESS</div>'
                '<div class="ew-sub">YOU SAW THE THIEF FOR 3 SECONDS · YOUR MEMORY IS THE ONLY WITNESS</div>')
        rank_bar = gr.HTML(elem_classes=["ew-rank"])

        with gr.Column(visible=True, elem_classes=["ew-card"]) as s_intro:
            intro_md = gr.Markdown()
            btn_glimpse = gr.Button("▶ ROLL THE FOOTAGE", variant="primary")

        with gr.Column(visible=False, elem_classes=["ew-card"]) as s_glimpse:
            glimpse_box = gr.HTML()
            btn_done = gr.Button("I SAW HIM →", variant="primary")

        with gr.Column(visible=False, elem_classes=["ew-card"]) as s_testimony:
            gr.Markdown("### Tell the sketch artist everything.\n*Face, hair, glasses, beard, hat, marks… anything you remember. English o castellano.*")
            testimony_tb = gr.Textbox(lines=4, label="Your testimony",
                                      placeholder="e.g. round face, bushy eyebrows, red beanie, sunglasses, big nose, looked smug...")
            btn_sketch = gr.Button("SEND TO SKETCH ARTIST", variant="primary")

        with gr.Column(visible=False, elem_classes=["ew-card"]) as s_lineup:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### The artist's sketch (from YOUR words)")
                    sketch_html = gr.HTML()
                    artist_md = gr.Markdown()
                    disguise_md = gr.Markdown()
                with gr.Column(scale=2):
                    gr.Markdown("### THE LINEUP — click the culprit")
                    lineup_gal = gr.Gallery(columns=4, height=300, allow_preview=False, label="")

        with gr.Column(visible=False, elem_classes=["ew-card"]) as s_verdict:
            verdict_html = gr.HTML()
            btn_next = gr.Button("NEXT CASE →", variant="primary")

    screens = [s_intro, s_glimpse, s_testimony, s_lineup, s_verdict]
    demo.load(start_case, state, [state, rank_bar, intro_md, *screens])
    btn_glimpse.click(show_glimpse, state, [glimpse_box, *screens])
    btn_done.click(to_testimony, state, screens)
    btn_sketch.click(submit_testimony, [state, testimony_tb],
                     [state, sketch_html, artist_md, disguise_md, lineup_gal, *screens])
    lineup_gal.select(pick_suspect, state,
                      [state, verdict_html, btn_next, *screens])
    btn_next.click(start_case, state, [state, rank_bar, intro_md, *screens])

if __name__ == "__main__":
    demo.launch()
