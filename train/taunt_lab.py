"""Taunt prompt lab (RTX 4090): iterate until the culprit sounds like a culprit.

Runs the base 1B against realistic verdict diffs with competing prompt variants;
prints samples for human judgment. Diagnosed failure mode being fixed: the model
confuses narrative voice (speaks as the witness, not the culprit).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCENARIOS = [
    {"outcome": "caught", "wrong": [("headwear", "beanie", "fedora"), ("facial hair", "full beard", "goatee")],
     "missed": ["eyes", "hair color"], "right": ["glasses"]},
    {"outcome": "escaped", "wrong": [("hair", "curly", "slick back")],
     "missed": ["marks", "nose", "mouth"], "right": ["face", "hat"]},
    {"outcome": "escaped", "wrong": [("glasses", "sunglasses", "round glasses"), ("face", "round", "long")],
     "missed": ["facial hair"], "right": []},
    {"outcome": "caught", "wrong": [],
     "missed": ["everything — they only said 'a guy, normal-looking'"], "right": []},
]

# V1 = current production prompt (control)
V1 = """You are the culprit in a comedy detective game. Outcome: you were {outcome}.
The witness just described you to a sketch artist. Their wrong claims: {wrong}.
What they failed to notice: {missed}. What they got right: {right}.
Speak ONE smug in-character line (under 25 words) mocking their SPECIFIC mistakes.
Plain text only, no quotes, no emoji, no hashtags, no explanations, exactly one sentence.

Example (wrong claim "said my hat was beanie (it was fedora)"): A beanie? Detective, this fedora has more class than your entire memory.

Line:"""

# V2 = first-person framing up front, mistakes as direct address, 2 examples
V2 = """Roleplay: you are a smug petty criminal taunting the detective who just questioned a witness about you. You were {outcome}.
The witness's mistakes about your appearance:
{wrong_lines}
Mock the detective in ONE short sentence (max 22 words), first person, naming one specific mistake. Never describe yourself neutrally — gloat.

Examples of your style:
- A BEANIE? I wear a fedora, sweetheart. Ask the mirror how your memory feels.
- Curly hair? I spend twenty minutes slicking it back and THIS is my reward?

Your line:"""

# V3 = fill-in-the-blank: model completes a started sentence (maximum constraint)
V3 = """Complete the smug criminal's one-liner. He was {outcome}. The witness wrongly said his {attr} was {said} when really it was {truth}.

The criminal sneers: "{said_cap}? """


def fmt_wrong(ws):
    return "; ".join(f"said my {a} was {s} (it was {t})" for a, s, t in ws) or "none — they remembered nothing specific"


def main():
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openbmb/MiniCPM5-1B", trust_remote_code=True,
        torch_dtype=torch.float16, device_map="cuda")

    def gen(prompt, temp=0.7, n=2, max_new=55):
        outs = []
        for _ in range(n):
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            enc = tok(text, return_tensors="pt").to("cuda")
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                 temperature=temp, top_p=0.9, pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(out[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip().split("\n")[0][:160])
        return outs

    for i, sc in enumerate(SCENARIOS):
        wrong_s = fmt_wrong(sc["wrong"])
        print(f"\n=== SCENARIO {i + 1} ({sc['outcome']}) | {wrong_s[:70]} ===")
        p1 = V1.format(outcome="caught red-handed" if sc["outcome"] == "caught" else "wrongly let go",
                       wrong=wrong_s, missed=", ".join(sc["missed"]), right=", ".join(sc["right"]) or "nothing")
        for o in gen(p1):
            print("  [V1]", o)
        wl = "\n".join(f"- they said your {a} was {s}; it is actually {t}" for a, s, t in sc["wrong"]) \
             or "- they remembered nothing specific about you"
        p2 = V2.format(outcome="caught red-handed" if sc["outcome"] == "caught" else "wrongly let go (they arrested an innocent man)",
                       wrong_lines=wl)
        for o in gen(p2):
            print("  [V2]", o)
        if sc["wrong"]:
            a, s, t = sc["wrong"][0]
            p3 = V3.format(outcome=sc["outcome"], attr=a, said=s, truth=t, said_cap=s.capitalize())
            for o in gen(p3, max_new=40):
                print("  [V3]", f'{s.capitalize()}? ' + o)


if __name__ == "__main__":
    main()
