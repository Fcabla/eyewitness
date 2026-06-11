"""LoRA fine-tune of MiniCPM5-1B as the EYEWITNESS testimony parser — on Modal.

Trains on the synthetic (testimony -> attribute JSON) dataset produced by
gen_dataset.py (ground truth by construction), merges the adapter, and pushes
the result to the HF Hub as a PUBLIC model (Well-Tuned badge requires a
published fine-tune that the app actually uses).

Run:  modal run train/train_modal.py --hub-repo Fcabla/MiniCPM5-1B-eyewitness
Then: set EYEWITNESS_MODEL_ID=<hub-repo> in the Space variables.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("eyewitness-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "datasets", "peft", "trl", "accelerate", "huggingface_hub")
)

DATASET_LOCAL = Path(__file__).parent / "dataset.jsonl"

SYSTEM = ("You are a police sketch-artist assistant. Extract ONLY what the witness "
          "said into the attribute JSON. Use null for anything not mentioned. "
          "Output only the JSON object.")


def to_chat(example: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f'Witness testimony: "{example["testimony"]}"'},
        {"role": "assistant", "content": json.dumps(example["labels"], ensure_ascii=False)},
    ]}


@app.function(image=image, gpu="A10G", timeout=5400,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def train(dataset_jsonl: str, hub_repo: str) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in dataset_jsonl.splitlines() if l.strip()]
    ds = Dataset.from_list([to_chat(r) for r in rows]).train_test_split(test_size=0.02, seed=7)

    base = "openbmb/MiniCPM5-1B"
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir="/tmp/out", num_train_epochs=2,
            per_device_train_batch_size=8, gradient_accumulation_steps=2,
            learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
            logging_steps=20, eval_strategy="steps", eval_steps=100,
            bf16=True, max_length=1024, report_to=[],
        ),
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("eval:", metrics)

    merged = trainer.model.merge_and_unload()
    merged.push_to_hub(hub_repo, private=False)
    tok.push_to_hub(hub_repo, private=False)
    return f"pushed to {hub_repo} | eval_loss={metrics.get('eval_loss'):.4f}"


@app.local_entrypoint()
def main(hub_repo: str = "Fcabla/MiniCPM5-1B-eyewitness"):
    data = DATASET_LOCAL.read_text()
    print(train.remote(data, hub_repo))
