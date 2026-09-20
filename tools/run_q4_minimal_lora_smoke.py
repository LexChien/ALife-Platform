#!/usr/bin/env python3
"""Minimal but real LoRA smoke for PLAN 23 Q4 on constrained hosts."""
from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.contracts import load_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dataset = ROOT / "training/datasets/clone_smoke_v1"
    out = ROOT / "runs/training/clone_lora_minimal_q4"
    if out.exists():
        raise SystemExit(f"Refuse to overwrite existing output: {out}")
    out.mkdir(parents=True)

    train_rows = load_rows(dataset / "train.jsonl", "clone")[:32]
    val_rows = load_rows(dataset / "validation.jsonl", "clone")[:8]
    started = time.time()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    base_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )

    def encode(row):
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": "Memory:\n" + row.get("context", "") + "\n\nQuestion:\n" + row["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        response = tokenizer.encode(row["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
        return {
            "input_ids": torch.tensor([prompt + response], dtype=torch.long),
            "labels": torch.tensor([[-100] * len(prompt) + response], dtype=torch.long),
        }

    train_ex = [encode(r) for r in train_rows]
    val_ex = [encode(r) for r in val_rows]

    def mean_loss(examples):
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for ex in examples:
                total += float(model(**ex, use_cache=False).loss)
        return total / max(len(examples), 1)

    baseline_val = mean_loss(val_ex)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    model.train()
    losses = []
    for epoch in range(1):
        for i, ex in enumerate(train_ex):
            opt.zero_grad(set_to_none=True)
            loss = model(**ex, use_cache=False).loss
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if i % 8 == 0:
                print(json.dumps({"stage": "train", "i": i, "loss": losses[-1]}), flush=True)
    adapted_val = mean_loss(val_ex)
    changed = sum(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters() if n in before)
    model.save_pretrained(out / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(out / "adapter")

    # one greedy sample without model.generate
    model.eval()
    sample = train_rows[0]
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": "Memory:\n" + sample.get("context", "") + "\n\nQuestion:\n" + sample["prompt"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = torch.tensor([prompt], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(32):
            logits = model(input_ids=ids, use_cache=False).logits[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == int(tokenizer.eos_token_id):
                break
    generation = tokenizer.decode(ids[0, len(prompt):], skip_special_tokens=True).strip()

    lineage = {
        "status": "trained_minimal_smoke",
        "model_id": "clone_lora_minimal_q4",
        "base_model": base_model,
        "base_revision": revision,
        "tuning_type": "lora_sft",
        "dataset_id": "clone_synthetic_smoke_v1",
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "train_rows_used": len(train_rows),
        "val_rows_used": len(val_rows),
        "epochs": 1,
        "baseline_val_nll": baseline_val,
        "adapted_val_nll": adapted_val,
        "changed_trainable_tensors": changed,
        "total_trainable_tensors": len(before),
        "elapsed_seconds": time.time() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sample_generation": generation,
        "scope": "Minimal CPU LoRA smoke on synthetic clone_smoke_v1 subset; not Gemma production weights.",
        "note": "Uses forward-only greedy decode because torch.generate hits broken distributed import in this venv.",
    }
    (out / "lineage.json").write_text(json.dumps(lineage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({
        "status": lineage["status"],
        "baseline_val_nll": baseline_val,
        "adapted_val_nll": adapted_val,
        "nll_improved": adapted_val < baseline_val,
        "adapter_weights_changed": changed > 0,
        "elapsed_seconds": lineage["elapsed_seconds"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "complete", **{k: lineage[k] for k in ["status", "baseline_val_nll", "adapted_val_nll", "changed_trainable_tensors", "elapsed_seconds"]}}, indent=2), flush=True)
    if changed <= 0:
        raise SystemExit("LoRA parameters did not change")


if __name__ == "__main__":
    main()
