"""Real, small pretrained-model LoRA training and honest held-out evaluation.

Model imports are lazy so dataset integrity checks work without an ML runtime.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import resource
import time

from training.datasets.contracts import load_rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    splits, cases, seen_rows, seen_subjects = {}, {}, set(), set()
    for split in ("train", "validation", "test"):
        spec = manifest["splits"][split]
        path, case_path = directory / spec["path"], directory / spec["cases"]
        if sha256(path) != spec["sha256"] or sha256(case_path) != spec["cases_sha256"]:
            raise ValueError(f"Dataset hash mismatch for {split}")
        rows = load_rows(path, "clone")
        metadata = json.loads(case_path.read_text())
        row_hashes = [hashlib.sha256((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()).hexdigest() for row in rows]
        subjects = {case["subject"] for case in metadata}
        if len(rows) != spec["rows"] or len(rows) != len(metadata) or row_hashes != spec["row_sha256"]:
            raise ValueError(f"Dataset row manifest mismatch for {split}")
        if seen_rows.intersection(row_hashes) or seen_subjects.intersection(subjects):
            raise ValueError("Data leakage across splits")
        seen_rows.update(row_hashes)
        seen_subjects.update(subjects)
        splits[split], cases[split] = rows, metadata
    return manifest, splits, cases


def prompt_ids(tokenizer, row):
    messages = [{"role": "system", "content": row["system"]},
                {"role": "user", "content": "Memory:\n" + row.get("context", "") + "\n\nQuestion:\n" + row["prompt"]}]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def encode_row(tokenizer, row, max_length):
    prompt = prompt_ids(tokenizer, row)
    response = tokenizer.encode(row["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prompt) + len(response) > max_length:
        raise ValueError(f"Example length {len(prompt) + len(response)} exceeds max_length={max_length}; targets must not be silently truncated")
    return {"input_ids": prompt + response, "labels": [-100] * len(prompt) + response}


def generation_metrics(rows, cases, outputs):
    if not (len(rows) == len(cases) == len(outputs)):
        raise ValueError("One actual model generation is required per held-out row")
    format_count, exact_count, semantic_count = 0, 0, 0
    per_kind = {}
    details = []
    for row, case, text in zip(rows, cases, outputs):
        # The answer body, never the injected persona header, must contain evidence.
        match = re.fullmatch(r"\[Lex\] tone=calm principles=honesty, privacy response=(.+)", text.strip(), re.DOTALL)
        body = match.group(1).strip() if match else text.strip()
        semantic = case["expected_body_contains"].casefold() in body.casefold() and case["body_must_not_contain"].casefold() not in body.casefold()
        valid = match is not None
        exact = text.strip() == row["response"].strip()
        format_count += valid
        exact_count += exact
        semantic_count += semantic
        group = per_kind.setdefault(case["kind"], {"correct": 0, "total": 0})
        group["correct"] += semantic
        group["total"] += 1
        details.append({"kind": case["kind"], "subject": case["subject"], "prompt": row["prompt"],
                        "context": row.get("context", ""), "target": row["response"], "generation": text,
                        "answer_body": body, "format_valid": valid, "exact_match": exact,
                        "body_grounding_correct": semantic})
    count = len(rows)
    return {"rows": count, "format_pass_rate": format_count / count, "exact_match_rate": exact_count / count,
            "body_grounding_accuracy": semantic_count / count, "per_kind": per_kind,
            "scope": "Actual deterministic generations on synthetic held-out memory values; body substring rubric, not an LLM judge.",
            "details": details}


def tensor_batch(torch, example):
    return {key: torch.tensor([value], dtype=torch.long) for key, value in example.items()}


def evaluate_loss(torch, model, examples):
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for example in examples:
            batch = tensor_batch(torch, example)
            weight = int((batch["labels"][:, 1:] != -100).sum())
            loss = float(model(**batch, use_cache=False).loss)
            total += loss * weight
            count += weight
    return {"response_token_nll": total / count, "scored_response_tokens": count, "rows": len(examples)}


def generate_rows(torch, model, tokenizer, rows, max_new_tokens):
    """Actual greedy decoding with KV caching; no distributed-runtime dependency."""
    model.eval()
    outputs = []
    eos_id = tokenizer.eos_token_id
    with torch.inference_mode():
        for row in rows:
            step_ids = torch.tensor([prompt_ids(tokenizer, row)], dtype=torch.long)
            generated_ids, cache = [], None
            for _ in range(max_new_tokens):
                result = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
                next_id = torch.argmax(result.logits[:, -1, :], dim=-1, keepdim=True)
                cache = result.past_key_values
                generated_ids.append(int(next_id.item()))
                if eos_id is not None and generated_ids[-1] == int(eos_id):
                    break
                step_ids = next_id
            outputs.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return outputs


def tree_hashes(directory):
    return {str(path.relative_to(directory)): sha256(path) for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("training/datasets/clone_smoke_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--revision", required=True, help="Immutable 40-character upstream model commit")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.revision):
        raise ValueError("Pin the base to an immutable Hugging Face commit SHA")
    if args.epochs < 1 or args.gradient_accumulation < 1:
        raise ValueError("epochs and gradient accumulation must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest, rows, cases = load_dataset(args.dataset)
    started = time.time()
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(args.base_model, revision=args.revision)
    card = info.card_data.to_dict() if info.card_data else {}
    license_id = card.get("license")
    if info.sha != args.revision or license_id != "apache-2.0":
        raise ValueError(f"Unexpected model identity or license: {info.sha}, {license_id}")
    readme = hf_hub_download(args.base_model, "README.md", revision=args.revision, local_files_only=args.local_files_only)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision, local_files_only=args.local_files_only)
    tokenizer.pad_token = tokenizer.eos_token
    examples = {split: [encode_row(tokenizer, row, args.max_length) for row in data] for split, data in rows.items()}
    model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=args.revision,
        torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager", local_files_only=args.local_files_only)
    source = {
        "model_id": "clone_smollm2_tiny_v1", "base_model": args.base_model, "base_revision": info.sha,
        "base_license": license_id, "base_model_card_sha256": sha256(readme),
        "model_card_url": f"https://huggingface.co/{args.base_model}/blob/{info.sha}/README.md",
        "tuning_type": "lora_sft", "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
        "dataset_sha256": manifest["dataset_sha256"], "split_hashes": {k: v["sha256"] for k, v in manifest["splits"].items()},
        "dataset_license": manifest["license"], "prompt_profile": "clone",
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "huggingface-hub", "safetensors")},
        "device": "cpu", "dtype": "float32", "scope": "Synthetic English tiny specialization smoke; not Gemma or production Clone weights.",
        "base_config": model.config.to_dict(), "status": "training",
    }
    save_json(args.output / "lineage.json", source)
    print(json.dumps({"stage": "baseline", "max_sequence_length": max(len(e["input_ids"]) for es in examples.values() for e in es)}), flush=True)
    baseline = {"test_loss": evaluate_loss(torch, model, examples["test"]),
                "validation_loss": evaluate_loss(torch, model, examples["validation"]),
                "generation": generation_metrics(rows["test"], cases["test"], generate_rows(torch, model, tokenizer, rows["test"], args.max_new_tokens))}
    save_json(args.output / "baseline.json", baseline)
    print(json.dumps({"stage": "baseline_complete", "test_nll": baseline["test_loss"]["response_token_nll"], "body_accuracy": baseline["generation"]["body_grounding_accuracy"], "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
    lora = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    source["trainable_parameters"] = sum(p.numel() for p in trainable)
    source["total_parameters"] = sum(p.numel() for p in model.parameters())
    initial_adapter = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    best_val, best_epoch = float("inf"), None
    history = []
    for epoch in range(args.epochs):
        order = list(range(len(examples["train"])))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for offset, index in enumerate(order):
            batch = tensor_batch(torch, examples["train"][index])
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became nonfinite")
            losses.append(float(loss.detach()))
            (loss / args.gradient_accumulation).backward()
            if (offset + 1) % args.gradient_accumulation == 0 or offset + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if offset == 0 or (offset + 1) % 16 == 0:
                print(json.dumps({"stage": "train", "epoch": epoch + 1, "example": offset + 1,
                                  "loss": sum(losses[-16:]) / len(losses[-16:]), "seconds": round(time.time() - started, 1), "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
        validation = evaluate_loss(torch, model, examples["validation"])
        value = validation["response_token_nll"]
        record = {"epoch": epoch + 1, "mean_train_loss": sum(losses) / len(losses), "validation": validation}
        history.append(record)
        if value < best_val:
            best_val, best_epoch = value, epoch + 1
            model.save_pretrained(args.output / "adapter", safe_serialization=True)
            tokenizer.save_pretrained(args.output / "adapter")
        save_json(args.output / "training_history.json", history)
        print(json.dumps({"stage": "validation", **record, "best_epoch": best_epoch}), flush=True)
    del optimizer, trainable, loss, batch
    gc.collect()
    model.gradient_checkpointing_disable()
    set_peft_model_state_dict(model, load_peft_weights(str(args.output / "adapter"), device="cpu"))
    source["adapter_update"] = {
        "changed_tensors": sum(not torch.equal(initial_adapter[name], parameter.detach()) for name, parameter in model.named_parameters() if name in initial_adapter),
        "total_tensors": len(initial_adapter),
        "l2_delta": sum(float((parameter.detach() - initial_adapter[name]).square().sum()) for name, parameter in model.named_parameters() if name in initial_adapter) ** 0.5,
    }
    del initial_adapter
    model.eval()
    adapted = {"selected_epoch": best_epoch, "selection_rule": "Minimum validation response-token NLL; test not used for checkpoint selection.",
               "test_loss": evaluate_loss(torch, model, examples["test"]),
               "generation": generation_metrics(rows["test"], cases["test"], generate_rows(torch, model, tokenizer, rows["test"], args.max_new_tokens))}
    save_json(args.output / "adapted.json", adapted)
    print(json.dumps({"stage": "adapted_complete", "test_nll": adapted["test_loss"]["response_token_nll"], "body_accuracy": adapted["generation"]["body_grounding_accuracy"]}), flush=True)
    probe = tensor_batch(torch, examples["test"][0])["input_ids"]
    with torch.inference_mode():
        reference_logits = model(input_ids=probe, use_cache=False).logits[:, -1, :].clone()
    reference_generation = adapted["generation"]["details"][0]["generation"]
    print(json.dumps({"stage": "export"}), flush=True)
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(args.output / "merged", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "merged")
    del model, merged
    gc.collect()
    print(json.dumps({"stage": "reload_adapter"}), flush=True)
    # Reload the independent adapter checkpoint into a freshly loaded base.
    base_reload = AutoModelForCausalLM.from_pretrained(args.base_model, revision=args.revision,
        torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager", local_files_only=True)
    reloaded_adapter = PeftModel.from_pretrained(base_reload, args.output / "adapter")
    reloaded_adapter.eval()
    with torch.inference_mode():
        adapter_logits = reloaded_adapter(input_ids=probe, use_cache=False).logits[:, -1, :]
        adapter_delta = float((reference_logits - adapter_logits).abs().max())
    adapter_generation = generate_rows(torch, reloaded_adapter, tokenizer, rows["test"][:1], args.max_new_tokens)[0]
    del reloaded_adapter, base_reload
    gc.collect()
    print(json.dumps({"stage": "reload_merged"}), flush=True)
    exported = AutoModelForCausalLM.from_pretrained(args.output / "merged", torch_dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager", local_files_only=True)
    exported.eval()
    with torch.inference_mode():
        exported_logits = exported(input_ids=probe, use_cache=False).logits[:, -1, :]
        merged_delta = float((reference_logits - exported_logits).abs().max())
    exported_generation = generate_rows(torch, exported, tokenizer, rows["test"][:1], args.max_new_tokens)[0]
    reload_check = {"adapter_max_abs_logit_difference": adapter_delta, "merged_max_abs_logit_difference": merged_delta,
                    "tolerance": 0.0001, "adapter_generation_equal": adapter_generation == reference_generation,
                    "merged_generation_equal": exported_generation == reference_generation,
                    "scope": "All-vocabulary next-token logits for one held-out probe and one complete deterministic generation.",
                    "passed": adapter_delta <= 0.0001 and merged_delta <= 0.0001 and adapter_generation == reference_generation and exported_generation == reference_generation}
    save_json(args.output / "reload_validation.json", reload_check)
    source.update({"status": "trained_evaluated_exported" if reload_check["passed"] else "reload_validation_failed",
                   "selected_epoch": best_epoch, "elapsed_seconds": time.time() - started,
                   "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "artifacts": {"adapter": tree_hashes(args.output / "adapter"), "merged": tree_hashes(args.output / "merged")},
                   "baseline_test_nll": baseline["test_loss"]["response_token_nll"],
                   "adapted_test_nll": adapted["test_loss"]["response_token_nll"],
                   "reload_passed": reload_check["passed"]})
    save_json(args.output / "lineage.json", source)
    save_json(args.output / "summary.json", {"status": source["status"], "baseline": {k:v for k,v in baseline.items() if k != "generation"},
        "baseline_generation": {k:v for k,v in baseline["generation"].items() if k != "details"},
        "adapted_test_loss": adapted["test_loss"], "adapted_generation": {k:v for k,v in adapted["generation"].items() if k != "details"},
        "reload_validation": reload_check, "lineage": "lineage.json", "elapsed_seconds": source["elapsed_seconds"]})
    print(json.dumps({"stage": "complete", "status": source["status"], "baseline_nll": source["baseline_test_nll"],
                      "adapted_nll": source["adapted_test_nll"], "reload": reload_check}), flush=True)
    if not reload_check["passed"]:
        raise RuntimeError("Reload equivalence failed")


if __name__ == "__main__":
    main()
