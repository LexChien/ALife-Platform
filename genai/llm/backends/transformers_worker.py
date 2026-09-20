"""One-request local CPU Transformer worker; stdout is exactly one JSON value."""
import json
import resource
import sys
import time


def main():
    payload = json.load(sys.stdin)
    started = time.monotonic()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from training.lora.clone_smoke import generate_rows, prompt_ids
    torch.set_num_threads(payload.get("threads", 2))
    tokenizer = AutoTokenizer.from_pretrained(payload["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(payload["model_path"], local_files_only=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager")
    row = payload["request"]
    row["context"] = row.get("context") or ""
    row["system"] = row.get("system") or ""
    length = len(prompt_ids(tokenizer, row))
    limit = int(getattr(model.config, "max_position_embeddings", 2048))
    if length + payload["max_tokens"] > limit:
        raise ValueError(f"Prompt and output budget exceed model context {limit}")
    output = generate_rows(torch, model, tokenizer, [row], payload["max_tokens"])[0]
    print(json.dumps({"text": output, "runtime": {"elapsed_seconds": time.monotonic() - started,
        "prompt_tokens": length, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "local_files_only": True, "weights": "merged_safetensors", "decoding": "greedy_kv_cache"}}))


if __name__ == "__main__":
    main()
