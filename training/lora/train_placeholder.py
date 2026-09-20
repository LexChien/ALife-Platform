"""SFT preflight only: validate/serialize data and record lineage; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.datasets.contracts import PROFILES, load_rows, render_example


def prepare(profile, dataset, output, base_model_reference):
    rows = load_rows(dataset, profile)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    examples = [render_example(profile, row) for row in rows]
    target = output / "prepared.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
    manifest = {
        "status": "prepared_not_trained", "profile": profile, "rows": len(rows),
        "base_model_reference": base_model_reference, "base_model_reference_verified": False,
        "dataset": str(Path(dataset).resolve()),
        "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        "prepared_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "trained_weights": None,
        "next_step": "Confirm a trainable base model and license, then configure an actual SFT/LoRA trainer.",
    }
    (output / "preflight.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model-reference", required=True,
                        help="Declared trainable model source; preflight does not verify or download it")
    args = parser.parse_args()
    print(json.dumps(prepare(args.profile, args.dataset, args.output, args.base_model_reference), indent=2))


if __name__ == "__main__":
    main()
