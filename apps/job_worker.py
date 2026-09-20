"""Execute one service job in isolation using the same engines as the CLIs."""
import argparse
import json
from pathlib import Path

from core.logger import build_run_summary, iso_now, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    target, config = request["target"], request["resolved_config"]
    directory = Path(request["run_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "resolved_config.json", config)
    started = iso_now()
    if target == "asal":
        from research.asal_engine.engine import ASALEngine
        result = ASALEngine(config, directory).run()
    elif target == "clone":
        from digital_clone.engine import DigitalCloneEngine
        result = DigitalCloneEngine(config, directory).run()
    elif target == "genai":
        from genai.multimodal.engine import GenAIEngine
        result = GenAIEngine(config, directory).run()
    else:
        raise ValueError(f"Unknown target: {target}")
    save_json(directory / "result.json", result)
    summary = build_run_summary(system=target, run_dir=directory,
                                config_path=request["config"], mode="service_job",
                                started_at=started, completed_at=iso_now(),
                                details=result.get("summary", result),
                                artifacts={"result": "result.json", "config": "resolved_config.json"})
    save_json(directory / "summary.json", summary)


if __name__ == "__main__":
    main()
