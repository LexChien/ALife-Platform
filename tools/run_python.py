#!/usr/bin/env python3
"""Launch the project interpreter with NVIDIA wheel library paths resolved."""
from pathlib import Path
import os
import sys


def runtime_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    libraries = []
    for packages in (root / ".venv" / "lib").glob("python*/site-packages"):
        libraries.extend(str(path) for path in (packages / "nvidia").glob("*/lib") if path.is_dir())
    if libraries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + [env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("ANONYMIZED_TELEMETRY", "False")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def main():
    root = Path(__file__).resolve().parents[1]
    python = root / ".venv" / "bin" / "python"
    if not python.exists():
        raise SystemExit("Project .venv missing; follow docs/ENVIRONMENT.md")
    os.execve(str(python), [str(python), *sys.argv[1:]], runtime_environment(root))


if __name__ == "__main__":
    main()
