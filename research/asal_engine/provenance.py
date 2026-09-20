"""Checksummed local provenance for exact trajectory replay."""
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SCORE_ATOL = 1e-7


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    paths = []
    for folder in ("research/asal_engine", "foundation_models", "core"):
        paths.extend(path for path in (ROOT / folder).rglob("*.py") if "history" not in path.parts)
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}


def provenance_snapshot():
    def git(*args):
        proc = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True, timeout=10, check=False)
        return proc.stdout.strip() if proc.returncode == 0 else None
    versions = {}
    for package in ("numpy", "Pillow", "torch", "open_clip_torch", "imageio"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"git_revision": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "source_sha256": source_hashes(), "python": platform.python_version(),
            "platform": platform.platform(), "packages": versions}
