"""Repository-relative paths. No absolute machine paths are stored."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    """Return the public repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


REPO_ROOT = repo_root()
CONFIGS_DIR = REPO_ROOT / "configs"
CACHED_DIR = REPO_ROOT / "results" / "cached"
GENERATED_DIR = REPO_ROOT / "results" / "generated"
RUNS_DIR = REPO_ROOT / "results" / "runs"


def cached_dir() -> Path:
    return repo_root() / "results" / "cached"


def generated_dir() -> Path:
    return repo_root() / "results" / "generated"


def generated_figures_dir() -> Path:
    return generated_dir() / "figures"


def configs_dir() -> Path:
    return repo_root() / "configs"


def ensure_output_tree(out: Path) -> Path:
    """Create the run-directory layout. Never points at ``results/cached``."""
    out = Path(out)
    cached = cached_dir().resolve()
    if out.resolve() == cached or cached in out.resolve().parents:
        raise ValueError("refusing to write a fresh run into results/cached/")
    for name in (
        "manifests",
        "checkpoints",
        "raw",
        "summaries",
        "figures",
        "logs",
        "reports",
        "status",
        "tables",
        "config",
    ):
        (out / name).mkdir(parents=True, exist_ok=True)
    return out


def new_run_dir(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = repo_root() / "results" / "runs" / f"{mode}_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest
