"""Shared job constructor."""

from __future__ import annotations

from typing import Any

from dynafront.hashing import config_hash, short_hash


def make_job(experiment: str, cfg: dict[str, Any], master_seed: int, seed: int = 0, tag: str = "") -> dict[str, Any]:
    ch = config_hash(cfg)
    extra = f"__{tag}" if tag else ""
    return {
        "job_id": f"{experiment}{extra}__{short_hash(cfg)}",
        "experiment": experiment,
        "config": cfg,
        "config_hash": ch,
        "seed": int(seed),
        "master_seed": master_seed,
    }
