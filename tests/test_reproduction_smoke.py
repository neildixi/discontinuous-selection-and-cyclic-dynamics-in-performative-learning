"""Smoke: enumerate jobs, refuse cached overwrites, run one tiny job."""

from __future__ import annotations

from pathlib import Path

import pytest

from dynafront.experiments import smoke_jobs
from dynafront.paths import cached_dir, ensure_output_tree
from dynafront.runner import run_jobs
from dynafront.seeds import MASTER_SEED
from dynafront.summaries import reconstruct_summaries


def test_refuse_to_write_into_cached() -> None:
    with pytest.raises(ValueError, match="cached"):
        ensure_output_tree(cached_dir())


def test_smoke_jobs_are_marked() -> None:
    families = smoke_jobs(MASTER_SEED)
    n = sum(len(jobs) for _, jobs, _ in families)
    assert n >= 10
    for _stage, jobs, _exec in families:
        assert all(j["job_id"].endswith("__smoke") for j in jobs)


def test_one_smoke_job_checkpoints(tmp_path: Path) -> None:
    stage, jobs, execute = smoke_jobs(MASTER_SEED)[0]
    out = tmp_path / "run"
    result = run_jobs(
        jobs[:1],
        execute,
        stage=stage,
        out=out,
        workers=1,
        resume=False,
        max_hours=0.1,
        master_seed=MASTER_SEED,
        skip_projection_gate=True,
    )
    assert result.get("failed", 0) == 0
    ckpt = out / "checkpoints" / stage / f"{jobs[0]['job_id']}.json"
    assert ckpt.exists()
    reconstruct_summaries(out)
    # resume skips the completed job
    result2 = run_jobs(
        jobs[:1],
        execute,
        stage=stage,
        out=out,
        workers=1,
        resume=True,
        max_hours=0.1,
        master_seed=MASTER_SEED,
        skip_projection_gate=True,
    )
    assert result2.get("skipped", 0) >= 1
