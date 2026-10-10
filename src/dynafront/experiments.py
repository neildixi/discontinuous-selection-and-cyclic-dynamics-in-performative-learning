"""Experiment-family registry, job enumeration, and smoke configurations.

A *job* is one configuration cell (one checkpoint). It is not the same as a
*system*, an initial condition, or a retained coverage example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from dynafront.exp_a import enumerate_jobs as a_jobs
from dynafront.exp_a import execute as a_exec
from dynafront.exp_b import enumerate_jobs as b_jobs
from dynafront.exp_b import execute as b_exec
from dynafront.exp_c import enumerate_jobs as c_jobs
from dynafront.exp_c import execute as c_exec
from dynafront.exp_d import (
    enumerate_bounded_jobs,
    enumerate_gaussian_jobs,
    execute_bounded,
    execute_gaussian,
)
from dynafront.exp_delayed import enumerate_jobs as delayed_jobs
from dynafront.exp_delayed import execute as delayed_exec
from dynafront.exp_gaussian_exact import enumerate_jobs as gauss_jobs
from dynafront.exp_gaussian_exact import execute as gauss_exec
from dynafront.exp_hysteresis import enumerate_jobs as hys_jobs
from dynafront.exp_hysteresis import execute as hys_exec
from dynafront.exp_itinerary import enumerate_jobs as itin_jobs
from dynafront.exp_itinerary import execute as itin_exec
from dynafront.exp_tanh import enumerate_jobs as tanh_jobs
from dynafront.exp_tanh import execute as tanh_exec
from dynafront.exp_unaligned import enumerate_jobs as unaligned_jobs
from dynafront.exp_unaligned import execute as unaligned_exec

JobList = list[dict[str, Any]]
ExecFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Stage:
    name: str
    enumerate_jobs: Callable[[int], JobList]
    execute: ExecFn
    family: str
    in_paper: bool
    unit: str
    workers_hint: int = 2
    bench_index: int = 0


STAGES: tuple[Stage, ...] = (
    Stage("synthetic_a", a_jobs, a_exec, "aligned_scalar", True, "parameter cells (jobs)", 2, 1),
    Stage("synthetic_b", b_jobs, b_exec, "aligned_latent", False, "matrix/seed cells (jobs)", 2, 360),
    Stage("synthetic_c", c_jobs, c_exec, "asymmetric_scalar", True, "systems (one job per system)", 2, 0),
    Stage("synthetic_d_bounded", enumerate_bounded_jobs, execute_bounded, "bounded_noise", False, "parameter cells (jobs)", 1, 0),
    Stage("synthetic_d_gaussian", enumerate_gaussian_jobs, execute_gaussian, "gaussian_observability", False, "parameter cells (jobs)", 2, 0),
    Stage("unaligned", unaligned_jobs, unaligned_exec, "unaligned_two_resolvent", True, "jobs (each job holds many systems)", 2, 0),
    Stage("tanh", tanh_jobs, tanh_exec, "tanh_selector", True, "parameter cells (jobs)", 2, 1),
    Stage("itinerary", itin_jobs, itin_exec, "finite_itineraries", True, "sample jobs (unconditional + coverage)", 1, 0),
    Stage("hysteresis", hys_jobs, hys_exec, "hysteresis_deadband", True, "parameter cells (jobs)", 2, 0),
    Stage("delayed", delayed_jobs, delayed_exec, "delayed_clipping", False, "parameter cells (jobs)", 2, 0),
    Stage("gaussian_exact", gauss_jobs, gauss_exec, "gaussian_exact_marginals", False, "parameter cells (jobs)", 2, 0),
)

STAGE_BY_NAME = {s.name: s for s in STAGES}
PAPER_STAGES = tuple(s.name for s in STAGES if s.in_paper)
FULL_STAGES = tuple(s.name for s in STAGES)
EXPECTED_JOB_COUNTS = {
    "synthetic_a": 325,
    "synthetic_b": 540,
    "synthetic_c": 900,
    "synthetic_d_bounded": 15,
    "synthetic_d_gaussian": 75,
    "unaligned": 12,
    "tanh": 85,
    "itinerary": 2,
    "hysteresis": 110,
    "delayed": 65,
    "gaussian_exact": 27,
}


def stages_for_mode(mode: str) -> tuple[Stage, ...]:
    mode = str(mode).lower()
    if mode == "quick":
        return ()
    if mode == "paper":
        return tuple(STAGE_BY_NAME[n] for n in PAPER_STAGES)
    if mode == "full":
        return STAGES
    raise ValueError(f"unknown mode {mode!r}; expected quick, paper, or full")


def count_jobs(mode: str, master_seed: int) -> dict[str, int]:
    if mode == "quick":
        n = sum(len(jobs) for _, jobs, _ in smoke_jobs(master_seed))
        return {"quick_smoke": n, "total": n}
    out = {}
    total = 0
    for st in stages_for_mode(mode):
        n = len(st.enumerate_jobs(master_seed))
        out[st.name] = n
        total += n
    out["total"] = total
    return out


def _tiny(job: dict[str, Any], **kw: Any) -> dict[str, Any]:
    j = dict(job)
    j["config"] = dict(j["config"])
    j["config"].update(kw)
    j["job_id"] = str(j["job_id"]) + "__smoke"
    return j


def smoke_jobs(master_seed: int) -> list[tuple[str, JobList, ExecFn]]:
    """Deterministic reduced-size jobs used by ``quick`` mode."""
    ja = [_tiny(j, n_ic=12, n_steps=60, burnin=20) for j in a_jobs(master_seed)[:3]]
    return [
        ("smoke_a", ja, a_exec),
        ("smoke_b", [_tiny(b_jobs(master_seed)[0], n_ic=4, n_steps=40, burnin=20, d=5)], b_exec),
        ("smoke_c", [_tiny(c_jobs(master_seed)[0], n_ic=8, n_steps=80, burnin=40)], c_exec),
        ("smoke_d_bounded", [_tiny(enumerate_bounded_jobs(master_seed)[0])], execute_bounded),
        ("smoke_unaligned", [_tiny(unaligned_jobs(master_seed)[0], n_systems=2)], unaligned_exec),
        ("smoke_tanh", [_tiny(tanh_jobs(master_seed)[1], n_ic=16)], tanh_exec),
        ("smoke_hysteresis", [_tiny(hys_jobs(master_seed)[0], n_ic=16)], hys_exec),
        ("smoke_delayed", [_tiny(delayed_jobs(master_seed)[0], n_pair=16)], delayed_exec),
        ("smoke_itinerary", [_tiny(itin_jobs(master_seed)[0], n=8, scan_budget=40)], itin_exec),
        ("smoke_gaussian_exact", [_tiny(gauss_jobs(master_seed)[1], n_paths=1500, n_steps=40)], gauss_exec),
    ]
