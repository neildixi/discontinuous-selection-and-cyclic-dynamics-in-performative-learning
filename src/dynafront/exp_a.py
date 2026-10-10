"""Synthetic experiment A: exact scalar phase diagram."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynafront.dynamics import initial_scores, simulate_scalar, two_step_rate_on_hard_orbit
from dynafront.hashing import config_hash, short_hash
from dynafront.metrics import classify_scalar_outcome, finite_stat
from dynafront.selectors import ratio_grid
from dynafront.theory import tau_c

RHOS = (0.1, 0.3, 0.5, 0.7, 0.9)
N_IC = 100
N_STEPS = 400
BURNIN = 200
BETA = 1.0
RATIOS = ratio_grid(31, 0.25, 2.0)


def enumerate_jobs(master_seed: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for rho in RHOS:
        cfg = {"selector": "hard", "rho": float(rho), "beta": BETA, "n_ic": N_IC, "n_steps": N_STEPS, "burnin": BURNIN}
        jobs.append(_job("synthetic_a", cfg, master_seed, seed=0))
        for ratio in RATIOS:
            cfg_c = {
                "selector": "continuous",
                "rho": float(rho),
                "beta": BETA,
                "ratio": float(ratio),
                "tau": float(ratio * tau_c(rho, BETA)),
                "n_ic": N_IC,
                "n_steps": N_STEPS,
                "burnin": BURNIN,
            }
            jobs.append(_job("synthetic_a", cfg_c, master_seed, seed=0))
            cfg_d = {
                "selector": "deadband",
                "rho": float(rho),
                "beta": BETA,
                "ratio": float(ratio),
                "h": float(ratio * tau_c(rho, BETA)),
                "n_ic": N_IC,
                "n_steps": N_STEPS,
                "burnin": BURNIN,
            }
            jobs.append(_job("synthetic_a", cfg_d, master_seed, seed=0))
    return jobs


def _job(experiment: str, cfg: dict[str, Any], master_seed: int, seed: int) -> dict[str, Any]:
    ch = config_hash(cfg)
    return {
        "job_id": f"{experiment}__{cfg['selector']}__rho{cfg['rho']:.3f}__{short_hash(cfg)}",
        "experiment": experiment,
        "config": cfg,
        "config_hash": ch,
        "seed": seed,
        "master_seed": master_seed,
    }


def execute(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    z0 = initial_scores(cfg["n_ic"])
    selector = cfg["selector"]
    z, a = simulate_scalar(
        z0,
        cfg["rho"],
        cfg["n_steps"],
        selector,
        beta=cfg["beta"],
        tau=cfg.get("tau"),
        h=cfg.get("h"),
    )
    cls = classify_scalar_outcome(z, a, cfg["rho"], cfg["beta"], cfg["burnin"])
    rates = two_step_rate_on_hard_orbit(z, cfg["rho"], cfg["beta"], cfg["burnin"])
    metrics = {
        "rho": cfg["rho"],
        "selector": selector,
        "ratio": cfg.get("ratio", np.nan),
        "tau": cfg.get("tau", np.nan),
        "h": cfg.get("h", np.nan),
        "tau_c": tau_c(cfg["rho"], cfg["beta"]),
        "A_z_mean": float(np.mean(cls["A_z"])),
        "A_z_norm_mean": float(np.mean(cls["A_z_norm"])),
        "D2_mean": float(np.mean(cls["D2"])),
        "D2_max": float(np.max(cls["D2"])),
        "terminal_abs_mean": float(np.mean(cls["terminal_abs"])),
        "switch_rate_mean": float(np.mean(cls["switch_rate"])),
        "two_step_rate_median": finite_stat(rates, np.median),
        "frac_zero": float(np.mean(cls["to_zero"])),
        "frac_cycle": float(np.mean(cls["to_cycle"])),
        "frac_unresolved": float(np.mean(cls["unresolved"])),
        "n_ic": int(z0.size),
    }
    return {
        "metrics": metrics,
        "arrays": {
            "z0": z0,
            "z_final": z[-1],
            "A_z": cls["A_z"],
            "A_z_norm": cls["A_z_norm"],
            "D2": cls["D2"],
            "terminal_abs": cls["terminal_abs"],
            "switch_rate": cls["switch_rate"],
            "two_step_rate": rates,
            "to_zero": cls["to_zero"],
            "to_cycle": cls["to_cycle"],
            "unresolved": cls["unresolved"],
        },
    }
