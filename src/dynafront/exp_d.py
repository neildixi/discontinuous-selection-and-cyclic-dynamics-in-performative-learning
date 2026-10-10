"""Synthetic experiment D: bounded tubes vs Gaussian observability."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynafront.dynamics import simulate_scalar_noisy
from dynafront.hashing import config_hash, short_hash
from dynafront.metrics import (
    amplitude,
    distance_to_hard_orbit,
    false_boundary_crossings,
    finite_stat,
    lag_corr,
    period_two_spectral_share,
    switch_rate,
)
from dynafront.seeds import default_rng
from dynafront.theory import bounded_noise_margin, tau_c

RHOS_B = (0.1, 0.5, 0.9)
N_STEPS_B = 400
BURNIN_B = 200
N_IC_B = 40

RHOS_G = (0.1, 0.5, 0.9)
LEVELS = (0.0, 0.05, 0.1, 0.25, 0.5)
N_SEEDS_G = 200
N_STEPS_G = 600
BURNIN_G = 300


def enumerate_bounded_jobs(master_seed: int) -> list[dict[str, Any]]:
    jobs = []
    # Settings on both sides of eps + delta/(1-rho) < tau_c
    for rho in RHOS_B:
        t = tau_c(rho, 1.0)
        settings = [
            ("inside", 0.1 * t, 0.1 * (1 - rho) * t),
            ("inside_tight", 0.25 * t, 0.25 * (1 - rho) * t),
            ("boundary_minus", 0.4 * t, 0.55 * (1 - rho) * t),
            ("outside", 0.6 * t, 0.6 * (1 - rho) * t),
            ("outside_large", 0.9 * t, 0.9 * (1 - rho) * t),
        ]
        for name, eps, delta in settings:
            cfg = {
                "rho": float(rho),
                "eps": float(eps),
                "delta": float(delta),
                "setting": name,
                "n_ic": N_IC_B,
                "n_steps": N_STEPS_B,
                "burnin": BURNIN_B,
                "kind": "bounded",
            }
            ch = config_hash(cfg)
            jobs.append(
                {
                    "job_id": f"synthetic_d_bounded__rho{rho}__{name}__{short_hash(cfg)}",
                    "experiment": "synthetic_d_bounded",
                    "config": cfg,
                    "config_hash": ch,
                    "seed": 0,
                    "master_seed": master_seed,
                }
            )
    return jobs


def enumerate_gaussian_jobs(master_seed: int) -> list[dict[str, Any]]:
    jobs = []
    for rho in RHOS_G:
        t = tau_c(rho, 1.0)
        for lo in LEVELS:
            for lt in LEVELS:
                cfg = {
                    "rho": float(rho),
                    "obs_level": float(lo),
                    "trans_level": float(lt),
                    "sigma_obs": float(lo * t),
                    "sigma_trans": float(lt * t),
                    "n_seeds": N_SEEDS_G,
                    "n_steps": N_STEPS_G,
                    "burnin": BURNIN_G,
                    "kind": "gaussian",
                }
                ch = config_hash(cfg)
                jobs.append(
                    {
                        "job_id": f"synthetic_d_gauss__rho{rho}__o{lo}__t{lt}__{short_hash(cfg)}",
                        "experiment": "synthetic_d_gaussian",
                        "config": cfg,
                        "config_hash": ch,
                        "seed": 0,
                        "master_seed": master_seed,
                    }
                )
    return jobs


def execute_bounded(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    rho, eps, delta = cfg["rho"], cfg["eps"], cfg["delta"]
    margin = bounded_noise_margin(rho, eps, delta)
    rng = default_rng(job["master_seed"], "synthetic_d_bounded", job["config_hash"], 0, "noise")
    n, T = cfg["n_ic"], cfg["n_steps"]
    t = tau_c(rho, 1.0)
    # Start near the positive orbit point.
    z0 = np.full(n, t) + rng.uniform(-0.05 * t, 0.05 * t, size=n)
    eps_arr = rng.uniform(-eps, eps, size=(T, n))
    eta = np.zeros((T + 1, n))
    eta[1:] = rng.uniform(-delta, delta, size=(T, n))
    z, a_obs, a_true = simulate_scalar_noisy(z0, rho, T, eps=eps_arr, eta=eta)
    R = float(margin["R"])
    post = z[cfg["burnin"] :]
    in_plus = (post >= t - R) & (post <= t + R)
    in_minus = (post >= -t - R) & (post <= -t + R)
    in_tubes = in_plus | in_minus
    # Alternation: consecutive post points should occupy opposite tubes once inside.
    alt_ok = []
    for i in range(n):
        seq = post[:, i]
        inside = in_tubes[:, i]
        if not np.all(inside):
            alt_ok.append(False)
            continue
        signs = np.sign(seq)
        alt_ok.append(bool(np.all(signs[1:] == -signs[:-1]) and np.all(np.abs(seq) >= t - R - 1e-12)))
    metrics = {
        **{k: (float(v) if not isinstance(v, bool) else v) for k, v in margin.items()},
        "rho": rho,
        "eps": eps,
        "delta": delta,
        "setting": cfg["setting"],
        "frac_in_tubes": float(np.mean(in_tubes)),
        "frac_traj_always_in_tubes": float(np.mean(np.all(in_tubes, axis=0))),
        "frac_alternating_in_tubes": float(np.mean(alt_ok)),
        "mean_dist_orbit": float(np.mean(distance_to_hard_orbit(post, rho))),
        "sufficient_bound": bool(margin["sufficient"]),
        "claim": "sufficient_forward_invariance_check_not_necessity",
    }
    return {"metrics": metrics}


def execute_gaussian(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    rho = cfg["rho"]
    n, T = cfg["n_seeds"], cfg["n_steps"]
    t = tau_c(rho, 1.0)
    rng = default_rng(job["master_seed"], "synthetic_d_gaussian", job["config_hash"], 0, "noise")
    z0 = np.full(n, t)
    eps = rng.normal(0.0, cfg["sigma_obs"], size=(T, n)) if cfg["sigma_obs"] > 0 else np.zeros((T, n))
    eta = np.zeros((T + 1, n))
    if cfg["sigma_trans"] > 0:
        eta[1:] = rng.normal(0.0, cfg["sigma_trans"], size=(T, n))
    z, a_obs, a_true = simulate_scalar_noisy(z0, rho, T, eps=eps, eta=eta)
    post = z[cfg["burnin"] :]
    a_post = a_obs[cfg["burnin"] :]
    sw = switch_rate(a_post)
    amp = amplitude(post)
    d2 = distance_to_hard_orbit(post, rho)
    c1 = lag_corr(post, 1)
    c2 = lag_corr(post, 2)
    p2 = period_two_spectral_share(post)
    false_x = false_boundary_crossings(a_obs, a_true)
    # Seed-level percentile CIs
    def _ci(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return (np.nan, np.nan, np.nan)
        lo, mid, hi = np.quantile(x, [0.025, 0.5, 0.975])
        return float(lo), float(mid), float(hi)

    metrics = {
        "rho": rho,
        "obs_level": cfg["obs_level"],
        "trans_level": cfg["trans_level"],
        "sigma_obs": cfg["sigma_obs"],
        "sigma_trans": cfg["sigma_trans"],
        "tau_c": t,
        "switch_rate_mean": float(np.mean(sw)),
        "switch_rate_lo": _ci(sw)[0],
        "switch_rate_hi": _ci(sw)[2],
        "lag1_mean": finite_stat(c1, np.mean),
        "lag1_lo": _ci(c1)[0],
        "lag1_hi": _ci(c1)[2],
        "lag2_mean": finite_stat(c2, np.mean),
        "lag2_lo": _ci(c2)[0],
        "lag2_hi": _ci(c2)[2],
        "amplitude_mean": float(np.mean(amp)),
        "amplitude_lo": _ci(amp)[0],
        "amplitude_hi": _ci(amp)[2],
        "D2_mean": float(np.mean(d2)),
        "D2_lo": _ci(d2)[0],
        "D2_hi": _ci(d2)[2],
        "false_cross_mean": float(np.mean(false_x)),
        "false_cross_lo": _ci(false_x)[0],
        "false_cross_hi": _ci(false_x)[2],
        "p2_share_mean": float(np.mean(p2)),
        "p2_share_lo": _ci(p2)[0],
        "p2_share_hi": _ci(p2)[2],
        "description": "gaussian_observability_not_bounded_theorem_validation",
    }
    return {"metrics": metrics, "arrays": {"switch_rate": sw, "p2_share": p2, "D2": d2}}
