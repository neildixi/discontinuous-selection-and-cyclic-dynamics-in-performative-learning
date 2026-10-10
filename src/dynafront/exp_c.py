"""Synthetic experiment C: asymmetric certificate specificity."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynafront.dynamics import simulate_asymmetric_scalar
from dynafront.hashing import config_hash, short_hash
from dynafront.metrics import detect_action_period, dwell_stats, finite_stat, switch_rate
from dynafront.seeds import default_rng
from dynafront.theory import (
    cross_image_conditions,
    equilibria_crossed,
    fixed_action_score_equilibrium,
    pair_sign_compatible,
    predicted_alternating_pair,
)

N_SYSTEMS = 300
N_IC = 20
N_STEPS = 500
BURNIN = 250
MAX_PERIOD = 16
STRATA = ("crossed_sign_compatible", "crossed_no_pair", "noncrossed")


def _classify_system(rho_p, rho_m, b_p, b_m) -> str:
    p_p = fixed_action_score_equilibrium(rho_p, b_p)
    p_m = fixed_action_score_equilibrium(rho_m, b_m)
    z_p, z_m = predicted_alternating_pair(rho_p, rho_m, b_p, b_m)
    crossed = equilibria_crossed(p_p, p_m)
    compatible = pair_sign_compatible(z_p, z_m)
    # Cross-image is equivalent to compatible pair under affine responses.
    if crossed and compatible:
        return "crossed_sign_compatible"
    if crossed and not compatible:
        return "crossed_no_pair"
    return "noncrossed"


def sample_systems(master_seed: int, config_hash_s: str) -> dict[str, list[dict[str, float]]]:
    rng = default_rng(master_seed, "synthetic_c", config_hash_s, 0, "rejection")
    buckets: dict[str, list[dict[str, float]]] = {s: [] for s in STRATA}
    attempts = 0
    max_attempts = 2_000_000
    while any(len(buckets[s]) < N_SYSTEMS for s in STRATA) and attempts < max_attempts:
        attempts += 1
        rho_p = float(rng.uniform(0.2, 0.9))
        rho_m = float(rng.uniform(0.2, 0.9))
        b_p = float(rng.uniform(-3.0, 3.0))
        b_m = float(rng.uniform(-3.0, 3.0))
        if abs(1.0 - rho_p * rho_m) < 1e-6:
            continue
        label = _classify_system(rho_p, rho_m, b_p, b_m)
        if len(buckets[label]) >= N_SYSTEMS:
            continue
        p_p = fixed_action_score_equilibrium(rho_p, b_p)
        p_m = fixed_action_score_equilibrium(rho_m, b_m)
        z_p, z_m = predicted_alternating_pair(rho_p, rho_m, b_p, b_m)
        buckets[label].append(
            {
                "rho_plus": rho_p,
                "rho_minus": rho_m,
                "b_plus": b_p,
                "b_minus": b_m,
                "p_plus": p_p,
                "p_minus": p_m,
                "z_plus": z_p,
                "z_minus": z_m,
                "crossed": float(equilibria_crossed(p_p, p_m)),
                "sign_compatible": float(pair_sign_compatible(z_p, z_m)),
                "cross_image": float(cross_image_conditions(rho_p, rho_m, b_p, b_m)),
            }
        )
    return buckets


def enumerate_jobs(master_seed: int) -> list[dict[str, Any]]:
    meta = {"n_systems": N_SYSTEMS, "n_ic": N_IC, "n_steps": N_STEPS, "burnin": BURNIN}
    ch_meta = config_hash(meta)
    buckets = sample_systems(master_seed, ch_meta)
    jobs = []
    for stratum, systems in buckets.items():
        for i, sys in enumerate(systems):
            cfg = {**meta, **sys, "stratum": stratum, "system_index": i}
            ch = config_hash(cfg)
            jobs.append(
                {
                    "job_id": f"synthetic_c__{stratum}__{i:03d}__{short_hash(cfg)}",
                    "experiment": "synthetic_c",
                    "config": cfg,
                    "config_hash": ch,
                    "seed": i,
                    "master_seed": master_seed,
                }
            )
    return jobs


def execute(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    rng = default_rng(job["master_seed"], "synthetic_c", job["config_hash"], cfg["system_index"], "z0")
    z0 = rng.uniform(-5.0, 5.0, size=cfg["n_ic"])
    z, a = simulate_asymmetric_scalar(
        z0, cfg["rho_plus"], cfg["rho_minus"], cfg["b_plus"], cfg["b_minus"], cfg["n_steps"]
    )
    z_post = z[cfg["burnin"] :]
    a_post = a[cfg["burnin"] :]
    periods = detect_action_period(a_post, MAX_PERIOD)
    mean_dwell, max_dwell = dwell_stats(a_post)
    sw = switch_rate(a_post)
    zp, zm = cfg["z_plus"], cfg["z_minus"]
    # Distance of even/odd tail to predicted pair, assigned by sign.
    tail = z_post[-40:]
    d_pair = np.mean(np.minimum(np.abs(tail - zp), np.abs(tail + 0) * 0 + np.abs(tail - zm)), axis=0)
    pos_vals = []
    neg_vals = []
    for i in range(cfg["n_ic"]):
        seq = tail[:, i]
        pos_vals.append(np.mean(seq[seq >= 0]) if np.any(seq >= 0) else np.nan)
        neg_vals.append(np.mean(seq[seq < 0]) if np.any(seq < 0) else np.nan)
    pos_vals = np.asarray(pos_vals)
    neg_vals = np.asarray(neg_vals)
    pair_err = finite_stat(np.concatenate([np.abs(pos_vals - zp), np.abs(neg_vals - zm)]), np.mean)
    # Empirical two-step factor on positive subsequence vs rho+ rho-
    emp = []
    for i in range(cfg["n_ic"]):
        seq = z_post[:, i]
        pos = np.where(seq[:-2] >= 0)[0]
        if pos.size < 4:
            emp.append(np.nan)
            continue
        den = seq[pos] - zp
        num = seq[pos + 2] - zp
        ok = np.abs(den) > 1e-8
        emp.append(float(np.median(num[ok] / den[ok])) if np.any(ok) else np.nan)
    # Fixed-action consistency for stratum 3: last action matches sign of last score and no switch
    last_a = a[-1]
    last_z = z[-1]
    self_consistent = ((last_z >= 0) & (last_a > 0) & (np.abs(last_z - cfg["p_plus"]) < 0.05)) | (
        (last_z < 0) & (last_a < 0) & (np.abs(last_z - cfg["p_minus"]) < 0.05)
    )
    metrics = {
        "stratum": cfg["stratum"],
        "system_index": cfg["system_index"],
        "rho_plus": cfg["rho_plus"],
        "rho_minus": cfg["rho_minus"],
        "b_plus": cfg["b_plus"],
        "b_minus": cfg["b_minus"],
        "p_plus": cfg["p_plus"],
        "p_minus": cfg["p_minus"],
        "z_plus": cfg["z_plus"],
        "z_minus": cfg["z_minus"],
        "crossed": cfg["crossed"],
        "sign_compatible": cfg["sign_compatible"],
        "cross_image": cfg["cross_image"],
        "frac_period_two": float(np.mean(periods == 2)),
        "frac_period_one": float(np.mean(periods == 1)),
        "frac_other_period": float(np.mean((periods > 2))),
        "frac_unresolved_period": float(np.mean(periods == 0)),
        "mean_detected_period": float(np.mean(periods[periods > 0])) if np.any(periods > 0) else np.nan,
        "switch_rate_mean": float(np.mean(sw)),
        "mean_dwell": float(np.mean(mean_dwell)),
        "max_dwell": float(np.max(max_dwell)),
        "pair_err": float(pair_err),
        "emp_two_step": finite_stat(emp, np.median),
        "theory_two_step": float(cfg["rho_plus"] * cfg["rho_minus"]),
        "frac_self_consistent_fixed": float(np.mean(self_consistent)),
        "global_pair_all_ic": float(np.mean(periods == 2) >= 0.95 and pair_err < 0.05),
    }
    return {"metrics": metrics}
