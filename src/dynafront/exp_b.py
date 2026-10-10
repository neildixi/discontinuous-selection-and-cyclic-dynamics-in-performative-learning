"""Synthetic experiment B: latent-dimension orbit verification."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynafront.dynamics import make_latent_matrix, simulate_latent
from dynafront.hashing import config_hash, short_hash
from dynafront.metrics import finite_stat
from dynafront.seeds import default_rng
from dynafront.theory import parity_states, tau_c

DIMS = (5, 20, 100)
RHOS = (0.1, 0.5, 0.9)
LATENT = (("fast", 0.2), ("mixed", 0.6), ("slow", 0.9))
N_MATRIX = 20
N_IC = 20
N_STEPS = 400
BURNIN = 200


def enumerate_jobs(master_seed: int) -> list[dict[str, Any]]:
    jobs = []
    for d in DIMS:
        for rho in RHOS:
            for name, radius in LATENT:
                for mseed in range(N_MATRIX):
                    cfg = {
                        "d": int(d),
                        "rho": float(rho),
                        "latent_regime": name,
                        "latent_radius": float(radius),
                        "matrix_seed": int(mseed),
                        "n_ic": N_IC,
                        "n_steps": N_STEPS,
                        "burnin": BURNIN,
                    }
                    ch = config_hash(cfg)
                    jobs.append(
                        {
                            "job_id": f"synthetic_b__d{d}__rho{rho}__{name}__m{mseed}__{short_hash(cfg)}",
                            "experiment": "synthetic_b",
                            "config": cfg,
                            "config_hash": ch,
                            "seed": mseed,
                            "master_seed": master_seed,
                        }
                    )
    return jobs


def execute(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    rng = default_rng(job["master_seed"], "synthetic_b", job["config_hash"], cfg["matrix_seed"], "matrix")
    A, u, v = make_latent_matrix(rng, cfg["d"], cfg["rho"], cfg["latent_radius"])
    cond = float(np.linalg.cond(np.eye(cfg["d"]) + A))
    spec_perp = float(np.linalg.norm(A - cfg["rho"] * np.outer(u, u), 2))
    warnings = []
    if cond > 1e8:
        warnings.append(f"ill_conditioned_IplusA:{cond:.3e}")
    if spec_perp >= 1.0:
        warnings.append(f"A_perp_not_strictly_contractive:{spec_perp:.6f}")
    x_plus, x_minus = parity_states(A, v)
    rng_x = default_rng(job["master_seed"], "synthetic_b", job["config_hash"], cfg["matrix_seed"], "x0")
    x0 = rng_x.normal(size=(cfg["n_ic"], cfg["d"]))
    import time as _time

    t0 = _time.time()
    x, z, a = simulate_latent(x0, A, u, v, cfg["n_steps"])
    runtime_sim = _time.time() - t0
    post_x = x[cfg["burnin"] :]  # [K, N, d]
    post_z = z[cfg["burnin"] :]
    z_plus = float(u @ x_plus)
    z_minus = float(u @ x_minus)
    t = tau_c(cfg["rho"], 1.0)
    # Parity: even/odd relative to first post-burnin index that is positive-score
    # Use action: when a=+1, state should approach x+; next state x-.
    # After burn-in, match each state to nearest parity vector.
    if post_x.size == 0:
        rel = np.full((0, cfg["n_ic"]), np.nan)
        score_err = np.full(cfg["n_ic"], np.nan)
    else:
        d_plus = np.linalg.norm(post_x - x_plus, axis=2)
        d_minus = np.linalg.norm(post_x - x_minus, axis=2)
        rel = np.minimum(d_plus, d_minus) / (np.linalg.norm(x_plus) + 1e-15)
        score_err = np.mean(np.minimum(np.abs(post_z - z_plus), np.abs(post_z - z_minus)), axis=0)
    # Empirical two-step state contraction vs ||A||^2 on deviations from the matched parity state.
    K = post_x.shape[0]
    emp_rates = []
    for i in range(cfg["n_ic"]):
        seq = post_x[:, i]
        # positive-score subsequence
        scores = z[cfg["burnin"] :, i]
        pos = np.where(scores[:-2] > 0)[0]
        if pos.size < 3:
            emp_rates.append(np.nan)
            continue
        e0 = seq[pos] - x_plus
        e2 = seq[pos + 2] - x_plus
        n0 = np.linalg.norm(e0, axis=1)
        n2 = np.linalg.norm(e2, axis=1)
        ok = n0 > 1e-10
        emp_rates.append(float(np.median(n2[ok] / n0[ok])) if np.any(ok) else np.nan)
    theory_rate = float(np.linalg.norm(A, 2) ** 2)
    metrics = {
        "d": cfg["d"],
        "rho": cfg["rho"],
        "latent_regime": cfg["latent_regime"],
        "latent_radius": cfg["latent_radius"],
        "matrix_seed": cfg["matrix_seed"],
        "rel_state_orbit_error": finite_stat(rel[-20:], np.mean),
        "score_orbit_error": finite_stat(score_err, np.mean),
        "emp_two_step": finite_stat(emp_rates, np.median),
        "theory_two_step": theory_rate,
        "tau_c": t,
        "z_plus": z_plus,
        "z_minus": z_minus,
        "cond_IplusA": cond,
        "A_perp_norm": spec_perp,
        "sim_runtime_sec": runtime_sim,
        "warnings": ";".join(warnings),
        "uTv": float(u @ v),
        "alignment_resid": float(np.linalg.norm(u @ A - cfg["rho"] * u)),
    }
    return {"metrics": metrics}
