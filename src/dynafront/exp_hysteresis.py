"""Phase 5: hysteresis versus neutral deadband on own normalized grids."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynafront.dynamics import initial_scores, simulate_hysteresis, simulate_scalar
from dynafront.jobs import make_job
from dynafront.metrics import dwell_stats, switch_rate
from dynafront import certificates as ora

RHOS = (0.0, 0.2, 0.5, 0.8, 0.95)
RATIOS = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.25, 1.5)
BETA = 1.0
N_IC = 80
MIN_SWITCHES = 20
MAX_H = 200_000


def enumerate_jobs(master_seed: int) -> list[dict[str, Any]]:
    jobs = []
    for rho in RHOS:
        t = ora.tau_c(rho, BETA)
        h_hys = ora.hysteresis_threshold(rho, BETA)
        for ratio in RATIOS:
            jobs.append(
                make_job(
                    "hysteresis",
                    {
                        "kind": "hysteresis",
                        "rho": float(rho),
                        "ratio": float(ratio),
                        "h": float(ratio * h_hys),
                        "h_hys": float(h_hys),
                        "tau_c": float(t),
                        "beta": BETA,
                        "n_ic": N_IC,
                    },
                    master_seed,
                    tag=f"hys_rho{rho}_r{ratio}",
                )
            )
            jobs.append(
                make_job(
                    "hysteresis",
                    {
                        "kind": "deadband",
                        "rho": float(rho),
                        "ratio": float(ratio),
                        "h": float(ratio * t),
                        "h_hys": float(h_hys),
                        "tau_c": float(t),
                        "beta": BETA,
                        "n_ic": N_IC,
                    },
                    master_seed,
                    tag=f"dead_rho{rho}_r{ratio}",
                )
            )
    return jobs


def _hysteresis_horizon(ratio: float, rho: float) -> int:
    if ratio >= 1.0:
        return 400
    dist = 1.0 - ratio
    if dist < 0.02:
        return 20000
    if dist < 0.1:
        return 8000
    return 2000


def execute(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    rho, h, beta = cfg["rho"], cfg["h"], cfg["beta"]
    ratio = cfg["ratio"]
    t = cfg["tau_c"]
    h_hys = cfg["h_hys"]
    z0 = initial_scores(cfg["n_ic"])
    # both previous actions for every score
    z0b = np.concatenate([z0, z0])
    a0 = np.concatenate([np.ones(z0.size), -np.ones(z0.size)])
    ratio_theory = ora.hysteresis_over_tau_c_ratio(rho)
    ratio_emp = h_hys / t if t > 0 else np.nan
    if cfg["kind"] == "deadband":
        z, a = simulate_scalar(z0, rho, 800, "deadband", beta=beta, h=h)
        term = np.abs(z[-1])
        from dynafront.metrics import classify_scalar_outcome

        cls = classify_scalar_outcome(z, a, rho, beta, 400)
        metrics = {
            "kind": "deadband",
            "rho": rho,
            "ratio": ratio,
            "h": h,
            "tau_c": t,
            "h_hys": h_hys,
            "threshold_ratio_theory": ratio_theory,
            "threshold_ratio_emp": ratio_emp,
            "frac_zero": float(np.mean(cls["to_zero"])),
            "frac_cycle": float(np.mean(cls["to_cycle"])),
            "frac_unresolved": float(np.mean(cls["unresolved"])),
            "mean_terminal": float(np.mean(term)),
            "regime": "super_or_critical_zero" if ratio >= 1.0 - 1e-15 else "subcritical_coexistence",
            "min_post_switches": np.nan,
            "max_future_switches": np.nan,
            "frac_at_most_one_switch": np.nan,
            "mean_dwell": np.nan,
            "max_dwell": np.nan,
            "horizon": 800,
            "censored": 0.0,
            "n_ic": int(z0.size),
        }
        return {"metrics": metrics}

    H0 = _hysteresis_horizon(ratio, rho)
    H = H0
    censored = 0.0
    z = a = None
    while True:
        z, a = simulate_hysteresis(z0b, a0, rho, h, H, beta=beta)
        sw_counts = np.sum(a[1:] != a[:-1], axis=0)
        # post-burn-in switches (last half)
        post = a[H // 2 :]
        post_sw = np.sum(post[1:] != post[:-1], axis=0)
        if ratio < 1.0 - 1e-15:
            if float(np.min(post_sw)) >= MIN_SWITCHES or H >= MAX_H:
                if float(np.min(post_sw)) < MIN_SWITCHES:
                    censored = 1.0
                break
            H = min(MAX_H, max(H * 2, H + 4000))
        else:
            break
    # future switches from t=0 including possible immediate switch
    n_switch = np.sum(a[1:] != a[:-1], axis=0)
    mean_dwell, max_dwell = dwell_stats(a[H // 4 :])
    # fixed-action equilibria: +1 -> z* = -beta/(1-rho) = -h_hys; -1 -> +h_hys
    z_plus_eq = -h_hys
    z_minus_eq = h_hys
    term = z[-1]
    d_fix = np.minimum(np.abs(term - z_plus_eq), np.abs(term - z_minus_eq))
    if ratio >= 1.0 - 1e-15:
        at_most_one = n_switch <= 1
        conv = d_fix <= 1e-6 * max(h_hys, 1.0)
        # last action frozen
        frozen = np.all(a[-20:] == a[-1], axis=0) if H >= 20 else np.ones(term.size, dtype=bool)
        success = float(np.mean(at_most_one & conv & frozen))
        regime = "critical_or_super_at_most_one_switch"
        min_post = float(np.min(post_sw))
        max_future = float(np.max(n_switch))
        frac_one = float(np.mean(at_most_one))
    else:
        success = float(np.mean(post_sw >= MIN_SWITCHES))
        regime = "subcritical_infinite_switches"
        min_post = float(np.min(post_sw))
        max_future = float(np.max(n_switch))
        frac_one = float(np.mean(n_switch <= 1))
    metrics = {
        "kind": "hysteresis",
        "rho": rho,
        "ratio": ratio,
        "h": h,
        "tau_c": t,
        "h_hys": h_hys,
        "threshold_ratio_theory": ratio_theory,
        "threshold_ratio_emp": ratio_emp,
        "frac_zero": np.nan,
        "frac_cycle": np.nan,
        "frac_unresolved": np.nan,
        "mean_terminal": float(np.mean(np.abs(term))),
        "regime": regime,
        "min_post_switches": min_post,
        "max_future_switches": max_future,
        "frac_at_most_one_switch": frac_one,
        "mean_dwell": float(np.mean(mean_dwell)),
        "max_dwell": float(np.max(max_dwell)),
        "success_frac": success,
        "horizon": H,
        "censored": censored,
        "n_ic": int(z0b.size),
        "rho_zero_equality_cell": float(rho == 0.0 and abs(ratio - 1.0) < 1e-15),
    }
    return {"metrics": metrics}
