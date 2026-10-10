"""Phase 6: delayed clipping, proved vs exploratory regions."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynafront.dynamics import simulate_delayed
from dynafront.jobs import make_job
from dynafront import certificates as ora

RHOS = (0.0, 0.2, 0.5, 0.8, 0.95)
K_BASE = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0, 1.01, 1.1, 1.25)
BETA = 1.0
N_BROAD = 81
N_RAND = 40
N_SMALL = 40


def _k_values(rho: float) -> list[float]:
    cut = 1.0 - float(rho)
    extra = [0.5 * max(cut, 0.02), max(0.02, cut - 0.04), cut, min(1.4, cut + 0.04)]
    vals = sorted({round(float(k), 10) for k in list(K_BASE) + extra if k > 0})
    return vals


def enumerate_jobs(master_seed: int) -> list[dict[str, Any]]:
    jobs = []
    for rho in RHOS:
        for k in _k_values(rho):
            tau = BETA / k
            cfg = {
                "rho": float(rho),
                "k": float(k),
                "tau": float(tau),
                "beta": BETA,
                "region": ora.delayed_region(rho, k),
            }
            jobs.append(make_job("delayed", cfg, master_seed, tag=f"rho{rho}_k{k}"))
    return jobs


def _grid_pairs(n: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    g = np.linspace(lo, hi, int(np.sqrt(n)))
    z0, zm1 = np.meshgrid(g, g)
    return z0.ravel()[:n], zm1.ravel()[:n]


def execute(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    rho, k, tau, beta = cfg["rho"], cfg["k"], cfg["tau"], cfg["beta"]
    region = ora.delayed_region(rho, k)
    roots = ora.delayed_jury_roots(rho, k)
    spec = float(np.max(np.abs(roots)))
    lip_delay = rho + k
    tau_c_inst = ora.tau_c(rho, beta)
    local_thr = beta  # delayed local threshold tau = beta <=> k=1

    z0b, zm1b = _grid_pairs(N_BROAD, -8.0, 8.0)
    # proved / unresolved: long horizon
    if region == "proved_global":
        H = 40000 if lip_delay >= 0.97 else 2500
        z = simulate_delayed(z0b, zm1b, rho, tau, H, beta=beta)
        mag = np.maximum(np.abs(z[-1]), np.abs(z[-2]))
        conv = mag < 1e-8
        pair0 = np.maximum(np.abs(z0b), np.abs(zm1b))
        pairT = mag
        contracted = pairT <= pair0 + 1e-12
        metrics_extra = {
            "frac_to_zero": float(np.mean(conv)),
            "max_terminal": float(np.max(mag)),
            "frac_sup_bound": float(np.mean(contracted)),
            "frac_growth_small": np.nan,
            "max_small_gain": np.nan,
            "label": "proved_global",
            "proved_gate": float(np.mean(conv) >= 1.0 - 1e-12),
            "exploratory": 0.0,
        }
    elif region == "locally_stable_globally_unresolved":
        H = 8000
        from dynafront.seeds import default_rng

        rng = default_rng(job["master_seed"], "delayed", job["config_hash"], 0, "rand")
        z0r = rng.uniform(-12, 12, size=N_RAND)
        zm1r = rng.uniform(-12, 12, size=N_RAND)
        z0 = np.concatenate([z0b, z0r])
        zm1 = np.concatenate([zm1b, zm1r])
        z = simulate_delayed(z0, zm1, rho, tau, H, beta=beta)
        mag = np.maximum(np.abs(z[-1]), np.abs(z[-2]))
        conv = mag < 1e-6
        decreasing = mag < np.maximum(np.abs(z[0]), np.abs(np.concatenate([zm1b, zm1r])))
        metrics_extra = {
            "frac_to_zero": float(np.mean(conv)),
            "max_terminal": float(np.max(mag)),
            "frac_sup_bound": np.nan,
            "frac_growth_small": np.nan,
            "max_small_gain": np.nan,
            "frac_decreasing": float(np.mean(decreasing)),
            "label": "exploratory_unresolved",
            "proved_gate": np.nan,
            "exploratory": 1.0,
        }
    elif region == "critical_not_locally_asymptotically_stable":
        H = 400
        # small linear-region pairs: |z| << tau
        amp = 1e-4 * tau
        z0s = amp * np.linspace(-1, 1, N_SMALL)
        zm1s = amp * np.linspace(1, -1, N_SMALL)
        z = simulate_delayed(z0s, zm1s, rho, tau, H, beta=beta)
        mag = np.maximum(np.abs(z[-1]), np.abs(z[-2]))
        mag0 = np.maximum(np.abs(z0s), np.abs(zm1s))
        # nondecaying: terminal not << initial
        gain = mag / (mag0 + 1e-18)
        metrics_extra = {
            "frac_to_zero": float(np.mean(mag < 1e-12)),
            "max_terminal": float(np.max(mag)),
            "frac_sup_bound": np.nan,
            "frac_growth_small": float(np.mean(gain > 0.5)),
            "max_small_gain": float(np.max(gain)),
            "median_small_gain": float(np.median(gain)),
            "label": "critical_not_locally_AS",
            "proved_gate": float(np.mean(mag < 1e-12) < 0.5),
            "exploratory": 0.0,
        }
    else:
        H = 80
        amp = 1e-4 * tau
        z0s = amp * np.linspace(-1, 1, N_SMALL)
        zm1s = 0.3 * amp * np.ones(N_SMALL)
        z = simulate_delayed(z0s, zm1s, rho, tau, H, beta=beta)
        mag = np.maximum(np.abs(z[-1]), np.abs(z[-2]))
        mag0 = np.maximum(np.abs(z0s), np.abs(zm1s))
        # growth before clipping: early steps
        early = np.maximum(np.abs(z[min(20, H)]), np.abs(z[min(19, H)]))
        gain = early / (mag0 + 1e-18)
        metrics_extra = {
            "frac_to_zero": float(np.mean(mag < 1e-8)),
            "max_terminal": float(np.max(mag)),
            "frac_sup_bound": np.nan,
            "frac_growth_small": float(np.mean(gain > 1.05)),
            "max_small_gain": float(np.max(gain)),
            "label": "locally_unstable",
            "proved_gate": float(np.max(gain) > 1.0),
            "exploratory": 0.0,
        }
    metrics = {
        "rho": rho,
        "k": k,
        "tau": tau,
        "region": region,
        "spectral_radius": spec,
        "rho_plus_k": lip_delay,
        "tau_c_instantaneous": tau_c_inst,
        "tau_local_delayed": local_thr,
        "horizon": H,
        "n_ic": int(z0b.size),
        **metrics_extra,
    }
    return {"metrics": metrics}
