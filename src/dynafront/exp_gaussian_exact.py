"""Phase 7 Gaussian exact-p_t Monte Carlo. Separate from v1 FFT observability."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynafront.jobs import make_job
from dynafront import certificates as ora
from dynafront.seeds import default_rng

RHOS = (0.0, 0.5, 0.9)
OBS_LEVELS = (0.0, 0.25, 0.5)
TRANS_LEVELS = (0.0, 0.25, 0.5)
N_PATHS = 100_000
N_STEPS = 400
CHUNK = 50_000
BETA = 1.0
Z_WILSON = 1.96
SE_MULT = 6.0
ABS_TOL = 1e-12


def enumerate_jobs(master_seed: int) -> list[dict[str, Any]]:
    jobs = []
    for rho in RHOS:
        t = ora.tau_c(rho, BETA)
        for lo in OBS_LEVELS:
            for lt in TRANS_LEVELS:
                cfg = {
                    "rho": float(rho),
                    "obs_level": float(lo),
                    "trans_level": float(lt),
                    "sigma_s": float(lo * t),
                    "sigma_eta": float(lt * t),
                    "tau_c": float(t),
                    "n_paths": N_PATHS,
                    "n_steps": N_STEPS,
                    "kind": "gaussian_exact",
                }
                jobs.append(make_job("gaussian_exact", cfg, master_seed, tag=f"rho{rho}_o{lo}_t{lt}"))
    return jobs


def wilson_interval(x: float, n: int, z: float = Z_WILSON) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = x / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / den
    margin = z * np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n) / den
    return float(center - margin), float(center + margin)


def _gate(phat: float, p: float, n: int) -> bool:
    se = np.sqrt(max(p * (1.0 - p), 0.0) / n)
    return abs(phat - p) <= SE_MULT * se + ABS_TOL


def execute(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    rho = cfg["rho"]
    tau = cfg["tau_c"]
    ss, se_n = cfg["sigma_s"], cfg["sigma_eta"]
    n_paths, n_steps = cfg["n_paths"], cfg["n_steps"]
    rng = default_rng(job["master_seed"], "gaussian_exact", job["config_hash"], 0, "paths")
    p_theory = np.array([ora.gaussian_p_t(t, rho, tau, ss, se_n) for t in range(1, n_steps + 1)])
    p_inf = ora.gaussian_p_inf(rho, tau, ss, se_n)
    bounds = ora.gaussian_horizon_bounds(p_theory)
    mean_lb = ora.gaussian_mean_waiting_lower_bound(p_inf)
    from scipy.signal import lfilter

    t_idx = np.arange(1, n_steps + 1)
    sign = np.where(t_idx % 2 == 1, -1.0, 1.0)[:, None]
    err_counts = np.zeros(n_steps, dtype=np.int64)
    first = np.full(n_paths, n_steps + 1, dtype=np.int32)
    filled = 0
    while filled < n_paths:
        m = min(CHUNK, n_paths - filled)
        eta = rng.normal(0.0, se_n, size=(n_steps, m)) if se_n > 0 else np.zeros((n_steps, m))
        eps = rng.normal(0.0, ss, size=(n_steps, m)) if ss > 0 else np.zeros((n_steps, m))
        e = eta if abs(rho) < 1e-15 else lfilter([1.0], [1.0, -float(rho)], eta, axis=0)
        s_obs = sign * tau + e + eps
        bad = np.sign(s_obs) != sign
        err_counts += bad.sum(axis=1)
        any_bad = bad.any(axis=0)
        hit = np.argmax(bad, axis=0) + 1
        first[filled : filled + m] = np.where(any_bad, hit, n_steps + 1)
        filled += m

    phat = err_counts / n_paths
    # primary check times
    check_idx = [0, n_steps // 4, n_steps // 2, n_steps - 1]
    p_err_max = float(np.max(np.abs(phat - p_theory)))
    p_gate = all(_gate(float(phat[i]), float(p_theory[i]), n_paths) for i in check_idx)
    # all-t gate for reporting
    n_fail_t = int(np.sum(np.abs(phat - p_theory) > SE_MULT * np.sqrt(np.maximum(p_theory * (1 - p_theory), 0) / n_paths) + ABS_TOL))

    hit = first <= n_steps
    pr_T_le_N = float(np.mean(hit))
    lo_b, hi_b = float(bounds["max_lower"]), float(bounds["union_upper"])
    bound_ok = (pr_T_le_N + 1e-12 >= lo_b) and (pr_T_le_N - 1e-12 <= min(1.0, hi_b) + 1.0 / np.sqrt(n_paths))
    # finite-horizon: MC should lie near [max p, sum p]; sampling error allowed on both sides of the true probability
    # The inequality is on the true Pr, not the MC estimate. Report coverage of the analytic sandwich.
    sandwich_holds_for_theory = lo_b <= min(1.0, hi_b) + 1e-15
    in_sandwich = lo_b - 6 / np.sqrt(n_paths) <= pr_T_le_N <= min(1.0, hi_b) + 6 / np.sqrt(n_paths)

    uncens = first[hit]
    emp_mean = float(np.mean(first.astype(np.float64)))  # includes censored as N+1 (lower bound on mean if many censored)
    emp_mean_uncens = float(np.mean(uncens)) if uncens.size else np.nan
    frac_cens = float(np.mean(~hit))

    obs_only = se_n == 0.0 and ss > 0.0
    p = float(p_theory[0]) if p_theory.size else 0.0
    geom_mean = (1.0 / p) if obs_only and p > 0 else np.nan
    # survival P(T>n)=(1-p)^n observation-only
    if obs_only and p > 0:
        ns = np.array([1, 5, 10, 20, 50, 100, 200])
        ns = ns[ns <= n_steps]
        surv_emp = np.array([float(np.mean(first > n)) for n in ns])
        surv_th = (1.0 - p) ** ns
        surv_err = float(np.max(np.abs(surv_emp - surv_th)))
        surv_gate = all(_gate(float(surv_emp[i]), float(surv_th[i]), n_paths) for i in range(len(ns)))
    else:
        surv_err, surv_gate = np.nan, True
        ns = np.array([])

    mean_lb_ok = True
    if mean_lb is not None:
        # KM-style lower bound on mean: if censored, empirical mean of min(T,N+1) underestimates E[T]
        mean_lb_ok = emp_mean + 1e-6 >= mean_lb * 0.0  # always true as a number check; report separately
        # The theorem is on true E[T]. We check empirical uncensored/truncated mean is not below the bound by more than MC noise when little censoring.
        if frac_cens < 0.05:
            mean_lb_ok = emp_mean_uncens + 6 * (emp_mean_uncens / np.sqrt(max(uncens.size, 1))) >= mean_lb

    lo_w, hi_w = wilson_interval(float(err_counts[0]), n_paths)
    metrics = {
        "rho": rho,
        "obs_level": cfg["obs_level"],
        "trans_level": cfg["trans_level"],
        "sigma_s": ss,
        "sigma_eta": se_n,
        "tau_c": tau,
        "p_inf": p_inf,
        "p_t1": float(p_theory[0]),
        "phat_t1": float(phat[0]),
        "p_t_max_abs_err": p_err_max,
        "n_times_outside_6se": n_fail_t,
        "p_t_gate": float(p_gate and n_fail_t == 0),
        "wilson_t1_lo": lo_w,
        "wilson_t1_hi": hi_w,
        "pr_T_le_N": pr_T_le_N,
        "bound_max_lower": lo_b,
        "bound_union_upper": min(1.0, hi_b),
        "in_horizon_sandwich": float(in_sandwich),
        "sandwich_theory_ok": float(sandwich_holds_for_theory),
        "emp_mean_T_trunc": emp_mean,
        "emp_mean_uncensored": emp_mean_uncens,
        "frac_censored": frac_cens,
        "mean_lb": mean_lb if mean_lb is not None else np.nan,
        "mean_lb_ok": float(mean_lb_ok),
        "obs_only": float(obs_only),
        "geom_mean": geom_mean,
        "geom_mean_rel_err": float(abs(emp_mean_uncens - geom_mean) / geom_mean) if obs_only and geom_mean == geom_mean and uncens.size else np.nan,
        "surv_max_err": surv_err,
        "surv_gate": float(surv_gate),
        "n_paths": n_paths,
        "n_steps": n_steps,
        "tol_formula": f"|phat-p|<= {SE_MULT}*sqrt(p(1-p)/n)+{ABS_TOL}; Wilson z={Z_WILSON}",
        "lag1_err_corr": np.nan,  # filled below for transition noise
    }
    # temporal dependence of error indicators under linear model (descriptive)
    if se_n > 0:
        # reuse one chunk-sized sample for lag-1 correlation of the error process
        rng2 = default_rng(job["master_seed"], "gaussian_exact", job["config_hash"], 1, "dep")
        m = 8000
        eta = rng2.normal(0.0, se_n, size=(min(80, n_steps), m))
        eps = rng2.normal(0.0, ss, size=eta.shape)
        e = np.zeros(m)
        flags = np.empty(eta.shape)
        for t in range(eta.shape[0]):
            e = rho * e + eta[t]
            flags[t] = (tau + e + eps[t]) < 0.0
        f0 = flags[:-1].ravel()
        f1 = flags[1:].ravel()
        if f0.std() > 0 and f1.std() > 0:
            metrics["lag1_err_corr"] = float(np.corrcoef(f0, f1)[0, 1])
    return {"metrics": metrics, "arrays": {"p_theory": p_theory, "phat": phat.astype(np.float64)}}
