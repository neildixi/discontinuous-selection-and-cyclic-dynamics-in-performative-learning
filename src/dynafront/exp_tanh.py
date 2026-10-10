"""Phase 3: tanh selector phase diagram."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynafront.dynamics import simulate_scalar, simulate_tanh, tanh_ics
from dynafront.jobs import make_job
from dynafront.metrics import finite_stat
from dynafront import certificates as ora

RHOS = (0.0, 0.2, 0.5, 0.8, 0.95)
RATIOS = (
    0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0, 1.005, 1.01, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0
)
BETA = 1.0
N_IC = 100
TOL = 1e-9


def enumerate_jobs(master_seed: int) -> list[dict[str, Any]]:
    jobs = []
    for rho in RHOS:
        for ratio in RATIOS:
            t = ora.tau_c(rho, BETA)
            cfg = {
                "rho": float(rho),
                "ratio": float(ratio),
                "tau": float(ratio * t),
                "beta": BETA,
                "n_ic": N_IC,
            }
            jobs.append(make_job("tanh", cfg, master_seed, tag=f"rho{rho}_r{ratio}"))
    return jobs


def _horizon(ratio: float) -> int:
    dist = abs(ratio - 1.0)
    if dist < 0.01:
        return 40000
    if dist < 0.05:
        return 12000
    return 4000


def execute(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    rho, tau, beta = cfg["rho"], cfg["tau"], cfg["beta"]
    ratio = cfg["ratio"]
    t = ora.tau_c(rho, beta)
    regime = ora.tanh_regime(tau, rho, beta)
    r_tau, resid = ora.tanh_amplitude(tau, rho, beta)
    lip = ora.clipped_lipschitz(rho, tau, beta)  # not tanh lip; tanh lip is |rho - beta/tau sech^2|
    # tanh global lip: max |rho - (beta/tau) sech^2| = max(|rho|, |rho - beta/tau|) at 0
    tanh_lip = max(abs(rho), abs(rho - beta / tau))
    z0 = tanh_ics(cfg["n_ic"])
    H = _horizon(ratio)
    z = simulate_tanh(z0, rho, tau, H, beta=beta)
    from dynafront.dynamics import simulate_scalar

    zc, _ = simulate_scalar(z0, rho, min(H, 4000), "continuous", beta=beta, tau=tau)
    clip_amp = 0.5 * (np.quantile(np.abs(zc[-min(200, zc.shape[0]-1):]), 0.95, axis=0) - np.quantile(np.abs(zc[-min(200, zc.shape[0]-1):]), 0.05, axis=0))
    # actually amplitude of signed series
    clip_amp = 0.5 * (np.quantile(zc[-min(200, zc.shape[0] - 1) :], 0.95, axis=0) - np.quantile(zc[-min(200, zc.shape[0] - 1) :], 0.05, axis=0))
    mag = np.abs(z)
    # distance to oracle set
    if r_tau == 0.0:
        dist = mag
        target = 0.0
    else:
        dist = np.minimum(np.abs(z - r_tau), np.abs(z + r_tau))
        target = r_tau
    # first time all ICs within TOL, else per-IC
    hit = dist <= max(TOL, 1e-8 * max(t, 1e-12))
    conv_time = np.full(z0.size, np.nan)
    for i in range(z0.size):
        w = np.where(hit[:, i])[0]
        if w.size:
            conv_time[i] = float(w[0])
    censored = np.isnan(conv_time)
    # near critical: if still decreasing, label censored not fail
    decreasing = mag[-1] < mag[H // 2]
    tail = z[-min(200, H) :]
    amp = 0.5 * (np.quantile(tail, 0.95, axis=0) - np.quantile(tail, 0.05, axis=0))
    term = np.abs(z[-1])
    if r_tau > 0:
        amp_err = np.abs(amp - r_tau) / t
        if np.any(~censored):
            amp_err = amp_err[~censored]
        sign_alt = np.mean(np.sign(z[-40::2]) * np.sign(z[-39::2]) < 0)
    else:
        amp_err = amp / t
        if np.any(~censored):
            amp_err = amp_err[~censored]
        sign_alt = np.nan
    # two-step multiplier at cycle for one IC near +r
    emp_mult = np.nan
    if r_tau > 1e-8:
        seq = z[:, np.argmin(np.abs(z0 - r_tau))]
        # positive subsequence near end
        pos = np.where(seq[:-2] > 0)[0]
        pos = pos[pos > H // 2]
        if pos.size > 5:
            num = seq[pos + 2] - r_tau
            den = seq[pos] - r_tau
            ok = np.abs(den) > 1e-10
            if np.any(ok):
                emp_mult = float(np.median(num[ok] / den[ok]))
    theory_mult = ora.tanh_two_step_multiplier(r_tau, tau, rho, beta) if r_tau > 0 else np.nan
    onset = ora.tanh_sqrt_onset(tau, rho, beta)
    metrics = {
        "rho": rho,
        "ratio": ratio,
        "tau": tau,
        "tau_c": t,
        "regime": regime,
        "r_tau": r_tau,
        "residual": resid,
        "tanh_lip": tanh_lip,
        "frac_converged": float(np.mean(~censored)),
        "frac_censored": float(np.mean(censored)),
        "frac_decreasing_if_censored": float(np.mean(decreasing[censored])) if np.any(censored) else np.nan,
        "median_conv_time": finite_stat(conv_time, np.median),
        "max_amp_err": float(np.max(amp_err)) if np.size(amp_err) else np.nan,
        "mean_amp": float(np.mean(amp)),
        "mean_amp_norm": float(np.mean(amp) / t),
        "mean_terminal": float(np.mean(term)),
        "emp_two_step": emp_mult,
        "theory_two_step": theory_mult,
        "onset": onset,
        "onset_rel_err": float(abs(r_tau - onset) / t) if r_tau > 0 else np.nan,
        "horizon": H,
        "n_ic": int(z0.size),
        "clipped_mean_amp_norm": float(np.mean(np.abs(clip_amp)) / t),
        "strict_decrease_nonzero": float(np.mean(mag[-1, np.abs(z0) > 1e-12] < mag[0, np.abs(z0) > 1e-12])) if regime.startswith("critical") else np.nan,
    }
    return {"metrics": metrics}
