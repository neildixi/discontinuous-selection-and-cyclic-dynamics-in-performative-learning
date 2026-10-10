"""Phase 2: genuinely unaligned two-resolvent systems."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynafront.dynamics import simulate_unaligned
from dynafront.jobs import make_job
from dynafront.metrics import finite_stat
from dynafront import certificates as ora
from dynafront.seeds import default_rng

DIMS = (2, 5, 20)
QUADRANTS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
N_PER_CELL = 36  # 36 * 3 dims = 108 per quadrant after pooling
ALIGN_REJECT = 1e-3
MARGIN = 0.20
RADIUS = 0.90
LOCAL_STEPS = 300
BROAD_STEPS = 250
BROAD_N = 16


def enumerate_jobs(master_seed: int) -> list[dict[str, Any]]:
    jobs = []
    for d in DIMS:
        for sfix, scyc in QUADRANTS:
            cfg = {
                "d": int(d),
                "sign_fix": int(sfix),
                "sign_cyc": int(scyc),
                "n_systems": N_PER_CELL,
                "margin": MARGIN,
                "radius": RADIUS,
                "align_reject": ALIGN_REJECT,
            }
            jobs.append(make_job("unaligned", cfg, master_seed, seed=0, tag=f"d{d}_f{sfix}_c{scyc}"))
    return jobs


def _stable_unaligned(rng, d: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(400):
        B = rng.normal(size=(d, d))
        nrm = np.linalg.norm(B, 2)
        if nrm <= 0:
            continue
        A = B * (RADIUS / nrm)
        u = rng.normal(size=d)
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            continue
        u = u / nu
        if ora.alignment_residual(A, u) < ALIGN_REJECT:
            continue
        if np.linalg.norm(A, 2) > RADIUS + 1e-12:
            continue
        return A, u
    return None


def execute(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    d = cfg["d"]
    rng = default_rng(job["master_seed"], "unaligned", job["config_hash"], 0, d, cfg["sign_fix"], cfg["sign_cyc"])
    rows = []
    accepted = 0
    aligned_rejected = 0
    attempts = 0
    while accepted < cfg["n_systems"] and attempts < 20000:
        attempts += 1
        pair = _stable_unaligned(rng, d)
        if pair is None:
            aligned_rejected += 1
            continue
        A, u = pair
        if ora.alignment_residual(A, u) < ALIGN_REJECT:
            aligned_rejected += 1
            continue
        v = ora.min_norm_v_for_resolvents(A, u, cfg["sign_fix"] * MARGIN, cfg["sign_cyc"] * MARGIN)
        cf = ora.chi_fix(A, u, v)
        cc = ora.chi_cyc(A, u, v)
        if abs(cf) < 0.5 * MARGIN or abs(cc) < 0.5 * MARGIN:
            continue
        if np.sign(cf) != cfg["sign_fix"] or np.sign(cc) != cfg["sign_cyc"]:
            continue
        xplus_f, xminus_f = ora.fixed_action_states(A, v)
        xplus_c, xminus_c = ora.alternating_pair_states(A, v)
        # frozen limits from random starts
        x0 = rng.normal(size=(4, d))
        xp, _, _ = simulate_unaligned(x0, A, u, v, 500, action="plus")
        xm, _, _ = simulate_unaligned(x0, A, u, v, 500, action="minus")
        err_fix_p = float(np.max(np.linalg.norm(xp[-1] - xplus_f, axis=1)))
        err_fix_m = float(np.max(np.linalg.norm(xm[-1] - xminus_f, axis=1)))
        err_cyc = float(np.linalg.norm(xplus_c + xminus_c))  # should be 0
        rel = max(err_fix_p, err_fix_m) / (np.linalg.norm(xplus_f) + 1e-15)
        # selector compatibility of formulas
        score_fp = float(u @ xplus_f)
        score_fm = float(u @ xminus_f)
        fix_compat_plus = score_fp > 0.0
        fix_compat_minus = score_fm < 0.0
        # crossed iff both invalid: chi_fix>0 => score of +1 state is -chi_fix < 0
        oracle_crossed = ora.fixed_points_crossed(cf)
        sim_crossed = (not fix_compat_plus) and (not fix_compat_minus)
        score_cp = float(u @ xplus_c)
        oracle_pair = ora.alternating_pair_compatible(cc)
        sim_pair = score_cp > 0.0
        # local tests
        local_fix_ok = np.nan
        local_cyc_ok = np.nan
        n_switch = np.nan
        if cf < 0:
            rad = 0.25 * min(abs(score_fp), abs(score_fm)) / (np.linalg.norm(u) + 1e-15)
            pert = rad * rng.normal(size=(8, d))
            pert *= rad / (np.linalg.norm(pert, axis=1, keepdims=True) + 1e-15)
            xloc = np.vstack([xplus_f + pert[:4], xminus_f + pert[4:]])
            xs, zs, a = simulate_unaligned(xloc, A, u, v, LOCAL_STEPS, action="selected")
            dplus = np.linalg.norm(xs[-1] - xplus_f, axis=1)
            dminus = np.linalg.norm(xs[-1] - xminus_f, axis=1)
            local_fix_ok = float(np.all(np.minimum(dplus, dminus) < 1e-6))
        if cc > 0:
            rad = 0.25 * abs(score_cp) / (np.linalg.norm(u) + 1e-15)
            pert = rng.normal(size=(8, d))
            pert *= rad / (np.linalg.norm(pert, axis=1, keepdims=True) + 1e-15)
            xloc = np.vstack([xplus_c + pert[:4], xminus_c + pert[4:]])
            xs, zs, a = simulate_unaligned(xloc, A, u, v, LOCAL_STEPS, action="selected")
            dplus = np.linalg.norm(xs[-1] - xplus_c, axis=1)
            dminus = np.linalg.norm(xs[-1] - xminus_c, axis=1)
            # after even steps should match starting parity approximately
            local_cyc_ok = float(np.all(np.minimum(dplus, dminus) < 1e-6))
        if cf > 0:
            x0b = rng.normal(size=(BROAD_N, d))
            xs, zs, a = simulate_unaligned(x0b, A, u, v, BROAD_STEPS, action="selected")
            sw = np.mean(a[1:] != a[:-1], axis=0)
            n_switch = float(np.min(np.sum(a[1:] != a[:-1], axis=0)))
        rows.append(
            dict(
                chi_fix=cf,
                chi_cyc=cc,
                align_resid=ora.alignment_residual(A, u),
                err_fix_rel=float(rel),
                err_cyc_norm=err_cyc,
                oracle_crossed=float(oracle_crossed),
                sim_crossed=float(sim_crossed),
                oracle_pair=float(oracle_pair),
                sim_pair=float(sim_pair),
                local_fix_ok=local_fix_ok,
                local_cyc_ok=local_cyc_ok,
                min_switches=n_switch,
                A_norm=float(np.linalg.norm(A, 2)),
            )
        )
        accepted += 1
    if not rows:
        return {"metrics": {"d": d, "n_accepted": 0, "attempts": attempts, "aligned_rejected": aligned_rejected}, "arrays": {}}
    arr = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}
    metrics = {
        "d": d,
        "sign_fix": cfg["sign_fix"],
        "sign_cyc": cfg["sign_cyc"],
        "n_accepted": accepted,
        "attempts": attempts,
        "aligned_rejected": aligned_rejected,
        "n_aligned_passed_filter": float(np.sum(arr["align_resid"] < ALIGN_REJECT)) if rows else 0.0,
        "fix_compat_agree": float(np.mean(arr["oracle_crossed"] == arr["sim_crossed"])) if rows else 0.0,
        "pair_compat_agree": float(np.mean(arr["oracle_pair"] == arr["sim_pair"])) if rows else 0.0,
        "max_rel_fix_err": float(np.max(arr["err_fix_rel"])) if rows else np.nan,
        "max_align_resid_min": float(np.min(arr["align_resid"])) if rows else np.nan,
        "frac_local_fix": finite_stat(arr["local_fix_ok"], np.mean) if rows else np.nan,
        "frac_local_cyc": finite_stat(arr["local_cyc_ok"], np.mean) if rows else np.nan,
        "min_switches_when_crossed": finite_stat(arr["min_switches"], np.min) if rows else np.nan,
    }
    return {"metrics": metrics, "arrays": arr}
