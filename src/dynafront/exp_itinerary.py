"""Extension C: exact finite itineraries. Certificates from dynafront.certificates."""

from __future__ import annotations

from typing import Any

import numpy as np

from dynafront import certificates as ora
from dynafront.dynamics import simulate_asymmetric_scalar
from dynafront.hashing import config_hash, short_hash
from dynafront.metrics import detect_action_period
from dynafront.seeds import default_rng

WORDS = ora.unique_primitive_words(8)
UNCOND_N = 2000
COVERAGE_PER = 25
SCAN_BUDGET = 80_000
LOCAL_N = 8
BROAD_N = 21
N_STEPS = 480
BURNIN = 240
MARGIN = 1e-8


def enumerate_jobs(master_seed: int) -> list[dict[str, Any]]:
    jobs = []
    cfg_u = {"sample": "unconditional", "n": UNCOND_N, "max_period": 8, "multiplier_protocol": "first_block_above_underflow"}
    jobs.append(
        {
            "job_id": f"itinerary_uncond__{short_hash(cfg_u)}",
            "experiment": "itinerary",
            "config": cfg_u,
            "config_hash": config_hash(cfg_u),
            "seed": 0,
            "master_seed": master_seed,
        }
    )
    cfg_c = {"sample": "coverage", "per_period": COVERAGE_PER, "scan_budget": SCAN_BUDGET, "max_period": 8, "multiplier_protocol": "first_block_above_underflow"}
    jobs.append(
        {
            "job_id": f"itinerary_coverage__{short_hash(cfg_c)}",
            "experiment": "itinerary",
            "config": cfg_c,
            "config_hash": config_hash(cfg_c),
            "seed": 1,
            "master_seed": master_seed,
        }
    )
    return jobs


def _draw(rng):
    rp = float(rng.uniform(0.1, 0.9))
    rm = float(rng.uniform(0.1, 0.9))
    bp = float(rng.uniform(-2.5, 2.5))
    bm = float(rng.uniform(-2.5, 2.5))
    return rp, rm, bp, bm


def _certificates(rp, rm, bp, bm):
    certs = []
    for w in WORDS:
        try:
            zs, R, B = ora.word_candidate_states(w, rp, rm, bp, bm)
        except ValueError:
            continue
        if ora.word_is_strict_orbit(zs, w, margin=MARGIN):
            certs.append((w, zs, R))
    return certs


def _stratum(rp, rm, bp, bm) -> str:
    pp, pm = ora.affine_fixed_points(rp, rm, bp, bm)
    zp, zm = ora.predicted_period_two_pair(rp, rm, bp, bm)
    crossed = pp < 0 < pm
    pair = zp > 0 > zm
    if crossed and pair:
        return "crossed_sign_compatible"
    if crossed:
        return "crossed_no_pair"
    return "noncrossed"


def _local_verify(w, zs, R, rp, rm, bp, bm, rng):
    """State error after contraction; multiplier from the first block while the
    perturbation is still well above underflow and still in the certified regions.
    """
    p = len(w)
    margin = float(np.min(np.abs(zs)))
    rad = 0.25 * max(margin - MARGIN, 1e-16)
    pert = rng.normal(size=LOCAL_N)
    pert = rad * pert / (np.max(np.abs(pert)) + 1e-15)
    z0 = zs[0] + pert
    n_block = 80 if abs(R) < 0.5 else 240
    z, a = simulate_asymmetric_scalar(z0, rp, rm, bp, bm, n_block * p)
    tail = z[(n_block // 2) * p :: p]
    err = float(np.max(np.abs(tail[-8:] - zs[0]))) if tail.size else np.nan
    expected = np.array([1.0 if s in {"+", "1", 1, True} else -1.0 for s in w])
    emp = []
    for i in range(z0.size):
        if not np.array_equal(a[:p, i], expected):
            continue
        e0 = z[0, i] - zs[0]
        e1 = z[p, i] - zs[0]
        if 1e-10 < abs(e0) < 0.5 * max(margin, 1e-12):
            emp.append(e1 / e0)
    if abs(R) <= 1e-14:
        merr = 0.0
    elif emp:
        merr = abs(float(np.median(emp)) - R)
    else:
        merr = np.nan
    return err, merr


def execute(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job["config"]
    rng = default_rng(job["master_seed"], "itinerary", job["config_hash"], job["seed"], cfg["sample"])
    z_grid = np.linspace(-4, 4, BROAD_N)
    if cfg["sample"] == "unconditional":
        n = cfg["n"]
        rows = []
        n_obs_without_cert = 0
        n_cert_local_fail = 0
        max_state = 0.0
        max_mult = 0.0
        strata = {"crossed_sign_compatible": 0, "crossed_no_pair": 0, "noncrossed": 0}
        period_hist = np.zeros(9)
        for i in range(n):
            rp, rm, bp, bm = _draw(rng)
            st = _stratum(rp, rm, bp, bm)
            strata[st] += 1
            certs = _certificates(rp, rm, bp, bm)
            periods = {len(w) for w, _, _ in certs}
            for w, zs, R in certs:
                err, merr = _local_verify(w, zs, R, rp, rm, bp, bm, rng)
                if not (err == err) or err > 1e-9:
                    n_cert_local_fail += 1
                else:
                    max_state = max(max_state, err)
                    if merr == merr and abs(R) > 1e-14 and abs(merr) < 1.0:
                        max_mult = max(max_mult, merr)
            z, a = simulate_asymmetric_scalar(z_grid, rp, rm, bp, bm, N_STEPS)
            per = detect_action_period(a[BURNIN:], 8)
            cert_keys = {ora.necklace_key(w) for w, _, _ in certs}
            for ic_i, p in enumerate(per):
                if 1 <= p <= 8:
                    period_hist[p] += 1
                    seq = a[BURNIN:, ic_i]
                    z_at = z[BURNIN:-1, ic_i]
                    block_z = z_at[-p:]
                    if float(np.min(np.abs(block_z))) < MARGIN:
                        continue
                    block = seq[-p:]
                    wobs = tuple("+" if x > 0 else "-" for x in block)
                    if ora.necklace_key(wobs) not in cert_keys:
                        n_obs_without_cert += 1
            rows.append(len(certs))
        metrics = {
            "sample": "unconditional",
            "n_systems": n,
            "n_words_enumerated": len(WORDS),
            "mean_n_certs": float(np.mean(rows)),
            "n_obs_without_cert": int(n_obs_without_cert),
            "n_cert_local_fail": int(n_cert_local_fail),
            "max_state_err": max_state,
            "max_mult_err": max_mult,
            "n_stratum1": strata["crossed_sign_compatible"],
            "n_stratum2": strata["crossed_no_pair"],
            "n_stratum3": strata["noncrossed"],
            **{f"obs_period_{p}": int(period_hist[p]) for p in range(1, 9)},
            "selection_rule": "none_unconditional_iid",
        }
        return {"metrics": metrics}

    # coverage: first qualifying systems with certified minimal period p=1..6
    found = {p: 0 for p in range(1, 7)}
    scanned = 0
    n_cert_local_fail = 0
    max_state = 0.0
    max_mult = 0.0
    first_seeds = {}
    while scanned < cfg["scan_budget"] and any(found[p] < cfg["per_period"] for p in found):
        scanned += 1
        rp, rm, bp, bm = _draw(rng)
        certs = _certificates(rp, rm, bp, bm)
        if not certs:
            continue
        # minimal certified period
        pmin = min(len(w) for w, _, _ in certs)
        if pmin > 6 or found[pmin] >= cfg["per_period"]:
            continue
        w, zs, R = min(certs, key=lambda t: len(t[0]))
        err, merr = _local_verify(w, zs, R, rp, rm, bp, bm, rng)
        if err == err:
            max_state = max(max_state, err)
        if not (err == err) or err > 1e-9:
            n_cert_local_fail += 1
            continue
        if merr == merr and abs(R) > 1e-14:
            max_mult = max(max_mult, merr)
        found[pmin] += 1
        if pmin not in first_seeds:
            first_seeds[pmin] = scanned  # first qualifying in seed order
    metrics = {
        "sample": "coverage",
        "scan_budget": cfg["scan_budget"],
        "scanned": scanned,
        "selection_rule": "first_qualifying_certified_minimal_period_in_seed_order",
        "n_cert_local_fail": n_cert_local_fail,
        "max_state_err": max_state,
        "max_mult_err": max_mult,
        **{f"n_period_{p}": found[p] for p in range(1, 7)},
        **{f"first_scan_index_period_{p}": first_seeds.get(p, -1) for p in range(1, 7)},
        "null_periods": ",".join(str(p) for p in range(1, 7) if found[p] < cfg["per_period"]),
    }
    return {"metrics": metrics}
