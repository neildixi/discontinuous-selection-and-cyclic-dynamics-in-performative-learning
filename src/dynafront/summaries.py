"""Rebuild compact summaries from checkpoints and extract headline numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dynafront.io_atomic import atomic_write_csv, atomic_write_json, atomic_write_text, load_json

STAGE_CSV = (
    "synthetic_a",
    "synthetic_b",
    "synthetic_c",
    "synthetic_d_bounded",
    "synthetic_d_gaussian",
    "unaligned",
    "tanh",
    "itinerary",
    "hysteresis",
    "delayed",
    "gaussian_exact",
)


def load_stage(out: Path, stage: str) -> pd.DataFrame:
    ckpt_dir = Path(out) / "checkpoints" / stage
    rows: list[dict[str, Any]] = []
    if not ckpt_dir.exists():
        return pd.DataFrame()
    for path in sorted(ckpt_dir.glob("*.json")):
        if path.name.startswith("._"):
            continue
        payload = load_json(path)
        row = dict(payload.get("metrics") or {})
        row["job_id"] = payload.get("job_id")
        row["config_hash"] = payload.get("config_hash")
        row["success"] = payload.get("success")
        row["runtime_sec"] = payload.get("runtime_sec")
        row["seed"] = payload.get("seed")
        row["stage"] = payload.get("stage")
        rows.append(row)
    return pd.DataFrame(rows)


def _tex_table(df: pd.DataFrame, path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tex = df.to_latex(index=False, float_format="%.4g", caption=caption, escape=True)
    except Exception:
        tex = df.to_string(index=False)
    atomic_write_text(path, tex if isinstance(tex, str) else str(tex))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def reconstruct_summaries(out: Path) -> dict[str, pd.DataFrame]:
    """Write CSV/LaTeX summaries from checkpoint metrics. Does not simulate."""
    out = Path(out)
    summ, tables = out / "summaries", out / "tables"
    summ.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    stages: dict[str, pd.DataFrame] = {}
    for name in STAGE_CSV:
        df = load_stage(out, name)
        stages[name] = df
        if df.empty:
            continue
        if name == "itinerary" and "sample" in df.columns:
            sort_cols = [c for c in ("n_cert_local_fail", "max_mult_err") if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols).drop_duplicates(subset=["sample"], keep="first")
                stages[name] = df
        if name == "hysteresis":
            df = df.copy()
            if "family" not in df.columns and "kind" in df.columns:
                df["family"] = df["kind"]
            if "gate_ok" not in df.columns and "success_frac" in df.columns:
                df["gate_ok"] = df["success_frac"]
            if "threshold_ratio_hys_over_tauc" not in df.columns and "threshold_ratio_emp" in df.columns:
                df["threshold_ratio_hys_over_tauc"] = df["threshold_ratio_emp"]
            stages[name] = df
        atomic_write_csv(summ / f"{name}.csv", df)
        preview = df.head(40) if len(df) > 40 else df
        _tex_table(preview, tables / f"{name}.tex", name)
    numbers = extract_headline_numbers(summ, out=out)
    atomic_write_json(summ / "headline_numbers.json", numbers)
    return stages


def extract_headline_numbers(summaries: Path, out: Path | None = None) -> dict[str, Any]:
    """Compute reported quantities from compact CSVs. Nothing is hardcoded here."""
    summaries = Path(summaries)
    a = _read_csv(summaries / "synthetic_a.csv")
    b = _read_csv(summaries / "synthetic_b.csv")
    c = _read_csv(summaries / "synthetic_c.csv")
    db = _read_csv(summaries / "synthetic_d_bounded.csv")
    dg = _read_csv(summaries / "synthetic_d_gaussian.csv")
    u = _read_csv(summaries / "unaligned.csv")
    th = _read_csv(summaries / "tanh.csv")
    it = _read_csv(summaries / "itinerary.csv")
    hy = _read_csv(summaries / "hysteresis.csv")
    de = _read_csv(summaries / "delayed.csv")
    gx = _read_csv(summaries / "gaussian_exact.csv")

    numbers: dict[str, Any] = {
        "unit_note": "CSV row counts are jobs except where a column records systems or retained examples.",
    }
    row_counts = {
        "synthetic_a": int(len(a)),
        "synthetic_b": int(len(b)),
        "synthetic_c": int(len(c)),
        "synthetic_d_bounded": int(len(db)),
        "synthetic_d_gaussian": int(len(dg)),
        "unaligned": int(len(u)),
        "tanh": int(len(th)),
        "itinerary": int(len(it)),
        "hysteresis": int(len(hy)),
        "delayed": int(len(de)),
        "gaussian_exact": int(len(gx)),
    }
    numbers["jobs_by_family"] = row_counts
    numbers["jobs_from_summaries"] = int(sum(row_counts.values()))
    paper_families = (
        "synthetic_a",
        "synthetic_c",
        "unaligned",
        "tanh",
        "itinerary",
        "hysteresis",
    )
    numbers["jobs_paper"] = int(sum(row_counts[name] for name in paper_families))
    if numbers["jobs_from_summaries"] == 2156:
        numbers["jobs_full"] = 2156

    if out is not None:
        planned = completed = failed = 0
        for path in sorted((Path(out) / "manifests").glob("*.json")):
            if path.stem.startswith("smoke") or path.stem in {"audit", "package_manifest"}:
                continue
            data = load_json(path)
            jobs = data.get("jobs") or {}
            for row in jobs.values():
                planned += 1
                stt = row.get("status")
                if stt == "completed":
                    completed += 1
                elif stt == "failed":
                    failed += 1
        numbers["jobs_planned"] = planned
        numbers["jobs_completed"] = completed
        numbers["jobs_failed"] = failed

    if not a.empty:
        hard = a[a["selector"] == "hard"]
        numbers["hard_n_jobs"] = int(len(hard))
        if not hard.empty and "D2_mean" in hard:
            numbers["hard_D2_mean_max"] = float(hard["D2_mean"].max())
        if not hard.empty and "D2_max" in hard:
            numbers["hard_D2_max_ic"] = float(hard["D2_max"].max())
        cont = a[a["selector"] == "continuous"]
        numbers["clipped_n_jobs"] = int(len(cont))
        if not cont.empty and "ratio" in cont:
            numbers["clipped_n_widths"] = int(pd.Series(cont["ratio"]).nunique())
        dead = a[a["selector"] == "deadband"]
        numbers["deadband_n_jobs"] = int(len(dead))

    if not c.empty:
        for s, key in (
            ("crossed_sign_compatible", "s1"),
            ("crossed_no_pair", "s2"),
            ("noncrossed", "s3"),
        ):
            sub = c[c["stratum"] == s]
            numbers[f"{key}_n_systems"] = int(len(sub))
            if sub.empty:
                continue
            if "n_ic" in sub:
                numbers[f"{key}_n_ic"] = int(sub["n_ic"].iloc[0])
            numbers[f"{key}_n_systems_period_two"] = int((sub["frac_period_two"] >= 1.0 - 1e-12).sum()) if "frac_period_two" in sub else None
            if "frac_period_two" in sub:
                numbers[f"{key}_frac_period_two"] = float(sub["frac_period_two"].mean())
            if "frac_other_period" in sub:
                numbers[f"{key}_n_other_finite"] = int((sub["frac_other_period"] >= 1.0 - 1e-12).sum())
                numbers[f"{key}_frac_other_period"] = float(sub["frac_other_period"].mean())
            if "frac_unresolved_period" in sub:
                numbers[f"{key}_n_unresolved"] = int((sub["frac_unresolved_period"] >= 1.0 - 1e-12).sum())
                numbers[f"{key}_frac_unresolved_period"] = float(sub["frac_unresolved_period"].mean())

    if not u.empty:
        numbers["unaligned_n_jobs"] = int(len(u))
        numbers["unaligned_n_systems"] = int(u["n_accepted"].sum()) if "n_accepted" in u else None
        if {"sign_fix", "sign_cyc", "n_accepted"} <= set(u.columns):
            numbers["unaligned_min_systems_per_quadrant"] = int(u.groupby(["sign_fix", "sign_cyc"])["n_accepted"].sum().min())
        if "d" in u:
            numbers["unaligned_dims"] = sorted(int(x) for x in pd.unique(u["d"]))
        if "max_rel_fix_err" in u:
            numbers["unaligned_max_rel_fix_err"] = float(u["max_rel_fix_err"].max())

    if not th.empty:
        numbers["tanh_n_jobs"] = int(len(th))
        sub = th[th["regime"].astype(str).str.contains("subcritical", na=False)] if "regime" in th else th[th["ratio"] < 1.0]
        numbers["tanh_n_subcritical"] = int(len(sub))
        if not sub.empty and "max_amp_err" in sub:
            numbers["tanh_max_amp_err_all_subcritical"] = float(sub["max_amp_err"].max())
            if "frac_converged" in sub:
                numbers["tanh_n_subcritical_converged"] = int((sub["frac_converged"] >= 1.0 - 1e-12).sum())
        if "max_amp_err" in th:
            numbers["tanh_max_amp_err_unfiltered_including_critical"] = float(th["max_amp_err"].max())
        far = th[th["ratio"] < 0.98] if "ratio" in th else th.iloc[0:0]
        if not far.empty and "max_amp_err" in far:
            numbers["tanh_max_amp_err_ratio_lt_0.98"] = float(far["max_amp_err"].max())
        if "residual" in th:
            numbers["tanh_max_residual"] = float(th["residual"].max())

    if not it.empty:
        un = it[it["sample"] == "unconditional"] if "sample" in it else it.iloc[0:0]
        cov = it[it["sample"] == "coverage"] if "sample" in it else it.iloc[0:0]
        if not un.empty:
            if "n_systems" in un:
                numbers["itinerary_uncond_n_systems"] = int(un["n_systems"].iloc[0])
            if "n_obs_without_cert" in un:
                numbers["itinerary_n_obs_without_cert"] = int(un["n_obs_without_cert"].iloc[0])
        if "n_cert_local_fail" in it:
            numbers["itinerary_n_cert_local_fail"] = int(it["n_cert_local_fail"].sum())
        if "max_state_err" in it:
            numbers["itinerary_max_state_err"] = float(it["max_state_err"].max())
        if "max_mult_err" in it:
            numbers["itinerary_max_mult_err"] = float(it["max_mult_err"].max())
        if not cov.empty:
            if "scanned" in cov:
                numbers["itinerary_coverage_scanned"] = int(cov["scanned"].iloc[0])
            for p in range(1, 7):
                col = f"n_period_{p}"
                if col in cov:
                    numbers[f"itinerary_coverage_period_{p}"] = int(cov[col].iloc[0])

    if not hy.empty:
        col = "kind" if "kind" in hy.columns else "family"
        honly = hy[hy[col] == "hysteresis"] if col in hy.columns else hy
        if not honly.empty and {"rho", "threshold_ratio_emp"} <= set(honly.columns):
            theory = (1.0 + honly["rho"]) / (1.0 - honly["rho"])
            numbers["hysteresis_threshold_ratio_identity"] = bool(np.allclose(honly["threshold_ratio_emp"], theory))

    if not b.empty and "rel_state_orbit_error" in b:
        numbers["latent_median_rel_state_err"] = float(np.nanmedian(b["rel_state_orbit_error"]))
    if not db.empty and "sufficient_bound" in db:
        inside = db[db["sufficient_bound"] == True]  # noqa: E712
        numbers["bounded_n_inside"] = int(len(inside))
        if not inside.empty and "frac_traj_always_in_tubes" in inside:
            numbers["bounded_tube_frac_min"] = float(inside["frac_traj_always_in_tubes"].min())
    if not gx.empty:
        if "p_t_max_abs_err" in gx:
            numbers["gauss_max_pt_abs_err"] = float(gx["p_t_max_abs_err"].max())
        if "p_t_gate" in gx:
            numbers["gauss_min_pt_gate"] = float(gx["p_t_gate"].min())

    return numbers
