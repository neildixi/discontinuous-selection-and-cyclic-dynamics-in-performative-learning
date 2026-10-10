"""Compare reconstructed headline numbers with cached reference values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dynafront.io_atomic import atomic_write_json, atomic_write_text, load_json
from dynafront.paths import cached_dir

# Exact quantities must match bit-for-bit after integer cast.
EXACT_KEYS = {
    "master_seed",
    "jobs_full",
    "jobs_paper",
    "clipped_n_widths",
    "unaligned_n_systems",
    "unaligned_min_systems_per_quadrant",
    "s1_n_systems",
    "s1_n_systems_period_two",
    "s2_n_systems",
    "s2_n_systems_period_two",
    "s2_n_other_finite",
    "s2_n_unresolved",
    "s3_n_systems",
    "itinerary_uncond_n_systems",
    "itinerary_n_obs_without_cert",
    "itinerary_n_cert_local_fail",
    "itinerary_coverage_period_1",
    "itinerary_coverage_period_2",
    "itinerary_coverage_period_3",
    "itinerary_coverage_period_4",
    "itinerary_coverage_period_5",
    "itinerary_coverage_period_6",
    "itinerary_coverage_scanned",
    "tanh_n_subcritical",
    "tanh_n_subcritical_converged",
}

FLOAT_SPECS: dict[str, dict[str, float]] = {
    "hard_D2_mean_max": {"atol": 1e-16, "rtol": 1e-6},
    "hard_D2_max_ic": {"atol": 5e-13, "rtol": 1e-4},
    "tanh_max_amp_err_all_subcritical": {"atol": 5e-16, "rtol": 0.05},
    "tanh_max_amp_err_ratio_lt_0.98": {"atol": 5e-16, "rtol": 0.05},
    "tanh_max_residual": {"atol": 1e-15, "rtol": 0.0},
    "unaligned_max_rel_fix_err": {"atol": 5e-16, "rtol": 0.2},
    "itinerary_max_state_err": {"atol": 1e-14, "rtol": 1e-4},
    "itinerary_max_mult_err": {"atol": 1e-14, "rtol": 1e-3},
    "s1_frac_period_two": {"atol": 1e-12, "rtol": 0.0},
    "s2_frac_period_two": {"atol": 1e-12, "rtol": 0.0},
    "s3_frac_period_two": {"atol": 1e-12, "rtol": 0.0},
}


def load_expected(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else cached_dir() / "expected_results.json"
    return load_json(path)


def _expected_value(entry: Any) -> Any:
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def compare_headlines(
    observed: dict[str, Any],
    expected: dict[str, Any] | None = None,
    *,
    mode: str = "full",
) -> dict[str, Any]:
    expected = expected if expected is not None else load_expected()
    rows: list[dict[str, Any]] = []
    failed = 0
    skipped = 0
    for key, spec in expected.items():
        if key.startswith("_"):
            continue
        if not isinstance(spec, dict):
            spec = {"value": spec, "compare": "float"}
        modes = spec.get("modes", ["paper", "full"])
        if mode == "quick" and "quick" not in modes:
            skipped += 1
            continue
        if mode == "paper" and "paper" not in modes and "full" not in modes:
            skipped += 1
            continue
        if mode == "paper" and spec.get("full_only"):
            skipped += 1
            continue
        want = spec.get("value")
        got = observed.get(key)
        compare = spec.get("compare")
        if compare is None:
            compare = "exact" if key in EXACT_KEYS else "float"
        status = "SKIP"
        detail = ""
        if got is None:
            if spec.get("optional") or mode != "full":
                status = "SKIP"
                skipped += 1
                detail = "not present in this mode"
            else:
                status = "FAIL"
                failed += 1
                detail = "missing observed value"
        elif compare == "exact":
            ok = got == want
            if not ok and isinstance(want, (int, np.integer)) and isinstance(got, (int, float, np.floating, np.integer)):
                ok = int(got) == int(want)
            status = "PASS" if ok else "FAIL"
            detail = f"observed={got} expected={want}"
            if not ok:
                failed += 1
        else:
            atol = float(spec.get("atol", FLOAT_SPECS.get(key, {}).get("atol", 1e-12)))
            rtol = float(spec.get("rtol", FLOAT_SPECS.get(key, {}).get("rtol", 1e-6)))
            ok = bool(np.isfinite(got) and np.isfinite(want) and np.isclose(float(got), float(want), atol=atol, rtol=rtol))
            status = "PASS" if ok else "FAIL"
            detail = f"observed={got} expected={want} atol={atol} rtol={rtol}"
            if not ok:
                failed += 1
        rows.append({"key": key, "status": status, "detail": detail})
    payload = {
        "mode": mode,
        "n_compared": len(rows),
        "n_failed": failed,
        "n_skipped": skipped,
        "passed": failed == 0,
        "rows": rows,
    }
    return payload


def write_validation_report(payload: dict[str, Any], dest: Path) -> None:
    dest = Path(dest)
    atomic_write_json(dest.with_suffix(".json"), payload)
    lines = [
        "# Validation report",
        "",
        f"- mode: `{payload['mode']}`",
        f"- compared: {payload['n_compared']}",
        f"- failed: **{payload['n_failed']}**",
        f"- passed: {payload['passed']}",
        "",
        "| key | status | detail |",
        "|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(f"| `{row['key']}` | {row['status']} | {row['detail']} |")
    atomic_write_text(dest.with_suffix(".md"), "\n".join(lines) + "\n")
