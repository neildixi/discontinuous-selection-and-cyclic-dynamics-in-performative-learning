"""Command-line interface for cached figures, notebooks, and reproduction."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from dynafront.experiments import (
    EXPECTED_JOB_COUNTS,
    count_jobs,
    smoke_jobs,
    stages_for_mode,
)
from dynafront.io_atomic import atomic_write_json, sha256_file
from dynafront.paths import (
    cached_dir,
    configs_dir,
    ensure_output_tree,
    generated_figures_dir,
    new_run_dir,
    repo_root,
)
from dynafront.plotting import rebuild_figures
from dynafront.runner import run_jobs
from dynafront.seeds import MASTER_SEED
from dynafront.summaries import extract_headline_numbers, reconstruct_summaries
from dynafront.validation import compare_headlines, load_expected, write_validation_report


def _load_config(mode: str, path: Path | None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else configs_dir() / f"{mode}.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    data = dict(data or {})
    data.setdefault("mode", mode)
    data.setdefault("master_seed", MASTER_SEED)
    data.setdefault("workers", 2)
    return data


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def cmd_figures(args: argparse.Namespace) -> int:
    source = Path(args.source) if args.source else cached_dir()
    dest = Path(args.dest) if args.dest else generated_figures_dir()
    if _inside(dest, cached_dir()):
        raise ValueError("refusing to write figures into results/cached/; use results/generated/figures/")
    print(f"Rebuilding figures from {source} -> {dest}")
    written = rebuild_figures(source, dest)
    print(f"Wrote {len(written)} figure stems (PDF + PNG each).")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    summaries = Path(args.summaries) if args.summaries else cached_dir() / "summaries"
    observed = extract_headline_numbers(summaries)
    expected = load_expected(Path(args.expected) if args.expected else None)
    payload = compare_headlines(observed, expected, mode=args.mode)
    dest = Path(args.report) if args.report else Path("validation_report.md")
    write_validation_report(payload, dest)
    print(f"compared={payload['n_compared']} failed={payload['n_failed']} passed={payload['passed']}")
    return 0 if payload["passed"] else 1


def cmd_notebooks(args: argparse.Namespace) -> int:
    from dynafront.paths import repo_root as root_fn

    nb_dir = Path(args.notebooks) if args.notebooks else root_fn() / "notebooks"
    import nbformat
    from nbclient import NotebookClient

    paths = sorted(nb_dir.glob("*.ipynb"))
    if not paths:
        raise SystemExit(f"no notebooks in {nb_dir}")
    failed = 0
    for path in paths:
        print(f"Executing {path.name}")
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(nb_dir)}})
        try:
            client.execute()
        except Exception as exc:
            print(f"FAIL {path.name}: {exc}")
            failed += 1
            continue
        nbformat.write(nb, path)
        print(f"OK {path.name}")
    return 0 if failed == 0 else 1


def _print_banner(mode: str, seed: int, n_jobs: int, out: Path, workers: int) -> None:
    print("=" * 64)
    print(f"DynaFront reproduce  mode={mode}")
    print(f"master_seed={seed}")
    print(f"jobs={n_jobs}  (jobs, not systems)")
    print(f"workers={workers}")
    print(f"output={out}")
    print("cached reference is never overwritten")
    print("=" * 64)


def cmd_reproduce(args: argparse.Namespace) -> int:
    cfg = _load_config(args.mode, Path(args.config) if args.config else None)
    seed = int(args.seed if args.seed is not None else cfg.get("master_seed", MASTER_SEED))
    workers = int(args.workers if args.workers is not None else cfg.get("workers", 2))
    if args.out:
        out = ensure_output_tree(Path(args.out))
    else:
        out = ensure_output_tree(new_run_dir(args.mode))
    shutil.copy2(configs_dir() / f"{args.mode}.yaml", out / "config" / f"{args.mode}.yaml")

    counts = count_jobs(args.mode, seed)
    n_jobs = int(counts["total"])
    _print_banner(args.mode, seed, n_jobs, out, workers)
    print("job counts by family:", json.dumps({k: v for k, v in counts.items() if k != "total"}))

    t_all = time.time()
    stage_results: list[dict[str, Any]] = []
    bench_rows: list[dict[str, Any]] = []

    if args.mode == "quick":
        families = smoke_jobs(seed)
        # short benchmark on the first smoke family
        t0 = time.time()
        first_stage, first_jobs, first_exec = families[0]
        run_jobs(
            first_jobs[:1],
            first_exec,
            stage=first_stage,
            out=out,
            workers=1,
            resume=True,
            max_hours=0.25,
            master_seed=seed,
            skip_projection_gate=True,
        )
        bench_s = time.time() - t0
        remaining = n_jobs - 1
        proj = remaining * bench_s / max(workers, 1)
        print(f"benchmark {bench_s:.3f}s; projected remaining ~{proj:.1f}s")
        atomic_write_json(
            out / "status" / "benchmark.json",
            {"bench_sec": bench_s, "projected_remaining_sec": proj, "n_jobs": n_jobs},
        )
        for stage, jobs, execute in families:
            r = run_jobs(
                jobs,
                execute,
                stage=stage,
                out=out,
                workers=1,
                resume=True,
                max_hours=args.max_hours,
                master_seed=seed,
                skip_projection_gate=True,
            )
            stage_results.append(r)
            if r.get("failed"):
                break
    else:
        stages = stages_for_mode(args.mode)
        for st in stages:
            want = EXPECTED_JOB_COUNTS[st.name]
            jobs = st.enumerate_jobs(seed)
            if len(jobs) != want:
                raise RuntimeError(f"{st.name} enumerated {len(jobs)} jobs, expected {want}")
            pick = jobs[min(max(st.bench_index, 0), len(jobs) - 1)]
            t0 = time.time()
            run_jobs(
                [pick],
                st.execute,
                stage=st.name,
                out=out,
                workers=1,
                resume=True,
                max_hours=0.5,
                master_seed=seed,
                skip_projection_gate=True,
            )
            bench_s = time.time() - t0
            proj = bench_s * len(jobs) / max(workers if st.workers_hint > 1 else 1, 1)
            bench_rows.append({"stage": st.name, "n_jobs": len(jobs), "bench_sec": bench_s, "projected_sec": proj})
            print(f"BENCH {st.name}: {bench_s:.3f}s x {len(jobs)} jobs -> ~{proj:.1f}s")
        total_proj = sum(r["projected_sec"] for r in bench_rows)
        print(f"Projected wall time after per-family benchmarks: {total_proj:.1f}s ({total_proj / 60.0:.2f} min)")
        atomic_write_json(
            out / "status" / "benchmark.json",
            {"projected_wall_sec": total_proj, "stages": bench_rows, "workers": workers},
        )
        if args.mode == "full" and args.skip_if_over_minutes is not None and total_proj / 60.0 > float(args.skip_if_over_minutes):
            print(
                f"STOP: projected {total_proj / 60.0:.1f} min exceeds {args.skip_if_over_minutes:.0f} min. "
                "Not starting the full suite."
            )
            atomic_write_json(out / "status" / "blocked.json", {"reason": "projection_exceeds_cap", "projected_sec": total_proj})
            return 2
        for st in stages:
            jobs = st.enumerate_jobs(seed)
            print(f"Running {st.name}: {len(jobs)} jobs ({st.unit})")
            r = run_jobs(
                jobs,
                st.execute,
                stage=st.name,
                out=out,
                workers=workers if st.workers_hint > 1 else 1,
                resume=True,
                max_hours=args.max_hours,
                master_seed=seed,
                skip_projection_gate=True,
            )
            stage_results.append(r)
            if r.get("failed"):
                break

    print("Reconstructing summaries from checkpoints")
    reconstruct_summaries(out)
    print("Rebuilding figures from reconstructed summaries")
    rebuild_figures(out, out / "figures")
    observed = extract_headline_numbers(out / "summaries", out=out)
    observed["master_seed"] = seed
    if args.mode == "full":
        observed["jobs_full"] = observed.get("jobs_completed", observed.get("jobs_from_summaries"))
    if args.mode == "paper":
        observed["jobs_paper"] = observed.get("jobs_completed", observed.get("jobs_from_summaries"))
    expected = load_expected()
    payload = compare_headlines(observed, expected, mode=args.mode)
    write_validation_report(payload, out / "reports" / "VALIDATION")
    atomic_write_json(out / "summaries" / "headline_numbers.json", observed)
    elapsed = time.time() - t_all
    failed_jobs = sum(int(r.get("failed") or 0) for r in stage_results)
    summary = {
        "mode": args.mode,
        "master_seed": seed,
        "output": str(out.relative_to(repo_root())) if out.is_relative_to(repo_root()) else str(out),
        "elapsed_sec": elapsed,
        "failed_jobs": failed_jobs,
        "comparison_passed": payload["passed"],
        "comparison_failed": payload["n_failed"],
        "n_jobs": n_jobs,
    }
    atomic_write_json(out / "status" / "final.json", summary)
    print(json.dumps(summary, indent=2))
    if failed_jobs or not payload["passed"]:
        return 1
    return 0


def verify_cached_manifest(root: Path | None = None) -> list[str]:
    root = cached_dir() if root is None else Path(root)
    manifest = root / "MANIFEST.sha256"
    failures: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split(None, 1)
        path = root / rel
        if not path.is_file():
            failures.append(f"missing {rel}")
        elif sha256_file(path) != digest:
            failures.append(f"mismatch {rel}")
    return failures


def cmd_verify_cached(_args: argparse.Namespace) -> int:
    failures = verify_cached_manifest()
    if failures:
        print("cached manifest check failed:")
        for item in failures:
            print(f"  {item}")
        return 1
    print("cached manifest OK")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else cached_dir()
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "MANIFEST.sha256":
            continue
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    text = "\n".join(lines) + "\n"
    dest = root / "MANIFEST.sha256"
    dest.write_text(text, encoding="utf-8")
    print(f"wrote {dest} ({len(lines)} files)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dynafront", description="DynaFront reproduction CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    fig = sub.add_parser("figures", help="Rebuild figures from cached summaries (no simulations)")
    fig.add_argument("--source", default=None)
    fig.add_argument("--dest", default=None)
    fig.set_defaults(func=cmd_figures)

    nb = sub.add_parser("notebooks", help="Execute the tutorial notebooks")
    nb.add_argument("--notebooks", default=None)
    nb.set_defaults(func=cmd_notebooks)

    cmp = sub.add_parser("compare", help="Compare summaries with expected_results.json")
    cmp.add_argument("--summaries", default=None)
    cmp.add_argument("--expected", default=None)
    cmp.add_argument("--mode", default="full")
    cmp.add_argument("--report", default=None)
    cmp.set_defaults(func=cmd_compare)

    rep = sub.add_parser("reproduce", help="Run quick, paper, or full experiments")
    rep.add_argument("--mode", choices=("quick", "paper", "full"), required=True)
    rep.add_argument("--config", default=None)
    rep.add_argument("--out", default=None, help="Fresh output directory (never results/cached)")
    rep.add_argument("--workers", type=int, default=None)
    rep.add_argument("--seed", type=int, default=None)
    rep.add_argument("--max-hours", type=float, default=2.0)
    rep.add_argument("--skip-if-over-minutes", type=float, default=None)
    rep.set_defaults(func=cmd_reproduce)

    man = sub.add_parser("manifest", help="Write SHA-256 manifest for cached artifacts")
    man.add_argument("--root", default=None)
    man.set_defaults(func=cmd_manifest)

    ver = sub.add_parser("verify-cached", help="Check results/cached against MANIFEST.sha256")
    ver.set_defaults(func=cmd_verify_cached)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
