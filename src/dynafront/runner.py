"""Resumable atomic-job runner with heartbeat, workers, and wall-clock cap."""

from __future__ import annotations

import os
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from dynafront.hashing import config_hash
from dynafront.io_atomic import atomic_write_json, sha256_file
from dynafront.logging_util import setup_logger, utc_now_iso
from dynafront.manifest import Manifest, validate_checkpoint
from dynafront.memory import current_rss_bytes, plan_workers
from dynafront.paths import ensure_output_tree

STOP = {"flag": False}


def _handle_signal(signum, _frame) -> None:
    STOP["flag"] = True


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


JobFn = Callable[[dict[str, Any]], dict[str, Any]]


def _write_runtime_estimate(out: Path, stage: str, n_unfinished: int, median_sec: float, workers: int) -> float:
    projected = (n_unfinished * median_sec) / max(workers, 1)
    lines = [
        f"# Runtime estimate — {stage}",
        "",
        f"- unfinished jobs: {n_unfinished}",
        f"- median benchmark job time: {median_sec:.4f} s",
        f"- effective workers: {workers}",
        f"- projected wall time: **{projected / 3600.0:.3f} h** ({projected:.1f} s)",
        "",
        "Formula: unfinished × median_job_time / workers.",
        "",
    ]
    dest = out / "reports" / "RUNTIME_ESTIMATE.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines))
    return projected


def run_jobs(
    jobs: list[dict[str, Any]],
    execute: JobFn,
    *,
    stage: str,
    out: Path,
    version: str | None = None,
    workers: int | None,
    resume: bool,
    max_hours: float | None,
    master_seed: int,
    logger_name: str | None = None,
    max_projected_hours: float = 12.0,
    benchmark_indices: list[int] | None = None,
    skip_projection_gate: bool = False,
    retries: int = 1,
) -> dict[str, Any]:
    out = ensure_output_tree(Path(out))
    logger = setup_logger(out / "logs" / f"{stage}.log", name=logger_name or f"dynafront.{stage}")
    install_signal_handlers()
    STOP["flag"] = False
    manifest = Manifest(out / "manifests" / f"{stage}.json")
    code_version = version or "public"
    for job in jobs:
        job.setdefault("code_version", code_version)
        job.setdefault("master_seed", master_seed)
        manifest.upsert_planned(job)

    pending = []
    for job in jobs:
        ckpt = out / "checkpoints" / stage / f"{job['job_id']}.json"
        if resume and manifest.is_complete(job["job_id"]):
            ok, reason = validate_checkpoint(ckpt)
            if ok:
                continue
            logger.info("Re-running %s: checkpoint invalid (%s)", job["job_id"], reason)
        pending.append(job)

    plan = plan_workers(workers)
    n_workers = plan.workers if len(pending) > 1 else 1
    logger.info("Stage %s: %d planned, %d pending, workers=%d (%s)", stage, len(jobs), len(pending), n_workers, plan.reason)

    t_cap = None if max_hours is None else time.time() + float(max_hours) * 3600.0
    job_times: list[float] = []
    completed = 0
    failed = 0
    skipped = len(jobs) - len(pending)
    started = time.time()

    def _heartbeat() -> None:
        elapsed = time.time() - started
        remaining = len(pending) - completed - failed
        med = float(sorted(job_times)[len(job_times) // 2]) if job_times else None
        eta = (remaining * med / max(n_workers, 1)) if med else None
        payload = {
            "stage": stage,
            "done": skipped + completed,
            "total": len(jobs),
            "failed": failed,
            "skipped": skipped,
            "elapsed_sec": elapsed,
            "rolling_job_sec": med,
            "eta_sec": eta,
            "rss_bytes": current_rss_bytes(),
            "timestamp": utc_now_iso(),
        }
        atomic_write_json(out / "status" / f"{stage}.json", payload)
        logger.info(
            "[heartbeat] %s %d/%d done skipped=%d fail=%d elapsed=%.0fs rolling=%.2fs eta=%s rss=%.1fMB",
            stage,
            skipped + completed,
            len(jobs),
            skipped,
            failed,
            elapsed,
            med or 0.0,
            f"{eta:.0f}s" if eta is not None else "?",
            current_rss_bytes() / (1024 * 1024),
        )

    # Benchmark a representative unfinished job before launching a large remaining set.
    if pending and not skip_projection_gate:
        bench_job = pending[0] if not benchmark_indices else jobs[benchmark_indices[0]]
        t0 = time.time()
        try:
            _run_one(bench_job, execute, out, stage, manifest, logger, retries=retries)
            bench_t = time.time() - t0
            # If the benchmark job was pending[0], don't rerun it.
            if pending and pending[0]["job_id"] == bench_job["job_id"]:
                pending = pending[1:]
                completed += 1
                job_times.append(bench_t)
        except Exception:
            bench_t = max(time.time() - t0, 1e-3)
        projected = _write_runtime_estimate(out, stage, len(pending), bench_t, n_workers)
        logger.info("Projected remaining wall time for %s: %.3f h", stage, projected / 3600.0)
        if projected / 3600.0 > max_projected_hours:
            msg = (
                f"STOP: projected {projected / 3600.0:.2f} h exceeds {max_projected_hours:.0f} h cap. "
                f"median_job={bench_t:.3f}s unfinished={len(pending)} workers={n_workers}. "
                f"Resume after approval with: python -m dynafront.cli reproduce --mode paper --workers {n_workers}"
            )
            logger.error(msg)
            report = out / "reports" / f"RUNTIME_BLOCK_{stage}.md"
            report.write_text(
                "\n".join(
                    [
                        f"# Runtime block — {stage}",
                        "",
                        f"- measured time per representative job: {bench_t:.4f} s",
                        f"- unfinished jobs: {len(pending)}",
                        f"- workers: {n_workers}",
                        f"- projected remaining: {projected / 3600.0:.3f} h",
                        f"- dominant cost: {stage} job `{bench_job['job_id']}`",
                        "- safe reductions: fewer seeds, fewer robustness cells, or a shorter measured horizon after explicit approval",
                        f"- resume command: `python -m dynafront.cli reproduce --mode paper --workers {n_workers}`",
                        "",
                    ]
                )
            )
            return {
                "stage": stage,
                "blocked": True,
                "projected_hours": projected / 3600.0,
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
            }

    last_beat = time.time()
    pbar = tqdm(total=len(jobs), initial=skipped, desc=stage, unit="job")

    def _after(job_id: str, runtime: float, ok: bool) -> None:
        nonlocal completed, failed, last_beat
        job_times.append(runtime)
        if ok:
            completed += 1
        else:
            failed += 1
        pbar.update(1)
        med = float(sorted(job_times)[len(job_times) // 2])
        remain = max(len(jobs) - skipped - completed - failed, 0)
        eta = remain * med / max(n_workers, 1)
        pbar.set_postfix(fail=failed, roll=f"{med:.2f}s", eta=f"{eta:.0f}s")
        if time.time() - last_beat >= 15:
            _heartbeat()
            last_beat = time.time()

    try:
        use_pool = n_workers > 1
        if use_pool:
            try:
                from concurrent.futures import ProcessPoolExecutor as _PPE  # noqa: F401
                _PPE(max_workers=n_workers).__enter__().__exit__(None, None, None)
            except Exception as exc:
                logger.warning("Process pool unavailable (%s); falling back to 1 worker", exc)
                use_pool = False
                n_workers = 1
        if not use_pool:
            for job in pending:
                if STOP["flag"] or (t_cap and time.time() > t_cap):
                    manifest.mark_interrupted(job["job_id"])
                    break
                t0 = time.time()
                ok = True
                try:
                    _run_one(job, execute, out, stage, manifest, logger, retries=retries)
                except Exception as exc:
                    ok = False
                    manifest.mark_failed(job["job_id"], "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                    logger.exception("Job failed: %s", job["job_id"])
                _after(job["job_id"], time.time() - t0, ok)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {}
                inflight = 0
                it = iter(pending)
                while True:
                    if STOP["flag"] or (t_cap and time.time() > t_cap):
                        for fut, job in list(futures.items()):
                            fut.cancel()
                            if not fut.done():
                                manifest.mark_interrupted(job["job_id"])
                        break
                    while inflight < n_workers:
                        try:
                            job = next(it)
                        except StopIteration:
                            break
                        if STOP["flag"]:
                            break
                        futures[pool.submit(_worker_execute, job, execute)] = job
                        inflight += 1
                    if not futures:
                        break
                    try:
                        fut = next(as_completed(list(futures.keys()), timeout=1.0))
                    except TimeoutError:
                        if time.time() - last_beat >= 15:
                            _heartbeat()
                            last_beat = time.time()
                        continue
                    job = futures.pop(fut)
                    inflight -= 1
                    try:
                        result, runtime = fut.result()
                        _save_result(job, result, runtime, out, stage, manifest)
                        _after(job["job_id"], runtime, True)
                    except Exception as exc:
                        manifest.mark_failed(job["job_id"], "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                        logger.exception("Job failed: %s", job["job_id"])
                        _after(job["job_id"], 0.0, False)
    finally:
        pbar.close()
        _heartbeat()

    counts = manifest.counts()
    logger.info("Stage %s finished: %s", stage, counts)
    return {
        "stage": stage,
        "blocked": False,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "counts": counts,
        "elapsed_sec": time.time() - started,
        "peak_rss_bytes": current_rss_bytes(),
    }


def _worker_execute(job: dict[str, Any], execute: JobFn) -> tuple[dict[str, Any], float]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    t0 = time.time()
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            result = execute(job)
            if attempt:
                result.setdefault("metrics", {})["retries"] = attempt
            return result, time.time() - t0
        except Exception as exc:
            last_exc = exc
            if attempt == 2:
                raise
    raise last_exc  # pragma: no cover


def _run_one(job: dict[str, Any], execute: JobFn, out: Path, stage: str, manifest: Manifest, logger, retries: int = 1) -> None:
    last_exc: Exception | None = None
    attempts = 1 + max(0, int(retries))
    t0 = time.time()
    for attempt in range(attempts):
        manifest.mark_running(job["job_id"])
        try:
            result = execute(job)
            if attempt:
                result.setdefault("metrics", {})["retries"] = attempt
            _save_result(job, result, time.time() - t0, out, stage, manifest)
            return
        except (OSError, TimeoutError, MemoryError) as exc:
            last_exc = exc
            logger.warning("Transient failure attempt %d/%d for %s: %s", attempt + 1, attempts, job["job_id"], exc)
            if attempt + 1 >= attempts:
                raise
    if last_exc is not None:
        raise last_exc


def _save_result(job: dict[str, Any], result: dict[str, Any], runtime: float, out: Path, stage: str, manifest: Manifest) -> None:
    ckpt_dir = out / "checkpoints" / stage
    raw_dir = out / "raw" / stage
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": job.get("experiment", stage),
        "stage": stage,
        "config": job.get("config", {}),
        "seed": job.get("seed"),
        "code_version": job.get("code_version"),
        "config_hash": job.get("config_hash") or config_hash(job.get("config", {})),
        "started_at": result.get("started_at") or utc_now_iso(),
        "completed_at": utc_now_iso(),
        "runtime_sec": float(runtime),
        "success": True,
        "exception": None,
        "metrics": result.get("metrics", {}),
        "arrays_path": None,
        "job_id": job["job_id"],
        "master_seed": job.get("master_seed"),
    }
    # Metrics-only checkpoints keep public runs compact; arrays are not persisted.
    payload["arrays_path"] = None
    ckpt_path = ckpt_dir / f"{job['job_id']}.json"
    if ckpt_path.exists():
        ok, _ = validate_checkpoint(ckpt_path)
        if ok:
            # Never overwrite a valid completed checkpoint.
            manifest.mark_completed(job["job_id"], str(ckpt_path), sha256_file(ckpt_path), runtime)
            return
    atomic_write_json(ckpt_path, payload)
    ok, reason = validate_checkpoint(ckpt_path)
    if not ok:
        raise RuntimeError(f"checkpoint validation failed for {job['job_id']}: {reason}")
    manifest.mark_completed(job["job_id"], str(ckpt_path), sha256_file(ckpt_path), runtime)
