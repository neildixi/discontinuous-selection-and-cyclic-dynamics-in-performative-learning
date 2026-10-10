"""Job manifest with atomic updates. Completed outputs are never overwritten."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pandas as pd

from dynafront.io_atomic import atomic_write_json, load_json, sha256_file
from dynafront.logging_util import utc_now_iso

REQUIRED_RESULT_KEYS = (
    "experiment",
    "stage",
    "config",
    "seed",
    "code_version",
    "config_hash",
    "started_at",
    "completed_at",
    "runtime_sec",
    "success",
)


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if self.path.exists():
            self.data = load_json(self.path)
        else:
            self.data = {"jobs": {}, "created_at": utc_now_iso()}
            self._flush()

    def _flush(self) -> None:
        atomic_write_json(self.path, self.data)
        rows = []
        for jid, row in self.data.get("jobs", {}).items():
            rec = {"job_id": jid, **{k: v for k, v in row.items() if k != "config"}}
            rec["config_json"] = str(row.get("config"))
            rows.append(rec)
        ledger = self.path.with_suffix(".csv")
        if rows:
            import pandas as pd
            from dynafront.io_atomic import atomic_write_csv
            atomic_write_csv(ledger, pd.DataFrame(rows))

    def upsert_planned(self, job: dict[str, Any]) -> None:
        jid = job["job_id"]
        with self._lock:
            existing = self.data["jobs"].get(jid)
            if existing and existing.get("status") == "completed" and existing.get("valid"):
                return
            if existing is None:
                self.data["jobs"][jid] = {
                    **job,
                    "status": "pending",
                    "valid": False,
                    "planned_at": utc_now_iso(),
                }
            self._flush()

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            row = self.data["jobs"][job_id]
            row["status"] = "running"
            row["started_at"] = utc_now_iso()
            self._flush()

    def mark_completed(self, job_id: str, result_path: str, checksum: str, runtime_sec: float) -> None:
        with self._lock:
            row = self.data["jobs"][job_id]
            row["status"] = "completed"
            row["valid"] = True
            row["result_path"] = result_path
            row["checksum"] = checksum
            row["runtime_sec"] = float(runtime_sec)
            row["completed_at"] = utc_now_iso()
            row["success"] = True
            self._flush()

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            row = self.data["jobs"][job_id]
            row["status"] = "failed"
            row["valid"] = False
            row["error"] = error
            row["completed_at"] = utc_now_iso()
            row["success"] = False
            self._flush()

    def mark_interrupted(self, job_id: str) -> None:
        with self._lock:
            row = self.data["jobs"][job_id]
            if row.get("status") == "completed" and row.get("valid"):
                return
            row["status"] = "interrupted"
            row["valid"] = False
            row["completed_at"] = utc_now_iso()
            self._flush()

    def is_complete(self, job_id: str) -> bool:
        with self._lock:
            row = self.data["jobs"].get(job_id) or {}
            return bool(row.get("status") == "completed" and row.get("valid"))

    def counts(self) -> dict[str, int]:
        with self._lock:
            jobs = self.data["jobs"].values()
            out = {"total": len(self.data["jobs"]), "pending": 0, "running": 0, "completed": 0, "failed": 0, "interrupted": 0}
            for row in jobs:
                st = row.get("status", "pending")
                out[st] = out.get(st, 0) + 1
            return out

    def jobs(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.data["jobs"])


def validate_checkpoint(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        payload = load_json(path)
    except Exception as exc:
        return False, f"unreadable json: {exc}"
    for key in REQUIRED_RESULT_KEYS:
        if key not in payload:
            return False, f"missing key {key}"
    if not payload.get("success"):
        return False, "success is false"
    metrics = payload.get("metrics") or {}
    for key, val in metrics.items():
        if isinstance(val, float) and val != val:  # NaN allowed in some metrics
            continue
        if isinstance(val, (list, tuple)):
            continue
    return True, "ok"


def checkpoint_checksum_ok(path: Path, expected: str | None) -> bool:
    if not expected:
        return True
    try:
        return sha256_file(path) == expected
    except Exception:
        return False
