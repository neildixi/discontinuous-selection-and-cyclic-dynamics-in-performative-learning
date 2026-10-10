"""Conservative worker selection from available memory."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ResourcePlan:
    n_cpu: int
    mem_bytes: int
    workers: int
    reason: str


def available_memory_bytes() -> int:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page > 0:
                return int(pages * page)
        except (ValueError, OSError):
            pass
    return 8 * 1024**3


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def plan_workers(
    requested: int | None,
    *,
    rss_per_worker_bytes: float | None = None,
    max_workers: int = 4,
    memory_fraction: float = 0.70,
) -> ResourcePlan:
    n_cpu = cpu_count()
    mem = available_memory_bytes()
    cap = min(max(1, max_workers), n_cpu)
    workers = cap if requested is None else max(1, min(int(requested), cap))
    reason = f"requested={requested} cpu={n_cpu} cap={cap}"
    if rss_per_worker_bytes and rss_per_worker_bytes > 0:
        mem_cap = max(1, int((memory_fraction * mem) / rss_per_worker_bytes))
        if mem_cap < workers:
            reason += f"; reduced {workers}->{mem_cap} to stay under {100 * memory_fraction:.0f}% RAM"
            workers = max(1, mem_cap)
    return ResourcePlan(n_cpu=n_cpu, mem_bytes=mem, workers=workers, reason=reason)


def current_rss_bytes() -> int:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys

        if sys.platform == "darwin":
            return int(usage)
        return int(usage * 1024)
    except Exception:
        return 0
