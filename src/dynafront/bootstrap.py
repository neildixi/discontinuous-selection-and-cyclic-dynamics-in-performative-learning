"""Seed-cluster percentile bootstrap. Resample complete trajectories, never timesteps."""

from __future__ import annotations

from typing import Callable

import numpy as np


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return (np.nan, np.nan, np.nan)
    lo, mid, hi = np.quantile(samples, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0])
    return float(lo), float(mid), float(hi)


def seed_cluster_bootstrap(
    values: np.ndarray,
    *,
    n_resample: int = 10_000,
    rng: np.random.Generator,
    stat: Callable[[np.ndarray], float] = np.mean,
    alpha: float = 0.05,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return {"lo": np.nan, "mid": np.nan, "hi": np.nan, "point": np.nan}
    point = float(stat(values))
    idx = rng.integers(0, n, size=(n_resample, n))
    draws = np.apply_along_axis(stat, 1, values[idx])
    lo, mid, hi = percentile_ci(draws, alpha)
    return {"lo": lo, "mid": mid, "hi": hi, "point": point}


def paired_diff_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resample: int = 10_000,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return seed_cluster_bootstrap(a - b, n_resample=n_resample, rng=rng, alpha=alpha)


def holm_adjust(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = pvalues[idx]
        val = min(1.0, (m - rank) * raw)
        running = max(running, val)
        adj[idx] = running
    return adj
