"""Selectors for the score-aligned model."""

from __future__ import annotations

import numpy as np

from dynafront.theory import tau_c


def sign_plus_at_zero(z: np.ndarray) -> np.ndarray:
    """Hard selector: a = sign(z), with a = +1 at a zero score (theory convention)."""
    a = np.sign(z)
    return np.where(a == 0.0, 1.0, a)


def clipped(z: np.ndarray, tau: float) -> np.ndarray:
    return np.clip(z / float(tau), -1.0, 1.0)


def deadband(z: np.ndarray, h: float) -> np.ndarray:
    a = sign_plus_at_zero(z)
    return np.where(np.abs(z) <= float(h), 0.0, a)


def ratio_grid(n: int = 31, lo: float = 0.25, hi: float = 2.0) -> np.ndarray:
    """Equally spaced ratios including the exact boundary value 1."""
    grid = np.linspace(lo, hi, int(n))
    if not np.any(np.isclose(grid, 1.0, atol=1e-15)):
        grid = np.sort(np.concatenate([grid, np.array([1.0])]))
    else:
        grid[np.argmin(np.abs(grid - 1.0))] = 1.0
    return grid.astype(np.float64)


def scales_from_tau_c(rho: float, ratios: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return np.asarray(ratios, dtype=np.float64) * tau_c(rho, beta)
