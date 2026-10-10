"""Vectorized score-aligned scalar and state dynamics."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from dynafront.selectors import clipped, deadband, sign_plus_at_zero

SelectorName = Literal["hard", "continuous", "deadband"]


def initial_scores(n: int = 100, core: tuple[float, float] = (-4.0, 4.0), wide: tuple[float, float] = (-20.0, 20.0)) -> np.ndarray:
    """Reproducible IC grid: dense core on [-4, 4] plus wide tails for both deadband basins."""
    n = int(n)
    n_core = min(n, max(3, int(round(0.6 * n))))
    if n >= 80:
        n_core = min(n, max(51, n_core))
    n_tail = n - n_core
    core_pts = np.linspace(core[0], core[1], n_core)
    # Include points inside a typical subcritical deadband and far outside the cycle.
    left = np.linspace(wide[0], core[0], n_tail // 2 + 1, endpoint=False)[1:]
    right = np.linspace(core[1], wide[1], n_tail - len(left) + 1, endpoint=True)[1:]
    z0 = np.concatenate([left, core_pts, right])
    if z0.size > n:
        # keep endpoints, zero, and uniform subsample
        keep = np.unique(np.round(np.linspace(0, z0.size - 1, n)).astype(int))
        z0 = z0[keep]
        if z0.size < n:
            extra = np.linspace(wide[0], wide[1], n - z0.size)
            z0 = np.sort(np.concatenate([z0, extra]))
    elif z0.size < n:
        extra = np.linspace(wide[0], wide[1], n - z0.size)
        z0 = np.sort(np.concatenate([z0, extra]))
    return z0.astype(np.float64)


def simulate_scalar(
    z0: np.ndarray,
    rho: float,
    n_steps: int,
    selector: SelectorName,
    *,
    beta: float = 1.0,
    tau: float | None = None,
    h: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized over initial conditions. Returns z[T+1, N], a[T, N]."""
    z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
    n = z0.size
    z = np.empty((n_steps + 1, n), dtype=np.float64)
    a = np.empty((n_steps, n), dtype=np.float64)
    z[0] = z0
    rho = float(rho)
    beta = float(beta)
    for t in range(n_steps):
        if selector == "hard":
            a[t] = sign_plus_at_zero(z[t])
        elif selector == "continuous":
            if tau is None:
                raise ValueError("tau required")
            a[t] = clipped(z[t], tau)
        elif selector == "deadband":
            if h is None:
                raise ValueError("h required")
            a[t] = deadband(z[t], h)
        else:
            raise ValueError(selector)
        z[t + 1] = rho * z[t] - beta * a[t]
    return z, a


def simulate_scalar_noisy(
    z0: np.ndarray,
    rho: float,
    n_steps: int,
    *,
    beta: float = 1.0,
    eps: np.ndarray | None = None,
    eta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z_{t+1} = rho z_t - beta sign(z_t + eps_t) + eta_{t+1}.

    eps, eta shapes: [T, N] (eta index 0 unused / aligned as eta_{t+1} at row t+1).
    Returns z, a_observed, a_true.
    """
    z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
    n = z0.size
    z = np.empty((n_steps + 1, n), dtype=np.float64)
    a_obs = np.empty((n_steps, n), dtype=np.float64)
    a_true = np.empty((n_steps, n), dtype=np.float64)
    z[0] = z0
    if eps is None:
        eps = np.zeros((n_steps, n))
    if eta is None:
        eta = np.zeros((n_steps + 1, n))
    for t in range(n_steps):
        a_true[t] = sign_plus_at_zero(z[t])
        a_obs[t] = sign_plus_at_zero(z[t] + eps[t])
        z[t + 1] = rho * z[t] - beta * a_obs[t] + eta[t + 1]
    return z, a_obs, a_true


def two_step_rate_on_hard_orbit(z: np.ndarray, rho: float, beta: float, burnin: int) -> np.ndarray:
    """Empirical (z_{t+2}-z^+)/(z_t-z^+) on the positive subsequence after burn-in."""
    from dynafront.theory import tau_c

    t = tau_c(rho, beta)
    post = z[burnin:]
    rates = []
    for i in range(post.shape[1]):
        seq = post[:, i]
        pos = np.where(seq[:-2] > 0)[0]
        if pos.size == 0:
            rates.append(np.nan)
            continue
        num = seq[pos + 2] - t
        den = seq[pos] - t
        ok = np.abs(den) > 1e-10
        if np.any(ok):
            rates.append(float(np.median(num[ok] / den[ok])))
        else:
            # Already on the orbit to numerical precision: two-step factor is exactly rho^2.
            rates.append(float(rho) ** 2)
    return np.asarray(rates, dtype=np.float64)


def make_latent_matrix(
    rng: np.random.Generator,
    d: int,
    rho: float,
    latent_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A = rho u u^T + A_perp with A_perp u = 0, u^T A_perp = 0, ||A_perp||_2 < 1."""
    u = np.zeros(d)
    u[0] = 1.0
    if d == 1:
        A = np.array([[float(rho)]], dtype=np.float64)
        v = np.array([1.0], dtype=np.float64)
        return A, u, v
    # Random matrix on the orthogonal complement, then rescale.
    B = rng.normal(size=(d - 1, d - 1))
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    target = min(0.99, max(1e-6, float(latent_radius)))
    s = s / (s.max() + 1e-15) * target
    A_lat = (U * s) @ Vt
    A = np.zeros((d, d), dtype=np.float64)
    A[0, 0] = float(rho)
    A[1:, 1:] = A_lat
    v = np.zeros(d, dtype=np.float64)
    v[0] = 1.0  # u^T v = 1
    # Mix a little of latent directions into v while keeping u^T v = 1
    extra = rng.normal(size=d)
    extra[0] = 0.0
    extra_norm = np.linalg.norm(extra)
    if extra_norm > 0:
        v = v + 0.25 * extra / extra_norm
    return A, u, v


def simulate_latent(
    x0: np.ndarray,
    A: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x_{t+1} = A x_t - a_t v, a_t = sign(u^T x_t), +1 at zero. x0: [N, d]."""
    x0 = np.asarray(x0, dtype=np.float64)
    n, d = x0.shape
    x = np.empty((n_steps + 1, n, d), dtype=np.float64)
    z = np.empty((n_steps + 1, n), dtype=np.float64)
    a = np.empty((n_steps, n), dtype=np.float64)
    x[0] = x0
    z[0] = x0 @ u
    for t in range(n_steps):
        a[t] = sign_plus_at_zero(z[t])
        x[t + 1] = x[t] @ A.T - a[t][:, None] * v[None, :]
        z[t + 1] = x[t + 1] @ u
    return x, z, a


def simulate_asymmetric_scalar(
    z0: np.ndarray,
    rho_plus: float,
    rho_minus: float,
    b_plus: float,
    b_minus: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
    n = z0.size
    z = np.empty((n_steps + 1, n), dtype=np.float64)
    a = np.empty((n_steps, n), dtype=np.float64)
    z[0] = z0
    for t in range(n_steps):
        pos = z[t] >= 0.0
        a[t] = np.where(pos, 1.0, -1.0)
        z[t + 1] = np.where(pos, rho_plus * z[t] + b_plus, rho_minus * z[t] + b_minus)
    return z, a


def simulate_tanh(z0: np.ndarray, rho: float, tau: float, n_steps: int, *, beta: float = 1.0) -> np.ndarray:
    z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
    z = np.empty((n_steps + 1, z0.size), dtype=np.float64)
    z[0] = z0
    k = float(beta)
    rho = float(rho)
    tau = float(tau)
    for t in range(n_steps):
        z[t + 1] = rho * z[t] - k * np.tanh(z[t] / tau)
    return z


def simulate_hysteresis(
    z0: np.ndarray,
    a0: np.ndarray,
    rho: float,
    h: float,
    n_steps: int,
    *,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Equality |z|=h retains the previous action."""
    z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
    prev = np.where(np.asarray(a0, dtype=np.float64).reshape(-1) >= 0.0, 1.0, -1.0)
    a = np.empty((n_steps, z0.size), dtype=np.float64)
    z = np.empty((n_steps + 1, z0.size), dtype=np.float64)
    z[0] = z0
    h = float(h)
    rho = float(rho)
    beta = float(beta)
    for t in range(n_steps):
        at = np.where(z[t] > h, 1.0, np.where(z[t] < -h, -1.0, prev))
        a[t] = at
        z[t + 1] = rho * z[t] - beta * at
        prev = at
    return z, a


def simulate_delayed(
    z0: np.ndarray,
    z_m1: np.ndarray,
    rho: float,
    tau: float,
    n_steps: int,
    *,
    beta: float = 1.0,
) -> np.ndarray:
    """z_{t+1} = rho z_t - beta clip(z_{t-1}/tau, -1, 1)."""
    prev = np.asarray(z_m1, dtype=np.float64).reshape(-1)
    cur = np.asarray(z0, dtype=np.float64).reshape(-1)
    z = np.empty((n_steps + 1, cur.size), dtype=np.float64)
    z[0] = cur
    rho = float(rho)
    tau = float(tau)
    beta = float(beta)
    for t in range(n_steps):
        nxt = rho * cur - beta * np.clip(prev / tau, -1.0, 1.0)
        z[t + 1] = nxt
        prev = cur
        cur = nxt
    return z


def tanh_ics(n: int = 100) -> np.ndarray:
    """Deterministic ICs: small, moderate, saturated; exclude exact zero."""
    small = np.linspace(-0.08, 0.08, 21)
    small = small[np.abs(small) > 1e-15]
    mid = np.linspace(-2.5, 2.5, 51)
    mid = mid[np.abs(mid) > 0.09]
    sat = np.concatenate([np.linspace(-25.0, -3.0, 20), np.linspace(3.0, 25.0, 20)])
    z = np.concatenate([small, mid, sat])
    if z.size > n:
        keep = np.unique(np.round(np.linspace(0, z.size - 1, n)).astype(int))
        z = z[keep]
    elif z.size < n:
        extra = np.linspace(-15.0, 15.0, n - z.size + 2)
        extra = extra[np.abs(extra) > 1e-12]
        z = np.sort(np.concatenate([z, extra]))[:n]
    return z.astype(np.float64)


def simulate_unaligned(
    x0: np.ndarray,
    A: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n_steps: int,
    action: str = "selected",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = np.asarray(x0, dtype=np.float64)
    n, d = x0.shape
    x = np.empty((n_steps + 1, n, d), dtype=np.float64)
    z = np.empty((n_steps + 1, n), dtype=np.float64)
    a = np.empty((n_steps, n), dtype=np.float64)
    x[0] = x0
    z[0] = x0 @ u
    for t in range(n_steps):
        if action == "plus":
            a[t] = 1.0
        elif action == "minus":
            a[t] = -1.0
        else:
            a[t] = sign_plus_at_zero(z[t])
        x[t + 1] = x[t] @ A.T - a[t][:, None] * v[None, :]
        z[t + 1] = x[t + 1] @ u
    return x, z, a
