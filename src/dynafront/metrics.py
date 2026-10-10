"""Trajectory metrics used in the paper experiments."""

from __future__ import annotations

import numpy as np

from dynafront.theory import tau_c


def finite_stat(values, reducer) -> float:
    """Reduce only finite entries so all-NaN slices do not warn."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(reducer(finite))


def amplitude(z_post: np.ndarray) -> np.ndarray:
    """A_z = (q0.95 - q0.05)/2 along time for each trajectory. z_post: [K, N]."""
    q95 = np.quantile(z_post, 0.95, axis=0)
    q05 = np.quantile(z_post, 0.05, axis=0)
    return 0.5 * (q95 - q05)


def distance_to_hard_orbit(z_post: np.ndarray, rho: float, beta: float = 1.0) -> np.ndarray:
    t = tau_c(rho, beta)
    return np.mean(np.minimum(np.abs(z_post - t), np.abs(z_post + t)), axis=0)


def terminal_distance_to_zero(z: np.ndarray) -> np.ndarray:
    return np.abs(z[-1])


def switch_rate(a_post: np.ndarray) -> np.ndarray:
    if a_post.shape[0] < 2:
        return np.zeros(a_post.shape[1])
    return np.mean(a_post[1:] != a_post[:-1], axis=0)


def lag_corr(z_post: np.ndarray, lag: int) -> np.ndarray:
    x = z_post[:-lag]
    y = z_post[lag:]
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    num = np.sum(x * y, axis=0)
    den = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    out = np.full(z_post.shape[1], np.nan)
    ok = den > 1e-15
    out[ok] = num[ok] / den[ok]
    return out


def period_two_spectral_share(z_post: np.ndarray) -> np.ndarray:
    """Power nearest frequency 1/2, divided by total nonzero-frequency power after centering."""
    x = z_post - z_post.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(x, axis=0)
    power = np.abs(spec) ** 2
    if power.shape[0] <= 1:
        return np.zeros(z_post.shape[1])
    nyquist_bin = power.shape[0] - 1
    total = np.sum(power[1:], axis=0)
    p2 = power[nyquist_bin]
    out = np.zeros(z_post.shape[1])
    ok = total > 1e-15
    out[ok] = p2[ok] / total[ok]
    return out


def classify_scalar_outcome(
    z: np.ndarray,
    a: np.ndarray,
    rho: float,
    beta: float,
    burnin: int,
    *,
    zero_tol: float = 1e-3,
    cycle_tol: float = 5e-3,
) -> dict[str, np.ndarray]:
    t = tau_c(rho, beta)
    z_post = z[burnin:]
    a_post = a[burnin:]
    amp = amplitude(z_post)
    d2 = distance_to_hard_orbit(z_post, rho, beta)
    term = terminal_distance_to_zero(z)
    sw = switch_rate(a_post)
    to_zero = (term <= zero_tol * max(t, 1e-12)) & (amp <= zero_tol * max(t, 1e-12))
    to_cycle = (d2 <= cycle_tol * max(t, 1e-12)) & (sw >= 0.9) & ~to_zero
    unresolved = ~(to_zero | to_cycle)
    return {
        "A_z": amp,
        "A_z_norm": amp / t,
        "D2": d2,
        "terminal_abs": term,
        "switch_rate": sw,
        "to_zero": to_zero.astype(np.float64),
        "to_cycle": to_cycle.astype(np.float64),
        "unresolved": unresolved.astype(np.float64),
    }


def detect_action_period(a_post: np.ndarray, max_period: int = 16) -> np.ndarray:
    """Smallest p in 1..max_period that exactly repeats on the second half; 0 if unresolved."""
    k, n = a_post.shape
    out = np.zeros(n, dtype=np.int32)
    tail = a_post[k // 2 :]
    m = tail.shape[0]
    for i in range(n):
        seq = tail[:, i]
        found = 0
        for p in range(1, min(max_period, m // 2) + 1):
            if m < 2 * p:
                continue
            if np.array_equal(seq[-2 * p : -p], seq[-p:]):
                # confirm over the available tail
                reps = m // p
                if reps < 2:
                    continue
                block = seq[-reps * p :].reshape(reps, p)
                if np.all(block == block[0]):
                    found = p
                    break
        out[i] = found
    return out


def dwell_stats(a_post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and max run length of constant action after burn-in."""
    k, n = a_post.shape
    mean_dwell = np.zeros(n)
    max_dwell = np.zeros(n)
    for i in range(n):
        seq = a_post[:, i]
        runs = []
        cur = 1
        for t in range(1, k):
            if seq[t] == seq[t - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        mean_dwell[i] = float(np.mean(runs))
        max_dwell[i] = float(np.max(runs))
    return mean_dwell, max_dwell


def false_boundary_crossings(a_obs: np.ndarray, a_true: np.ndarray) -> np.ndarray:
    return np.mean(a_obs != a_true, axis=0)


def policy_probability_churn(pi: np.ndarray) -> float:
    """pi: [K, n_ref] approval probabilities on a frozen reference panel."""
    if pi.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(pi[1:] - pi[:-1])))


def independent_bernoulli_churn(p: np.ndarray) -> float:
    """Expected switch probability if a_k ~ Bern(p_k) independently. p in [0,1]."""
    p = np.asarray(p, dtype=np.float64)
    if p.size < 2:
        return 0.0
    return float(np.mean(p[:-1] * (1 - p[1:]) + (1 - p[:-1]) * p[1:]))


def common_uniform_bernoulli_churn(p: np.ndarray) -> float:
    """Minimal expected switch under common-uniform coupling: |p_{k+1}-p_k|."""
    p = np.asarray(p, dtype=np.float64)
    if p.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(p))))


def fixed_policy_bernoulli_baseline(p: float) -> dict[str, float]:
    """Constant marginal p: independent switch 2p(1-p); common-uniform switch 0."""
    p = float(p)
    return {"independent": 2.0 * p * (1.0 - p), "common_uniform": 0.0}
