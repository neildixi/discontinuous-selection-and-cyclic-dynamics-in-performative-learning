"""Independent theorem oracle.

This module computes closed-form predictions from the verified theory notes.
It must not import or call simulation update functions.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Score-aligned scalar hard / clipped / deadband
# ---------------------------------------------------------------------------


def tau_c(rho: float, beta: float = 1.0) -> float:
    rho = float(rho)
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 <= rho < 1")
    return float(beta) / (1.0 + rho)


def hard_cycle_scores(rho: float, beta: float = 1.0) -> tuple[float, float]:
    t = tau_c(rho, beta)
    return t, -t


def two_step_hard_multiplier(rho: float) -> float:
    return float(rho) ** 2


def clipped_regime(tau: float, rho: float, beta: float = 1.0, *, rel_tol: float = 0.0) -> str:
    t = tau_c(rho, beta)
    tau = float(tau)
    if tau > t * (1.0 + rel_tol):
        return "supercritical_global_zero"
    if tau < t * (1.0 - rel_tol):
        return "subcritical_hard_cycle"
    return "critical_neutral_continuum"


def clipped_lipschitz(rho: float, tau: float, beta: float = 1.0) -> float:
    """Global Lipschitz constant of the piecewise-affine clipped map."""
    m_inner = float(rho) - float(beta) / float(tau)
    return float(max(abs(rho), abs(m_inner)))


def deadband_threshold(rho: float, beta: float = 1.0) -> float:
    return tau_c(rho, beta)


def hysteresis_threshold(rho: float, beta: float = 1.0) -> float:
    return float(beta) / (1.0 - float(rho))


def hysteresis_over_tau_c_ratio(rho: float) -> float:
    return (1.0 + float(rho)) / (1.0 - float(rho))


# ---------------------------------------------------------------------------
# Unaligned multidimensional two-resolvent
# ---------------------------------------------------------------------------


def chi_fix(A: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    d = A.shape[0]
    return float(u @ np.linalg.solve(np.eye(d) - A, v))


def chi_cyc(A: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    d = A.shape[0]
    return float(u @ np.linalg.solve(np.eye(d) + A, v))


def fixed_action_states(A: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x^{*,+} = -(I-A)^{-1} v  (action +1),  x^{*,-} = +(I-A)^{-1} v (action -1)."""
    d = A.shape[0]
    w = np.linalg.solve(np.eye(d) - A, v)
    return -w, w


def alternating_pair_states(A: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x^+ = (I+A)^{-1} v, x^- = -(I+A)^{-1} v."""
    d = A.shape[0]
    w = np.linalg.solve(np.eye(d) + A, v)
    return w, -w


def fixed_points_crossed(chi_f: float) -> bool:
    return bool(chi_f > 0.0)


def alternating_pair_compatible(chi_c: float) -> bool:
    return bool(chi_c > 0.0)


def alignment_residual(A: np.ndarray, u: np.ndarray) -> float:
    """Relative residual of u^T A against its projection onto u^T."""
    u = np.asarray(u, dtype=np.float64)
    row = u @ A
    denom = float(u @ u)
    rho_hat = float(row @ u) / denom if denom > 0 else 0.0
    proj = rho_hat * u
    num = float(np.linalg.norm(row - proj))
    den = float(np.linalg.norm(row))
    if den < 1e-15:
        return 0.0
    return num / den


def min_norm_v_for_resolvents(
    A: np.ndarray,
    u: np.ndarray,
    target_chi_fix: float,
    target_chi_cyc: float,
) -> np.ndarray:
    d = A.shape[0]
    p = np.linalg.solve(np.eye(d) - A.T, u)  # (I-A)^{-T} u  so p^T v = u^T (I-A)^{-1} v
    q = np.linalg.solve(np.eye(d) + A.T, u)
    M = np.column_stack([p, q])  # d x 2
    gram = M.T @ M
    rhs = np.array([target_chi_fix, target_chi_cyc], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    return M @ coef


# ---------------------------------------------------------------------------
# Tanh selector
# ---------------------------------------------------------------------------


def tanh_regime(tau: float, rho: float, beta: float = 1.0, *, rel_tol: float = 0.0) -> str:
    t = tau_c(rho, beta)
    tau = float(tau)
    if tau > t * (1.0 + rel_tol):
        return "supercritical_global_zero_exponential"
    if tau < t * (1.0 - rel_tol):
        return "subcritical_amplitude_cycle"
    return "critical_zero_nonexponential"


def tanh_amplitude(tau: float, rho: float, beta: float = 1.0) -> tuple[float, float]:
    """Return (r_tau, residual). r_tau = 0 at/above threshold."""
    t = tau_c(rho, beta)
    tau = float(tau)
    rho = float(rho)
    beta = float(beta)
    if tau >= t:
        return 0.0, 0.0

    def f(r: float) -> float:
        return (1.0 + rho) * r - beta * np.tanh(r / tau)

    # Unique positive root: f(0)=0, f'(0)<0, f -> +inf.
    lo = 1e-16
    hi = max(10.0 * t, 10.0 * tau, 1.0)
    for _ in range(80):
        if f(hi) > 0.0:
            break
        hi *= 2.0
    # Move lo until f(lo) < 0
    if f(lo) >= 0.0:
        # extremely close to threshold
        r = t * np.sqrt(max(3.0 * (1.0 - tau / t), 0.0))
        return float(r), float(abs(f(r)))
    r = float(brentq(f, lo, hi, xtol=1e-16, rtol=1e-14, maxiter=200))
    return r, float(abs(f(r)))


def tanh_sqrt_onset(tau: float, rho: float, beta: float = 1.0) -> float:
    t = tau_c(rho, beta)
    x = 1.0 - float(tau) / t
    if x <= 0.0:
        return 0.0
    return t * np.sqrt(3.0 * x)


def tanh_two_step_multiplier(r_tau: float, tau: float, rho: float, beta: float = 1.0) -> float:
    sech2 = 1.0 / np.cosh(r_tau / float(tau)) ** 2
    gp = float(rho) - (float(beta) / float(tau)) * sech2
    return float(gp * gp)


# ---------------------------------------------------------------------------
# Affine finite itineraries
# ---------------------------------------------------------------------------


def word_multiplier(word: Sequence[str], rho_plus: float, rho_minus: float) -> float:
    r = 1.0
    for w in word:
        r *= rho_plus if w in {"+", "1", 1, True} else rho_minus
    return float(r)


def word_offset(word: Sequence[str], rho_plus: float, rho_minus: float, b_plus: float, b_minus: float) -> float:
    """B_w = sum_j b_{w_j} prod_{ell=j+1}^{p-1} rho_{w_ell}."""
    p = len(word)
    acc = 0.0
    for j in range(p):
        b = b_plus if word[j] in {"+", "1", 1, True} else b_minus
        prod = 1.0
        for ell in range(j + 1, p):
            prod *= rho_plus if word[ell] in {"+", "1", 1, True} else rho_minus
        acc += b * prod
    return float(acc)


def word_candidate_states(
    word: Sequence[str],
    rho_plus: float,
    rho_minus: float,
    b_plus: float,
    b_minus: float,
) -> tuple[np.ndarray, float, float]:
    R = word_multiplier(word, rho_plus, rho_minus)
    B = word_offset(word, rho_plus, rho_minus, b_plus, b_minus)
    if abs(1.0 - R) < 1e-15:
        raise ValueError("word multiplier is 1")
    z0 = B / (1.0 - R)
    zs = np.empty(len(word), dtype=np.float64)
    zs[0] = z0
    z = z0
    for j, w in enumerate(word[:-1]):
        if w in {"+", "1", 1, True}:
            z = rho_plus * z + b_plus
        else:
            z = rho_minus * z + b_minus
        zs[j + 1] = z
    return zs, float(R), float(B)


def word_is_strict_orbit(zs: np.ndarray, word: Sequence[str], *, margin: float = 1e-8) -> bool:
    for z, w in zip(zs, word):
        if w in {"+", "1", 1, True}:
            if not (z > margin):
                return False
        else:
            if not (z < -margin):
                return False
    return True


def is_primitive(word: Sequence[str]) -> bool:
    p = len(word)
    for d in range(1, p):
        if p % d == 0 and tuple(word) == tuple(word[:d]) * (p // d):
            return False
    return True


def necklace_key(word: Sequence[str]) -> tuple:
    w = tuple(word)
    rots = [w[i:] + w[:i] for i in range(len(w))]
    return min(rots)


def unique_primitive_words(max_period: int = 8) -> list[tuple[str, ...]]:
    seen: set[tuple] = set()
    out: list[tuple[str, ...]] = []
    for p in range(1, max_period + 1):
        for mask in range(1 << p):
            w = tuple("+" if (mask >> j) & 1 else "-" for j in range(p))
            if not is_primitive(w):
                continue
            key = necklace_key(w)
            if key in seen:
                continue
            seen.add(key)
            out.append(w)
    return out


def predicted_period_two_pair(rho_plus: float, rho_minus: float, b_plus: float, b_minus: float) -> tuple[float, float]:
    den = 1.0 - float(rho_plus) * float(rho_minus)
    z_plus = (float(rho_minus) * float(b_plus) + float(b_minus)) / den
    z_minus = (float(b_plus) + float(rho_plus) * float(b_minus)) / den
    return float(z_plus), float(z_minus)


def affine_fixed_points(rho_plus: float, rho_minus: float, b_plus: float, b_minus: float) -> tuple[float, float]:
    return b_plus / (1.0 - rho_plus), b_minus / (1.0 - rho_minus)


# ---------------------------------------------------------------------------
# Delayed clipping
# ---------------------------------------------------------------------------


def delayed_k(tau: float, beta: float = 1.0) -> float:
    return float(beta) / float(tau)


def delayed_region(rho: float, k: float) -> str:
    rho = float(rho)
    k = float(k)
    if k > 1.0:
        return "locally_unstable"
    if abs(k - 1.0) <= 1e-15:
        return "critical_not_locally_asymptotically_stable"
    # k < 1: locally exponentially stable
    if rho + k < 1.0:
        return "proved_global"
    return "locally_stable_globally_unresolved"


def delayed_jury_roots(rho: float, k: float) -> np.ndarray:
    """Characteristic polynomial of the inner linear delay map: r^2 - rho r + k = 0."""
    return np.roots([1.0, -float(rho), float(k)])


# ---------------------------------------------------------------------------
# Gaussian observability (exact marginals / bounds; not a metastability exponent)
# ---------------------------------------------------------------------------


def gaussian_sigma_t(t: int, rho: float, sigma_s: float, sigma_eta: float) -> float:
    rho = float(rho)
    if t <= 0:
        return float(sigma_s)
    if abs(rho) < 1e-15:
        trans = float(sigma_eta) ** 2
    else:
        trans = (float(sigma_eta) ** 2) * (1.0 - rho ** (2 * t)) / (1.0 - rho ** 2)
    return float(np.sqrt(float(sigma_s) ** 2 + trans))


def gaussian_p_t(t: int, rho: float, tau: float, sigma_s: float, sigma_eta: float) -> float:
    s = gaussian_sigma_t(t, rho, sigma_s, sigma_eta)
    if s <= 0.0:
        return 0.0
    return float(ndtr(-float(tau) / s))


def gaussian_p_inf(rho: float, tau: float, sigma_s: float, sigma_eta: float) -> float:
    rho = float(rho)
    trans = 0.0 if abs(sigma_eta) < 1e-18 else (float(sigma_eta) ** 2) / (1.0 - rho ** 2 if abs(rho) > 1e-15 else 1.0)
    s = float(np.sqrt(float(sigma_s) ** 2 + trans))
    if s <= 0.0:
        return 0.0
    return float(ndtr(-float(tau) / s))


def gaussian_horizon_bounds(p: np.ndarray) -> dict[str, float]:
    """max p_t <= Pr(T<=N) <= sum p_t <= N p_inf, using p[1:] as t=1..N (p[0] unused)."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("p must be 1d")
    # allow p indexed from t=1 as p[0]... or from 0
    vals = p[p == p]
    s = float(np.sum(vals))
    m = float(np.max(vals)) if vals.size else 0.0
    return {"union_upper": s, "max_lower": m}


def gaussian_mean_waiting_lower_bound(p_inf: float) -> float | None:
    if p_inf <= 0.0:
        return None
    if p_inf <= 0.5:
        return 1.0 / (4.0 * p_inf)
    return None


def phi_cdf(x: float) -> float:
    return float(norm.cdf(x))
