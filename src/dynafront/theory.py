"""Closed-form objects used by the paper theorems.

- Crossed equilibria imply infinitely many switches, not period two.
- The affine pair (z+, z-) with z+ > 0 > z- is the global period-two certificate
  and is equivalent to the cross-image conditions f+(f-(0)) < 0 < f-(f+(0)).
- Clipped selector and deadband share the exact global threshold tau_c = beta/(1+rho).
- Below the threshold they have different basin structures.
- Bounded-noise tubes apply to bounded perturbations, not automatically to Gaussians.
"""

from __future__ import annotations

import numpy as np


def tau_c(rho: float, beta: float = 1.0) -> float:
    return float(beta) / (1.0 + float(rho))


def hard_orbit_scores(rho: float, beta: float = 1.0) -> tuple[float, float]:
    t = tau_c(rho, beta)
    return t, -t


def fixed_action_score_equilibrium(rho_a: float, b_a: float) -> float:
    return float(b_a) / (1.0 - float(rho_a))


def predicted_alternating_pair(rho_plus: float, rho_minus: float, b_plus: float, b_minus: float) -> tuple[float, float]:
    den = 1.0 - float(rho_plus) * float(rho_minus)
    z_plus = (float(rho_minus) * float(b_plus) + float(b_minus)) / den
    z_minus = (float(b_plus) + float(rho_plus) * float(b_minus)) / den
    return float(z_plus), float(z_minus)


def equilibria_crossed(p_plus: float, p_minus: float) -> bool:
    return bool(p_plus < 0.0 < p_minus)


def pair_sign_compatible(z_plus: float, z_minus: float) -> bool:
    return bool(z_plus > 0.0 > z_minus)


def cross_image_conditions(rho_plus: float, rho_minus: float, b_plus: float, b_minus: float) -> bool:
    """Affine form of f+(f-(0)) < 0 and f-(f+(0)) > 0."""
    f_plus_of_f_minus_0 = rho_plus * b_minus + b_plus
    f_minus_of_f_plus_0 = rho_minus * b_plus + b_minus
    return bool(f_plus_of_f_minus_0 < 0.0 and f_minus_of_f_plus_0 > 0.0)


def bounded_noise_margin(rho: float, eps: float, delta: float, beta: float = 1.0) -> dict[str, float | bool]:
    r = float(delta) / (1.0 - float(rho))
    t = tau_c(rho, beta)
    return {
        "R": r,
        "tau_c": t,
        "eps_plus_R": float(eps) + r,
        "sufficient": bool(float(eps) + r < t),
    }


def parity_states(A: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x^+ = (I+A)^{-1} v, x^- = -(I+A)^{-1} v."""
    d = A.shape[0]
    w = np.linalg.solve(np.eye(d) + A, v)
    return w, -w


def inner_slope_continuous(rho: float, tau: float, beta: float = 1.0) -> float:
    return float(rho) - float(beta) / float(tau)
