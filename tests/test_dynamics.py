"""Simulator checks against independently stated closed forms."""

from __future__ import annotations

import numpy as np

from dynafront.certificates import tau_c
from dynafront.dynamics import simulate_asymmetric_scalar, simulate_scalar
from dynafront.metrics import distance_to_hard_orbit
from dynafront.theory import (
    cross_image_conditions,
    hard_orbit_scores,
    predicted_alternating_pair,
)


def test_hard_orbit_attracts_and_two_step_is_rho_squared() -> None:
    rho, beta = 0.5, 1.0
    z_plus, z_minus = hard_orbit_scores(rho, beta)
    assert np.isclose(z_plus, beta / (1 + rho))
    assert np.isclose(z_minus, -beta / (1 + rho))
    z, _a = simulate_scalar(np.array([3.0, -2.0, 0.1, -0.7]), rho, 200, "hard", beta=beta)
    d = distance_to_hard_orbit(z[100:], rho, beta)
    assert np.all(d < 1e-10)
    t = tau_c(rho, beta)
    z_tr, _ = simulate_scalar(np.array([3.0]), rho, 40, "hard", beta=beta)
    seq = z_tr[:, 0]
    pos = np.where(seq[:-2] > 0)[0]
    den = seq[pos] - t
    num = seq[pos + 2] - t
    ok = np.abs(den) > 1e-10
    rate = np.median(num[ok] / den[ok])
    assert np.isclose(rate, rho**2, atol=1e-8)


def test_asymmetric_predicted_pair() -> None:
    rho_p, rho_m, b_p, b_m = 0.4, 0.6, -1.2, 0.9
    zp, zm = predicted_alternating_pair(rho_p, rho_m, b_p, b_m)
    assert zp > 0 > zm
    assert np.isclose(rho_p * zp + b_p, zm)
    assert np.isclose(rho_m * zm + b_m, zp)
    assert cross_image_conditions(rho_p, rho_m, b_p, b_m)
    z, _a = simulate_asymmetric_scalar(np.linspace(-4, 4, 21), rho_p, rho_m, b_p, b_m, 400)
    tail = z[200:]
    pos = [np.mean(tail[tail[:, i] >= 0, i]) for i in range(tail.shape[1]) if np.any(tail[:, i] >= 0)]
    neg = [np.mean(tail[tail[:, i] < 0, i]) for i in range(tail.shape[1]) if np.any(tail[:, i] < 0)]
    assert np.allclose(pos, zp, atol=1e-8)
    assert np.allclose(neg, zm, atol=1e-8)


def test_frozen_policy_is_a_contraction() -> None:
    rho, beta = 0.5, 1.0
    z = 2.5
    for _ in range(40):
        z = rho * z - beta
    assert abs(z + beta / (1 - rho)) < 1e-10
