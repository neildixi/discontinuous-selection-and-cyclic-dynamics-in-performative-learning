"""Selector maps and threshold identities."""

from __future__ import annotations

import numpy as np

from dynafront.certificates import hysteresis_threshold, tau_c
from dynafront.dynamics import simulate_delayed, simulate_hysteresis, simulate_scalar, simulate_tanh
from dynafront.selectors import clipped, deadband, ratio_grid, sign_plus_at_zero


def test_hard_sign_plus_at_zero() -> None:
    z = np.array([-1.0, 0.0, 1.0])
    assert np.array_equal(sign_plus_at_zero(z), np.array([-1.0, 1.0, 1.0]))


def test_clipped_grid_has_thirty_two_widths() -> None:
    g = ratio_grid(31, 0.25, 2.0)
    assert g.size == 32
    assert np.any(np.isclose(g, 1.0))


def test_clipped_and_deadband_maps() -> None:
    z = np.array([-2.0, -0.1, 0.0, 0.1, 2.0])
    c = clipped(z, 1.0)
    assert np.allclose(c, np.array([-1.0, -0.1, 0.0, 0.1, 1.0]))
    d = deadband(z, 0.5)
    assert d[1] == 0.0 and d[2] == 0.0
    assert d[0] == -1.0 and d[-1] == 1.0


def test_hysteresis_equality_retains_previous_at_rho0() -> None:
    z0 = np.array([1.0, -1.0, 0.0, 2.0])
    a0 = np.array([1.0, -1.0, 1.0, -1.0])
    _z, a = simulate_hysteresis(z0, a0, 0.0, 1.0, 5, beta=1.0)
    assert a[0, 0] == 1.0
    assert a[0, 1] == -1.0
    assert np.all(a[:, 0] == 1.0)


def test_tanh_subcritical_matches_threshold() -> None:
    rho, beta = 0.0, 1.0
    t = tau_c(rho, beta)
    z_super, _ = simulate_scalar(np.array([2.0]), rho, 200, "continuous", tau=1.5 * t, beta=beta)
    assert np.abs(z_super[-1, 0]) < 1e-8
    z = simulate_tanh(np.array([2.0]), rho, 0.5, 400, beta=beta)
    assert np.abs(z[-1, 0]) > 0.1


def test_delayed_proved_region_to_zero() -> None:
    rho, k = 0.2, 0.5
    tau = 1.0 / k
    z = simulate_delayed(np.array([3.0, -2.0]), np.array([-1.0, 4.0]), rho, tau, 400)
    assert np.max(np.abs(z[-1])) < 1e-8
    assert hysteresis_threshold(0.5, 1.0) == 2.0
