"""Independent theorem-oracle checks. This module must not import the simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dynafront import certificates as ora


def test_certificates_source_does_not_call_simulator() -> None:
    src = Path(ora.__file__).read_text(encoding="utf-8")
    assert "simulate" not in src
    assert "dynafront.dynamics" not in src


def test_tau_c_and_hard_cycle() -> None:
    assert ora.tau_c(0.0, 1.0) == 1.0
    assert np.isclose(ora.tau_c(0.5, 1.0), 1.0 / 1.5)
    zp, zm = ora.hard_cycle_scores(0.5, 1.0)
    assert np.isclose(zp, 2.0 / 3.0) and np.isclose(zm, -2.0 / 3.0)
    assert ora.two_step_hard_multiplier(0.5) == 0.25


def test_clipped_and_deadband_regimes() -> None:
    rho, beta = 0.5, 1.0
    t = ora.tau_c(rho, beta)
    assert ora.clipped_regime(1.5 * t, rho, beta) == "supercritical_global_zero"
    assert ora.clipped_regime(t, rho, beta) == "critical_neutral_continuum"
    assert ora.clipped_regime(0.5 * t, rho, beta) == "subcritical_hard_cycle"
    assert ora.clipped_lipschitz(rho, 2 * t, beta) < 1.0
    assert ora.deadband_threshold(rho, beta) == t
    assert np.isclose(ora.hysteresis_threshold(0.5, 1.0), 2.0)
    assert np.isclose(ora.hysteresis_over_tau_c_ratio(0.5), 3.0)


def test_tanh_amplitude_equation() -> None:
    r, res = ora.tanh_amplitude(0.5, 0.0, 1.0)
    assert res < 1e-12
    assert np.isclose(r, np.tanh(2 * r), atol=1e-10)
    r0, res0 = ora.tanh_amplitude(1.0, 0.0, 1.0)
    assert r0 == 0.0 and res0 == 0.0
    assert np.isclose(ora.tanh_sqrt_onset(0.99, 0.0, 1.0), np.sqrt(0.03))


def test_unaligned_resolvents() -> None:
    A = np.array([[0.2, 0.1], [0.0, 0.3]])
    u = np.array([1.0, 0.0])
    v = np.array([1.0, 0.5])
    cf = ora.chi_fix(A, u, v)
    xp, xm = ora.fixed_action_states(A, v)
    ap, am = ora.alternating_pair_states(A, v)
    assert np.allclose(xp, -np.linalg.solve(np.eye(2) - A, v))
    assert np.allclose(ap, np.linalg.solve(np.eye(2) + A, v))
    assert np.isclose(u @ xp, -cf)
    v2 = ora.min_norm_v_for_resolvents(A, u, 0.4, -0.3)
    assert np.isclose(ora.chi_fix(A, u, v2), 0.4, atol=1e-10)
    assert np.isclose(ora.chi_cyc(A, u, v2), -0.3, atol=1e-10)


def test_delayed_and_gaussian_oracles() -> None:
    assert ora.delayed_region(0.2, 0.5) == "proved_global"
    assert ora.delayed_region(0.2, 1.0) == "critical_not_locally_asymptotically_stable"
    assert ora.delayed_region(0.2, 1.2) == "locally_unstable"
    p = ora.gaussian_p_t(1, 0.5, 1.0, 0.2, 0.0)
    assert 0.0 < p < 1.0
