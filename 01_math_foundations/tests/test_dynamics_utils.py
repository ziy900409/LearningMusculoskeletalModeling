"""Regression tests for :mod:`dynamics_utils` (Stage 1, math foundations).

These enforce the four correctness claims that the notes make by hand:

1. **Energy conservation** -- the passive double pendulum meets the Stage-1
   milestone (relative energy range < 1 % over 10 s).
2. **Mass-matrix structure** -- ``M(q)`` is symmetric and positive definite for
   every configuration (so forward dynamics is well posed).
3. **Closed-form agreement** -- the point-mass special case reproduces the
   ``M, C qd, g`` expressions written in ``notes.md`` Section 3.3.
4. **Forward / inverse consistency** -- ``inverse_dynamics`` is the exact left
   inverse of ``forward_dynamics``.

Plus small unit tests for the ``energy_drift`` normalisation fix.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dynamics_utils import energy_drift, planar_chain


# --------------------------------------------------------------------------------------
# Fixtures: build each symbolic chain once per module (the SymPy derivation is the slow
# part, so we amortise it across tests).
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def point_mass_dp():
    """Classic point-mass double pendulum: c = L, I = 0, g = 9.81."""
    return planar_chain(n=2, m=[1.0, 1.0], L=[1.0, 1.0])


@pytest.fixture(scope="module")
def compound_dp():
    """Compound (rigid-link) double pendulum with COM inset and finite inertia."""
    return planar_chain(n=2, m=[1.3, 0.7], L=[1.1, 0.9], c=[0.5, 0.4], I=[0.05, 0.03])


# --------------------------------------------------------------------------------------
# 1. Energy conservation -- the Stage-1 milestone
# --------------------------------------------------------------------------------------
def test_energy_conserved_double_pendulum(point_mass_dp):
    dp = point_mass_dp
    y0 = np.array([np.radians(120), np.radians(-10), 0.0, 0.0])
    sol = solve_ivp(dp.state_derivative, (0, 10), y0, method="DOP853",
                    rtol=1e-10, atol=1e-12, max_step=0.01, dense_output=True)
    assert sol.success
    Q, QD = sol.y[:2].T, sol.y[2:].T
    _, _, E = dp.energies(Q, QD)
    d = energy_drift(E)
    # Milestone is < 1 %; a tight DOP853 run should beat it by many orders.
    assert d["rel_range_pct"] < 1.0
    assert d["rel_range_pct"] < 1e-3


# --------------------------------------------------------------------------------------
# 2. Mass matrix is symmetric positive definite everywhere
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("fixture_name", ["point_mass_dp", "compound_dp"])
def test_mass_matrix_symmetric_positive_definite(fixture_name, request):
    dp = request.getfixturevalue(fixture_name)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, size=dp.n)
        qd = rng.uniform(-3.0, 3.0, size=dp.n)
        M = dp.M(q, qd)
        assert np.allclose(M, M.T, atol=1e-10)          # symmetric
        assert np.all(np.linalg.eigvalsh(M) > 0.0)      # positive definite


# --------------------------------------------------------------------------------------
# 3. Point-mass special case matches the closed form in notes.md Section 3.3
# --------------------------------------------------------------------------------------
def _closed_form_point_mass(m1, m2, L1, L2, g, q, qd):
    """M(q), C(q,qd) qd and g(q) exactly as written in notes.md Section 3.3."""
    q1, q2 = q
    qd1, qd2 = qd
    d = q1 - q2
    M = np.array([[(m1 + m2) * L1 ** 2, m2 * L1 * L2 * np.cos(d)],
                  [m2 * L1 * L2 * np.cos(d), m2 * L2 ** 2]])
    Cqd = np.array([m2 * L1 * L2 * qd2 ** 2 * np.sin(d),
                    -m2 * L1 * L2 * qd1 ** 2 * np.sin(d)])
    grav = np.array([(m1 + m2) * g * L1 * np.sin(q1),
                     m2 * g * L2 * np.sin(q2)])
    return M, Cqd, grav


def test_point_mass_matches_closed_form(point_mass_dp):
    dp = point_mass_dp
    m1 = m2 = L1 = L2 = 1.0
    g = dp.params["g"]
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, size=2)
        qd = rng.uniform(-3.0, 3.0, size=2)
        M_ref, Cqd_ref, g_ref = _closed_form_point_mass(m1, m2, L1, L2, g, q, qd)
        assert np.allclose(dp.M(q, qd), M_ref, atol=1e-10)
        assert np.allclose(dp.gravity(q, qd), g_ref, atol=1e-10)
        # forcing = C(q,qd) qd + g(q)
        assert np.allclose(dp.forcing(q, qd), Cqd_ref + g_ref, atol=1e-10)


# --------------------------------------------------------------------------------------
# 4. Forward <-> inverse dynamics are exact inverses
# --------------------------------------------------------------------------------------
def test_forward_inverse_roundtrip(compound_dp):
    dp = compound_dp
    rng = np.random.default_rng(2)
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, size=2)
        qd = rng.uniform(-3.0, 3.0, size=2)

        # tau -> qdd -> tau
        tau = rng.uniform(-5.0, 5.0, size=2)
        qdd = dp.forward_dynamics(q, qd, tau)
        assert np.allclose(dp.inverse_dynamics(q, qd, qdd), tau, atol=1e-8)

        # qdd -> tau -> qdd
        qdd0 = rng.uniform(-5.0, 5.0, size=2)
        tau_rec = dp.inverse_dynamics(q, qd, qdd0)
        assert np.allclose(dp.forward_dynamics(q, qd, tau_rec), qdd0, atol=1e-8)


@pytest.mark.parametrize("fixture_name", ["point_mass_dp", "compound_dp"])
def test_coriolis_matrix_reproduces_velocity_term(fixture_name, request):
    """C(q,qd) @ qd must equal the velocity-dependent term forcing - gravity."""
    dp = request.getfixturevalue(fixture_name)
    rng = np.random.default_rng(4)
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, size=dp.n)
        qd = rng.uniform(-3.0, 3.0, size=dp.n)
        Cqd = dp.coriolis_matrix(q, qd) @ qd
        assert np.allclose(Cqd, dp.forcing(q, qd) - dp.gravity(q, qd), atol=1e-9)


@pytest.mark.parametrize("fixture_name", ["point_mass_dp", "compound_dp"])
def test_mdot_minus_2c_is_skew_symmetric(fixture_name, request):
    """The passivity property ``Mdot - 2C`` skew-symmetric (energy conservation)."""
    dp = request.getfixturevalue(fixture_name)
    rng = np.random.default_rng(5)
    eps = 1e-6
    for _ in range(30):
        q = rng.uniform(-np.pi, np.pi, size=dp.n)
        qd = rng.uniform(-3.0, 3.0, size=dp.n)
        C = dp.coriolis_matrix(q, qd)
        # Mdot = sum_k (dM/dq_k) qd_k, via central differences of M(q).
        Mdot = np.zeros((dp.n, dp.n))
        for k in range(dp.n):
            dq = np.zeros(dp.n)
            dq[k] = eps
            Mdot += (dp.M(q + dq, qd) - dp.M(q - dq, qd)) / (2 * eps) * qd[k]
        S = Mdot - 2 * C
        assert np.allclose(S, -S.T, atol=1e-5)


def test_inverse_dynamics_of_passive_trajectory_is_zero(point_mass_dp):
    """The passive chain needs zero torque to follow its own accelerations."""
    dp = point_mass_dp
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, size=2)
        qd = rng.uniform(-3.0, 3.0, size=2)
        qdd_passive = dp.forward_dynamics(q, qd, tau=None)
        assert np.allclose(dp.inverse_dynamics(q, qd, qdd_passive),
                           np.zeros(2), atol=1e-8)


# --------------------------------------------------------------------------------------
# 5. energy_drift normalisation (the C3 fix)
# --------------------------------------------------------------------------------------
def test_energy_drift_basic():
    E = 5.0 + np.linspace(0.0, 0.05, 100)   # 0.05 J drift on a ~5 J scale
    d = energy_drift(E)
    assert d["rel_range"] == pytest.approx(0.05 / np.max(np.abs(E)), rel=1e-9)


def test_energy_drift_not_capped_when_E0_near_zero():
    # E0 ~ 0 (datum choice) but a big swing: relative range must exceed 1, which
    # the old max(|E0|, ptp) normalisation silently clipped to exactly 1.
    E = np.array([1e-6, 1e-6, 1.0, -1.0, 1e-6])
    d = energy_drift(E)
    assert d["rel_range"] == pytest.approx(2.0, rel=1e-9)   # ptp=2, scale=max|E|=1


def test_energy_drift_scale_override():
    E = np.array([0.0, 0.1, -0.1, 0.0])
    d = energy_drift(E, scale=10.0)
    assert d["rel_range"] == pytest.approx(0.2 / 10.0, rel=1e-9)
