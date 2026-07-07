"""Regression tests for :mod:`hamiltonian_utils` (Stage 1, Part 3: Hamiltonian mechanics).

These enforce, pointwise and over random samples, the claims the notes make by hand for
the phase-space (Hamiltonian) formulation:

1. **Legendre transform** -- ``p = M(q) qd`` and ``qd = M(q)^{-1} p`` are mutual inverses,
   and the Hamiltonian ``H(q, p)`` equals the Lagrangian total energy ``T + V``.
2. **Canonical equations** -- ``pdot`` assembled from the Christoffel Coriolis term equals
   the independent finite-difference gradient ``-dH/dq``; the Hamiltonian flow integrates
   the *same* trajectory as the Lagrangian flow (with and without joint torques); energy
   is conserved for the passive chain.
3. **Poisson brackets** -- the canonical relations ``{q_i, p_j} = delta_ij`` hold, and
   ``{q_i, H} = qdot_i``, ``{p_i, H} = pdot_i`` reproduce the canonical equations.
4. **Symplectic integrators** -- ``symplectic_euler`` and ``leapfrog`` are area-preserving
   (unit-determinant one-step map) while ``forward_euler`` is not; symplectic schemes keep
   the energy in a bounded band over long horizons whereas forward Euler drifts; leapfrog
   is second-order and time-reversible.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dynamics_utils import planar_chain, limb_chain
from hamiltonian_utils import (
    conjugate_momentum, velocity_from_momentum, hamiltonian,
    hamilton_state_derivative, poisson_bracket,
    symplectic_euler, leapfrog, forward_euler, rk4,
    pendulum_field, harmonic_oscillator_field, energy_band, SeparableSystem,
)


@pytest.fixture(scope="module")
def dp():
    """Point-mass double pendulum (the Part 1 workhorse)."""
    return planar_chain(n=2, m=[1.0, 1.2], L=[1.0, 0.8])


@pytest.fixture(scope="module")
def leg():
    """A compound (rigid-link) two-segment limb -- non-trivial M(q), c<L, I>0."""
    return limb_chain(("thigh", "shank"), body_mass=70.0, height=1.75)


def _random_states(n, count, rng, scale=1.5):
    for _ in range(count):
        yield rng.uniform(-scale, scale, size=n), rng.uniform(-scale, scale, size=n)


# --------------------------------------------------------------------------------------
# 1. Legendre transform:  p = M qd  <->  qd = M^{-1} p,  and  H = T + V
# --------------------------------------------------------------------------------------
def test_legendre_round_trip(dp):
    rng = np.random.default_rng(0)
    for q, qd in _random_states(dp.n, 100, rng):
        p = conjugate_momentum(dp, q, qd)
        assert np.allclose(velocity_from_momentum(dp, q, p), qd, atol=1e-9)


def test_hamiltonian_equals_total_energy(dp, leg):
    rng = np.random.default_rng(1)
    for chain in (dp, leg):
        for q, qd in _random_states(chain.n, 60, rng):
            p = conjugate_momentum(chain, q, qd)
            assert np.isclose(hamiltonian(chain, q, p), chain.total_energy(q, qd),
                              rtol=1e-10, atol=1e-10)


def test_momentum_is_symmetric_positive_form(dp):
    # p = M qd with M symmetric positive definite -> kinetic energy 1/2 p . qd >= 0
    rng = np.random.default_rng(2)
    for q, qd in _random_states(dp.n, 50, rng):
        p = conjugate_momentum(dp, q, qd)
        T = 0.5 * p @ qd
        assert T >= -1e-12
        assert np.isclose(T, dp.kinetic_energy(q, qd), rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------------------
# 2. Canonical equations
# --------------------------------------------------------------------------------------
def test_pdot_equals_minus_dH_dq(dp, leg):
    """The assembled ``pdot = C^T qd - g`` must equal an independent -dH/dq (finite diff)."""
    rng = np.random.default_rng(3)
    eps = 1e-6
    for chain in (dp, leg):
        for q, qd in _random_states(chain.n, 40, rng):
            p = conjugate_momentum(chain, q, qd)
            z = np.concatenate([q, p])
            pdot = hamilton_state_derivative(chain, 0.0, z)[chain.n:]
            # numerical -dH/dq at fixed p
            grad = np.zeros(chain.n)
            for i in range(chain.n):
                dq = np.zeros(chain.n); dq[i] = eps
                grad[i] = (hamiltonian(chain, q + dq, p)
                           - hamiltonian(chain, q - dq, p)) / (2 * eps)
            assert np.allclose(pdot, -grad, atol=1e-5)


def test_qdot_equals_dH_dp(dp):
    """qdot from the canonical equations equals dH/dp = M^{-1} p (finite diff)."""
    rng = np.random.default_rng(4)
    eps = 1e-6
    for q, qd in _random_states(dp.n, 40, rng):
        p = conjugate_momentum(dp, q, qd)
        qdot = hamilton_state_derivative(dp, 0.0, np.concatenate([q, p]))[:dp.n]
        grad = np.zeros(dp.n)
        for i in range(dp.n):
            dp_ = np.zeros(dp.n); dp_[i] = eps
            grad[i] = (hamiltonian(dp, q, p + dp_)
                       - hamiltonian(dp, q, p - dp_)) / (2 * eps)
        assert np.allclose(qdot, grad, atol=1e-5)
        assert np.allclose(qdot, qd, atol=1e-9)          # and it recovers the velocity


def test_hamiltonian_matches_lagrangian_trajectory(dp):
    """Integrating the canonical equations gives the same motion as the Lagrangian form."""
    q0 = np.array([np.radians(100.0), np.radians(-20.0)])
    qd0 = np.array([0.3, -0.5])
    p0 = conjugate_momentum(dp, q0, qd0)
    tspan, ts = (0.0, 8.0), np.linspace(0.0, 8.0, 400)

    solL = solve_ivp(dp.state_derivative, tspan, np.concatenate([q0, qd0]),
                     method="DOP853", rtol=1e-11, atol=1e-12, dense_output=True)
    solH = solve_ivp(lambda t, z: hamilton_state_derivative(dp, t, z), tspan,
                     np.concatenate([q0, p0]), method="DOP853",
                     rtol=1e-11, atol=1e-12, dense_output=True)
    qL, qH = solL.sol(ts)[:2], solH.sol(ts)[:2]
    assert np.max(np.abs(qL - qH)) < 1e-6


def test_hamiltonian_matches_lagrangian_with_torque(dp):
    """The equivalence also holds under a prescribed joint torque tau_fn."""
    def tau_fn(t, q, qd):
        return np.array([0.5 * np.sin(2.0 * t), -0.2 * qd[1]])   # drive + damping

    q0 = np.array([0.4, -0.3]); qd0 = np.array([0.0, 0.1])
    p0 = conjugate_momentum(dp, q0, qd0)
    tspan, ts = (0.0, 6.0), np.linspace(0.0, 6.0, 300)

    solL = solve_ivp(lambda t, y: dp.state_derivative(t, y, tau_fn), tspan,
                     np.concatenate([q0, qd0]), method="DOP853",
                     rtol=1e-11, atol=1e-12, dense_output=True)
    solH = solve_ivp(lambda t, z: hamilton_state_derivative(dp, t, z, tau_fn), tspan,
                     np.concatenate([q0, p0]), method="DOP853",
                     rtol=1e-11, atol=1e-12, dense_output=True)
    assert np.max(np.abs(solL.sol(ts)[:2] - solH.sol(ts)[:2])) < 1e-6


def test_energy_conserved_under_canonical_flow(dp):
    """H is conserved along the passive canonical flow (Hamiltonian is autonomous)."""
    q0 = np.array([np.radians(130.0), np.radians(-15.0)])
    p0 = conjugate_momentum(dp, q0, np.zeros(2))
    sol = solve_ivp(lambda t, z: hamilton_state_derivative(dp, t, z), (0.0, 10.0),
                    np.concatenate([q0, p0]), method="DOP853",
                    rtol=1e-11, atol=1e-12, dense_output=True)
    ts = np.linspace(0.0, 10.0, 1000)
    Z = sol.sol(ts)
    H = np.array([hamiltonian(dp, Z[:2, k], Z[2:, k]) for k in range(len(ts))])
    assert energy_band(H)["rel_range_pct"] < 1e-3          # milestone-grade conservation


# --------------------------------------------------------------------------------------
# 3. Poisson brackets
# --------------------------------------------------------------------------------------
def test_canonical_poisson_relations():
    """{q_i, p_j} = delta_ij and {q_i, q_j} = {p_i, p_j} = 0."""
    rng = np.random.default_rng(5)
    n = 3
    q = rng.uniform(-1, 1, n); p = rng.uniform(-1, 1, n)
    for i in range(n):
        for j in range(n):
            qi = (lambda i: (lambda qq, pp: qq[i]))(i)
            pj = (lambda j: (lambda qq, pp: pp[j]))(j)
            qj = (lambda j: (lambda qq, pp: qq[j]))(j)
            pi = (lambda i: (lambda qq, pp: pp[i]))(i)
            assert np.isclose(poisson_bracket(qi, pj, q, p), 1.0 if i == j else 0.0,
                              atol=1e-6)
            assert np.isclose(poisson_bracket(qi, qj, q, p), 0.0, atol=1e-6)
            assert np.isclose(poisson_bracket(pi, pj, q, p), 0.0, atol=1e-6)


def test_poisson_bracket_generates_canonical_equations(dp):
    """{q_i, H} = qdot_i and {p_i, H} = pdot_i -- the bracket *is* the time evolution."""
    rng = np.random.default_rng(6)
    H = lambda qq, pp: hamiltonian(dp, qq, pp)
    for q, qd in _random_states(dp.n, 20, rng, scale=1.0):
        p = conjugate_momentum(dp, q, qd)
        deriv = hamilton_state_derivative(dp, 0.0, np.concatenate([q, p]))
        qdot, pdot = deriv[:dp.n], deriv[dp.n:]
        for i in range(dp.n):
            qi = (lambda i: (lambda qq, pp: qq[i]))(i)
            pi = (lambda i: (lambda qq, pp: pp[i]))(i)
            assert np.isclose(poisson_bracket(qi, H, q, p), qdot[i], atol=1e-4)
            assert np.isclose(poisson_bracket(pi, H, q, p), pdot[i], atol=1e-4)


def test_energy_poisson_commutes_with_itself(dp):
    """{H, H} = 0 (energy conservation as a bracket statement)."""
    rng = np.random.default_rng(7)
    H = lambda qq, pp: hamiltonian(dp, qq, pp)
    for q, qd in _random_states(dp.n, 20, rng):
        p = conjugate_momentum(dp, q, qd)
        assert np.isclose(poisson_bracket(H, H, q, p), 0.0, atol=1e-6)


# --------------------------------------------------------------------------------------
# 4. Symplectic integrators
# --------------------------------------------------------------------------------------
def _one_step_map(integ, system, dt):
    """2x2 one-step Jacobian of a *linear* separable system about the origin fixed point."""
    def step(z):
        Q, P = integ(system.force, [z[0]], [z[1]], dt, 1, inv_mass=system.inv_mass)
        return np.array([Q[1, 0], P[1, 0]])
    col0 = step(np.array([1.0, 0.0]))
    col1 = step(np.array([0.0, 1.0]))
    return np.column_stack([col0, col1])


def test_symplectic_maps_preserve_area():
    """Harmonic oscillator: symplectic Euler and leapfrog have unit-determinant maps."""
    ho = harmonic_oscillator_field(k=1.7, m=0.9)
    dt = 0.1
    for integ in (symplectic_euler, leapfrog):
        A = _one_step_map(integ, ho, dt)
        assert np.isclose(np.linalg.det(A), 1.0, atol=1e-12)


def test_forward_euler_inflates_area():
    """Forward Euler is NOT symplectic: its map determinant exceeds 1."""
    ho = harmonic_oscillator_field(k=1.7, m=0.9)
    A = _one_step_map(forward_euler, ho, 0.1)
    det = np.linalg.det(A)
    assert det > 1.0
    assert np.isclose(det, 1.0 + 1.7 / 0.9 * 0.1**2, atol=1e-12)   # 1 + (k/m) dt^2


def test_symplectic_energy_bounded_forward_euler_drifts():
    """Over a long horizon symplectic schemes stay in a band; forward Euler drifts away."""
    pend = pendulum_field(m=1.0, L=1.0)
    th0, p0, dt, N = np.radians(60.0), 0.0, 0.01, 20000     # 200 s of swinging
    Qs, Ps = symplectic_euler(pend.force, [th0], [p0], dt, N, inv_mass=pend.inv_mass)
    Ql, Pl = leapfrog(pend.force, [th0], [p0], dt, N, inv_mass=pend.inv_mass)
    Qf, Pf = forward_euler(pend.force, [th0], [p0], dt, N, inv_mass=pend.inv_mass)

    band_s = energy_band(pend.energies(Qs, Ps))["rel_range_pct"]
    band_l = energy_band(pend.energies(Ql, Pl))["rel_range_pct"]
    band_f = energy_band(pend.energies(Qf, Pf))["rel_range_pct"]

    assert band_l < 0.5                       # leapfrog tightly bounded
    assert band_s < 5.0                       # symplectic Euler bounded (1st order)
    assert band_f > 10.0 * band_l             # forward Euler drifts far more


def test_symplectic_energy_does_not_grow_secularly():
    """A symplectic integrator's energy error does not grow with the horizon."""
    pend = pendulum_field(m=1.0, L=1.0)
    th0, p0, dt = np.radians(70.0), 0.0, 0.01
    short = energy_band(pend.energies(*leapfrog(pend.force, [th0], [p0], dt, 5000,
                                                inv_mass=pend.inv_mass)))["rel_max_dev"]
    long = energy_band(pend.energies(*leapfrog(pend.force, [th0], [p0], dt, 40000,
                                               inv_mass=pend.inv_mass)))["rel_max_dev"]
    assert long < 3.0 * short                 # bounded, not linear in time


def test_leapfrog_is_second_order():
    """Halving dt cuts the leapfrog global error by ~4x (order-2 convergence)."""
    ho = harmonic_oscillator_field(k=1.0, m=1.0)   # exact period 2 pi
    q0, p0, T = 1.0, 0.0, 6.0

    def err(dt):
        N = int(round(T / dt))
        Q, P = leapfrog(ho.force, [q0], [p0], dt, N, inv_mass=ho.inv_mass)
        tN = N * dt
        exact = np.array([q0 * np.cos(tN), -q0 * np.sin(tN)])   # m=k=1
        return np.linalg.norm([Q[-1, 0] - exact[0], P[-1, 0] - exact[1]])

    ratio = err(0.02) / err(0.01)
    assert 3.5 < ratio < 4.5


def test_leapfrog_time_reversible():
    """Integrating forward then backward (negate p, run, negate p) returns to the start."""
    pend = pendulum_field(m=1.0, L=1.0)
    th0, p0, dt, N = np.radians(80.0), 0.2, 0.005, 4000
    Q, P = leapfrog(pend.force, [th0], [p0], dt, N, inv_mass=pend.inv_mass)
    # reverse momentum and integrate forward: retraces the path
    Qb, Pb = leapfrog(pend.force, [Q[-1, 0]], [-P[-1, 0]], dt, N, inv_mass=pend.inv_mass)
    assert np.isclose(Qb[-1, 0], th0, atol=1e-9)
    assert np.isclose(Pb[-1, 0], -p0, atol=1e-9)


def test_integrators_match_reference_pendulum_trajectory():
    """For small dt the symplectic schemes agree with a tight solve_ivp reference."""
    pend = pendulum_field(m=1.0, L=1.0)
    th0, p0, dt, N = np.radians(45.0), 0.0, 0.002, 2500      # 5 s
    Q, P = leapfrog(pend.force, [th0], [p0], dt, N, inv_mass=pend.inv_mass)

    def rhs(t, z):
        return [pend.inv_mass * z[1], float(pend.force(z[0])[0])]
    ref = solve_ivp(rhs, (0.0, N * dt), [th0, p0], method="DOP853",
                    rtol=1e-12, atol=1e-13, dense_output=True)
    qref, pref = ref.sol(N * dt)
    assert np.isclose(Q[-1, 0], qref, atol=1e-4)
    assert np.isclose(P[-1, 0], pref, atol=1e-4)


def test_separable_system_energies_shape():
    """SeparableSystem.energies returns one energy per (q, p) sample and is conserved-ish."""
    ho = harmonic_oscillator_field()
    Q, P = leapfrog(ho.force, [1.0], [0.0], 0.01, 500, inv_mass=ho.inv_mass)
    E = ho.energies(Q, P)
    assert E.shape == (501,)
    assert energy_band(E)["rel_range_pct"] < 0.1
