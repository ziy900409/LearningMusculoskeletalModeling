r"""
hamiltonian_utils.py
====================

Reusable utilities for Stage 1, Part 3 (math foundations) of the
musculoskeletal-modeling learning path: **Hamiltonian mechanics and phase space**.

Part 1 (``dynamics_utils.py``) built the *second-order* Lagrangian equations of motion
``M(q) qdd + C(q,qd) qd + g(q) = tau`` in the ``(q, qd)`` state.  This module recasts the
same mechanics as a *first-order* flow on **phase space** ``(q, p)``, where the
**conjugate momentum** ``p = dL/dqd = M(q) qd`` replaces the velocity.  The payoff is
threefold:

* **Symmetry.** Hamilton's canonical equations ``qdot = dH/dp``, ``pdot = -dH/dq`` treat
  position and momentum on the same footing; energy ``H = T + V`` becomes the generator
  of the motion.
* **Structure.** The flow is *symplectic* (it preserves the canonical 2-form, hence
  phase-space volume -- Liouville's theorem).  Integrators that respect this structure
  (:func:`symplectic_euler`, :func:`leapfrog`) have **no secular energy drift**, unlike
  a generic Runge-Kutta method -- the deep reason behind the Part 1 remark that
  *structure beats order* (see ``notes.md`` Sections 6.3 and 28).
* **Control.** The costate/adjoint of optimal control obeys ``lambdadot = -dH/dx`` -- the
  *same* equation as ``pdot`` -- so the momentum of mechanics is the shadow price of the
  effort-minimising simulations in Stage 6 (static optimisation) and Stage 8 (Moco).

Two layers are provided:

1. **Canonical dynamics of a** :class:`~dynamics_utils.PlanarChain`.
   :func:`conjugate_momentum` / :func:`velocity_from_momentum` are the (invertible)
   Legendre transform; :func:`hamiltonian` is the total mechanical energy expressed in
   ``(q, p)``; :func:`hamilton_state_derivative` is the canonical-equations right-hand
   side for ``scipy.integrate.solve_ivp``.  The momentum rate is assembled from the
   *existing* Coriolis and gravity terms via the skew-symmetry identity
   ``Mdot = C + C.T`` (Part 1, Section 3.5):

       pdot = d/dt(M qd) = Mdot qd + M qdd
            = (C + C.T) qd + (tau - C qd - g)
            = C(q, qd).T @ qd - g(q) + tau,

   so the Hamiltonian and Lagrangian formulations reuse one derivation and are checked
   against each other pointwise in the tests.

2. **Separable-Hamiltonian symplectic integrators** for ``H = 1/2 p^T W p + V(q)``
   (constant inverse-mass ``W``): :func:`symplectic_euler`, :func:`leapfrog`
   (Stormer-Verlet) and, as the non-symplectic foil, :func:`forward_euler` and
   :func:`rk4`.  :func:`pendulum_field` and :func:`harmonic_oscillator_field` supply
   ready-made separable systems for the phase-portrait, energy-drift and
   Liouville demonstrations.

Convention
----------
Phase-space state is stacked as ``z = [q, p]`` (each length ``n``), mirroring the
``[q, qd]`` layout of Part 1 so the two right-hand sides are drop-in comparable.  Angles
follow the Part 1 convention (measured from the downward vertical, ``q = 0`` hanging).

Author: Stage-1 notes, musculoskeletal-modeling learning path.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:                                    # runs both as a package-less script and via pytest
    from dynamics_utils import PlanarChain
except ImportError:                     # pragma: no cover - only for static type checking
    PlanarChain = "PlanarChain"         # type: ignore

G_DEFAULT = 9.81  # gravitational acceleration [m s^-2] (matches dynamics_utils)


# --------------------------------------------------------------------------------------
# Legendre transform:  (q, qd)  <->  (q, p),  with  p = M(q) qd
# --------------------------------------------------------------------------------------
def conjugate_momentum(chain: "PlanarChain", q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    r"""Generalised (conjugate) momentum ``p = dL/dqd = M(q) qd``.

    This is the forward Legendre transform that maps the Lagrangian velocity state
    ``(q, qd)`` to the Hamiltonian phase-space state ``(q, p)``.  For a mechanical
    system the kinetic energy is the quadratic form ``T = 1/2 qd^T M(q) qd``, so
    ``p = dT/dqd = M(q) qd`` -- the mass matrix turns velocity into momentum.
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    return chain.M(q, qd) @ qd


def velocity_from_momentum(chain: "PlanarChain", q: np.ndarray, p: np.ndarray) -> np.ndarray:
    r"""Inverse Legendre transform ``qd = M(q)^{-1} p``.

    Well posed because ``M(q)`` is symmetric positive definite (the kinetic energy is a
    positive-definite quadratic form) -- the same condition that makes forward dynamics
    solvable in Part 1.  Left-inverse of :func:`conjugate_momentum`.
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.linalg.solve(chain.M(q, np.zeros_like(q)), p)


def hamiltonian(chain: "PlanarChain", q: np.ndarray, p: np.ndarray) -> float:
    r"""Hamiltonian ``H(q, p) = 1/2 p^T M(q)^{-1} p + V(q) = T + V``.

    For a scleronomic system (time-independent constraints, velocity-independent
    potential) the Hamiltonian *is* the total mechanical energy, now expressed in the
    momentum ``p`` rather than the velocity ``qd``.  Equal to
    ``chain.total_energy(q, qd)`` with ``qd = M(q)^{-1} p``.
    """
    qd = velocity_from_momentum(chain, q, p)
    return chain.total_energy(q, qd)


def hamilton_state_derivative(chain: "PlanarChain", t: float, z: np.ndarray,
                              tau_fn: Callable | None = None) -> np.ndarray:
    r"""Right-hand side of Hamilton's canonical equations for ``solve_ivp``.

    With phase-space state ``z = [q, p]`` the canonical equations are

        qdot =  dH/dp = M(q)^{-1} p
        pdot = -dH/dq + tau = C(q, qd).T @ qd - g(q) + tau,

    where the momentum rate uses the Part 1 skew-symmetry identity ``Mdot = C + C.T``
    (Christoffel-form Coriolis matrix) to reuse the existing ``C`` and ``g`` terms:
    ``pdot = Mdot qd + M qdd = (C + C.T) qd + (tau - C qd - g) = C.T qd - g + tau``.
    (Equivalently ``-dH/dq_i = 1/2 qd^T (dM/dq_i) qd - g_i``, which equals ``(C.T qd)_i``
    for the Christoffel ``C``.)

    This integrates the *same* physics as :meth:`PlanarChain.state_derivative`; the tests
    confirm the two trajectories agree.  ``tau_fn(t, q, qd) -> ndarray`` supplies optional
    joint torques (default: passive chain).
    """
    n = chain.n
    q = np.asarray(z[:n], dtype=float)
    p = np.asarray(z[n:], dtype=float)
    qd = np.linalg.solve(chain.M(q, np.zeros(n)), p)
    tau = np.zeros(n) if tau_fn is None else np.asarray(tau_fn(t, q, qd), dtype=float)
    C = chain.coriolis(q, qd)
    g = chain.gravity(q, qd)
    pdot = C.T @ qd - g + tau
    return np.concatenate([qd, pdot])


# --------------------------------------------------------------------------------------
# Poisson bracket (numerical)
# --------------------------------------------------------------------------------------
def poisson_bracket(f: Callable, g: Callable, q: np.ndarray, p: np.ndarray,
                    eps: float = 1e-6) -> float:
    r"""Canonical Poisson bracket ``{f, g}`` evaluated at ``(q, p)`` (central differences).

        {f, g} = sum_i ( df/dq_i dg/dp_i - df/dp_i dg/dq_i ).

    ``f`` and ``g`` are scalar callables ``f(q, p) -> float``.  The bracket packages the
    whole of Hamiltonian dynamics: the time evolution of any observable is
    ``df/dt = {f, H} + df/dt|_explicit``, so ``{f, H} = 0`` means ``f`` is a **conserved
    quantity** (e.g. ``{H, H} = 0`` is energy conservation, and a momentum conjugate to a
    cyclic coordinate satisfies ``{p_i, H} = 0``).  The canonical relations
    ``{q_i, p_j} = delta_ij`` are the defining structure of phase space.
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    n = q.size

    def grad(h, x, other, along_q):
        gr = np.zeros(n)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            if along_q:
                gr[i] = (h(x + dx, other) - h(x - dx, other)) / (2 * eps)
            else:
                gr[i] = (h(other, x + dx) - h(other, x - dx)) / (2 * eps)
        return gr

    df_dq = grad(f, q, p, along_q=True)
    df_dp = grad(f, p, q, along_q=False)
    dg_dq = grad(g, q, p, along_q=True)
    dg_dp = grad(g, p, q, along_q=False)
    return float(df_dq @ dg_dp - df_dp @ dg_dq)


# --------------------------------------------------------------------------------------
# Separable-Hamiltonian integrators:  H(q, p) = 1/2 p^T W p + V(q)
# --------------------------------------------------------------------------------------
# ``force(q) = -dV/dq`` is the generalised force; ``W`` is the (constant) inverse mass, so
# ``qdot = dH/dp = W p``.  ``W`` may be a scalar, a 1-D diagonal, or a full matrix.
def _apply_inv_mass(W, p: np.ndarray) -> np.ndarray:
    """Compute ``W p`` for scalar / diagonal / full inverse-mass ``W``."""
    W = np.asarray(W, dtype=float)
    if W.ndim == 0:
        return float(W) * p
    if W.ndim == 1:
        return W * p
    return W @ p


def symplectic_euler(force: Callable, q0, p0, dt: float, n_steps: int,
                     inv_mass=1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Semi-implicit (symplectic) Euler for a separable ``H = 1/2 p^T W p + V(q)``.

        p_{k+1} = p_k + dt * force(q_k)          (kick, using the old position)
        q_{k+1} = q_k + dt * W p_{k+1}            (drift, using the NEW momentum)

    The single crossed update (new ``p`` drives the ``q`` step) is what makes the map
    **symplectic**: its Jacobian has determinant exactly 1, so it preserves phase-space
    area.  It is only first-order accurate, yet its energy error stays **bounded** for all
    time (it exactly conserves a nearby "shadow" Hamiltonian) instead of drifting -- the
    contrast with :func:`forward_euler` is the whole point of Section 28.

    Returns ``(Q, P)`` each of shape ``(n_steps + 1, dim)``.
    """
    q = np.atleast_1d(np.asarray(q0, dtype=float)).astype(float)
    p = np.atleast_1d(np.asarray(p0, dtype=float)).astype(float)
    Q = np.empty((n_steps + 1, q.size))
    P = np.empty((n_steps + 1, p.size))
    Q[0], P[0] = q, p
    for k in range(n_steps):
        p = p + dt * np.atleast_1d(np.asarray(force(q), dtype=float))
        q = q + dt * _apply_inv_mass(inv_mass, p)
        Q[k + 1], P[k + 1] = q, p
    return Q, P


def leapfrog(force: Callable, q0, p0, dt: float, n_steps: int,
             inv_mass=1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Stormer-Verlet / velocity-Verlet integrator (kick-drift-kick).

        p_{1/2} = p_k     + (dt/2) force(q_k)
        q_{k+1} = q_k     + dt * W p_{1/2}
        p_{k+1} = p_{1/2} + (dt/2) force(q_{k+1}).

    Second-order, **symplectic** and **time-reversible**.  Like :func:`symplectic_euler`
    it has no secular energy drift, but the symmetric half-kicks give a much smaller
    (still bounded) oscillation.  This is the workhorse of molecular dynamics and the
    template for the *variational integrators* used in trajectory optimisation.

    Returns ``(Q, P)`` each of shape ``(n_steps + 1, dim)``.
    """
    q = np.atleast_1d(np.asarray(q0, dtype=float)).astype(float)
    p = np.atleast_1d(np.asarray(p0, dtype=float)).astype(float)
    Q = np.empty((n_steps + 1, q.size))
    P = np.empty((n_steps + 1, p.size))
    Q[0], P[0] = q, p
    for k in range(n_steps):
        p_half = p + 0.5 * dt * np.atleast_1d(np.asarray(force(q), dtype=float))
        q = q + dt * _apply_inv_mass(inv_mass, p_half)
        p = p_half + 0.5 * dt * np.atleast_1d(np.asarray(force(q), dtype=float))
        Q[k + 1], P[k + 1] = q, p
    return Q, P


def forward_euler(force: Callable, q0, p0, dt: float, n_steps: int,
                  inv_mass=1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Explicit (forward) Euler -- the **non-symplectic** baseline.

        q_{k+1} = q_k + dt * W p_k
        p_{k+1} = p_k + dt * force(q_k)      (both updates use the OLD state)

    Its Jacobian has determinant ``1 + O(dt^2) > 1`` for an oscillator, so it inflates
    phase-space area every step and the energy grows without bound (secular drift).  Kept
    only to make that failure visible next to the symplectic schemes.
    """
    q = np.atleast_1d(np.asarray(q0, dtype=float)).astype(float)
    p = np.atleast_1d(np.asarray(p0, dtype=float)).astype(float)
    Q = np.empty((n_steps + 1, q.size))
    P = np.empty((n_steps + 1, p.size))
    Q[0], P[0] = q, p
    for k in range(n_steps):
        f = np.atleast_1d(np.asarray(force(q), dtype=float))
        q_new = q + dt * _apply_inv_mass(inv_mass, p)
        p_new = p + dt * f
        q, p = q_new, p_new
        Q[k + 1], P[k + 1] = q, p
    return Q, P


def rk4(force: Callable, q0, p0, dt: float, n_steps: int,
        inv_mass=1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Classical 4th-order Runge-Kutta on the separable Hamiltonian flow.

    Highly accurate over short horizons, but **not symplectic**: over long integrations
    its energy error, though tiny per step, accumulates as a slow secular drift.  A good
    illustration that *order of accuracy* and *long-time structure preservation* are
    different properties (Section 28 / Section 6.3).
    """
    q = np.atleast_1d(np.asarray(q0, dtype=float)).astype(float)
    p = np.atleast_1d(np.asarray(p0, dtype=float)).astype(float)
    Q = np.empty((n_steps + 1, q.size))
    P = np.empty((n_steps + 1, p.size))
    Q[0], P[0] = q, p

    def deriv(qq, pp):
        return _apply_inv_mass(inv_mass, pp), np.atleast_1d(np.asarray(force(qq), float))

    for k in range(n_steps):
        k1q, k1p = deriv(q, p)
        k2q, k2p = deriv(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p)
        k3q, k3p = deriv(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p)
        k4q, k4p = deriv(q + dt * k3q, p + dt * k3p)
        q = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
        p = p + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        Q[k + 1], P[k + 1] = q, p
    return Q, P


# --------------------------------------------------------------------------------------
# Ready-made separable systems for the demonstrations
# --------------------------------------------------------------------------------------
class SeparableSystem:
    """A separable Hamiltonian ``H = 1/2 p^T W p + V(q)`` bundled for the integrators.

    Attributes
    ----------
    force : callable
        Generalised force ``-dV/dq``.
    inv_mass : float | ndarray
        Constant inverse mass ``W`` (so ``qdot = W p``).
    energy : callable
        ``energy(q, p) -> float`` total mechanical energy along a trajectory.
    """

    def __init__(self, force, inv_mass, energy):
        self.force = force
        self.inv_mass = inv_mass
        self.energy = energy

    def energies(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Total energy sampled along a ``(Q, P)`` trajectory."""
        return np.array([self.energy(q, p) for q, p in zip(np.atleast_2d(Q),
                                                           np.atleast_2d(P))])


def pendulum_field(m: float = 1.0, L: float = 1.0, g: float = G_DEFAULT) -> SeparableSystem:
    r"""Simple (point-mass) pendulum as a separable Hamiltonian system.

    Angle ``theta`` from the downward vertical, momentum ``p = m L^2 thetadot``:

        H(theta, p) = p^2 / (2 m L^2) + m g L (1 - cos theta),
        force(theta) = -dH/dtheta = -m g L sin theta,   W = 1 / (m L^2).

    The phase portrait has closed **libration** orbits (swinging) inside the
    **separatrix** and open **rotation** orbits (going over the top) outside it -- the
    canonical picture of a 1-DOF conservative system (Section 24).
    """
    inv_mass = 1.0 / (m * L * L)

    def force(theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        return -m * g * L * np.sin(theta)

    def energy(theta, p):
        theta = float(np.ravel(theta)[0])
        p = float(np.ravel(p)[0])
        return 0.5 * inv_mass * p * p + m * g * L * (1.0 - np.cos(theta))

    return SeparableSystem(force, inv_mass, energy)


def harmonic_oscillator_field(k: float = 1.0, m: float = 1.0) -> SeparableSystem:
    r"""Linear harmonic oscillator ``H = p^2/(2m) + 1/2 k q^2`` (exact test case).

    Because the flow is linear, each integrator's one-step map is a *constant* matrix,
    so its area-preservation (symplecticity) can be checked exactly from the determinant
    of that matrix -- used in the tests to certify the integrators.
    """
    inv_mass = 1.0 / m

    def force(q):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        return -k * q

    def energy(q, p):
        q = float(np.ravel(q)[0])
        p = float(np.ravel(p)[0])
        return 0.5 * inv_mass * p * p + 0.5 * k * q * q

    return SeparableSystem(force, inv_mass, energy)


def energy_band(E: np.ndarray) -> dict:
    r"""Summarise how an integrator conserves energy over a trajectory.

    Reports the peak deviation from ``E0`` and the peak-to-peak spread, both normalised
    by ``max_t |E|``.  For a **symplectic** integrator these stay bounded as the horizon
    grows (energy oscillates in a band); for a non-symplectic one they grow roughly
    linearly with time (secular drift).  Mirrors :func:`dynamics_utils.energy_drift`.
    """
    E = np.asarray(E, dtype=float)
    scale = max(float(np.max(np.abs(E))), 1e-12)
    return {
        "E0": float(E[0]),
        "abs_max_dev": float(np.max(np.abs(E - E[0]))),
        "rel_max_dev": float(np.max(np.abs(E - E[0])) / scale),
        "rel_range": float(np.ptp(E) / scale),
        "rel_range_pct": float(100.0 * np.ptp(E) / scale),
    }


if __name__ == "__main__":
    # ---- self-test 1: Hamiltonian dynamics == Lagrangian dynamics (double pendulum) ----
    from scipy.integrate import solve_ivp
    from dynamics_utils import planar_chain

    dp = planar_chain(n=2, m=[1.0, 1.0], L=[1.0, 1.0])       # point masses
    q0 = np.array([np.radians(120.0), np.radians(-10.0)])
    qd0 = np.zeros(2)
    p0 = conjugate_momentum(dp, q0, qd0)

    # Legendre round trip
    rt = np.linalg.norm(velocity_from_momentum(dp, q0, p0) - qd0)
    # Hamiltonian equals Lagrangian total energy
    dH = abs(hamiltonian(dp, q0, p0) - dp.total_energy(q0, qd0))
    print("Hamiltonian vs Lagrangian (double pendulum):")
    print(f"  Legendre round-trip ||qd - M^-1 M qd|| = {rt:.2e}")
    print(f"  |H(q,p) - (T+V)|                        = {dH:.2e}")

    solL = solve_ivp(dp.state_derivative, (0, 10), np.concatenate([q0, qd0]),
                     method="DOP853", rtol=1e-11, atol=1e-12, dense_output=True)
    solH = solve_ivp(lambda t, z: hamilton_state_derivative(dp, t, z), (0, 10),
                     np.concatenate([q0, p0]), method="DOP853",
                     rtol=1e-11, atol=1e-12, dense_output=True)
    ts = np.linspace(0, 10, 500)
    qL = solL.sol(ts)[:2]
    qH = solH.sol(ts)[:2]
    print(f"  max |q_Lagrange(t) - q_Hamilton(t)| over 10 s = {np.max(np.abs(qL - qH)):.2e}")

    # ---- self-test 2: symplectic vs non-symplectic energy behaviour (pendulum) ---------
    pend = pendulum_field(m=1.0, L=1.0)
    th0, pp0, dt, N = np.radians(150.0), 0.0, 0.02, 3000    # 60 s, large-amplitude swing
    for name, integ in [("symplectic Euler", symplectic_euler),
                        ("leapfrog", leapfrog),
                        ("forward Euler", forward_euler),
                        ("rk4", rk4)]:
        Q, P = integ(pend.force, [th0], [pp0], dt, N, inv_mass=pend.inv_mass)
        band = energy_band(pend.energies(Q, P))
        print(f"  {name:16s}: energy peak-to-peak over 60 s = {band['rel_range_pct']:.3e} %")
