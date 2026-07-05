r"""
dynamics_utils.py
=================

Reusable utilities for Stage 1 (math foundations) of the musculoskeletal-modeling
learning path.

The centrepiece is :func:`planar_chain`, which derives the equations of motion of an
``n``-link **planar pendulum chain** from the Lagrangian, symbolically (SymPy), and
returns fast numeric callables (NumPy) for:

* the mass / inertia matrix ``M(q)``
* the Coriolis / centrifugal term ``C(q, qd) @ qd``
* the gravity term ``g(q)``
* kinetic energy ``T``, potential energy ``V`` and total energy ``E``
* forward dynamics ``qdd = M(q)^{-1} (tau - C(q,qd) qd - g(q))``

This single abstraction covers the whole of Stage 1:

* a *point-mass* double pendulum (physics toy)      -> ``n=2`` with ``c=L``, ``I=0``
* a *compound* (rigid-link) double pendulum          -> ``n=2`` with COM + inertia
* a two-segment **limb** (thigh+shank, arm+forearm)  -> ``n=2`` with anthropometry
* the triple-pendulum milestone exercise             -> ``n=3``

Convention
----------
Absolute segment angles ``q_i`` are measured from the **downward vertical**, so a chain
hanging straight down is ``q = 0``.  Using *absolute* (not relative) angles makes each
segment's angular velocity simply ``qd_i``, which keeps the kinetic energy clean.  The
proximal joint of segment ``i`` sits at the distal end of segment ``i-1``:

    X_i = sum_{k<=i} L_k sin(q_k)          (joint i, horizontal)
    Y_i = sum_{k<=i} -L_k cos(q_k)         (joint i, vertical, +y up)

The centre of mass of segment ``i`` is a distance ``c_i`` (0 <= c_i <= L_i) along the
segment from its proximal joint.

The manipulator form
    M(q) qdd + C(q, qd) qd + g(q) = tau
is exactly the equation you meet again in Stage 2 (multibody dynamics), Stage 3
(OpenSim inverse dynamics: tau = M qdd + C qd + g - J^T F_ext) and Stage 6/8 (static
optimisation / Moco).  Deriving it once, by hand-checked SymPy, is the point of Stage 1.

Author: Stage-1 notes, musculoskeletal-modeling learning path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import sympy as sp

G_DEFAULT = 9.81  # gravitational acceleration [m s^-2]


# --------------------------------------------------------------------------------------
# Symbolic derivation
# --------------------------------------------------------------------------------------
@dataclass
class PlanarChain:
    """Container for the symbolic and numeric model of an ``n``-link planar chain.

    Attributes
    ----------
    n : int
        Number of links.
    q, qd, qdd : list[sympy.Symbol]
        Generalised coordinates and their first/second time derivatives.
    M_sym, forcing_sym : sympy.Matrix
        Symbolic mass matrix and the combined ``C qd + g`` vector.
    T_sym, V_sym : sympy.Expr
        Symbolic kinetic and potential energy.
    params : dict
        The numeric inertial parameters baked into the lambdified callables
        (``m``, ``L``, ``c``, ``I`` per link and ``g``).
    """

    n: int
    q: list
    qd: list
    qdd: list
    M_sym: sp.Matrix
    forcing_sym: sp.Matrix      # C(q, qd) qd + g(q)
    gravity_sym: sp.Matrix      # g(q)
    T_sym: sp.Expr
    V_sym: sp.Expr
    params: dict

    # numeric callables (filled in by planar_chain)
    M: Callable
    forcing: Callable
    gravity: Callable
    kinetic_energy: Callable
    potential_energy: Callable

    # ---- numeric dynamics -------------------------------------------------------------
    def forward_dynamics(self, q: np.ndarray, qd: np.ndarray,
                         tau: np.ndarray | None = None) -> np.ndarray:
        """Return the joint accelerations ``qdd`` for the given state and joint torques.

        Solves ``M(q) qdd = tau - C(q,qd) qd - g(q)`` for ``qdd``.
        ``tau=None`` means the passive (unforced) chain.
        """
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        if tau is None:
            tau = np.zeros(self.n)
        rhs = np.asarray(tau, dtype=float) - self.forcing(q, qd)
        return np.linalg.solve(self.M(q, qd), rhs)

    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray,
                         qdd: np.ndarray) -> np.ndarray:
        """Return the joint torques ``tau`` that produce a prescribed motion.

        Evaluates the manipulator form ``tau = M(q) qdd + C(q,qd) qd + g(q)``.
        This is the exact left inverse of :meth:`forward_dynamics`
        (``inverse_dynamics(q, qd, forward_dynamics(q, qd, tau)) == tau``) and is
        the analogue of OpenSim's ``InverseDynamicsTool``: given kinematics
        ``q, qd, qdd`` (e.g. from motion capture) it returns the net joint moments.
        """
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)
        return self.M(q, qd) @ qdd + self.forcing(q, qd)

    def state_derivative(self, t: float, y: np.ndarray,
                         tau_fn: Callable | None = None) -> np.ndarray:
        """RHS for ``scipy.integrate.solve_ivp`` with state ``y = [q, qd]``.

        ``tau_fn(t, q, qd) -> ndarray`` supplies optional joint torques.
        """
        q = y[: self.n]
        qd = y[self.n:]
        tau = None if tau_fn is None else tau_fn(t, q, qd)
        qdd = self.forward_dynamics(q, qd, tau)
        return np.concatenate([qd, qdd])

    def total_energy(self, q: np.ndarray, qd: np.ndarray) -> float:
        """Total mechanical energy ``E = T + V`` (conserved for the passive chain)."""
        return float(self.kinetic_energy(q, qd) + self.potential_energy(q, qd))

    def energies(self, Q: np.ndarray, QD: np.ndarray) -> tuple:
        """Vectorised energies over a trajectory.

        ``Q``, ``QD`` have shape ``(n_steps, n)``.  Returns ``(T, V, E)`` arrays.
        """
        T = np.array([self.kinetic_energy(q, qd) for q, qd in zip(Q, QD)])
        V = np.array([self.potential_energy(q, qd) for q, qd in zip(Q, QD)])
        return T, V, T + V

    # ---- forward kinematics (for plotting / animation) --------------------------------
    def joint_positions(self, q: np.ndarray) -> np.ndarray:
        """Cartesian coordinates of joints 0..n. Returns array of shape ``(n+1, 2)``."""
        q = np.asarray(q, dtype=float)
        L = self.params["L"]
        x = np.concatenate([[0.0], np.cumsum(L * np.sin(q))])
        y = np.concatenate([[0.0], np.cumsum(-L * np.cos(q))])
        return np.column_stack([x, y])


def planar_chain(n: int = 2,
                 m: Sequence[float] | None = None,
                 L: Sequence[float] | None = None,
                 c: Sequence[float] | None = None,
                 I: Sequence[float] | None = None,
                 g: float = G_DEFAULT) -> PlanarChain:
    """Derive and lambdify the equations of motion of an ``n``-link planar chain.

    Parameters
    ----------
    n : int
        Number of links.
    m, L : sequence of float, optional
        Segment masses [kg] and lengths [m]. Default: all ones.
    c : sequence of float, optional
        Distance of each segment's centre of mass from its proximal joint [m].
        Default: point mass at the distal end (``c = L``) -> classic pendulum.
    I : sequence of float, optional
        Moment of inertia of each segment about its own centre of mass
        [kg m^2]. Default: 0 (point mass).
    g : float
        Gravitational acceleration [m s^-2].

    Returns
    -------
    PlanarChain
        With symbolic expressions and fast numeric callables attached.

    Notes
    -----
    The derivation is fully symbolic, so it is exact (no finite differencing) and
    self-documenting: inspect ``chain.M_sym`` or ``chain.T_sym`` to see the algebra.
    """
    m = np.ones(n) if m is None else np.asarray(m, dtype=float)
    L = np.ones(n) if L is None else np.asarray(L, dtype=float)
    c = L.copy() if c is None else np.asarray(c, dtype=float)
    I = np.zeros(n) if I is None else np.asarray(I, dtype=float)
    for name, arr in {"m": m, "L": L, "c": c, "I": I}.items():
        if len(arr) != n:
            raise ValueError(f"parameter {name!r} must have length n={n}, got {len(arr)}")

    t = sp.symbols("t", real=True)
    q_f = [sp.Function(f"q{i+1}")(t) for i in range(n)]          # q_i(t)
    qd_s = sp.symbols(f"qd1:{n+1}", real=True)                    # generic velocity syms
    qdd_s = sp.symbols(f"qdd1:{n+1}", real=True)                  # generic accel syms
    q_s = sp.symbols(f"q1:{n+1}", real=True)                      # generic pos syms
    ms = sp.symbols(f"m1:{n+1}", positive=True)
    Ls = sp.symbols(f"L1:{n+1}", positive=True)
    cs = sp.symbols(f"c1:{n+1}", nonnegative=True)
    Is = sp.symbols(f"I1:{n+1}", nonnegative=True)
    gsym = sp.symbols("g", positive=True)

    # ---- forward kinematics of centres of mass ---------------------------------------
    # proximal joint positions accumulate along the chain
    Xj = [sp.Integer(0)]
    Yj = [sp.Integer(0)]
    for i in range(n):
        Xj.append(Xj[-1] + Ls[i] * sp.sin(q_f[i]))
        Yj.append(Yj[-1] - Ls[i] * sp.cos(q_f[i]))

    T = sp.Integer(0)
    V = sp.Integer(0)
    for i in range(n):
        xc = Xj[i] + cs[i] * sp.sin(q_f[i])
        yc = Yj[i] - cs[i] * sp.cos(q_f[i])
        vx = sp.diff(xc, t)
        vy = sp.diff(yc, t)
        omega = sp.diff(q_f[i], t)                 # absolute angular velocity = qd_i
        T += sp.Rational(1, 2) * ms[i] * (vx**2 + vy**2) + sp.Rational(1, 2) * Is[i] * omega**2
        V += ms[i] * gsym * yc

    Lag = sp.expand_trig(sp.simplify(T - V))

    # ---- Euler-Lagrange:  d/dt (dL/dqd_i) - dL/dq_i = tau_i ---------------------------
    q_dot = [sp.diff(f, t) for f in q_f]
    q_ddot = [sp.diff(f, t, 2) for f in q_f]
    EOM = []
    for i in range(n):
        dL_dqd = sp.diff(Lag, q_dot[i])
        term = sp.diff(dL_dqd, t) - sp.diff(Lag, q_f[i])
        EOM.append(sp.simplify(term))
    EOM = sp.Matrix(EOM)

    # substitute the time-functions with plain symbols so we can lambdify
    subs_map = {}
    for i in range(n):
        subs_map[q_ddot[i]] = qdd_s[i]
        subs_map[q_dot[i]] = qd_s[i]
        subs_map[q_f[i]] = q_s[i]
    EOM_s = EOM.subs(subs_map)

    # M(q) = d(EOM)/d(qdd);  forcing = EOM|_{qdd=0} = C(q,qd) qd + g(q)
    M_sym = EOM_s.jacobian(sp.Matrix(qdd_s))
    M_sym = sp.simplify(M_sym)
    forcing_sym = sp.simplify(EOM_s.subs({a: 0 for a in qdd_s}))
    gravity_sym = sp.simplify(forcing_sym.subs({v: 0 for v in qd_s}))

    T_sym = sp.simplify(T.subs(subs_map))
    V_sym = sp.simplify(V.subs(subs_map))

    # ---- bake numeric parameters and lambdify ----------------------------------------
    num = {}
    for i in range(n):
        num[ms[i]] = float(m[i]); num[Ls[i]] = float(L[i])
        num[cs[i]] = float(c[i]); num[Is[i]] = float(I[i])
    num[gsym] = float(g)

    def _lam_matrix(expr):
        e = expr.subs(num)
        f = sp.lambdify((q_s, qd_s), e, "numpy")
        return lambda q, qd: np.asarray(f(np.asarray(q, float), np.asarray(qd, float)),
                                        dtype=float)

    def _lam_vec(expr):
        e = expr.subs(num)
        f = sp.lambdify((q_s, qd_s), sp.Matrix(e), "numpy")
        return lambda q, qd: np.asarray(f(np.asarray(q, float),
                                          np.asarray(qd, float)), dtype=float).ravel()

    M_num = _lam_matrix(M_sym)
    forcing_num = _lam_vec(forcing_sym)
    gravity_num = _lam_vec(gravity_sym)
    T_fun = sp.lambdify((q_s, qd_s), T_sym.subs(num), "numpy")
    V_fun = sp.lambdify((q_s, qd_s), V_sym.subs(num), "numpy")

    chain = PlanarChain(
        n=n, q=list(q_s), qd=list(qd_s), qdd=list(qdd_s),
        M_sym=M_sym, forcing_sym=forcing_sym, gravity_sym=gravity_sym,
        T_sym=T_sym, V_sym=V_sym,
        params={"m": m, "L": L, "c": c, "I": I, "g": g},
        M=M_num, forcing=forcing_num, gravity=gravity_num,
        kinetic_energy=lambda q, qd: float(T_fun(np.asarray(q, float),
                                                 np.asarray(qd, float))),
        potential_energy=lambda q, qd: float(V_fun(np.asarray(q, float),
                                                   np.asarray(qd, float))),
    )
    return chain


# --------------------------------------------------------------------------------------
# Anthropometric helper: turn a limb into planar_chain parameters
# --------------------------------------------------------------------------------------
# Segment inertial parameters as fractions, in the spirit of the regression tables in
# Winter (2009) and Zatsiorsky (2002). These are *illustrative teaching defaults*, not a
# substitute for subject-specific scaling (that arrives properly in Stage 3, OpenSim).
#   mass_frac : segment mass / total body mass
#   com_frac  : COM distance from PROXIMAL joint / segment length
#   rog_frac  : radius of gyration about COM / segment length
_SEGMENTS = {
    #                 mass_frac  com_frac  rog_frac   typical_length_frac_of_height
    "thigh":   dict(mass_frac=0.100, com_frac=0.433, rog_frac=0.323, len_frac=0.245),
    "shank":   dict(mass_frac=0.0465, com_frac=0.433, rog_frac=0.302, len_frac=0.246),
    "upperarm":dict(mass_frac=0.028, com_frac=0.436, rog_frac=0.322, len_frac=0.186),
    "forearm": dict(mass_frac=0.016, com_frac=0.430, rog_frac=0.303, len_frac=0.146),
}


def limb_chain(segments=("thigh", "shank"), body_mass=70.0, height=1.75,
               g: float = G_DEFAULT) -> PlanarChain:
    """Build a :class:`PlanarChain` for a multi-segment limb using anthropometry.

    Example
    -------
    >>> leg = limb_chain(("thigh", "shank"), body_mass=70, height=1.75)
    >>> leg.params["m"]      # thigh, shank masses [kg]

    The defaults follow Winter-style / Zatsiorsky-style segment regression ratios and
    are meant for illustration; subject-specific inertia comes from OpenSim scaling in
    Stage 3.
    """
    m, L, c, I = [], [], [], []
    for s in segments:
        p = _SEGMENTS[s]
        seg_len = p["len_frac"] * height
        seg_mass = p["mass_frac"] * body_mass
        com = p["com_frac"] * seg_len
        inertia = seg_mass * (p["rog_frac"] * seg_len) ** 2   # I = m (k)^2 about COM
        m.append(seg_mass); L.append(seg_len); c.append(com); I.append(inertia)
    return planar_chain(n=len(segments), m=m, L=L, c=c, I=I, g=g)


# --------------------------------------------------------------------------------------
# Energy-drift diagnostic (the Stage-1 milestone metric)
# --------------------------------------------------------------------------------------
def energy_drift(E: np.ndarray, scale: float | None = None) -> dict:
    """Summarise energy conservation over a trajectory.

    Returns absolute and *relative* drift.  Relative drift is normalised by an
    energy *scale* that is independent of the drift being measured -- by default
    ``max_t |E(t)|``.  (An earlier version divided by ``max(|E0|, ptp(E))``; the
    ``ptp`` term put the quantity being measured into the denominator, which
    silently capped ``rel_range`` at 1 and made the metric meaningless whenever
    ``E0`` sat near zero because of the potential-energy datum.)

    Parameters
    ----------
    E : array_like
        Total mechanical energy sampled along the trajectory.
    scale : float, optional
        Override the normalising energy scale.  Useful when ``E`` crosses zero
        (arbitrary potential datum): pass e.g. the kinetic-energy amplitude so
        the relative figure reflects the true energy scale of the motion.

    Notes
    -----
    The Stage-1 self-assessment target is ``rel_range < 1%`` for a passive chain
    integrated for ~10 s.
    """
    E = np.asarray(E, dtype=float)
    E0 = E[0]
    if scale is None:
        scale = np.max(np.abs(E))
    scale = max(float(scale), 1e-12)
    return {
        "E0": float(E0),
        "abs_max_dev": float(np.max(np.abs(E - E0))),
        "rel_max_dev": float(np.max(np.abs(E - E0)) / scale),
        "rel_range": float(np.ptp(E) / scale),
        "rel_range_pct": float(100.0 * np.ptp(E) / scale),
    }


if __name__ == "__main__":
    # quick self-test: passive point-mass double pendulum, energy should be conserved
    from scipy.integrate import solve_ivp

    dp = planar_chain(n=2, m=[1.0, 1.0], L=[1.0, 1.0])  # c=L, I=0 -> point masses
    y0 = np.array([np.radians(120), np.radians(-10), 0.0, 0.0])
    sol = solve_ivp(dp.state_derivative, (0, 10), y0, method="DOP853",
                    rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.01)
    Q, QD = sol.y[:2].T, sol.y[2:].T
    _, _, E = dp.energies(Q, QD)
    d = energy_drift(E)
    print("double pendulum, 10 s, DOP853 rtol=1e-10")
    print(f"  E0 = {d['E0']:.6f} J,  relative energy range = {d['rel_range_pct']:.3e} %")
    print("  M(q0) =\n", dp.M(Q[0], QD[0]))
