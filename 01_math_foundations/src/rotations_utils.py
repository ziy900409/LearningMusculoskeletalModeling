r"""
rotations_utils.py
==================

Reusable utilities for Stage 1, Part 2 (math foundations) of the
musculoskeletal-modeling learning path: **the representation of 3D rotations**.

Where Part 1 (``dynamics_utils.py``) built the *planar* equations of motion, real
skeletons rotate in 3D: the shoulder and hip are ball (3-DOF) joints, motion-capture
and IMU pipelines report orientations, and every OpenSim body frame is related to its
parent by a rotation.  This module gives fast, hand-checked NumPy implementations of the
three equivalent languages for a rotation and the maps between them:

* the **rotation group** ``SO(3)`` -- orthogonal matrices with ``det = +1``
  (:func:`exp_so3`, :func:`log_so3`, :func:`hat`, :func:`vee`);
* **unit quaternions** ``S^3`` -- the double cover of ``SO(3)``
  (:func:`quat_multiply`, :func:`quat_rotate`, :func:`quat_exp`, :func:`quat_log`);
* **axis-angle / the exponential coordinates** ``so(3)`` -- a rotation vector
  ``w = theta * axis`` (:func:`axis_angle_to_quat`, :func:`quat_to_axis_angle`);

plus the conversions ``quat <-> matrix`` (:func:`quat_to_matrix`, :func:`matrix_to_quat`),
spherical linear interpolation :func:`slerp`, and the **orientation kinematics**
``qdot = 1/2 q (X) omega`` (:func:`quat_derivative`, :func:`quat_kinematics_rhs`) that
integrate an angular velocity into an orientation.

Conventions
-----------
* **Quaternions are scalar-first**: ``q = [w, x, y, z]`` represents
  ``w + x i + y j + z k``.  (SciPy's ``Rotation`` is scalar-*last* ``[x, y, z, w]``;
  :func:`quat_to_scipy` / :func:`quat_from_scipy` bridge the two.)
* **Hamilton** (not JPL) multiplication: ``i j = k``, ``j k = i``, ``k i = j``.
* **Active** rotations of column vectors: ``v' = R v`` rotates the vector ``v`` within a
  fixed frame.  Composition ``R(q1 (X) q2) = R(q1) R(q2)`` means *apply ``q2`` first*.
* Rotation vectors / angular velocities are ordinary ``(3,)`` arrays in radians.

Every claim these functions make (orthogonality, the double cover ``R(q) = R(-q)``, the
group homomorphism, the exp/log round trip, ``qdot = 1/2 q (X) omega``) is checked
pointwise in ``tests/test_rotations_utils.py`` -- including against
``scipy.spatial.transform.Rotation`` -- in the same "verify before you trust it" spirit
as the energy-conservation check of Part 1.

Author: Stage-1 notes, musculoskeletal-modeling learning path.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# so(3): the hat / vee isomorphism between R^3 and 3x3 skew-symmetric matrices
# --------------------------------------------------------------------------------------
def hat(w: np.ndarray) -> np.ndarray:
    r"""Map a vector ``w`` in R^3 to the skew-symmetric matrix ``[w]_x`` in so(3).

    ``[w]_x v = w x v`` (the cross product) for every ``v``::

              [  0   -w3   w2 ]
        [w]_x=[  w3   0   -w1 ]
              [ -w2   w1   0  ]
    """
    w = np.asarray(w, dtype=float).ravel()
    if w.shape != (3,):
        raise ValueError(f"hat expects a length-3 vector, got shape {w.shape}")
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee(S: np.ndarray) -> np.ndarray:
    r"""Inverse of :func:`hat`: extract ``w`` from a skew-symmetric ``[w]_x``.

    The skew part is used, so small numerical asymmetry in ``S`` is tolerated.
    """
    S = np.asarray(S, dtype=float)
    if S.shape != (3, 3):
        raise ValueError(f"vee expects a 3x3 matrix, got shape {S.shape}")
    return 0.5 * np.array([S[2, 1] - S[1, 2],
                           S[0, 2] - S[2, 0],
                           S[1, 0] - S[0, 1]])


# --------------------------------------------------------------------------------------
# SO(3): exponential and logarithm (Rodrigues' rotation formula)
# --------------------------------------------------------------------------------------
def exp_so3(w: np.ndarray) -> np.ndarray:
    r"""Exponential map ``so(3) -> SO(3)``: rotation vector ``w`` to matrix ``R``.

    With ``theta = ||w||`` and unit axis ``n = w / theta``, Rodrigues' formula is

        R = I + (sin theta) [n]_x + (1 - cos theta) [n]_x^2
          = I + A [w]_x + B [w]_x^2,   A = sin(theta)/theta,  B = (1-cos theta)/theta^2.

    The ``A, B`` coefficients are evaluated by their Taylor series near ``theta = 0`` so
    the identity rotation is handled without a divide-by-zero.
    """
    w = np.asarray(w, dtype=float).ravel()
    if w.shape != (3,):
        raise ValueError(f"exp_so3 expects a length-3 vector, got shape {w.shape}")
    theta2 = float(w @ w)
    K = hat(w)
    if theta2 < _EPS:
        # A -> 1 - theta^2/6, B -> 1/2 - theta^2/24 ; leading terms suffice at machine eps
        A = 1.0 - theta2 / 6.0
        B = 0.5 - theta2 / 24.0
    else:
        theta = np.sqrt(theta2)
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta2
    return np.eye(3) + A * K + B * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    r"""Logarithm map ``SO(3) -> so(3)``: matrix ``R`` to rotation vector ``w``.

    Returns ``w = theta * axis`` with ``theta`` in ``[0, pi]``.  Uses

        theta = arccos((tr R - 1)/2),   w = theta/(2 sin theta) * vee(R - R^T)

    with dedicated branches for ``theta ~ 0`` (``w = vee(R - R^T)/2``) and the
    ``theta ~ pi`` antipode (where ``sin theta -> 0`` and the axis is read off the
    diagonal of ``R + I``).  The axis at ``theta = pi`` is defined only up to sign; a
    deterministic sign convention is applied.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"log_so3 expects a 3x3 matrix, got shape {R.shape}")
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:                       # near identity: first-order log
        return vee(R - R.T)                # = (theta/sin theta ~ 1) * vee(R - R^T)

    if np.pi - theta < 1e-8:               # near pi: sin theta ~ 0, axis from R + I
        # R + I = 2 n n^T ; the largest diagonal gives the most accurate column.
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(A[k, k])
        axis = axis / np.linalg.norm(axis)
        # deterministic sign: first non-negligible component positive
        for c in axis:
            if abs(c) > 1e-9:
                if c < 0:
                    axis = -axis
                break
        return theta * axis

    return (theta / (2.0 * np.sin(theta))) * vee(R - R.T)


# --------------------------------------------------------------------------------------
# Quaternion algebra (Hamilton, scalar-first [w, x, y, z])
# --------------------------------------------------------------------------------------
def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r"""Hamilton product ``q1 (X) q2`` of two quaternions ``[w, x, y, z]``.

    For unit quaternions this composes rotations: ``R(q1 (X) q2) = R(q1) R(q2)``, i.e.
    ``q2`` is applied to a vector first, then ``q1``.
    """
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate ``q* = [w, -x, -y, -z]``.  For a unit quaternion this is its inverse."""
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_norm(q: np.ndarray) -> float:
    """Euclidean norm ``||q||`` of the quaternion viewed as a 4-vector."""
    return float(np.linalg.norm(np.asarray(q, dtype=float)))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q / ||q||`` (project back onto the unit sphere ``S^3``)."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < _EPS:
        raise ValueError("cannot normalize a (near-)zero quaternion")
    return q / n


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse ``q^{-1} = q* / ||q||^2`` (equals the conjugate for a unit quaternion)."""
    q = np.asarray(q, dtype=float)
    return quat_conjugate(q) / float(q @ q)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""Rotate a 3-vector ``v`` by the unit quaternion ``q``: ``v' = q (0, v) q*``.

    Equivalent to ``quat_to_matrix(q) @ v`` (active rotation) but avoids forming the
    matrix.  ``q`` is normalized defensively.
    """
    q = quat_normalize(q)
    v = np.asarray(v, dtype=float).ravel()
    p = np.array([0.0, v[0], v[1], v[2]])
    return quat_multiply(quat_multiply(q, p), quat_conjugate(q))[1:]


# --------------------------------------------------------------------------------------
# Axis-angle / exponential coordinates <-> unit quaternion
# --------------------------------------------------------------------------------------
def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    r"""Unit quaternion for a rotation of ``angle`` [rad] about ``axis``.

    ``q = (cos(angle/2), sin(angle/2) * axis_hat)``.  ``axis`` need not be unit length;
    a (near-)zero axis returns the identity quaternion.
    """
    axis = np.asarray(axis, dtype=float).ravel()
    n = np.linalg.norm(axis)
    if n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    half = 0.5 * float(angle)
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def quat_exp(w: np.ndarray) -> np.ndarray:
    r"""Exponential map ``so(3) -> S^3``: rotation vector ``w`` to a unit quaternion.

    ``q = (cos(theta/2), sinc(theta/2)/2 * w)`` with ``theta = ||w||``; the half-angle
    sinc is Taylor-expanded near ``theta = 0``.  Satisfies
    ``quat_to_matrix(quat_exp(w)) == exp_so3(w)``.
    """
    w = np.asarray(w, dtype=float).ravel()
    theta = float(np.linalg.norm(w))
    half = 0.5 * theta
    if theta < 1e-8:
        # sin(half)/theta -> 1/2 - theta^2/48 ; keep the leading term at machine eps
        coeff = 0.5 - theta * theta / 48.0
    else:
        coeff = np.sin(half) / theta
    return np.concatenate([[np.cos(half)], coeff * w])


def quat_log(q: np.ndarray) -> np.ndarray:
    r"""Logarithm map ``S^3 -> so(3)``: unit quaternion to rotation vector ``w``.

    Returns ``w = theta * axis`` with ``theta`` in ``[0, pi]``.  The sign of ``q`` is
    normalized so that ``w`` = 0 near the identity (the antipodal ``-q`` is mapped to the
    same rotation, consistent with the double cover).
    """
    q = quat_normalize(q)
    if q[0] < 0:                    # pick the representative with w >= 0 (double cover)
        q = -q
    vnorm = float(np.linalg.norm(q[1:]))
    if vnorm < 1e-8:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(vnorm, q[0])
    return (theta / vnorm) * q[1:]


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(axis, angle)`` with unit ``axis`` and ``angle`` in ``[0, pi]``.

    Identity rotations return the arbitrary axis ``[1, 0, 0]`` with ``angle = 0``.
    """
    w = quat_log(q)
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return w / angle, angle


# --------------------------------------------------------------------------------------
# Quaternion <-> rotation matrix
# --------------------------------------------------------------------------------------
def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    r"""Rotation matrix ``R(q)`` of a (defensively normalized) unit quaternion.

        R = [ 1-2(y^2+z^2)   2(xy-wz)     2(xz+wy)   ]
            [ 2(xy+wz)       1-2(x^2+z^2) 2(yz-wx)   ]
            [ 2(xz-wy)       2(yz+wx)     1-2(x^2+y^2)]
    """
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    r"""Unit quaternion of a rotation matrix (Shepperd's numerically stable method).

    The component with the largest magnitude is computed first (from the trace or a
    diagonal entry) and the rest are derived from it, avoiding cancellation.  The
    returned quaternion has ``w >= 0`` (one of the two double-cover representatives).
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"matrix_to_quat expects a 3x3 matrix, got shape {R.shape}")
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0            # s = 4w
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0   # s = 4x
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0   # s = 4y
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0   # s = 4z
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return quat_normalize(q)


# --------------------------------------------------------------------------------------
# Intrinsic Euler angles (for the gimbal-lock demonstration)
# --------------------------------------------------------------------------------------
_AXES = {"x": np.array([1.0, 0.0, 0.0]),
         "y": np.array([0.0, 1.0, 0.0]),
         "z": np.array([0.0, 0.0, 1.0])}


def euler_to_matrix(angles, seq: str = "zyx", intrinsic: bool = True) -> np.ndarray:
    r"""Rotation matrix from a 3-angle Euler / Cardan sequence.

    ``seq`` is a 3-character string such as ``"zyx"`` or ``"yxy"`` (the ISB shoulder
    sequence).  ``intrinsic=True`` (the OpenSim / ISB convention) rotates about
    successively rotated body axes, giving ``R = R_{a1}(t1) R_{a2}(t2) R_{a3}(t3)``.
    """
    angles = np.asarray(angles, dtype=float).ravel()
    if len(seq) != 3 or len(angles) != 3:
        raise ValueError("euler_to_matrix needs a 3-char seq and 3 angles")
    mats = [exp_so3(a * _AXES[ax]) for ax, a in zip(seq, angles)]
    if intrinsic:
        return mats[0] @ mats[1] @ mats[2]
    return mats[2] @ mats[1] @ mats[0]


def euler_rate_matrix(angles, seq: str = "zyx", intrinsic: bool = True) -> np.ndarray:
    r"""Map from Euler-angle rates to the **space** angular velocity: ``omega_s = E d/dt``.

    For intrinsic ``R = R_{a1} R_{a2} R_{a3}`` the world-frame angular velocity is

        omega_s = a1 * dtheta1 + (R_{a1} a2) * dtheta2 + (R_{a1} R_{a2} a3) * dtheta3,

    so column ``k`` of ``E`` is the ``k``-th rotation axis **as currently oriented in the
    world** (``R_1 ... R_{k-1} a_k``).  ``E`` becomes **singular at gimbal lock** -- its
    determinant vanishes when two of those axes align (e.g. the middle Cardan angle at
    +/- pi/2) -- which is exactly why Euler angles are a poor state for 3D orientation.
    The notebook plots ``1/|det E|`` (or the condition number) blowing up there.  (Body-
    and space-frame maps differ by ``R``, which is orthogonal, so ``|det E|`` -- and hence
    the location of the singularity -- is the same in either frame.)
    """
    angles = np.asarray(angles, dtype=float).ravel()
    if intrinsic:
        R = np.eye(3)
        cols = []
        for ax, a in zip(seq, angles):
            cols.append(R @ _AXES[ax])            # world-frame axis of the k-th rotation
            R = R @ exp_so3(a * _AXES[ax])
        return np.column_stack(cols)
    raise NotImplementedError("only the intrinsic convention is implemented")


# --------------------------------------------------------------------------------------
# Spherical linear interpolation (SLERP)
# --------------------------------------------------------------------------------------
def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    r"""Spherical linear interpolation on ``S^3`` between unit quaternions.

        slerp(q0, q1; t) = sin((1-t) Omega)/sin Omega * q0 + sin(t Omega)/sin Omega * q1,

    where ``cos Omega = q0 . q1``.  This traces the **great-circle geodesic** on ``S^3``,
    which projects to the constant-angular-velocity rotation between the two orientations
    -- the reason it is the standard way to interpolate motion-capture orientations.

    ``t`` may be a scalar or a 1-D array (vectorised over the interpolation parameter).
    The sign of ``q1`` is flipped when ``q0 . q1 < 0`` so the *shorter* arc is taken
    (``q`` and ``-q`` are the same rotation).  Near-parallel inputs fall back to
    normalized linear interpolation.
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(q0 @ q1)
    if dot < 0.0:                      # shorter arc: -q1 is the same rotation as q1
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))

    if dot > 1.0 - 1e-9:               # almost identical: linear interp + renormalize
        out = (1.0 - t_arr)[:, None] * q0 + t_arr[:, None] * q1
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
    else:
        Omega = np.arccos(dot)
        sinO = np.sin(Omega)
        a = np.sin((1.0 - t_arr) * Omega) / sinO
        b = np.sin(t_arr * Omega) / sinO
        out = a[:, None] * q0 + b[:, None] * q1

    return out[0] if np.isscalar(t) or np.ndim(t) == 0 else out


# --------------------------------------------------------------------------------------
# Orientation kinematics:  qdot = 1/2 q (X) omega  (body)  or  1/2 omega (X) q  (space)
# --------------------------------------------------------------------------------------
def quat_derivative(q: np.ndarray, omega: np.ndarray, frame: str = "body") -> np.ndarray:
    r"""Time derivative of an orientation quaternion given an angular velocity.

    * ``frame="body"``  : ``omega`` in body coordinates, ``qdot = 1/2 q (X) (0, omega)``.
    * ``frame="space"`` : ``omega`` in world coordinates, ``qdot = 1/2 (0, omega) (X) q``.

    These are the quaternion equivalents of ``Rdot = R [omega_b]_x`` and
    ``Rdot = [omega_s]_x R`` respectively.  Integrating this ODE turns a measured angular
    velocity (e.g. an IMU gyro signal) into an orientation.
    """
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float).ravel()
    omega_q = np.array([0.0, omega[0], omega[1], omega[2]])
    if frame == "body":
        return 0.5 * quat_multiply(q, omega_q)
    if frame == "space":
        return 0.5 * quat_multiply(omega_q, q)
    raise ValueError("frame must be 'body' or 'space'")


def quat_kinematics_rhs(omega_fn: Callable, frame: str = "body",
                        renormalize: bool = True) -> Callable:
    r"""Build a ``scipy.integrate.solve_ivp`` right-hand side for orientation.

    ``omega_fn(t) -> (3,)`` supplies the angular velocity.  The returned ``f(t, q)``
    evaluates :func:`quat_derivative`; with ``renormalize=True`` a Baumgarte-style term
    ``+ k (1 - ||q||^2) q`` is added so the numerical solution is gently pulled back onto
    the unit sphere instead of drifting off it (the standard fix for the fact that a
    generic ODE step leaves ``S^3``).
    """
    k = 1.0

    def f(t: float, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        dq = quat_derivative(q, omega_fn(t), frame=frame)
        if renormalize:
            dq = dq + k * (1.0 - float(q @ q)) * q
        return dq

    return f


# --------------------------------------------------------------------------------------
# Random rotations (uniform on SO(3)) -- handy for tests and Monte-Carlo checks
# --------------------------------------------------------------------------------------
def random_quaternion(rng: np.random.Generator | None = None) -> np.ndarray:
    """Unit quaternion drawn uniformly from ``S^3`` (Haar measure on ``SO(3)``).

    Uses Shoemake's method: three uniforms mapped to a point on the 3-sphere.
    """
    rng = np.random.default_rng() if rng is None else rng
    u1, u2, u3 = rng.uniform(size=3)
    s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    q = np.array([s1 * np.sin(2 * np.pi * u2),
                  s1 * np.cos(2 * np.pi * u2),
                  s2 * np.sin(2 * np.pi * u3),
                  s2 * np.cos(2 * np.pi * u3)])
    # Shoemake orders as (x, y, z, w); reorder to scalar-first and fix the sign.
    q = np.array([q[3], q[0], q[1], q[2]])
    return q if q[0] >= 0 else -q


def random_rotation(rng: np.random.Generator | None = None) -> np.ndarray:
    """Rotation matrix drawn uniformly from ``SO(3)`` (via :func:`random_quaternion`)."""
    return quat_to_matrix(random_quaternion(rng))


# --------------------------------------------------------------------------------------
# SciPy bridge (scalar-first [w,x,y,z]  <->  SciPy scalar-last [x,y,z,w])
# --------------------------------------------------------------------------------------
def quat_to_scipy(q: np.ndarray) -> np.ndarray:
    """Reorder a scalar-first ``[w, x, y, z]`` to SciPy's scalar-last ``[x, y, z, w]``."""
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def quat_from_scipy(q: np.ndarray) -> np.ndarray:
    """Reorder SciPy's scalar-last ``[x, y, z, w]`` to scalar-first ``[w, x, y, z]``."""
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


def is_rotation(R: np.ndarray, atol: float = 1e-9) -> bool:
    """True if ``R`` is a proper rotation: ``R^T R = I`` and ``det R = +1``."""
    R = np.asarray(R, dtype=float)
    return bool(np.allclose(R.T @ R, np.eye(3), atol=atol)
                and np.isclose(np.linalg.det(R), 1.0, atol=atol))


if __name__ == "__main__":
    # quick self-test: exp/log and quat/matrix round trips agree, and qdot = 1/2 q (X) w
    rng = np.random.default_rng(0)
    max_rt = 0.0
    for _ in range(1000):
        w = rng.uniform(-np.pi, np.pi, size=3)
        R = exp_so3(w)
        # exp/log round trip
        max_rt = max(max_rt, np.linalg.norm(exp_so3(log_so3(R)) - R))
        # quaternion path agrees with the matrix path
        q = quat_exp(w)
        max_rt = max(max_rt, np.linalg.norm(quat_to_matrix(q) - R))
        # matrix -> quat -> matrix
        max_rt = max(max_rt, np.linalg.norm(quat_to_matrix(matrix_to_quat(R)) - R))
    print(f"rotation round-trip max error over 1000 samples: {max_rt:.2e}")

    # double cover: q and -q are the same rotation
    q = random_quaternion(rng)
    print("R(q) == R(-q):", np.allclose(quat_to_matrix(q), quat_to_matrix(-q)))

    # homomorphism: R(q1 (X) q2) == R(q1) R(q2)
    q1, q2 = random_quaternion(rng), random_quaternion(rng)
    lhs = quat_to_matrix(quat_multiply(q1, q2))
    rhs = quat_to_matrix(q1) @ quat_to_matrix(q2)
    print("R(q1 (X) q2) == R(q1) R(q2):", np.allclose(lhs, rhs))
