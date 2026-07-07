"""Regression tests for :mod:`rotations_utils` (Stage 1, Part 2: rotations).

These enforce, pointwise and over random samples, the correctness claims the notes
make by hand for the three languages of a 3D rotation:

1. **SO(3) structure** -- ``exp_so3`` produces proper rotations (``R^T R = I``,
   ``det R = +1``) and ``log_so3`` is its inverse (exp/log round trip).
2. **Double cover** -- ``R(q) = R(-q)``: the antipodal unit quaternions are the same
   rotation, and ``matrix_to_quat`` / ``quat_to_matrix`` round-trip.
3. **Group homomorphism** -- ``R(q1 (X) q2) = R(q1) R(q2)`` and ``quat_rotate`` agrees
   with the matrix action.
4. **Agreement with SciPy** -- every conversion matches
   ``scipy.spatial.transform.Rotation`` (an independent implementation).
5. **SLERP** -- hits its endpoints and moves at constant angular velocity
   (equal-angle increments), i.e. it is the geodesic on ``S^3``.
6. **Orientation kinematics** -- ``qdot = 1/2 q (X) omega`` recovers the angular
   velocity that generated a trajectory, and integrating it keeps ``||q|| = 1``.
"""
from __future__ import annotations

import numpy as np
import pytest

from rotations_utils import (
    axis_angle_to_quat, euler_rate_matrix, euler_to_matrix, exp_so3, hat,
    is_rotation, log_so3, matrix_to_quat, quat_conjugate, quat_derivative,
    quat_exp, quat_from_scipy, quat_kinematics_rhs, quat_log, quat_multiply,
    quat_rotate, quat_to_axis_angle, quat_to_matrix, quat_to_scipy,
    random_quaternion, random_rotation, slerp, vee,
)

try:
    from scipy.spatial.transform import Rotation as ScipyRotation
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    _HAVE_SCIPY = False

requires_scipy = pytest.mark.skipif(not _HAVE_SCIPY, reason="scipy not available")


# --------------------------------------------------------------------------------------
# hat / vee
# --------------------------------------------------------------------------------------
def test_hat_is_cross_product_and_skew():
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = rng.uniform(-3, 3, size=3)
        v = rng.uniform(-3, 3, size=3)
        assert np.allclose(hat(w) @ v, np.cross(w, v))
        assert np.allclose(hat(w), -hat(w).T)            # skew-symmetric
        assert np.allclose(vee(hat(w)), w)               # vee inverts hat


# --------------------------------------------------------------------------------------
# 1. SO(3) structure: exp_so3 gives proper rotations; log_so3 inverts it
# --------------------------------------------------------------------------------------
def test_exp_so3_is_proper_rotation():
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = rng.uniform(-np.pi, np.pi, size=3)
        R = exp_so3(w)
        assert is_rotation(R)

def test_exp_so3_identity_at_zero():
    assert np.allclose(exp_so3(np.zeros(3)), np.eye(3))
    assert np.allclose(exp_so3(np.array([1e-10, 0, 0])), np.eye(3), atol=1e-9)

def test_exp_log_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(300):
        # keep |theta| < pi so the returned w matches the input axis-angle exactly
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = rng.uniform(0, np.pi - 1e-3)
        w = theta * axis
        R = exp_so3(w)
        assert np.allclose(log_so3(R), w, atol=1e-9)
        assert np.allclose(exp_so3(log_so3(R)), R, atol=1e-9)

def test_log_so3_handles_pi_rotation():
    for axis in np.eye(3):
        R = exp_so3(np.pi * axis)
        w = log_so3(R)
        # exp of the recovered log reproduces R even though the axis sign is ambiguous
        assert np.allclose(exp_so3(w), R, atol=1e-9)
        assert np.isclose(np.linalg.norm(w), np.pi, atol=1e-8)


# --------------------------------------------------------------------------------------
# 2. Quaternion <-> matrix, and the double cover
# --------------------------------------------------------------------------------------
def test_quat_to_matrix_is_rotation():
    rng = np.random.default_rng(3)
    for _ in range(200):
        q = random_quaternion(rng)
        assert is_rotation(quat_to_matrix(q))

def test_double_cover_q_and_minus_q():
    rng = np.random.default_rng(4)
    for _ in range(200):
        q = random_quaternion(rng)
        assert np.allclose(quat_to_matrix(q), quat_to_matrix(-q))

def test_matrix_quat_roundtrip():
    rng = np.random.default_rng(5)
    for _ in range(300):
        R = random_rotation(rng)
        assert np.allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-9)

def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(6)
    for _ in range(300):
        q = random_quaternion(rng)          # already has w >= 0
        q2 = matrix_to_quat(quat_to_matrix(q))
        assert np.allclose(q, q2, atol=1e-8)

def test_quat_exp_matches_exp_so3():
    rng = np.random.default_rng(7)
    for _ in range(300):
        w = rng.uniform(-np.pi, np.pi, size=3)
        assert np.allclose(quat_to_matrix(quat_exp(w)), exp_so3(w), atol=1e-10)

def test_quat_log_inverts_quat_exp():
    rng = np.random.default_rng(8)
    for _ in range(300):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        theta = rng.uniform(0, np.pi - 1e-3)
        w = theta * axis
        assert np.allclose(quat_log(quat_exp(w)), w, atol=1e-9)


# --------------------------------------------------------------------------------------
# 3. Group homomorphism and the vector action
# --------------------------------------------------------------------------------------
def test_composition_is_homomorphism():
    rng = np.random.default_rng(9)
    for _ in range(200):
        q1, q2 = random_quaternion(rng), random_quaternion(rng)
        lhs = quat_to_matrix(quat_multiply(q1, q2))
        rhs = quat_to_matrix(q1) @ quat_to_matrix(q2)
        assert np.allclose(lhs, rhs, atol=1e-10)

def test_quat_rotate_matches_matrix_action():
    rng = np.random.default_rng(10)
    for _ in range(200):
        q = random_quaternion(rng)
        v = rng.uniform(-5, 5, size=3)
        assert np.allclose(quat_rotate(q, v), quat_to_matrix(q) @ v, atol=1e-10)

def test_quat_rotate_preserves_length():
    rng = np.random.default_rng(11)
    for _ in range(200):
        q = random_quaternion(rng)
        v = rng.uniform(-5, 5, size=3)
        assert np.isclose(np.linalg.norm(quat_rotate(q, v)), np.linalg.norm(v))

def test_conjugate_undoes_rotation():
    rng = np.random.default_rng(12)
    for _ in range(200):
        q = random_quaternion(rng)
        v = rng.uniform(-5, 5, size=3)
        back = quat_rotate(quat_conjugate(q), quat_rotate(q, v))
        assert np.allclose(back, v, atol=1e-10)


# --------------------------------------------------------------------------------------
# 4. Agreement with SciPy (independent implementation, scalar-last convention)
# --------------------------------------------------------------------------------------
@requires_scipy
def test_quat_to_matrix_matches_scipy():
    rng = np.random.default_rng(13)
    for _ in range(200):
        q = random_quaternion(rng)
        R_scipy = ScipyRotation.from_quat(quat_to_scipy(q)).as_matrix()
        assert np.allclose(quat_to_matrix(q), R_scipy, atol=1e-12)

@requires_scipy
def test_exp_so3_matches_scipy_rotvec():
    rng = np.random.default_rng(14)
    for _ in range(200):
        w = rng.uniform(-np.pi, np.pi, size=3)
        R_scipy = ScipyRotation.from_rotvec(w).as_matrix()
        assert np.allclose(exp_so3(w), R_scipy, atol=1e-12)

@requires_scipy
def test_axis_angle_matches_scipy():
    rng = np.random.default_rng(15)
    for _ in range(200):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        angle = rng.uniform(0, np.pi - 1e-3)
        q = axis_angle_to_quat(axis, angle)
        R_scipy = ScipyRotation.from_rotvec(angle * axis).as_matrix()
        assert np.allclose(quat_to_matrix(q), R_scipy, atol=1e-12)

@requires_scipy
def test_euler_to_matrix_matches_scipy_intrinsic():
    rng = np.random.default_rng(16)
    for seq in ("zyx", "xyz", "yxy", "zxz"):
        for _ in range(30):
            angles = rng.uniform(-2, 2, size=3)
            # SciPy uppercase = intrinsic (rotating axes), matching euler_to_matrix
            R_scipy = ScipyRotation.from_euler(seq.upper(), angles).as_matrix()
            assert np.allclose(euler_to_matrix(angles, seq, intrinsic=True),
                               R_scipy, atol=1e-10)


# --------------------------------------------------------------------------------------
# 5. SLERP: endpoints and constant angular velocity (geodesic on S^3)
# --------------------------------------------------------------------------------------
def test_slerp_endpoints():
    rng = np.random.default_rng(17)
    for _ in range(50):
        q0, q1 = random_quaternion(rng), random_quaternion(rng)
        assert np.allclose(quat_to_matrix(slerp(q0, q1, 0.0)), quat_to_matrix(q0))
        assert np.allclose(quat_to_matrix(slerp(q0, q1, 1.0)), quat_to_matrix(q1))

def test_slerp_constant_angular_velocity():
    rng = np.random.default_rng(18)
    for _ in range(30):
        q0, q1 = random_quaternion(rng), random_quaternion(rng)
        ts = np.linspace(0, 1, 21)
        qs = slerp(q0, q1, ts)
        # relative rotation angle between successive samples must be equal
        angles = []
        for a, b in zip(qs[:-1], qs[1:]):
            rel = quat_multiply(quat_conjugate(a), b)
            angles.append(quat_to_axis_angle(rel)[1])
        angles = np.array(angles)
        assert np.allclose(angles, angles.mean(), atol=1e-9)

def test_slerp_stays_on_unit_sphere():
    rng = np.random.default_rng(19)
    q0, q1 = random_quaternion(rng), random_quaternion(rng)
    qs = slerp(q0, q1, np.linspace(0, 1, 50))
    assert np.allclose(np.linalg.norm(qs, axis=1), 1.0, atol=1e-12)

@requires_scipy
def test_slerp_matches_scipy():
    rng = np.random.default_rng(20)
    from scipy.spatial.transform import Slerp
    q0, q1 = random_quaternion(rng), random_quaternion(rng)
    key = ScipyRotation.from_quat([quat_to_scipy(q0), quat_to_scipy(q1)])
    sc = Slerp([0.0, 1.0], key)
    for t in np.linspace(0, 1, 11):
        R_ours = quat_to_matrix(slerp(q0, q1, t))
        assert np.allclose(R_ours, sc([t]).as_matrix()[0], atol=1e-10)


# --------------------------------------------------------------------------------------
# 6. Orientation kinematics: qdot = 1/2 q (X) omega
# --------------------------------------------------------------------------------------
def test_quat_derivative_matches_matrix_kinematics():
    """qdot = 1/2 q (X) omega  <=>  Rdot = R [omega_b]_x  (body frame)."""
    rng = np.random.default_rng(21)
    for _ in range(100):
        q = random_quaternion(rng)
        omega = rng.uniform(-3, 3, size=3)
        # finite-difference d/dt of R for the two definitions must agree
        dq = quat_derivative(q, omega, frame="body")
        eps = 1e-6
        q_plus = q + eps * dq
        q_plus /= np.linalg.norm(q_plus)
        Rdot_fd = (quat_to_matrix(q_plus) - quat_to_matrix(q)) / eps
        Rdot_body = quat_to_matrix(q) @ hat(omega)
        assert np.allclose(Rdot_fd, Rdot_body, atol=1e-4)

def test_space_vs_body_frame_relation():
    """omega_s = R omega_b, so space and body derivatives describe the same motion."""
    rng = np.random.default_rng(22)
    for _ in range(100):
        q = random_quaternion(rng)
        omega_b = rng.uniform(-3, 3, size=3)
        omega_s = quat_to_matrix(q) @ omega_b
        assert np.allclose(quat_derivative(q, omega_b, "body"),
                           quat_derivative(q, omega_s, "space"), atol=1e-12)

def test_integrated_kinematics_recovers_orientation():
    """Integrating qdot = 1/2 q (X) omega_b with constant omega_b gives exp(t omega_b)."""
    from scipy.integrate import solve_ivp
    omega_b = np.array([0.4, -0.9, 1.3])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    rhs = quat_kinematics_rhs(lambda t: omega_b, frame="body")
    sol = solve_ivp(rhs, (0, 3.0), q0, rtol=1e-10, atol=1e-12, dense_output=True)
    assert sol.success
    qT = sol.y[:, -1]
    qT /= np.linalg.norm(qT)
    R_expected = exp_so3(3.0 * omega_b)         # constant body rate -> matrix exponential
    assert np.allclose(quat_to_matrix(qT), R_expected, atol=1e-7)
    # the constraint ||q|| = 1 is preserved along the whole trajectory
    norms = np.linalg.norm(sol.y, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-5)


# --------------------------------------------------------------------------------------
# 7. Euler-angle gimbal lock: the rate map becomes singular
# --------------------------------------------------------------------------------------
def test_euler_rate_matrix_singular_at_gimbal_lock():
    # zyx Cardan angles lose a DOF when the middle (y) angle is +/- pi/2
    E_ok = euler_rate_matrix([0.3, 0.4, 0.5], seq="zyx")
    E_lock = euler_rate_matrix([0.3, np.pi / 2, 0.5], seq="zyx")
    assert abs(np.linalg.det(E_ok)) > 1e-2
    assert abs(np.linalg.det(E_lock)) < 1e-9        # singular at gimbal lock

def test_euler_rate_matrix_maps_rates_to_space_omega():
    """omega_s = E(angles) @ angle_rates must match a finite-difference of R."""
    rng = np.random.default_rng(23)
    seq = "zyx"
    for _ in range(50):
        angles = rng.uniform(-1.0, 1.0, size=3)     # away from gimbal lock
        rates = rng.uniform(-1.0, 1.0, size=3)
        omega_pred = euler_rate_matrix(angles, seq) @ rates
        eps = 1e-6
        R0 = euler_to_matrix(angles, seq)
        R1 = euler_to_matrix(angles + eps * rates, seq)
        omega_fd = vee(((R1 - R0) / eps) @ R0.T)     # space-frame omega = vee(Rdot R^T)
        assert np.allclose(omega_pred, omega_fd, atol=1e-4)


# --------------------------------------------------------------------------------------
# 8. SciPy-bridge round trip
# --------------------------------------------------------------------------------------
def test_scipy_bridge_roundtrip():
    rng = np.random.default_rng(24)
    for _ in range(50):
        q = random_quaternion(rng)
        assert np.allclose(quat_from_scipy(quat_to_scipy(q)), q)
