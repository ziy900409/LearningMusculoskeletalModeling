"""Physical checks independent of a forward/inverse round trip."""
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from single_joint import SingleJoint, reference_motion


def test_hand_calculated_static_load():
    model = SingleJoint()
    assert model.inertia == pytest.approx(0.30625)
    assert model.inverse_dynamics(np.pi / 2, 0, 0) == pytest.approx(9.442125)
    assert model.inverse_dynamics(0, 0, 0) == 0
    assert model.inverse_dynamics(-np.pi / 2, 0, 0) == pytest.approx(-9.442125)


def test_constant_torque_matches_analytic_solution():
    model = SingleJoint(gravity=0, damping=0)
    times = np.linspace(0, 2, 201)
    result = model.simulate(times, [0.2, -0.1], lambda t: 0.4)
    expected = 0.2 - 0.1 * times + 0.4 / (2 * model.inertia) * times**2
    np.testing.assert_allclose(result.y[0], expected, atol=1e-10)


def test_energy_conservation_and_dissipation():
    times = np.linspace(0, 5, 1001)
    model = SingleJoint(damping=0)
    result = model.simulate(times, [0.35, 0], lambda t: 0)
    energy = model.energy(*result.y)
    assert np.max(np.abs(energy - energy[0])) / (2 * model.gravity_coefficient) < 1e-7
    damped = replace(model, damping=0.08)
    result = damped.simulate(times, [0.35, 0], lambda t: 0)
    energy = damped.energy(*result.y)
    assert np.max(np.diff(energy)) < 1e-9
    assert energy[-1] < energy[0] * 0.5


def test_prescribed_motion_reconstructed_from_torque():
    model = SingleJoint()
    times = np.linspace(0, 2, 301)
    q, velocity, _ = reference_motion(times)
    result = model.simulate(times, [q[0], velocity[0]],
                            lambda t: model.inverse_dynamics(*reference_motion(t)))
    np.testing.assert_allclose(result.y[0], q, atol=1e-6, rtol=0)


def test_small_angle_period():
    model = SingleJoint(damping=0)
    omega = np.sqrt(model.gravity_coefficient / model.inertia)
    times = np.linspace(0, 2 * np.pi / omega, 301)
    result = model.simulate(times, [1e-4, 0], lambda t: 0)
    np.testing.assert_allclose(result.y[0], 1e-4 * np.cos(omega * times), atol=1e-10)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        SingleJoint(damping=-1)
    with pytest.raises(ValueError):
        SingleJoint().simulate([0, 0], [0, 0], lambda t: 0)
