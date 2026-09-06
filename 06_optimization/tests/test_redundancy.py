import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from redundancy import TwoMuscles


def test_hand_calculations():
    model = TwoMuscles()
    assert model.interval(8) == (150, 200)
    np.testing.assert_allclose(model.solve(8), [160, 80])
    np.testing.assert_allclose(model.solve(8, (1/300**2, 1/100**2)), [7200/37, 400/37])


@pytest.mark.parametrize('torque', [0, 2, 8, 10, 12, 13.9, 14])
@pytest.mark.parametrize('weights', [(1, 1), (1, 9), (9, 1)])
def test_solver_and_feasibility(torque, weights):
    model = TwoMuscles()
    force = model.solve(torque, weights)
    np.testing.assert_allclose(model.solve_numerically(torque, weights), force, atol=1e-5)
    assert np.dot(model.arms, force) == pytest.approx(torque, abs=1e-10)
    assert np.all(force >= -1e-10) and np.all(force <= model.limits)


def test_boundary_and_impossibility():
    model = TwoMuscles()
    np.testing.assert_allclose(model.solve(12), [250, 100])
    np.testing.assert_allclose(model.solve(14), [300, 100])
    for torque in [-1, 16]:
        assert model.interval(torque) is None
        with pytest.raises(ValueError, match='Infeasible'):
            model.solve(torque)
        with pytest.raises(ValueError, match='Infeasible'):
            model.solve_numerically(torque)


def test_objective_and_nullspace():
    model = TwoMuscles()
    assert np.dot(model.arms, [1, -2]) == 0
    for weights in [(1, 1), (1, 9)]:
        answer = model.solve(8, weights)
        grid = model.forces(np.linspace(150, 200, 10001), 8)
        assert np.dot(weights, answer**2) <= np.min(np.asarray(weights) @ grid**2) + 1e-8
        np.testing.assert_allclose(model.solve(8, np.array(weights)*100), answer)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        TwoMuscles(arms=(-0.04, 0.02))
    with pytest.raises(ValueError):
        TwoMuscles().solve(8, (0, 1))
