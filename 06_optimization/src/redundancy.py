"""Two synergist muscles at fixed posture. Forces N, arms m, torque N m."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class TwoMuscles:
    arms: tuple = (0.04, 0.02)
    limits: tuple = (300.0, 100.0)

    def __post_init__(self):
        for values in (self.arms, self.limits):
            if np.shape(values) != (2,) or not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
                raise ValueError('Two finite positive arms and limits are required.')

    @property
    def capacity(self):
        return float(np.dot(self.arms, self.limits))

    def interval(self, torque):
        """Feasible F1 interval, or None when equality cannot be satisfied."""
        if not np.isfinite(torque):
            raise ValueError('Torque must be finite.')
        r1, r2 = self.arms
        u1, u2 = self.limits
        if torque < 0 or torque > self.capacity:
            return None
        return max(0.0, (torque - r2 * u2) / r1), min(u1, torque / r1)

    def forces(self, first_force, torque):
        """Recover F2 from equilibrium; caller checks bounds separately."""
        return np.array([first_force, (torque - self.arms[0] * first_force) / self.arms[1]])

    @staticmethod
    def weights(values):
        values = np.asarray(values, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError('Weights must be two finite positive numbers.')
        return values / np.max(values)

    def solve(self, torque, weights=(1, 1)):
        """Analytic constrained optimum for positive diagonal quadratic weights."""
        w1, w2 = self.weights(weights)
        interval = self.interval(torque)
        if interval is None:
            raise ValueError(f'Infeasible torque: expected 0 <= torque <= {self.capacity:g} N m.')
        r1, r2 = self.arms
        unconstrained = w2 * r1 * torque / (w1 * r2**2 + w2 * r1**2)
        return self.forces(np.clip(unconstrained, *interval), torque)

    def solve_numerically(self, torque, weights=(1, 1)):
        """SLSQP comparison in capacity-normalized force coordinates."""
        weights = self.weights(weights)
        interval = self.interval(torque)
        if interval is None:
            raise ValueError('Infeasible torque.')
        limits = np.asarray(self.limits)
        # At either extreme the feasible set is a single point; no optimization needed.
        if torque == 0 or torque == self.capacity:
            return self.forces(interval[0], torque)
        coefficient = weights * limits**2
        coefficient /= coefficient.max()
        constraint = np.asarray(self.arms) * limits / self.capacity
        initial = self.forces(np.mean(interval), torque) / limits
        result = minimize(lambda z: np.dot(coefficient, z**2), initial,
                          jac=lambda z: 2 * coefficient * z,
                          bounds=[(0, 1), (0, 1)], method='SLSQP',
                          constraints={'type': 'eq',
                                       'fun': lambda z: constraint @ z - torque / self.capacity,
                                       'jac': lambda z: constraint},
                          options={'ftol': 1e-12, 'maxiter': 200})
        if not result.success:
            raise RuntimeError(result.message)
        force = result.x * limits
        if abs(np.dot(self.arms, force) - torque) > 1e-7 or np.any(force < -1e-7) or np.any(force > limits + 1e-7):
            raise RuntimeError('Solver output failed feasibility checks.')
        return force
