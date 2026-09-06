"""Fixed-pivot planar segment and point load; SI units, angle from downward."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SingleJoint:
    """Synthetic teaching parameters, not subject-specific anthropometry."""

    mass: float = 1.5
    length: float = 0.35
    com_distance: float = 0.175
    com_inertia: float = 0.0153125
    load_mass: float = 2.0
    gravity: float = 9.81
    damping: float = 0.08

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not all(np.isfinite(x) for x in values):
            raise ValueError('Parameters must be finite.')
        if self.mass <= 0 or self.length <= 0 or self.com_inertia <= 0:
            raise ValueError('Mass, length and COM inertia must be positive.')
        if not 0 <= self.com_distance <= self.length:
            raise ValueError('COM distance must lie on the segment.')
        if min(self.load_mass, self.gravity, self.damping) < 0:
            raise ValueError('Load mass, gravity and damping must be nonnegative.')

    @property
    def inertia(self):
        """Moment of inertia about the fixed pivot, kg m^2."""
        return self.com_inertia + self.mass * self.com_distance**2 + self.load_mass * self.length**2

    @property
    def gravity_coefficient(self):
        """Amplitude K of gravitational torque, N m."""
        return (self.mass * self.com_distance + self.load_mass * self.length) * self.gravity

    def torque_terms(self, q, velocity, acceleration):
        """Return left-hand equation terms (NOT three applied torques), N m."""
        return (self.inertia * np.asarray(acceleration),
                self.damping * np.asarray(velocity),
                self.gravity_coefficient * np.sin(q))

    def inverse_dynamics(self, q, velocity, acceleration):
        """Required ideal actuator torque, N m."""
        return sum(self.torque_terms(q, velocity, acceleration))

    def energy(self, q, velocity):
        """Mechanical energy in J; potential zero at downward posture."""
        return 0.5 * self.inertia * np.asarray(velocity)**2 + self.gravity_coefficient * (1 - np.cos(q))

    def simulate(self, times, initial_state, torque, rtol=1e-10, atol=1e-12):
        """Integrate [q, velocity] with torque(t), sampling at increasing times."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError('Times must be a finite, strictly increasing vector.')

        def rhs(t, state):
            q, velocity = state
            return [velocity, (torque(t) - self.damping * velocity - self.gravity_coefficient * np.sin(q)) / self.inertia]

        result = solve_ivp(rhs, (times[0], times[-1]), initial_state,
                           t_eval=times, method='DOP853', rtol=rtol, atol=atol)
        if not result.success:
            raise RuntimeError(result.message)
        return result


def reference_motion(t):
    """Prescribed 30-60 degree movement; return q, qdot, qddot in SI units."""
    t = np.asarray(t)
    return (np.pi / 4 + np.pi / 12 * np.sin(np.pi * t),
            np.pi**2 / 12 * np.cos(np.pi * t),
            -np.pi**3 / 12 * np.sin(np.pi * t))
