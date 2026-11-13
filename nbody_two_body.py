"""Independent Kepler reference for isolated, non-radial Newtonian two-body motion.

Universal-variable f/g propagation solves a scalar Kepler equation independently
at each requested time. It does not advance the numerical trajectory in steps.
Elliptic, parabolic and hyperbolic states, unequal masses, 3D rotations and
uniform center-of-mass translation share the same reference.
"""
import numpy as np
from nbody_kepler import binary_state


def _stumpff(z):
    z = np.asarray(z, dtype=float)
    c, s = np.empty_like(z), np.empty_like(z)
    small = np.abs(z) < 1e-3
    q = z[small]
    c[small] = .5-q/24+q*q/720-q**3/40320+q**4/3628800
    s[small] = 1/6-q/120+q*q/5040-q**3/362880+q**4/39916800
    positive = (~small) & (z > 0)
    x = np.sqrt(z[positive])
    c[positive] = 2*np.sin(x/2)**2/z[positive]
    s[positive] = (x-np.sin(x))/x**3
    negative = (~small) & (z < 0)
    x = np.sqrt(-z[negative])
    with np.errstate(over='ignore', invalid='ignore'):
        c[negative] = 2*np.sinh(x/2)**2/x**2
        s[negative] = (np.sinh(x)-x)/x**3
    return c, s


class TwoBodyReference:
    def __init__(self, stars, gravity=1.):
        if len(stars) != 2:
            raise ValueError('An isolated two-body reference requires exactly two masses.')
        self.mass = np.array([s['mass'] for s in stars], dtype=float)
        self.pos = np.array([s['pos'] for s in stars], dtype=float)
        self.vel = np.array([s['v'] for s in stars], dtype=float)
        self.total_mass = self.mass.sum()
        self.mu = gravity*self.total_mass
        self.r0 = self.pos[1]-self.pos[0]
        self.v0 = self.vel[1]-self.vel[0]
        self.radius = float(np.linalg.norm(self.r0))
        if self.radius <= 0 or self.mu <= 0:
            raise ValueError('Two-body reference requires positive masses and noncoincident positions.')
        angular = np.cross(self.r0, self.v0)
        if np.linalg.norm(angular) == 0:
            raise ValueError('Radial collision motion is singular; no regular conic reference is provided.')
        self.rv = float(np.dot(self.r0, self.v0))
        self.alpha = 2/self.radius - float(np.dot(self.v0, self.v0))/self.mu
        self.com_pos = np.sum(self.mass[:, None]*self.pos, axis=0)/self.total_mass
        self.com_vel = np.sum(self.mass[:, None]*self.vel, axis=0)/self.total_mass
        self.eccentricity = float(np.linalg.norm(np.cross(self.v0, angular)/self.mu-self.r0/self.radius))
        self.length_scale = float(np.sqrt(np.mean(np.sum((self.pos-self.com_pos)**2, axis=1))))

    def positions(self, times):
        times = np.asarray(times, dtype=float)
        if not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError('Reference times must be finite and nonnegative.')
        root_mu = np.sqrt(self.mu)
        # Reduce complete revolutions only for relative elliptic motion. COM
        # translation always uses the original physical time.
        relative_times = times
        if self.alpha > 0:
            period = 2*np.pi/(root_mu*self.alpha**1.5)
            relative_times = np.remainder(times, period)
        target = root_mu*relative_times

        def equation(chi):
            c, s = _stumpff(self.alpha*chi**2)
            with np.errstate(over='ignore', invalid='ignore'):
                value = self.rv/root_mu*chi**2*c + (1-self.alpha*self.radius)*chi**3*s + self.radius*chi-target
            return value

        lo = np.zeros_like(times)
        hi = np.maximum(np.sqrt(self.radius), target/self.radius)
        # For hyperbolae start from a logarithmic bracket to avoid exp overflow.
        if self.alpha < 0:
            hi = np.maximum(np.sqrt(self.radius), np.log1p(target/self.radius)*2/np.sqrt(-self.alpha))
        for _ in range(128):
            value = equation(hi)
            if np.isnan(value).any():
                raise ValueError('Two-body analytic reference exceeds its floating-point range.')
            grow = value < 0
            if not grow.any():
                break
            hi = np.where(grow, hi*2, hi)
        else:
            raise ValueError('Could not bracket the universal Kepler equation.')
        for _ in range(64):
            mid = (lo+hi)/2
            value = equation(mid)
            lo = np.where(value < 0, mid, lo)
            hi = np.where(value >= 0, mid, hi)
        chi = (lo+hi)/2
        c, s = _stumpff(self.alpha*chi**2)
        f = 1-chi**2*c/self.radius
        g = relative_times-chi**3*s/root_mu
        relative = f[..., None]*self.r0+g[..., None]*self.v0
        com = self.com_pos+times[..., None]*self.com_vel
        result = np.stack((com-self.mass[1]/self.total_mass*relative,
                           com+self.mass[0]/self.total_mass*relative), axis=-2)
        result = np.where((times == 0)[..., None, None], self.pos, result)
        if not np.isfinite(result).all():
            raise ValueError('Non-finite two-body reference.')
        return result


def record_position_error(times, trajectories, analytic, length_scale, eccentricity):
    """Time integral of squared piecewise-linear position-error vectors.

    Evaluate analytic positions at every actual recorded time first. This avoids
    the lower-order interpolation of the numerical orbit to a separate grid.
    The interval integral is still a quadrature model, not exact continuous error.
    """
    times = np.asarray(times, dtype=float)
    error = np.asarray(trajectories).transpose(1, 0, 2)-analytic
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Position RMS needs increasing times and at least two records.')
    left, right = error[:-1], error[1:]
    # Positive form of (|left|^2 + left.dot(right) + |right|^2)/3.
    interval_mean = (np.sum((left+right)**2, axis=2)
                     + np.sum(left**2, axis=2)+np.sum(right**2, axis=2))/6
    per_star_squared = np.sum(np.diff(times)[:,None]*interval_mean, axis=0)/(times[-1]-times[0])
    per_star = np.sqrt(per_star_squared)
    rms = float(np.sqrt(np.mean(per_star_squared)))
    return dict(reference_kind='analytic_kepler', metric_version=2,
                rmse=rms, nrmse=rms/length_scale,
                per_star_rmse=per_star.tolist(), per_star_nrmse=(per_star/length_scale).tolist(),
                samples=len(times), time_start=float(times[0]), time_end=float(times[-1]),
                eccentricity=eccentricity, length_scale=length_scale,
                normalization='initial unweighted RMS distance from center of mass',
                metric='time-weighted RMS over all recorded position-error vectors and both bodies',
                quadrature='exact squared piecewise-linear error-vector integral per recorded interval',
                interpolation='no resampling of numerical positions; reference evaluated at actual record times',
                units='code length')


def preset_position_error(times, trajectories, eccentricity):
    result = record_position_error(times, trajectories, binary_state(times, eccentricity)[0],
                                   (1+eccentricity)/2, eccentricity)
    result.update(reference_model='preset_kepler_ellipse',
                  reference_solver='Kepler equation with 64 bisections, not numerical time integration',
                  masses=[1.,1.], G=1., semi_major_axis_relative=1.)
    return result


def two_body_position_error(times, trajectories, velocities, stars, gravity=1.):
    reference = TwoBodyReference(stars, gravity)
    result = record_position_error(times, trajectories, reference.positions(times),
                                   reference.length_scale, reference.eccentricity)
    result.update(reference_model='universal_kepler_initial_state', G=gravity,
                  masses=reference.mass.tolist(),
                  reference_solver='universal Kepler equation with 64 bracketed bisections')
    return result
