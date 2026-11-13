"""Order 100/150/200 Newtonian Taylor steps, float64 single-thread Numba.

The h-independent time scale keeps coefficients within a better numeric range:
tau=(t-t0)/s, R_k=s**k*r_k. Both physical position and velocity have degree p;
R_(p+1) supplies the last velocity term. The application owns all adaptation.
The original unscaled order-45 implementation is deliberately kept separate.
"""
import numpy as np
import numba
from numba import njit

HIGH_ORDERS = (100, 150, 200)


class TaylorCoefficientError(RuntimeError):
    """Coefficient generation failed, independently of candidate evaluation."""
    taylor_coefficient_failure = True


@njit(nogil=True, fastmath=False, cache=True)
def taylor_trial(pos, vel, mass, gravity, h, order):
    """Return candidate pos/vel and status: 0=OK, 1=start, 2=coeff, 3=value.

    No input arrays are mutated. A polynomial evaluation failure is distinct
    from coefficient failure, which cannot in general be cured by shrinking h.
    """
    if order != 100 and order != 150 and order != 200:
        raise ValueError('Taylor order must be 100, 150 or 200.')
    n = len(mass)
    scale = 1.0
    for i in range(n):
        for j in range(i+1, n):
            dist2 = 0.0
            for d in range(3):
                diff = pos[j, d]-pos[i, d]
                dist2 += diff*diff
            if dist2 <= 0.0 or not np.isfinite(dist2):
                return pos.copy(), vel.copy(), 1
            local = np.sqrt(dist2*np.sqrt(dist2)/(gravity*(mass[i]+mass[j])))
            scale = min(scale, local)
    if scale <= 0.0 or not np.isfinite(scale) or scale*scale == 0.0:
        return pos.copy(), vel.copy(), 2
    r = np.zeros((n, 3, order+2), dtype=np.float64)
    r[:, :, 0] = pos
    r[:, :, 1] = vel*scale
    pairs = n*(n-1)//2
    delta = np.zeros((pairs, 3, order), dtype=np.float64)
    q = np.zeros((pairs, order), dtype=np.float64)
    b = np.zeros((pairs, order), dtype=np.float64)
    for k in range(order):
        acc = np.zeros((n, 3), dtype=np.float64)
        pair = 0
        for i in range(n):
            for j in range(i+1, n):
                for d in range(3):
                    delta[pair, d, k] = r[j, d, k]-r[i, d, k]
                qk = 0.0
                for d in range(3):
                    for l in range(k+1):
                        qk += delta[pair, d, l]*delta[pair, d, k-l]
                q[pair, k] = qk
                if k == 0:
                    denominator = qk*np.sqrt(qk)
                    if denominator <= 0.0 or not np.isfinite(denominator):
                        return pos.copy(), vel.copy(), 2
                    b[pair, 0] = 1.0/denominator
                else:
                    bk = 0.0
                    for term in range(1, k+1):
                        bk += (term+2*k)*q[pair, term]*b[pair, k-term]
                    b[pair, k] = -bk/(2*k*q[pair, 0])
                if not np.isfinite(qk) or not np.isfinite(b[pair, k]):
                    return pos.copy(), vel.copy(), 2
                for d in range(3):
                    force = 0.0
                    for l in range(k+1):
                        force += delta[pair, d, l]*b[pair, k-l]
                    force *= gravity
                    force *= scale*scale
                    acc[i, d] += mass[j]*force
                    acc[j, d] -= mass[i]*force
                pair += 1
        for i in range(n):
            for d in range(3):
                r[i, d, k+2] = acc[i, d]/((k+1)*(k+2))
                if not np.isfinite(r[i, d, k+2]):
                    return pos.copy(), vel.copy(), 2
    x = h/scale
    new_pos, new_vel = np.empty_like(pos), np.empty_like(vel)
    for i in range(n):
        for d in range(3):
            p, v = r[i, d, order], (order+1)*r[i, d, order+1]
            for k in range(order-1, -1, -1):
                p = p*x+r[i, d, k]
                v = v*x+(k+1)*r[i, d, k+1]
            new_pos[i, d], new_vel[i, d] = p, v/scale
    if not np.isfinite(new_pos).all() or not np.isfinite(new_vel).all():
        return new_pos, new_vel, 3
    return new_pos, new_vel, 0


def warmup_taylor_backend(order=100):
    """Compile on disposable data, outside the application's compute timer."""
    if isinstance(order, (bool, np.bool_)) or order not in HIGH_ORDERS:
        raise ValueError('Taylor order must be 100, 150 or 200.')
    _, _, status = taylor_trial(np.array([[-1., 0., 0.], [1., 0., 0.]]),
                                np.array([[0., -1., 0.], [0., 1., 0.]]),
                                np.array([4., 4.]), 1., .01, int(order))
    if status:
        raise RuntimeError('High-order Taylor CPU preparation failed.')


class Float64HighOrderTaylorNBody:
    def __init__(self, masses, positions, velocities, gravity=1.,
                 cancel_check=lambda: None, order=100):
        if isinstance(order, (bool, np.bool_)) or order not in HIGH_ORDERS:
            raise ValueError('Taylor order must be 100, 150 or 200.')
        self.order = int(order)
        self.mass = np.array(masses, dtype=np.float64, order='C', copy=True)
        self.pos = np.array(positions, dtype=np.float64, order='C', copy=True)
        self.vel = np.array(velocities, dtype=np.float64, order='C', copy=True)
        self.gravity = float(gravity)
        if (self.mass.ndim != 1 or not len(self.mass) or self.pos.shape != (len(self.mass), 3)
                or self.vel.shape != self.pos.shape or not np.all(self.mass > 0)
                or not all(np.isfinite(a).all() for a in (self.mass, self.pos, self.vel))
                or not np.isfinite(self.gravity) or self.gravity <= 0):
            raise ValueError('Taylor requires finite N x 3 states, positive masses and positive G.')
        self.cancel_check = cancel_check
        self.trials, self.time = 0, 0.0
        self.cancel_check()
        warmup_taylor_backend(self.order)
        self.cancel_check()

    def checkpoint(self):
        return self.pos.copy(), self.vel.copy(), self.time

    def restore(self, checkpoint):
        pos, vel, self.time = checkpoint
        self.pos, self.vel = pos.copy(), vel.copy()

    def advance(self, h):
        h = float(h)
        if not np.isfinite(h) or h <= 0:
            raise ValueError('Taylor requires a finite positive step.')
        self.cancel_check()
        self.trials += 1  # Failed candidates are real work; restore never rewinds this.
        pos, vel, status = taylor_trial(self.pos, self.vel, self.mass, self.gravity, h, self.order)
        self.cancel_check()
        if status == 1:
            raise ValueError(f'Taylor{self.order}: coincident particles or out-of-range starting separation.')
        if status == 2:
            raise TaylorCoefficientError(f'Taylor{self.order} float64 coefficients overflowed despite time scaling; '
                               'shrinking dt alone may not resolve this state. Inspect the encounter or use 80-digit CPU.')
        if status == 3:
            raise FloatingPointError(f'Taylor{self.order} produced a non-finite trial state; reduce dt.')
        self.pos, self.vel = pos, vel
        self.time += h

    def report(self):
        return dict(taylor_order=self.order, taylor_precision='float64', taylor_mantissa_bits=53,
                    taylor_decimal_precision=15, taylor_numba_version=numba.__version__,
                    taylor_trial_evaluations=self.trials, taylor_internal_time=self.time,
                    taylor_final_positions=self.pos.tolist(), taylor_final_velocities=self.vel.tolist(),
                    taylor_time_scaled_coefficients=True,
                    taylor_coefficient_scale='s=min(1,min_pairs sqrt(distance^3/(G*(mi+mj)))); independent of trial h',
                    taylor_precision_scope='IEEE float64 throughout (53-bit significand, about 15-16 decimal digits); not an error bound.',
                    taylor_execution=f'CPU single-thread Numba; float64; fastmath=False; degree-{self.order} '
                                     'Taylor position/velocity polynomials; time-scaled coefficients; no native controller')
