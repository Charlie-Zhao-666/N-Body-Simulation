"""True order-45 Taylor recurrence in hardware float64, serial Numba, no fastmath.

Same coefficient recurrence as nbody_taylor45.TaylorNBody. No native error
controller; outer application owns h, rejection and pre-probe. Compilation and
disposable warm-up occur before timing. No mpmath dependency on this path.
"""
import numpy as np
import numba
from numba import njit

ORDER = 45


@njit(nogil=True, fastmath=False, cache=True)
def taylor45_trial(pos, vel, mass, gravity, h):
    n = len(mass)
    r = np.zeros((n, 3, ORDER+2), dtype=np.float64)
    r[:, :, 0] = pos
    r[:, :, 1] = vel
    pairs = n*(n-1)//2
    delta = np.zeros((pairs, 3, ORDER), dtype=np.float64)
    q = np.zeros((pairs, ORDER), dtype=np.float64)
    b = np.zeros((pairs, ORDER), dtype=np.float64)
    for k in range(ORDER):
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
                    if qk <= 0.0 or not np.isfinite(qk):
                        raise ValueError("Taylor45 float64: coincident particles or out-of-range separation.")
                    b[pair, 0] = 1.0/(qk*np.sqrt(qk))
                else:
                    bk = 0.0
                    for term in range(1, k+1):
                        bk += (term+2*k)*q[pair, term]*b[pair, k-term]
                    b[pair, k] = -bk/(2*k*q[pair, 0])
                for d in range(3):
                    force = 0.0
                    for l in range(k+1):
                        force += delta[pair, d, l]*b[pair, k-l]
                    force *= gravity
                    acc[i, d] += mass[j]*force
                    acc[j, d] -= mass[i]*force
                pair += 1
        for i in range(n):
            for d in range(3):
                r[i, d, k+2] = acc[i, d]/((k+1)*(k+2))
                if not np.isfinite(r[i, d, k+2]):
                    raise RuntimeError("Taylor45 float64 coefficients overflowed. Use 80-digit precision or inspect close encounters.")
    new_pos = np.empty_like(pos)
    new_vel = np.empty_like(vel)
    for i in range(n):
        for d in range(3):
            p, v = r[i, d, ORDER], (ORDER+1)*r[i, d, ORDER+1]
            for k in range(ORDER-1, -1, -1):
                p = p*h+r[i, d, k]
                v = v*h+(k+1)*r[i, d, k+1]
            new_pos[i, d], new_vel[i, d] = p, v
    return new_pos, new_vel


def warmup_taylor45_backend():
    """Compile a disposable, regular binary; do not touch any user's initial state."""
    taylor45_trial(np.array([[-1.,0.,0.],[1.,0.,0.]]),
                   np.array([[0.,-1.,0.],[0.,1.,0.]]), np.array([4.,4.]), 1., .01)


class Float64TaylorNBody:
    def __init__(self, masses, positions, velocities, gravity=1., cancel_check=lambda: None):
        self.mass = np.ascontiguousarray(masses, dtype=np.float64)
        self.pos = np.array(positions, dtype=np.float64, order='C', copy=True)
        self.vel = np.array(velocities, dtype=np.float64, order='C', copy=True)
        self.gravity = float(gravity)
        if (not len(self.mass) or self.mass.ndim != 1 or self.pos.shape != (len(self.mass),3)
                or self.vel.shape != self.pos.shape or not np.all(self.mass>0)
                or not all(np.isfinite(a).all() for a in (self.mass,self.pos,self.vel))
                or not np.isfinite(self.gravity) or self.gravity<=0):
            raise ValueError("Taylor45 requires finite N x 3 states, positive masses and positive G.")
        self.cancel_check = cancel_check
        self.trials, self.time = 0, 0.0
        # Warm regular dummy data instead of evaluating a potentially singular
        # high-order series of the user's close-encounter initial state.
        self.cancel_check()
        warmup_taylor45_backend()
        self.cancel_check()

    def checkpoint(self):
        return self.pos.copy(), self.vel.copy(), self.time

    def restore(self, checkpoint):
        pos, vel, self.time = checkpoint
        self.pos, self.vel = pos.copy(), vel.copy()

    def advance(self, h):
        h = float(h)
        if not np.isfinite(h) or h<=0:
            raise ValueError("Taylor45 requires a finite positive step.")
        self.cancel_check()
        pos, vel = taylor45_trial(self.pos, self.vel, self.mass, self.gravity, h)
        self.cancel_check()
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            raise RuntimeError("Taylor45 float64 produced non-finite states. Reduce dt or inspect close encounters.")
        self.pos, self.vel = pos, vel
        self.time += h
        self.trials += 1

    def report(self):
        return dict(taylor_order=ORDER, taylor_precision='float64', taylor_mantissa_bits=53,
                    taylor_decimal_precision=15, taylor_numba_version=numba.__version__,
                    taylor_trial_evaluations=self.trials, taylor_internal_time=self.time,
                    taylor_final_positions=self.pos.tolist(), taylor_final_velocities=self.vel.tolist(),
                    taylor_precision_scope='IEEE float64 throughout (53-bit significand, about 15-16 decimal digits); not a guaranteed error bound.',
                    taylor_execution='CPU single-thread Numba; float64; fastmath=False; degree-45 Taylor position/velocity polynomials; no native controller')
