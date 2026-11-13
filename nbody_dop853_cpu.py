"""Numba DOP853 stages, embedded norm and step factor, float64/no fastmath.

Coefficients come from the installed SciPy DOP853. Controller constants and
error norm follow scipy.integrate._ivp.rk, with forward-time integration.
Python retains cancellation between trials and the existing recording interface.
"""
import numpy as np
from numba import njit
from scipy.integrate import DOP853
from nbody_array_cpu import fields_serial, fields_parallel, thread_scope

A = np.ascontiguousarray(DOP853.A)
B = np.ascontiguousarray(DOP853.B)
E3 = np.ascontiguousarray(DOP853.E3)
E5 = np.ascontiguousarray(DOP853.E5)


def _make_kernels(fields):
    @njit(nogil=True, fastmath=False)
    def rhs(y, mass, gravity):
        n = len(mass)
        pos = y[:3*n].reshape(n, 3)
        vel = y[3*n:].reshape(n, 3)
        acceleration, _, _ = fields(pos, vel, mass, gravity, False)
        return np.concatenate((y[3*n:], acceleration.ravel()))

    @njit(nogil=True, fastmath=False)
    def trial(y, f, mass, gravity, h, adaptive, rtol, atol, rejected_before):
        K = np.empty((13, len(y)), dtype=np.float64)
        K[0] = f
        for stage in range(1, 12):
            # Autonomous gravity: stage times do not appear in the RHS.
            intermediate = y + h*np.dot(A[stage, :stage], K[:stage])
            K[stage] = rhs(intermediate, mass, gravity)
        y_new = y + h*np.dot(B, K[:12])
        f_new = rhs(y_new, mass, gravity)
        K[12] = f_new
        if not adaptive:
            return y_new, f_new, 0., 1., True
        scale = atol + np.maximum(np.abs(y), np.abs(y_new))*rtol
        err5 = np.dot(K.T, E5)/scale
        err3 = np.dot(K.T, E3)/scale
        norm5 = np.linalg.norm(err5)**2
        norm3 = np.linalg.norm(err3)**2
        error = 0. if norm5 == 0. and norm3 == 0. else abs(h)*norm5/np.sqrt((norm5+.01*norm3)*len(y))
        accepted = error < 1.
        if accepted:
            factor = 10. if error == 0. else min(10., .9*error**(-1./8.))
            if rejected_before:
                factor = min(1., factor)
        else:
            factor = max(.2, .9*error**(-1./8.)) if np.isfinite(error) else .2
        return y_new, f_new, error, factor, accepted

    return rhs, trial


rhs_serial, trial_serial = _make_kernels(fields_serial)
rhs_parallel, trial_parallel = _make_kernels(fields_parallel)


def warmup_dop853_backend(parallel=False):
    """Compile disposable float64 signatures; no user state advances."""
    import numba
    with thread_scope():
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        mass = np.ones(2)
        y = np.array([-.5,0.,0., .5,0.,0., 0.,-.5,0., 0.,.5,0.])
        rhs, trial = (rhs_parallel, trial_parallel) if parallel else (rhs_serial, trial_serial)
        f = rhs(y, mass, 1.)
        for adaptive in (False, True):
            trial(y, f, mass, 1., .001, adaptive, 1e-9, 1e-12, False)


class CompiledDOP853:
    def __init__(self, y, state, config, cancel_check):
        self.rhs, self.trial = ((rhs_parallel, trial_parallel) if state.parallel
                               else (rhs_serial, trial_serial))
        self.mass, self.gravity = state.mass, state.gravity
        self.y = y.copy()
        self.f = self.rhs(self.y, self.mass, self.gravity)
        self.adaptive = config['adaptive_enabled']
        self.max_step, self.bound = config['dt'], config['total_time']
        self.rtol, self.atol = config['dop853_rtol'], config['dop853_atol']
        self.next_step = min(self.max_step, self.bound)
        self.cancel_check = cancel_check
        self.nfev = 1
        self.rejected_trials = 0
        self.last_error_norm = None
        # Compile the actual signature before the shared simulation clock starts.
        # The returned disposable trial never replaces self.y/self.f.
        self.trial(self.y, self.f, self.mass, self.gravity, 0., self.adaptive,
                   self.rtol, self.atol, False)

    def advance(self, time, fixed_h=None):
        minimum = 10*abs(np.nextafter(time, np.inf)-time)
        if self.adaptive:
            proposal = (self.max_step if self.next_step > self.max_step else
                        minimum if self.next_step < minimum else self.next_step)
        else:
            proposal = fixed_h
        rejected = False
        while True:
            self.cancel_check()
            if self.adaptive and proposal < minimum:
                raise RuntimeError(f'DOP853 failed at t={time:.12g}: required step is below floating-point time spacing.')
            new_time = min(time+proposal, self.bound)
            h = new_time-time if self.adaptive else fixed_h
            if h <= 0 or new_time <= time:
                raise RuntimeError('DOP853 cannot advance at floating-point time resolution.')
            y, f, error, factor, accepted = self.trial(
                self.y, self.f, self.mass, self.gravity, float(h), self.adaptive,
                self.rtol, self.atol, rejected)
            self.nfev += 12
            if accepted:
                self.y, self.f = y, f
                self.next_step = h*factor
                self.last_error_norm = float(error) if self.adaptive else None
                return new_time, h
            self.rejected_trials += 1
            rejected = True
            proposal = h*factor
