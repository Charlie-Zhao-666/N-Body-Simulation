"""Optional float64 CPU array solvers; also loaded by GUI startup warmup.

The serial and parallel solvers share arithmetic and target-star summation order.
Each parallel iteration owns one output row, so there are no shared force writes.
"""
from contextlib import contextmanager
import math
import numpy as np
import numba
from numba import njit, prange


@contextmanager
def thread_scope():
    """Restore the calling thread's Numba mask after success, cancellation or error."""
    previous = numba.get_num_threads()
    try:
        yield
    finally:
        numba.set_num_threads(previous)


def stars_to_arrays(stars):
    """Copy the dictionary initial state once; never alias the original arrays."""
    masses = np.array([s['mass'] for s in stars], dtype=np.float64)
    positions = np.array([s['pos'] for s in stars], dtype=np.float64, order='C')
    velocities = np.array([s['v'] for s in stars], dtype=np.float64, order='C')
    if (not len(masses) or positions.shape != (len(masses), 3)
            or velocities.shape != positions.shape or not np.all(masses > 0)
            or not all(np.isfinite(a).all() for a in (masses, positions, velocities))):
        raise ValueError('Array CPU requires finite positions/velocities and positive masses.')
    return masses, positions, velocities


def _fields_impl(pos, vel, mass, gravity, need_times):
    n = len(mass)
    acceleration = np.empty((n, 3), dtype=np.float64)
    potential = np.empty(n, dtype=np.float64)
    timescales = np.empty(n, dtype=np.float64)
    for i in prange(n):
        ax, ay, az, u, tau = 0.0, 0.0, 0.0, 0.0, np.inf
        for j in range(n):
            if i == j:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r2 = dx*dx + dy*dy + dz*dz
            if r2 <= 0.0 or not np.isfinite(r2):
                ax, ay, az, u, tau = np.nan, np.nan, np.nan, np.nan, np.nan
                break
            r = math.sqrt(r2)
            factor = gravity * mass[j] / (r2 * r)
            ax += factor * dx
            ay += factor * dy
            az += factor * dz
            # Ordered pairs appear twice, hence half the potential per target.
            u -= 0.5 * gravity * mass[i] * mass[j] / r
            if need_times:
                mu = gravity * (mass[i] + mass[j])
                if mu > 0.0:
                    tau = min(tau, r / math.sqrt(mu / r))
                vx = vel[j, 0] - vel[i, 0]
                vy = vel[j, 1] - vel[i, 1]
                vz = vel[j, 2] - vel[i, 2]
                speed = math.sqrt(vx*vx + vy*vy + vz*vz)
                if speed > 0.0:
                    tau = min(tau, r / speed)
        acceleration[i, 0] = ax
        acceleration[i, 1] = ay
        acceleration[i, 2] = az
        potential[i] = u
        timescales[i] = tau
    return acceleration, potential, timescales


# prange becomes a serial range in the first dispatcher. No fast-math or softening.
fields_serial = njit(nogil=True, fastmath=False)(_fields_impl)
fields_parallel = njit(nogil=True, parallel=True, fastmath=False)(_fields_impl)


def _make_advance(fields):
    @njit(nogil=True, fastmath=False)
    def advance(pos, vel, mass, acceleration, h, gravity, method, need_times):
        if method == 1:
            half_vel = vel + 0.5 * h * acceleration
            new_pos = pos + h * half_vel
            new_acceleration, potential, timescales = fields(new_pos, half_vel, mass, gravity, False)
            new_vel = half_vel + 0.5 * h * new_acceleration
            if need_times:
                # Encounter time scales require the completed velocity, not the half kick.
                new_acceleration, potential, timescales = fields(new_pos, new_vel, mass, gravity, True)
        else:
            a1, v1 = acceleration, vel
            v2 = v1 + 0.5 * h * a1
            a2, _, _ = fields(pos + 0.5 * h * v1, v2, mass, gravity, False)
            v3 = v1 + 0.5 * h * a2
            a3, _, _ = fields(pos + 0.5 * h * v2, v3, mass, gravity, False)
            v4 = v1 + h * a3
            a4, _, _ = fields(pos + h * v3, v4, mass, gravity, False)
            new_pos = pos + h * (v1 + 2*v2 + 2*v3 + v4) / 6
            new_vel = vel + h * (a1 + 2*a2 + 2*a3 + a4) / 6
            new_acceleration, potential, timescales = fields(new_pos, new_vel, mass, gravity, need_times)
        return new_pos, new_vel, new_acceleration, potential, timescales
    return advance


advance_serial = _make_advance(fields_serial)
advance_parallel = _make_advance(fields_parallel)


def warmup_cpu_backend(parallel=False):
    """Prepare production float64 signatures on disposable two-body data.

    Array length and thread count do not specialize these signatures. Exercise
    both integrators and adaptive branches without touching a user's state.
    """
    stars = [dict(mass=1.0, pos=np.array([-0.5, 0.0, 0.0]), v=np.array([0.0, -0.5, 0.0])),
             dict(mass=1.0, pos=np.array([0.5, 0.0, 0.0]), v=np.array([0.0, 0.5, 0.0]))]
    with thread_scope():
        for adaptive in (False, True):
            ArrayCPUState(stars, parallel=parallel,
                          threads=min(2, numba.config.NUMBA_NUM_THREADS), adaptive=adaptive)


class ArrayCPUState:
    """Array state implementing the same recording interface as the GPU state."""
    def __init__(self, stars, gravity=1.0, parallel=False, threads=0, adaptive=False):
        self.mass, self.pos, self.vel = stars_to_arrays(stars)
        self.host_masses = self.mass
        self.gravity = float(gravity)
        if not np.isfinite(self.gravity) or self.gravity <= 0:
            raise ValueError('Gravity must be finite and positive.')
        self.adaptive = bool(adaptive)
        self.parallel = bool(parallel)
        self.thread_limit = numba.config.NUMBA_NUM_THREADS
        if parallel:
            self.thread_count = int(threads) if threads else min(len(self.mass), self.thread_limit)
            if not 1 <= self.thread_count <= self.thread_limit:
                raise ValueError(f'CPU threads must be 0 (auto) or 1..{self.thread_limit} in this Python process.')
            numba.set_num_threads(self.thread_count)
        else:
            self.thread_count = 1
        self.fields = fields_parallel if parallel else fields_serial
        self._advance = advance_parallel if parallel else advance_serial
        self.acceleration, potential, timescales = self.fields(self.pos, self.vel, self.mass, self.gravity, self.adaptive)
        self._snapshot(potential, timescales)
        # Compile integrator signatures on disposable data, before the simulation timer.
        # Positions/velocities returned here are discarded; no real initial state advances.
        for method in (1, 2):
            self._advance(self.pos, self.vel, self.mass, self.acceleration,
                          0.0, self.gravity, method, self.adaptive)

    def _snapshot(self, potential, timescales):
        kinetic = float(0.5 * np.sum(self.mass[:, None] * self.vel**2))
        potential_energy = float(np.sum(potential))
        self.energy = kinetic + potential_energy
        self.energy_scale = kinetic - potential_energy
        self.fastest_timescale = float(np.min(timescales))
        if (not np.isfinite(self.energy) or not np.isfinite(self.pos).all()
                or not np.isfinite(self.vel).all() or not np.isfinite(self.acceleration).all()
                or np.isnan(self.fastest_timescale)):
            raise RuntimeError('Array CPU encountered coincident particles or a non-finite state; reduce dt or check initial conditions.')
        # These arrays are replaced, never mutated, by the next advance. Historical
        # acceleration and velocity samples therefore remain independent snapshots.
        self.host_pos = self.pos
        self.host_vel = self.vel
        self.host_acceleration = self.acceleration

    def propose_step(self, max_dt, eta, previous_proposal):
        proposal = min(max_dt, eta * self.fastest_timescale)
        if previous_proposal is not None:
            proposal = min(proposal, 1.25 * previous_proposal)
        return proposal

    def advance(self, h, method):
        self.pos, self.vel, self.acceleration, potential, timescales = self._advance(
            self.pos, self.vel, self.mass, self.acceleration, float(h), self.gravity, int(method), self.adaptive)
        self._snapshot(potential, timescales)
        return self.host_acceleration, self.energy

    def diagnostics(self):
        momentum = self.mass[:, None] * self.vel
        total_mass = self.mass.sum()
        return dict(energy=self.energy, momentum=momentum.sum(axis=0),
                    angular_momentum=np.cross(self.pos, momentum).sum(axis=0),
                    com_pos=(self.mass[:, None] * self.pos).sum(axis=0) / total_mass,
                    com_v=momentum.sum(axis=0) / total_mass)
