"""Taylor integration of unsoftened Newtonian point-mass gravity.

Automatic power-series recurrence, NOT RK45. Position and velocity are each
truncated after h**p, p in {45,100,150,200}. The application adapter exposes
float64 snapshots for the float64, Python/mpmath and compiled MPFR cores.
No embedded error estimator: callers own step selection and rejection.
"""
import numpy as np

ORDER = 45
DECIMAL_PRECISION = 80
TAYLOR_ORDERS = (45, 100, 150, 200)


class TaylorStepRejected(RuntimeError):
    """Finite starting coefficients, but a trial state/diagnostic is nonfinite."""


class TaylorNBody:
    """CPU arbitrary-precision Taylor core; masses/positions may also be strings."""

    def __init__(self, masses, positions, velocities, gravity=1,
                 dps=DECIMAL_PRECISION, cancel_check=lambda: None, order=ORDER):
        if isinstance(order, (bool, np.bool_)) or order not in TAYLOR_ORDERS:
            raise ValueError('Taylor order must be 45, 100, 150 or 200.')
        self.order = int(order)
        try:
            import mpmath
        except ImportError as exc:
            raise RuntimeError("Taylor requires mpmath. Install requirements-taylor45.txt in this Python environment.") from exc
        self.ctx = mpmath.mp.clone()
        self.ctx.dps = dps
        self.version = mpmath.__version__
        self.mass = [self.ctx.mpf(m) for m in masses]
        self.pos = [[self.ctx.mpf(x) for x in row] for row in positions]
        self.vel = [[self.ctx.mpf(x) for x in row] for row in velocities]
        self.gravity = self.ctx.mpf(gravity)
        self.n = len(self.mass)
        if (self.n < 1 or len(self.pos) != self.n or len(self.vel) != self.n
                or any(len(row) != 3 for row in self.pos + self.vel)):
            raise ValueError(f"Taylor{self.order} requires N masses and N x 3 positions and velocities.")
        if (not self.ctx.isfinite(self.gravity) or self.gravity <= 0
                or any(not self.ctx.isfinite(m) or m <= 0 for m in self.mass)
                or any(not self.ctx.isfinite(x) for row in self.pos + self.vel for x in row)):
            raise ValueError(f"Taylor{self.order} needs finite states, positive masses and positive G.")
        self.cancel_check = cancel_check
        self.trials = 0
        self.coefficient_generations = 0
        self._coefficient_key = self._cached_coefficients = None
        self.time = self.ctx.zero
        self.initial_energy = self.energy()
        self._initial_energy_scale = 2 * self._kinetic_energy() - self.initial_energy
        self._accepted_duration = self._squared_error_integral = self._last_energy_offset = self.ctx.zero

    def checkpoint(self):
        # mpf objects are immutable. Counters measure real work and never rewind.
        return ([row[:] for row in self.pos], [row[:] for row in self.vel], self.time)

    def restore(self, checkpoint):
        pos, vel, time = checkpoint
        self.pos, self.vel = [row[:] for row in pos], [row[:] for row in vel]
        self.time = time

    def _kinetic_energy(self):
        return sum(self.mass[i] * sum(v*v for v in self.vel[i]) / 2 for i in range(self.n))

    def energy(self):
        c = self.ctx
        kinetic = self._kinetic_energy()
        potential = c.zero
        for i in range(self.n):
            for j in range(i + 1, self.n):
                q = sum((self.pos[j][d] - self.pos[i][d])**2 for d in range(3))
                if q == 0:
                    raise ValueError(f"Taylor{self.order}: coincident point masses; Newtonian gravity is singular.")
                potential -= self.gravity * self.mass[i] * self.mass[j] / c.sqrt(q)
        return kinetic + potential

    def record_accepted_energy(self, h):
        """Called once by the Lab after all trials/probes for this step finish."""
        h = self.ctx.mpf(h)
        previous = self._last_energy_offset
        current = self.energy() - self.initial_energy
        self._squared_error_integral += h * (previous*previous + previous*current + current*current) / 3
        self._accepted_duration += h
        self._last_energy_offset = current

    def coefficients(self):
        """r_k = r^(k)/k!. Extra r_(p+1) supplies the degree-p velocity term."""
        c, n = self.ctx, self.n
        r = [[[c.zero for _ in range(self.order + 2)] for _ in range(3)] for _ in range(n)]
        for i in range(n):
            for d in range(3):
                r[i][d][0], r[i][d][1] = self.pos[i][d], self.vel[i][d]
        # d = r_j-r_i, q=d.d, b=q^(-3/2); all are Taylor coefficient arrays.
        pairs = [(i, j, [[], [], []], [], []) for i in range(n) for j in range(i+1, n)]
        for k in range(self.order):
            self.cancel_check()
            acc = [[c.zero]*3 for _ in range(n)]
            for i, j, delta, q, b in pairs:
                self.cancel_check()
                for d in range(3):
                    delta[d].append(r[j][d][k] - r[i][d][k])
                q.append(sum(delta[d][l]*delta[d][k-l] for d in range(3) for l in range(k+1)))
                if k == 0:
                    if q[0] == 0:
                        raise ValueError(f"Taylor{self.order}: coincident point masses; reduce dt or inspect the initial state.")
                    b.append(1 / (q[0]*c.sqrt(q[0])))
                else:
                    # From q*b' = alpha*q'*b, alpha=-3/2. Integer arithmetic
                    # in the weights avoids rounded floating tableau constants.
                    b.append(-sum((i_term + 2*k)*q[i_term]*b[k-i_term]
                                  for i_term in range(1, k+1)) / (2*k*q[0]))
                for d in range(3):
                    force = self.gravity * sum(delta[d][l]*b[k-l] for l in range(k+1))
                    acc[i][d] += self.mass[j]*force
                    acc[j][d] -= self.mass[i]*force
            for i in range(n):
                for d in range(3):
                    r[i][d][k+2] = acc[i][d] / ((k+1)*(k+2))
        return r

    def advance(self, h):
        c = self.ctx
        h = c.mpf(h)
        if not c.isfinite(h) or h <= 0:
            raise ValueError(f"Taylor{self.order} needs a finite positive step.")
        self.cancel_check()
        # The coefficients depend on the expansion point, never on the proposed h.
        # Value keys also handle public state edits and restores from older checkpoints.
        key = (self.order, c.prec, self.gravity, tuple(self.mass),
               tuple(map(tuple, self.pos)), tuple(map(tuple, self.vel)))
        if key != self._coefficient_key:
            self._coefficient_key = self._cached_coefficients = None
            self._cached_coefficients = self.coefficients()
            self._coefficient_key = key
            self.coefficient_generations += 1
        r = self._cached_coefficients
        pos, vel = [[c.zero]*3 for _ in range(self.n)], [[c.zero]*3 for _ in range(self.n)]
        for i in range(self.n):
            for d in range(3):
                # Horner summation of two degree-p polynomials; r_(p+1) is NOT
                # added to the position polynomial (which would change the method).
                p, v = r[i][d][self.order], (self.order+1)*r[i][d][self.order+1]
                for k in range(self.order-1, -1, -1):
                    p = p*h + r[i][d][k]
                    v = v*h + (k+1)*r[i][d][k+1]
                pos[i][d], vel[i][d] = p, v
        if any(not c.isfinite(x) for row in pos+vel for x in row):
            raise RuntimeError(f"Taylor{self.order} produced non-finite states; reduce dt.")
        self.pos, self.vel = pos, vel
        self.time += h
        self.trials += 1

    def report(self):
        show = lambda x: self.ctx.nstr(x, self.ctx.dps)
        energy_metrics = {}
        if self._accepted_duration:
            rmse = self.ctx.sqrt(self._squared_error_integral / self._accepted_duration)
            energy_metrics['taylor_internal_energy_rmse'] = show(rmse)
            energy_metrics['taylor_internal_energy_nrmse'] = (
                show(rmse / self._initial_energy_scale) if self._initial_energy_scale else None)
        return dict(taylor_order=self.order, taylor_precision='mpmath', taylor_decimal_precision=self.ctx.dps,
                    taylor_mantissa_bits=self.ctx.prec, taylor_coefficient_generations=self.coefficient_generations,
                    taylor_mpmath_version=self.version, taylor_trial_evaluations=self.trials,
                    taylor_final_positions=[[show(x) for x in row] for row in self.pos],
                    taylor_final_velocities=[[show(x) for x in row] for row in self.vel],
                    taylor_internal_time=show(self.time),
                    taylor_initial_energy=show(self.initial_energy),
                    taylor_final_energy_difference=show(self.energy()-self.initial_energy),
                    taylor_precision_scope=f"{self.ctx.dps}-digit working precision for states and accepted-node energy RMS. Lab input, step controllers, time labels, plots and ordinary RMS use float64.",
                    taylor_execution=f"CPU serial mpmath; degree-{self.order} Taylor polynomials for position and velocity; no native step controller",
                    **energy_metrics)


class TaylorStepper:
    """Application interface, including rejected-step / pre-probe rollback."""

    def __init__(self, stars, gravity, acceleration_fn, energy_fn, cancel_check,
                 backend='cpu_mp', state=None, order=ORDER):
        if isinstance(order, (bool, np.bool_)) or order not in TAYLOR_ORDERS:
            raise ValueError('Taylor order must be 45, 100, 150 or 200.')
        self.order = int(order)
        self.stars = stars
        self.state = state
        self.acceleration_fn, self.energy_fn = acceleration_fn, energy_fn
        if backend == 'cpu_array':
            if self.order == 45:
                from nbody_taylor45_cpu import Float64TaylorNBody
                core_type = Float64TaylorNBody
            else:
                from nbody_taylor_cpu import Float64HighOrderTaylorNBody
                core_type = Float64HighOrderTaylorNBody
        elif backend in ('cpu', 'cpu_mp'):
            # cpu was the legacy high-precision-only backend identifier.
            core_type = TaylorNBody
        elif backend in ('cpu_mpfr', 'cpu_mpfr_113'):
            from nbody_taylor_mpfr import MPFRTaylorNBody
            core_type = MPFRTaylorNBody
        else:
            raise ValueError(f'Unsupported Taylor CPU backend: {backend}')
        kwargs = {} if backend == 'cpu_array' and self.order == 45 else {'order': self.order}
        if backend == 'cpu_mpfr_113':
            kwargs['precision_bits'] = 113
        self.core = core_type([s['mass'] for s in stars], [s['pos'] for s in stars],
                             [s['v'] for s in stars], gravity=gravity, cancel_check=cancel_check, **kwargs)

    def checkpoint(self):
        return self.core.checkpoint()

    def restore(self, checkpoint):
        self.core.restore(checkpoint)

    def advance(self, time, h):
        try:
            return self._advance(time, h)
        except (FloatingPointError, OverflowError) as exc:
            if self.order == 45:
                raise
            raise TaylorStepRejected(f'Taylor{self.order}: non-finite trial evaluation; reduce dt.') from exc

    def _advance(self, time, h):
        self.core.advance(h)
        with np.errstate(over='raise', invalid='raise'):
            pos, vel = np.array(self.core.pos, dtype=float), np.array(self.core.vel, dtype=float)
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            error = RuntimeError if self.order == 45 else TaylorStepRejected
            raise error(f"Taylor{self.order} state exceeds float64 display range; reduce dt.")
        if self.state is not None:
            state = self.state
            state.pos, state.vel = pos, vel
            state.acceleration, potential, timescales = state.fields(
                pos, vel, state.mass, state.gravity, state.adaptive)
            try:
                state._snapshot(potential, timescales)
            except RuntimeError as exc:
                if self.order == 45:
                    raise
                if (not np.isfinite(pos).all() or not np.isfinite(vel).all()
                        or not np.isfinite(state.acceleration).all() or not np.isfinite(potential).all()
                        or np.isnan(timescales).any() or not np.isfinite(state.energy)):
                    raise TaylorStepRejected(f'Taylor{self.order}: non-finite trial diagnostics; reduce dt.') from exc
                raise
            acceleration, energy = state.host_acceleration, state.energy
        else:
            for s, p, v in zip(self.stars, pos, vel):
                s['pos'], s['v'] = p, v
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                acceleration, energy = self.acceleration_fn(self.stars), self.energy_fn(self.stars)
        if not np.isfinite(acceleration).all() or not np.isfinite(energy):
            error = RuntimeError if self.order == 45 else TaylorStepRejected
            raise error(f"Taylor{self.order} encountered non-finite diagnostics; reduce dt or inspect close encounters.")
        return time+h, h, acceleration, energy


# Keep the old import for saved tools and callers, while new callers use a neutral name.
Taylor45Stepper = TaylorStepper
