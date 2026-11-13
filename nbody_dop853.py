"""DOP853: SciPy native control or the same eighth-order tableau at fixed h.

SciPy supplies the published coefficients; no coefficients are copied or rounded.
Array CPU uses Numba stages/error control; dictionary/GPU retain SciPy stages.
Only accepted endpoints are committed to the application's physical state.
"""
import numpy as np


class DOP853Stepper:
    def __init__(self, stars, state, config, acceleration_fn, energy_fn, cancel_check):
        try:
            import scipy
            from scipy.integrate import DOP853
        except ImportError as exc:
            raise RuntimeError("DOP853 requires SciPy in the Python running this program. Install scipy, or select KDK/RK4.") from exc
        self.version = scipy.__version__
        self.stars, self.state = stars, state
        self.acceleration_fn, self.energy_fn = acceleration_fn, energy_fn
        self.cancel_check = cancel_check
        self.n = len(stars)
        self.nfev = 0
        # External controllers supply h themselves; never run the embedded controller too.
        self.adaptive = config["adaptive_enabled"] and config.get("adaptive_algorithm", "dop853") == "dop853"
        self.y = np.concatenate((np.array([s['pos'] for s in stars]).ravel(),
                                 np.array([s['v'] for s in stars]).ravel()))
        self.solver = None
        self.compiled = None
        self.accepted_error_norms = []
        if config['compute_backend'] in ('cpu_array', 'cpu_parallel'):
            from nbody_dop853_cpu import CompiledDOP853
            self.compiled = CompiledDOP853(self.y, state, dict(config, adaptive_enabled=self.adaptive), cancel_check)
            self.nfev = self.compiled.nfev
        elif self.adaptive:
            class CountedDOP853(DOP853):
                rejected_trials = 0
                last_error_norm = None

                def _estimate_error_norm(inner, K, h, scale):
                    # Instrument only; retain SciPy's estimator and controller unchanged.
                    error = super()._estimate_error_norm(K, h, scale)
                    inner.last_error_norm = float(error)
                    inner.rejected_trials += int(not error < 1)
                    return error

            self.solver = CountedDOP853(
                self.rhs, 0.0, self.y, config['total_time'],
                first_step=min(config['dt'], config['total_time']), max_step=config['dt'],
                rtol=config['dop853_rtol'], atol=config['dop853_atol'])
        else:
            self.A, self.B, self.C = DOP853.A, DOP853.B, DOP853.C
            self.K = np.empty((DOP853.n_stages, len(self.y)), dtype=float)
            self.f = self.rhs(0.0, self.y)

    def rhs(self, t, y):
        self.cancel_check()
        self.nfev += 1
        pos, vel = y.reshape(2, self.n, 3)
        state = self.state
        if state is None:
            temporary = [dict(mass=s['mass'], pos=p, v=v)
                         for s, p, v in zip(self.stars, pos, vel)]
            acceleration = self.acceleration_fn(temporary)
        elif hasattr(state, 'cp'):
            cp = state.cp
            acceleration = cp.asnumpy(state.fields(cp.asarray(pos), cp.asarray(vel), False)[0])
        else:
            acceleration = state.fields(pos, vel, state.mass, state.gravity, False)[0]
        return np.concatenate((vel.ravel(), acceleration.ravel()))

    def advance(self, time, h):
        self.cancel_check()
        if self.compiled is not None:
            new_time, h = self.compiled.advance(time, h)
            self.y = self.compiled.y
            self.nfev = self.compiled.nfev
            if self.adaptive:
                self.accepted_error_norms.append(self.compiled.last_error_norm)
        elif self.adaptive:
            message = self.solver.step()
            if self.solver.status == 'failed':
                raise RuntimeError(f"DOP853 failed at t={self.solver.t:.12g}: {message}")
            new_time = float(self.solver.t)
            h = float(self.solver.step_size)
            self.y = self.solver.y.copy()
            self.accepted_error_norms.append(self.solver.last_error_norm)
        else:
            # Pure explicit eighth-order RK step: no estimator, rejection, or h controller.
            self.K[0] = self.f
            for stage in range(1, len(self.B)):
                intermediate = self.y + h * np.dot(self.A[stage, :stage], self.K[:stage])
                self.K[stage] = self.rhs(time + self.C[stage] * h, intermediate)
            self.y = self.y + h * np.dot(self.B, self.K)
            self.f = self.rhs(time + h, self.y)
            new_time = time + h
        if not np.isfinite(self.y).all():
            raise RuntimeError('DOP853 produced non-finite values. Reduce dt or inspect close encounters.')
        pos, vel = self.y.reshape(2, self.n, 3)
        state = self.state
        if state is None:
            for star, p, v in zip(self.stars, pos, vel):
                star['pos'], star['v'] = p.copy(), v.copy()
            acceleration, energy = self.acceleration_fn(self.stars), self.energy_fn(self.stars)
        elif hasattr(state, 'cp'):
            state.pos, state.vel = state.cp.asarray(pos), state.cp.asarray(vel)
            state.refresh_snapshot()
            acceleration, energy = state.host_acceleration, state.energy
        else:
            state.pos, state.vel = pos.copy(), vel.copy()
            state.acceleration, potential, times = state.fields(state.pos, state.vel, state.mass, state.gravity, False)
            state._snapshot(potential, times)
            acceleration, energy = state.host_acceleration, state.energy
        if not np.isfinite(energy) or not np.isfinite(acceleration).all():
            raise RuntimeError('DOP853 encountered non-finite energy or acceleration.')
        return new_time, h, acceleration, energy

    @property
    def rejected_trials(self):
        if self.compiled is not None:
            return self.compiled.rejected_trials
        return self.solver.rejected_trials if self.adaptive else 0

    def checkpoint(self):
        """Save only physical integrator state; cost counters must include retries/probes."""
        if self.adaptive:
            raise RuntimeError('Native DOP853 manages its own rejected trials.')
        f = self.compiled.f if self.compiled is not None else self.f
        return self.y.copy(), f.copy()

    def restore(self, checkpoint):
        if self.adaptive:
            raise RuntimeError('Native DOP853 manages its own rejected trials.')
        y, f = checkpoint
        self.y = y.copy()
        if self.compiled is not None:
            self.compiled.y, self.compiled.f = self.y, f.copy()
        else:
            self.f = f.copy()

    @property
    def trial_evaluations(self):
        # One initial RHS, then twelve per DOP853 trial. Diagnostic forces are separate.
        return (self.nfev - 1) // 12

    @property
    def execution_description(self):
        stages = ('Numba float64 fastmath=False: CPU stages/force' if self.compiled is not None else
                  'SciPy coefficients/NumPy CPU stages; selected backend gravity')
        control = ('; native embedded error norm and step factor enabled' if self.adaptive else
                   '; embedded estimator/control disabled; caller supplies every trial h')
        return stages + control + '; Python cancellation/recording'
