"""Step energy error band: a pure accept/retry decision module (standard library only).

The controller decides, for one trial step from a fixed accepted start, whether
to accept the trial or to retry from the *same* start with another width.  It
never touches physical state and writes no log; the caller saves and restores
the integrator state around every trial.

Quantities (absolute code energy units, float64 total K+U snapshots):

* ``delta_energy = E_trial - E_start``  (signed single-step change)
* ``drift        = E_trial - E_initial`` (signed offset from the run's initial energy)
* ``roundoff_screen = 8 eps max(scale_start, scale_trial)`` with scale = K+|U|.

Rules (pilot.py strict branch + pilot_guarded.py, Codex/2026-11-10-local-energy-band):

1. Non-finite candidate energy/scale: never accepted; shrink as a high-side
   reading (factor .5 when the drift limit is on, .2 when it is off).
2. Drift limit G>0 checked first: ``|drift| > G`` rejects even an in-band step.
   G == 0 skips every cumulative effect (only the signed drift is reported).
3. ``|delta| > U``: retry smaller.
4. ``|delta| < L``: retry larger from the same start, except (a) the final
   clipped step (``at_end``), which waives only L, and (b) the cumulative
   waiver: this start already had a drift rejection whose saved signed room
   is below L. The current candidate must still pass U/G, but may have zero
   or inward delta. This is the cumulative-priority exception in the prototype.
5. Otherwise accept; the caller keeps the accepted h as the next first trial.

Before any of these, a width with ``t + h == t`` in float64 fails with
``clock_resolution`` (no accept, no retry); proposals are checked the same way.

Width proposals.  ``h_new = h (target/|delta|)**(1/q)`` with
``target = sqrt(L U)`` and ``q = order + 1``, clipped to [.2, 2].  q is the
*predicted* local exponent of the leading single-step energy change from one
start; it is a proposal heuristic only.  It is not a rigorous error bound: the
leading term may cancel for a scalar like energy, and float64 readings near the
roundoff screen carry no slope information.  Hence every proposal is re-measured
and acceptance depends only on the measured change.  Readings at or below the
roundoff screen on the low side grow by a bounded factor 2.  The drift branch
aims at ``min(target, room/2)`` with factor clipped to [.2, .995].  Once a start
has both a too-small and a too-large width, proposals stay within the inner
10%..90% (logarithmic) part of that bracket, else fall back to its geometric
midpoint.

Arithmetic.  Under ordinary float64 conditions the operations and their order
are those of the prototypes, so the trial sequence is reproduced bit for bit.
Only when an intermediate would overflow, underflow to zero/subnormal, or make
no progress does the module switch to logarithmic forms or explicit guards.
"""
from dataclasses import dataclass
import math
import numbers
import sys

__all__ = ["BandDecision", "EnergyBandController"]

_EPS = sys.float_info.epsilon
_FMIN = sys.float_info.min          # smallest positive normal float
_FMAX = sys.float_info.max
_LOW_CLIP, _HIGH_CLIP = .2, 2.      # pilot.py ratio limits
_DRIFT_CAP = .995                   # pilot_guarded.py drift-branch upper clip
_LOW_MIN_GROW = 1.01                # pilot_guarded.py: low side must grow
_SCREEN_FACTOR = 8 * _EPS           # 2**-49, exact scaling
_LOG_FMAX = math.log(_FMAX)


@dataclass(frozen=True)
class BandDecision:
    """Immutable result of one assessed trial.

    action: "accept" | "retry" | "fail".
    reason: stable string, one of
        accept: "in_band", "final_low", "drift_priority_low"
        retry:  "nonfinite", "drift_limit", "above_upper", "below_lower"
        fail:   "trial_limit", "bracket_exhausted", "no_progress", "clock_resolution"
    next_h: next trial width (positive float) for retry, otherwise None.
    """
    action: str
    reason: str
    next_h: object
    delta_energy: float
    drift: float
    roundoff_screen: float
    attempt: int


def _real(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _normal(x):
    """True for a finite float of normal magnitude (no overflow/underflow)."""
    return _FMIN <= abs(x) <= _FMAX


def _sqrt_product(a, b):
    """sqrt(a*b) for positive finite a, b; prototype form when a*b is normal."""
    p = a * b
    if _normal(p):
        return math.sqrt(p)
    # Each root is normal-range or subnormal-safe; the product cannot overflow
    # (sqrt(max)**2 <= max) and stays >= min(a, b) up to rounding.
    return math.sqrt(a) * math.sqrt(b)


def _power_ratio(num, den, inv_q):
    """(num/den)**inv_q for positive finite num, den; safe log form at extremes."""
    r = num / den
    if _normal(r):
        return r ** inv_q
    x = (math.log(num) - math.log(den)) * inv_q
    # Bounded: exp would raise OverflowError (e.g. order=1, huge band, tiny
    # err) before the caller's clip; FMAX is clipped there like any large ratio.
    return _FMAX if x >= _LOG_FMAX else math.exp(x)


def _log_interp(low, high, frac):
    """low*(high/low)**frac (pilot form), log form if the ratio is not normal."""
    r = high / low
    if _normal(r):
        return low * r ** frac
    return math.exp(math.log(low) + frac * (math.log(high) - math.log(low)))


class EnergyBandController:
    """Same-start two-sided retry controller for the single-step energy change."""

    def __init__(self, lower=1e-13, upper=1e-11, drift_limit=0., order=45, max_trials=64):
        lower = _real("lower", lower)
        upper = _real("upper", upper)
        drift_limit = _real("drift_limit", drift_limit)
        if not 0. < lower < upper:
            raise ValueError(f"need 0 < lower < upper, got {lower!r}, {upper!r}")
        if drift_limit < 0.:
            raise ValueError(f"drift_limit must be >= 0, got {drift_limit!r}")
        self._lower = lower
        self._upper = upper
        self._drift = drift_limit + 0.          # normalise -0.0
        self._order = _positive_int("order", order)
        self._max_trials = _positive_int("max_trials", max_trials)
        self._q = self._order + 1
        self._inv_q = 1. / self._q
        # Clamp only matters when rounding at extreme magnitudes leaves [L, U].
        self._target = min(upper, max(lower, _sqrt_product(lower, upper)))
        self._t = self._e0 = self._e = self._scale = 0.
        self._low_h, self._high_h = 0., math.inf
        self._blocked_room = None
        self._attempt = 0
        self._started = False
        self._closed = True

    # read-only parameters ----------------------------------------------------
    lower = property(lambda self: self._lower)
    upper = property(lambda self: self._upper)
    drift_limit = property(lambda self: self._drift)
    order = property(lambda self: self._order)
    max_trials = property(lambda self: self._max_trials)
    q = property(lambda self: self._q, doc="Predicted local exponent order+1.")
    target = property(lambda self: self._target, doc="Geometric band midpoint sqrt(L U).")
    attempts = property(lambda self: self._attempt)
    bracket = property(lambda self: (self._low_h, self._high_h),
                       doc="(largest too-small h, smallest too-large h) at this start.")
    blocked_room = property(lambda self: self._blocked_room)

    def begin_step(self, t, energy_initial, energy_start, scale_start):
        """Start a new accepted node; clears bracket, blocked room and trial count."""
        t = _real("t", t)
        e0 = _real("energy_initial", energy_initial)
        e = _real("energy_start", energy_start)
        scale = _real("scale_start", scale_start)
        if scale < 0.:
            raise ValueError(f"scale_start must be >= 0, got {scale!r}")
        self._t, self._e0, self._e, self._scale = t, e0, e, scale
        self._low_h, self._high_h = 0., math.inf
        self._blocked_room = None
        self._attempt = 0
        self._started = True
        self._closed = False

    def assess(self, h, energy_trial, scale_trial, at_end=False):
        """Judge one trial of width h from the current start. Pure w.r.t. physics."""
        if not self._started or self._closed:
            raise RuntimeError("call begin_step before assess (a start ends at accept/fail)")
        if isinstance(h, bool) or not isinstance(h, numbers.Real):
            raise TypeError(f"h must be a real number, got {h!r}")
        h = float(h)
        if not (math.isfinite(h) and h > 0.):
            raise ValueError(f"h must be finite and positive, got {h!r}")
        self._attempt += 1
        attempt = self._attempt
        trial_time = self._t + h
        clock_ok = math.isfinite(trial_time) and trial_time > self._t
        lo, hi, guard = self._lower, self._upper, self._drift
        e, e0 = self._e, self._e0

        en = float(energy_trial)
        sn = float(scale_trial)
        finite = math.isfinite(en) and math.isfinite(sn)
        de = en - e
        drift = en - e0
        screen = _SCREEN_FACTOR * max(sn, self._scale) if finite else math.inf
        err = abs(de) if math.isfinite(de) else math.inf

        def done(action, reason, next_h=None):
            if action != "retry":
                self._closed = True
            return BandDecision(action, reason, next_h, de, drift, screen, attempt)

        if not clock_ok:
            # This h cannot advance the float64 clock: never accept or retry it.
            return done("fail", "clock_resolution")
        if not finite:
            # Cannot be judged; treat as a too-large width.  No direction info.
            mult = .5 if guard > 0. else _LOW_CLIP
            self._high_h = min(h, self._high_h)
            reason, side = "nonfinite", -1
        elif guard > 0. and abs(drift) > guard:
            # Signed one-sided room: larger when the change heads back to E0.
            direction = math.copysign(1., de) if de != 0. else 0.
            offset = e - e0
            room = max(0., guard - (direction * offset if direction else abs(offset)))
            self._blocked_room = room
            local_target = min(self._target, room * .5)
            if local_target == 0. or not math.isfinite(err) or err == 0.:
                mult = .5
            else:
                mult = min(_DRIFT_CAP, max(_LOW_CLIP, _power_ratio(local_target, err, self._inv_q)))
            self._high_h = min(h, self._high_h)
            reason, side = "drift_limit", -1
        elif err > hi:
            if math.isfinite(err):
                mult = min(_HIGH_CLIP, max(_LOW_CLIP, _power_ratio(self._target, err, self._inv_q)))
            else:
                mult = _LOW_CLIP
            self._high_h = min(h, self._high_h)
            reason, side = "above_upper", -1
        elif err < lo:
            if at_end:
                return done("accept", "final_low")
            desired = self._target
            if self._blocked_room is not None:
                if self._blocked_room < lo:
                    return done("accept", "drift_priority_low")
                desired = max(lo, min(self._target, .5 * self._blocked_room))
            if err <= screen:
                mult = _HIGH_CLIP   # unresolved reading: bounded growth only
            else:
                mult = min(_HIGH_CLIP, _power_ratio(desired, err, self._inv_q))
            if mult <= 1.:
                mult = _LOW_MIN_GROW
            self._low_h = max(self._low_h, h)
            reason, side = "below_lower", +1
        else:
            return done("accept", "in_band")

        next_h = min(h * mult, _FMAX)
        low_h, high_h = self._low_h, self._high_h
        if low_h > 0. and math.isfinite(high_h):
            lo_trial = _log_interp(low_h, high_h, .1)
            hi_trial = _log_interp(low_h, high_h, .9)
            if not lo_trial <= next_h <= hi_trial:
                next_h = _sqrt_product(low_h, high_h)
            if not low_h < next_h < high_h:
                return done("fail", "bracket_exhausted")
        else:
            # Degenerate rounding guards; inactive whenever the product moves h.
            if side < 0 and not next_h < h:
                next_h = h * _DRIFT_CAP
            if not 0. < next_h <= _FMAX or (next_h <= h if side > 0 else next_h >= h):
                return done("fail", "no_progress")
        next_time = self._t + next_h
        if not math.isfinite(next_time) or not next_time > self._t:
            return done("fail", "clock_resolution")
        if attempt >= self._max_trials:
            return done("fail", "trial_limit")
        return done("retry", reason, next_h)
