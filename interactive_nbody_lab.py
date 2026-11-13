# Start CPU preparation before NumPy/Matplotlib imports or any window/archive setup.
# Imported/headless callers remain lazy; NBodyApp starts preparation when constructed.
from nbody_cpu_startup import CPUStartupPreparation

_CPU_STARTUP = CPUStartupPreparation() if __name__ == "__main__" else None

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection  # 2026-11-11: drift-limit bound lines (display only)
from matplotlib.ticker import (FixedLocator, FuncFormatter, NullLocator, LogLocator,
                               MaxNLocator, ScalarFormatter, LogFormatterSciNotation,
                               Locator, SymmetricalLogLocator)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import proj3d
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import sys
import copy
import json
import secrets
import csv
from io import BytesIO
from decimal import Decimal
from datetime import datetime
from dateutil.tz import gettz


import queue
import threading
from stable_orbits import STABLE_ORBITS as BASE_STABLE_ORBITS
from nbody_kepler import (KEPLER_BINARY_ORBIT, DEFAULT_ECCENTRICITY, binary_stars,
                          binary_position_error, validate_eccentricity)
from nbody_two_body import two_body_position_error, preset_position_error
from nbody_position_analysis import PositionAnalysisMixin, position_export_arrays

# Extend only the interactive catalog; the original program and testers retain the base catalog.
# Li & Liao (2019), Collisionless periodic orbits in the free-fall three-body problem.
# Retain the authors' coordinates without shifting the center of mass; the published initial
# values have about 10 decimal places, so exact periodic closure is not guaranteed.
# Preserve the historical free-fall entry at menu #16 when the base catalog grows.
STABLE_ORBITS = [*BASE_STABLE_ORBITS[:15], {
    "name": "Free-fall F1 (1, 0.8, 0.8) · Li-Liao",
    "N": 3,
    "seed": 16,  # Historical menu identifier, not a random initial-condition seed.
    "masses": [1.0, 0.8, 0.8],
    "positions": [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0],
                  [0.0009114239, 0.3019805958, 0.0]],
    "velocities": [[0.0, 0.0, 0.0] for _ in range(3)],
    "period": 1.8286248401,
    "dt": 1.8286248401 / 40000,
    "tot_time": 40000,
    "recommended_method": "2",
    "source": "https://numericaltank.sjtu.edu.cn/three-body/free-fall-3b/free-fall-3b-movies.htm",
    "note": "Masses: 1, 0.8, 0.8; initially at rest. RK4 selected; start with one period and adaptive stepping off.",
}, *[dict(orbit, seed=index + 17)
      for index, orbit in enumerate(BASE_STABLE_ORBITS[15:])]]

G = 1.0
TAYLOR_METHOD_ORDERS = {"4": 45, "5": 100, "6": 150, "7": 200}
METHOD_CHOICES = {"1": "Leapfrog", "2": "RK4", "3": "DOP853",
                  **{key: f"Taylor{order}" for key, order in TAYLOR_METHOD_ORDERS.items()}}
METHOD_NAMES = {"1": "Leapfrog (KDK)", "2": "Runge-Kutta 4th order", "3": "DOP853 (eighth order)",
                **{key: f"Taylor{order} ({order}th order)" for key, order in TAYLOR_METHOD_ORDERS.items()}}
STABLE_ORBITS.append(dict(KEPLER_BINARY_ORBIT, seed=len(STABLE_ORBITS)+1))


def is_kepler_config(config):
    index = config.get("orbit_index")
    return (config.get("initial_mode") == "orbit" and isinstance(index, int)
            and 0 <= index < len(STABLE_ORBITS) and STABLE_ORBITS[index].get("kepler_binary", False))

COMPUTE_BACKENDS = {
    "cpu": "CPU (dictionary / original)",
    "cpu_array": "CPU (array / single thread)",
    "cpu_parallel": "CPU (array / multi-thread)",
    "gpu": "GPU (NVIDIA / CuPy)",
    "cpu_mp": "CPU (mpmath / 80 digits)",
    "cpu_mpfr": "CPU (MPFR / 80 digits)",
    "cpu_mpfr_113": "CPU (MPFR / 34 digits)",
}
BACKEND_CODES = {label: code for code, label in COMPUTE_BACKENDS.items()}
BACKEND_CODES["CPU (80-digit / Taylor)"] = "cpu_mp"
BACKEND_CODES["CPU (80-digit / Taylor45)"] = "cpu_mp"  # Historical label.
TAYLOR_CPU_LABEL = COMPUTE_BACKENDS["cpu_mp"]
TAYLOR_FLOAT_LABEL = "CPU (float64 / single thread)"
BACKEND_CODES[TAYLOR_FLOAT_LABEL] = "cpu_array"
TAYLOR_MP_BACKENDS = ("cpu_mpfr_113", "cpu_mpfr", "cpu_mp")
STANDARD_BACKEND_LABELS = [label for code, label in COMPUTE_BACKENDS.items() if code not in TAYLOR_MP_BACKENDS]
TAYLOR_BACKEND_LABELS = [TAYLOR_FLOAT_LABEL, *(COMPUTE_BACKENDS[code] for code in TAYLOR_MP_BACKENDS)]


def sync_integrator_backend(owner, variable, widget, taylor45):
    """Switch precision explicitly; remember each integrator family's last device."""
    current = variable.get()
    if taylor45:
        if current not in TAYLOR_BACKEND_LABELS:
            owner._backend_before_taylor = current
            current = getattr(owner, "_taylor_backend_selection", TAYLOR_FLOAT_LABEL)
        variable.set(current)
        owner._taylor_backend_selection = current
        widget.configure(values=TAYLOR_BACKEND_LABELS, state="readonly")
    else:
        if current in TAYLOR_BACKEND_LABELS:
            current = getattr(owner, "_backend_before_taylor", COMPUTE_BACKENDS["cpu_parallel"])
        variable.set(current)
        widget.configure(values=STANDARD_BACKEND_LABELS, state="readonly")
# Retired identifiers are retained only to read historical experiment metadata.
WIDE_ALGORITHM = "work_skill_test_wide"
WIDE_DEFINITION = "endpoint_max_displacement_work_wide_v1"

ADAPTIVE_ALGORITHMS = {
    "timescale": "Time-scale method",
    "acceleration_ratio": "Acceleration ratio",
    "work_step": "work scale",
    "work_step_align": "work scale align",
    "work_step_average": "work scale average",
    "work_step_predicted": "work scale predicted",
}
ADAPTIVE_ALGORITHMS["dop853"] = "DOP853 native error control"
ADAPTIVE_ALGORITHMS["step_energy_band"] = "Step energy error band"
ENERGY_BAND_DEFAULTS = {"step_energy_lower": 1e-13, "step_energy_upper": 1e-11, "energy_drift_limit": 0.0}
INTEGRATOR_ORDERS = {"1": 2, "2": 4, "3": 8, **TAYLOR_METHOD_ORDERS}

def energy_band_summary(metadata):
    cfg = metadata.get("config", metadata)
    low = cfg.get("step_energy_lower", 1e-13)
    high = cfg.get("step_energy_upper", 1e-11)
    guard = cfg.get("energy_drift_limit", 0.)
    g = "Off" if guard == 0 else f"{guard:g}"
    return (f"Step energy error band [{low:g}, {high:g}]; drift={g} (accepted nodes); "
            f"trials={metadata.get('step_energy_trial_evaluations', 'NA')}; "
            f"rejected={metadata.get('step_energy_rejected_trials', 'NA')}")

ADAPTIVE_CODES = {label: code for code, label in ADAPTIVE_ALGORITHMS.items()}
WORK_ALGORITHM_CODES = ("work_step", "work_step_align", "work_step_average", "work_step_predicted")
WORK_ALIGNED_CODES = ("work_step_align",)
RETIRED_WORK_ALGORITHMS = {
    "work_skill_test": "Work scale test",
    "work_skill_test_align": "Work scale test align",
    WIDE_ALGORITHM: "Work scale test — wide protection",
}
WORK_HISTORY_NAMES = ({code: ADAPTIVE_ALGORITHMS[code].capitalize() for code in WORK_ALGORITHM_CODES}
                      | RETIRED_WORK_ALGORITHMS)
WORK_HISTORY_CODES = tuple(WORK_HISTORY_NAMES)
# Read old display names without changing algorithm IDs or immutable run records.
WORK_HISTORY_ALIASES = {
    "WorkStep": "work_step", "WorkSkillTest": "work_skill_test",
    **{name.replace("scale", "skill"): code for code, name in WORK_HISTORY_NAMES.items()},
}
WORK_DEFAULT_THRESHOLD = 0.01
WORK_DEFAULT_FACTOR = 1.1
WORK_DEFAULT_MAX_POWER = 200
DOP853_DEFAULT_RTOL = 3e-13
DOP853_DEFAULT_ATOL = 3e-16
WORK_PREPROBE_LABEL = "Encounter Pre-probe"
WORK_LEGACY_DISPLACEMENT_DEFINITION = "constant_acceleration_displacement_v1"
WORK_LEGACY_SYMMETRIC_DEFINITION = "endpoint_max_displacement_work_v1"
WORK_DISPLACEMENT_DEFINITION = "start_state_displacement_work_v2"
WORK_SYMMETRIC_DEFINITION = "same_time_endpoint_max_displacement_work_v2"
WORK_AVERAGE_DEFINITION = "endpoint_average_acceleration_displacement_work_v1"
WORK_PREDICTED_LEGACY_DEFINITION = "previous_state_extrapolated_acceleration_displacement_work_v1"
WORK_PREDICTED_DEFINITION = "step_ratio_extrapolated_acceleration_displacement_work_v2"


def work_algorithm_from_definition(definition, fallback=None):
    return {WORK_LEGACY_DISPLACEMENT_DEFINITION: "work_step",
            WORK_LEGACY_SYMMETRIC_DEFINITION: "work_skill_test",
            WIDE_DEFINITION: WIDE_ALGORITHM,
            WORK_AVERAGE_DEFINITION: "work_step_average",
            WORK_PREDICTED_LEGACY_DEFINITION: "work_step_predicted",
            WORK_PREDICTED_DEFINITION: "work_step_predicted",
            WORK_DISPLACEMENT_DEFINITION: "work_step_align",
            WORK_SYMMETRIC_DEFINITION: "work_skill_test_align"}.get(definition, fallback)


def velocity_acceleration_projections(accelerations, velocities):
    """Signed scalar components at the same recorded time; zero direction -> NaN.

    a_along_v = a dot v_hat (code length / time**2)
    v_along_a = v dot a_hat (code length / time)
    Leading dimensions are preserved, normally [time, star].
    """
    a = np.asarray(accelerations, dtype=float)
    v = np.asarray(velocities, dtype=float)
    if a.shape != v.shape or a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError("Projection inputs must have matching (..., 3) shapes.")
    a_norm = np.hypot.reduce(a, axis=-1)
    v_norm = np.hypot.reduce(v, axis=-1)
    # Normalize first: avoid under/overflow from squaring or multiplying tiny/large vectors.
    a_hat = np.full_like(a, np.nan)
    v_hat = np.full_like(v, np.nan)
    np.divide(a, a_norm[..., None], out=a_hat, where=a_norm[..., None] > 0)
    np.divide(v, v_norm[..., None], out=v_hat, where=v_norm[..., None] > 0)
    return np.sum(a * v_hat, axis=-1), np.sum(v * a_hat, axis=-1)


def acceleration_velocity_projection_product(accelerations, velocities):
    """|a| * |v along a| = |a dot v|, in code length^2/time^3.

    Use the dot product directly so a=0 gives zero even though its direction,
    and therefore the separate velocity projection, is undefined.
    """
    a = np.asarray(accelerations, dtype=float)
    v = np.asarray(velocities, dtype=float)
    if a.shape != v.shape or a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError("Product inputs must have matching (..., 3) shapes.")
    return np.abs(np.sum(a * v, axis=-1))


def work_step_indicator(accelerations, velocities, step_sizes):
    """Trial-end |a[n] dot v[n]| * incoming accepted h[n]; initial state undefined."""
    product = acceleration_velocity_projection_product(accelerations, velocities)
    h = np.asarray(step_sizes, dtype=float)
    if (product.ndim < 1 or h.ndim != 1 or len(h) == 0 or len(h) != product.shape[0]
            or not np.isfinite(h).all() or h[0] != 0 or np.any(h[1:] <= 0)):
        raise ValueError("Work indicator requires matching accepted step sizes, with initial zero and positive later steps.")
    values = product * h.reshape((len(h),) + (1,) * (product.ndim - 1))
    values[0] = np.nan
    return values


def normalized_work_power(accelerations, velocities, masses, energy_scale):
    """Per-star m |a dot v| / (K0 + |U0|), in inverse code time."""
    product = acceleration_velocity_projection_product(accelerations, velocities)
    mass = np.asarray(masses, dtype=float)
    scale = float(energy_scale)
    if mass.ndim > 1 or (mass.ndim == 1 and (product.ndim == 0 or len(mass) != product.shape[-1])):
        raise ValueError("Normalized work requires masses matching the star axis.")
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("Normalized work requires positive finite masses and initial energy scale.")
    return product * (mass / scale)


def normalized_work_step_indicator(accelerations, velocities, step_sizes, masses, energy_scale):
    """Dimensionless per-star work scale; h[n] is the accepted step ending at t[n]."""
    values = normalized_work_power(accelerations, velocities, masses, energy_scale)
    h = np.asarray(step_sizes, dtype=float)
    if (values.ndim < 1 or h.ndim != 1 or len(h) == 0 or len(h) != values.shape[0]
            or not np.isfinite(h).all() or h[0] != 0 or np.any(h[1:] <= 0)):
        raise ValueError("Work indicator requires matching accepted step sizes, with initial zero and positive later steps.")
    # Match the controller's arithmetic order at threshold boundaries.
    values *= h.reshape((len(h),) + (1,) * (values.ndim - 1))
    values[0] = np.nan
    return values


def normalized_displacement_work(accelerations, initial_velocities, step_size, masses, energy_scale):
    """m (|a_eff dot v_start| h + 0.5 |a_eff|² h²) / S_E.

    The caller selects start, end, endpoint-mean or accepted-history-predicted
    acceleration. This scale is not signed exact work or an integration error.
    """
    a = np.asarray(accelerations, dtype=float)
    power = normalized_work_power(a, initial_velocities, masses, energy_scale)
    h = np.asarray(step_size, dtype=float)
    if not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError("Displacement work requires finite nonnegative step sizes.")
    if h.ndim:
        if h.ndim != 1 or power.ndim < 1 or len(h) != power.shape[0]:
            raise ValueError("Displacement work step sizes must match the time axis.")
        h = h.reshape((len(h),) + (1,) * (power.ndim - 1))
    acceleration_squared = np.sum(a * a, axis=-1)
    return (power + .5 * acceleration_squared * (np.asarray(masses) / energy_scale) * h) * h


def select_work_acceleration(initial_acceleration, final_acceleration, *, aligned=False, average=False,
                             predicted=False, previous_acceleration=None, step_size=None,
                             previous_step_size=None, predicted_legacy=False):
    """Select a vector before projection/squaring; predicted ignores the trial end.

    Extrapolate the accepted-history slope to the candidate midpoint h/2.
    Rejected trials do not change that slope. Missing history uses current a.
    Historical v1 records explicitly retain their unscaled half difference.
    """
    if sum((bool(aligned), bool(average), bool(predicted))) > 1:
        raise ValueError("Choose only one acceleration-selection variant.")
    a0, a1 = np.asarray(initial_acceleration), np.asarray(final_acceleration)
    if a0.shape != a1.shape:
        raise ValueError("Start/end accelerations must have the same shape.")
    if predicted:
        prior = a0 if previous_acceleration is None else np.asarray(previous_acceleration)
        if prior.shape != a0.shape:
            raise ValueError("Previous accepted acceleration must match the current shape.")
        if predicted_legacy:
            return a0 + .5*(a0-prior)
        if previous_acceleration is None or previous_step_size is None:
            return a0.copy()
        if step_size is None:
            raise ValueError("Predicted acceleration requires the actual candidate step size.")
        h, prior_h = np.broadcast_arrays(np.asarray(step_size, dtype=float),
                                         np.asarray(previous_step_size, dtype=float))
        if (not np.isfinite(h).all() or not np.isfinite(prior_h).all()
                or np.any(h < 0) or np.any(prior_h < 0)):
            raise ValueError("Prediction intervals must be finite and nonnegative.")
        ratio = np.divide(h, prior_h, out=np.zeros_like(h), where=prior_h > 0)
        if ratio.ndim:
            if ratio.ndim != 1 or len(ratio) != a0.shape[0]:
                raise ValueError("Prediction intervals must match the acceleration time axis.")
            ratio = ratio.reshape((len(ratio),) + (1,)*(a0.ndim-1))
        return a0 + .5*ratio*(a0-prior)
    return .5*a0 + .5*a1 if average else a0 if aligned else a1


def displacement_work_step_indicator(accelerations, velocities, step_sizes, masses, energy_scale, *, legacy=False, average=False, predicted=False, predicted_legacy=False):
    """Accepted incoming h with start v and selected start/end/mean acceleration."""
    a, v, h = np.asarray(accelerations), np.asarray(velocities), np.asarray(step_sizes, dtype=float)
    if (a.shape != v.shape or a.ndim < 2 or a.shape[-1] != 3 or h.ndim != 1
            or len(h) != a.shape[0] or len(h) == 0 or not np.isfinite(h).all()
            or h[0] != 0 or np.any(h[1:] <= 0)):
        raise ValueError("Displacement work requires matching state records and accepted steps, with initial zero.")
    values = np.full(a.shape[:-1], np.nan, dtype=float)
    if len(h) > 1:
        prior = np.concatenate((a[:1], a[:-2]), axis=0) if predicted else None
        chosen = select_work_acceleration(a[:-1], a[1:], aligned=not legacy and not average and not predicted,
                                         average=average, predicted=predicted, previous_acceleration=prior,
                                         step_size=h[1:], previous_step_size=h[:-1], predicted_legacy=predicted_legacy)
        values[1:] = normalized_displacement_work(chosen, v[:-1], h[1:], masses, energy_scale)
    return values


def displacement_work_step_terms(accelerations, velocities, step_sizes, masses, energy_scale, *, legacy=False, average=False, predicted=False, predicted_legacy=False):
    """Decompose the actual selected monitor into its linear and quadratic h terms."""
    a, v, h = np.asarray(accelerations), np.asarray(velocities), np.asarray(step_sizes, dtype=float)
    if (a.shape != v.shape or a.ndim < 2 or a.shape[-1] != 3 or h.ndim != 1
            or len(h) != a.shape[0] or len(h) == 0 or not np.isfinite(h).all()
            or h[0] != 0 or np.any(h[1:] <= 0)):
        raise ValueError("Work terms require matching state records and accepted steps, with initial zero.")
    velocity_term = np.full(a.shape[:-1], np.nan, dtype=float)
    acceleration_term = np.full_like(velocity_term, np.nan)
    if len(h) > 1:
        prior = np.concatenate((a[:1], a[:-2]), axis=0) if predicted else None
        chosen = select_work_acceleration(a[:-1], a[1:], aligned=not legacy and not average and not predicted,
                                         average=average, predicted=predicted, previous_acceleration=prior,
                                         step_size=h[1:], previous_step_size=h[:-1], predicted_legacy=predicted_legacy)
        normalized_power = normalized_work_power(chosen, v[:-1], masses, energy_scale)
        incoming = h[1:].reshape((len(h)-1,) + (1,) * (normalized_power.ndim-1))
        velocity_term[1:] = normalized_power * incoming
        acceleration_term[1:] = .5 * np.sum(chosen*chosen, axis=-1) * (np.asarray(masses)/energy_scale) * incoming**2
    return velocity_term, acceleration_term


def symmetric_displacement_work(initial_accelerations, initial_velocities, final_accelerations,
                                final_velocities, step_size, masses, energy_scale, *, legacy=False):
    """max(Q_forward, Q_reverse) per particle; reverse velocity signs cancel under abs.

    Symmetric under swapping endpoints and negating both velocities. This does
    not by itself make the integrator or the adaptive controller time reversible.
    """
    a0, v0, a1, v1 = (np.asarray(value) for value in
                       (initial_accelerations, initial_velocities, final_accelerations, final_velocities))
    if not (a0.shape == v0.shape == a1.shape == v1.shape):
        raise ValueError("Symmetric work requires matching start/end acceleration and velocity shapes.")
    forward = normalized_displacement_work(a1 if legacy else a0, v0, step_size, masses, energy_scale)
    reverse = normalized_displacement_work(a0 if legacy else a1, v1, step_size, masses, energy_scale)
    return np.maximum(forward, reverse)


def bidirectional_work_step_indicators(accelerations, velocities, step_sizes, masses, energy_scale, *, legacy=False):
    """Forward, reverse, and selected max for every accepted incoming step."""
    a, v, h = np.asarray(accelerations), np.asarray(velocities), np.asarray(step_sizes, dtype=float)
    forward = displacement_work_step_indicator(a, v, h, masses, energy_scale, legacy=legacy)
    reverse = np.full_like(forward, np.nan)
    if len(h) > 1:
        reverse[1:] = normalized_displacement_work(a[:-1] if legacy else a[1:], v[1:], h[1:], masses, energy_scale)
    return forward, reverse, np.maximum(forward, reverse)


def symmetric_work_step_indicator(accelerations, velocities, step_sizes, masses, energy_scale, *, legacy=False):
    return bidirectional_work_step_indicators(accelerations, velocities, step_sizes, masses, energy_scale, legacy=legacy)[2]


def symmetric_work_step_terms(accelerations, velocities, step_sizes, masses, energy_scale, *, legacy=False):
    """Qv/Qa from the winning direction, not independent maxima of the two terms."""
    a, v, h = np.asarray(accelerations), np.asarray(velocities), np.asarray(step_sizes, dtype=float)
    fv, fa = displacement_work_step_terms(a, v, h, masses, energy_scale, legacy=legacy)
    forward, reverse, selected = bidirectional_work_step_indicators(a, v, h, masses, energy_scale, legacy=legacy)
    rv, ra = np.full_like(fv, np.nan), np.full_like(fa, np.nan)
    if len(h) > 1:
        incoming = h[1:].reshape((len(h)-1,) + (1,)*(fv.ndim-1))
        reverse_a = a[:-1] if legacy else a[1:]
        rv[1:] = normalized_work_power(reverse_a, v[1:], masses, energy_scale)*incoming
        ra[1:] = .5*np.sum(reverse_a**2, axis=-1)*(np.asarray(masses)/energy_scale)*incoming**2
    use_reverse = reverse > forward
    return dict(forward=forward, reverse=reverse, indicator=selected, reverse_selected=use_reverse,
                velocity_term=np.where(use_reverse, rv, fv), acceleration_term=np.where(use_reverse, ra, fa))


def displacement_work_convention(symmetric=False, legacy=False, average=False, predicted=False, predicted_legacy=False):
    """Exact record indexing; retired symmetric definitions remain readable."""
    if predicted:
        return (("a_pred=a_i[n-1]+0.5*(a_i[n-1]-a_i[max(n-2,0)]); " if predicted_legacy else
                 "a_pred=a_i[n-1]+0.5*(h[n]/h[n-1])*(a_i[n-1]-a_i[n-2]); ") +
                "m_i*(abs(a_pred dot v_i[n-1])*h[n]+0.5*|a_pred|^2*h[n]^2)/S_E; "
                + ("historical accepted-history half difference, no step-ratio rescaling; " if predicted_legacy else
                   "candidate step divided by previous accepted step; no extra average; ") +
                "first step uses initial a; ")
    if average:
        return ("a_bar=(a_i[n-1]+a_i[n])/2; "
                "m_i*(abs(a_bar dot v_i[n-1])*h[n]+0.5*|a_bar|^2*h[n]^2)/S_E; "
                "arithmetic vector average before projection and squaring; magnitudes added, no cancellation between terms; ")
    forward_a = "a_i[n]" if legacy else "a_i[n-1]"
    reverse_a = "a_i[n-1]" if legacy else "a_i[n]"
    forward = f"m_i*(abs({forward_a} dot v_i[n-1])*h[n]+0.5*|{forward_a}|^2*h[n]^2)/S_E"
    reverse = f"m_i*(abs({reverse_a} dot v_i[n])*h[n]+0.5*|{reverse_a}|^2*h[n]^2)/S_E"
    return ((f"max(Q_forward,Q_reverse); Q_forward={forward}; Q_reverse={reverse}; " if symmetric else forward + "; ")
            + ("mixed-time states; " if legacy else "same-time acceleration and velocity at each evaluated endpoint; ")
            + "magnitudes added, no cancellation; ")


def work_algorithm_note(algorithm):
    if algorithm in RETIRED_WORK_ALGORITHMS:
        return RETIRED_WORK_ALGORITHMS[algorithm] + ": retired; available only in historical records."
    average = algorithm == "work_step_average"
    predicted = algorithm == "work_step_predicted"
    acceleration = "a_pred" if predicted else "a_bar" if average else "a_start" if algorithm in WORK_ALIGNED_CODES else "a_end"
    return (ADAPTIVE_ALGORITHMS[algorithm] + ": "
            + ("a_pred=a_current+0.5*(h/h_previous)*(a_current-a_previous), accepted states only; first step uses a_current; h is the actual candidate interval; no second average; " if predicted else "")
            + ("a_bar=(a_start+a_end)/2, arithmetic vector average; " if average else "")
            + f"Q=max_i m_i*(|{acceleration} dot v_start|*h+0.5*|{acceleration}|^2*h^2)/S_E. "
            + "S_E=K(0)+|U(0)| stays fixed; work scale, not an energy-error estimate. "
            + "Above threshold: restore and retry h/N. Below threshold/N: restore and retry N*h, capped at initial dt. "
            + "Every enlarged trial is checked; failed larger bands block repeated growth at the same start. "
            + "At maximum exponent K, accept and mark over-limit steps. Defaults: threshold=0.01, N=10, K=3. "
            + ("Opposite endpoint accelerations can cancel in the vector average; no endpoint symmetry or reversibility is claimed." if average else ""))


def projection_export_arrays(accelerations, velocities, times=None, step_sizes=None, metadata=None):
    a_along_v, v_along_a = velocity_acceleration_projections(accelerations, velocities)
    arrays = dict(acceleration_along_velocity=a_along_v, velocity_along_acceleration=v_along_a,
                 speed=np.hypot.reduce(np.asarray(velocities, dtype=float), axis=-1),
                 speed_convention="Euclidean norm of each star's velocity; shape [time, star]; code length/time",
                 acceleration_velocity_product=acceleration_velocity_projection_product(accelerations, velocities),
                 acceleration_velocity_product_convention=
                 "|a| * |v along a| = |a dot v|; nonnegative; shape [time, star]; "
                 "code length^2/time^3; zero acceleration gives zero by dot-product continuation",
                projection_convention="signed scalar components; shape [time, star]; "
                "a dot v_hat and v dot a_hat; zero direction is NaN; "
                "code length/time^2 and code length/time, respectively")
    if times is not None:
        ratio, next_dt = relative_speed_change_ratio(accelerations, velocities, times)
        arrays.update(relative_speed_change_ratio=ratio, next_step_sizes=next_dt,
                      relative_speed_change_convention="abs(a_parallel[n]) * (t[n+1]-t[n]) / speed[n]; "
                      "dimensionless first-order estimate, not measured speed change or integration error; "
                      "zero speed and final record are NaN")
        projection_ratio, _ = acceleration_step_velocity_ratio(accelerations, velocities, times)
        arrays.update(acceleration_step_velocity_ratio=projection_ratio,
                      acceleration_step_velocity_ratio_convention=
                      "|a[n]| * (t[n+1]-t[n]) / signed(v[n] dot a_hat[n]); dimensionless; "
                      "zero acceleration, zero velocity projection and final record are NaN; "
                      "diagnostic ratio, not integration error")
        incoming_steps = np.r_[0., np.diff(times)] if step_sizes is None else step_sizes
        indicator = work_step_indicator(accelerations, velocities, incoming_steps)
        arrays.update(work_step_indicator=indicator,
                      work_step_indicator_max=np.max(indicator, axis=1) if indicator.ndim > 1 else indicator.copy(),
                      work_step_indicator_convention="abs(a[n] dot v[n]) * accepted h[n] ending at t[n]; "
                      "shape [time, star]; max is over all stars; initial state NaN; code length^2/time^2")
        if metadata and metadata.get("work_indicator_normalization") == "initial_energy_scale":
            masses = np.asarray([star["mass"] for star in metadata["initial_stars"]])
            definition = metadata.get("work_indicator_definition")
            legacy = definition in (WORK_LEGACY_DISPLACEMENT_DEFINITION, WORK_LEGACY_SYMMETRIC_DEFINITION, WIDE_DEFINITION)
            predicted = definition in (WORK_PREDICTED_DEFINITION, WORK_PREDICTED_LEGACY_DEFINITION)
            predicted_legacy = definition == WORK_PREDICTED_LEGACY_DEFINITION
            average = definition == WORK_AVERAGE_DEFINITION
            displacement = definition in (WORK_DISPLACEMENT_DEFINITION, WORK_LEGACY_DISPLACEMENT_DEFINITION, WORK_AVERAGE_DEFINITION, WORK_PREDICTED_DEFINITION, WORK_PREDICTED_LEGACY_DEFINITION)
            symmetric = definition in (WORK_SYMMETRIC_DEFINITION, WORK_LEGACY_SYMMETRIC_DEFINITION, WIDE_DEFINITION)
            compute = symmetric_work_step_indicator if symmetric else displacement_work_step_indicator if displacement else normalized_work_step_indicator
            normalized = compute(accelerations, velocities, incoming_steps, masses, metadata["energy_scale"],
                                 **({"legacy": legacy, **({"average": True} if average else {}), **({"predicted": True, "predicted_legacy": predicted_legacy} if predicted else {})} if symmetric or displacement else {}))
            arrays.update(specific_work_step_indicator=indicator,
                          specific_work_step_indicator_convention=arrays["work_step_indicator_convention"],
                          work_step_indicator=normalized,
                          work_step_indicator_max=np.max(normalized, axis=1),
                          work_step_indicator_convention=
                          (displacement_work_convention(symmetric, legacy, average, predicted, predicted_legacy) if symmetric or displacement else
                           "m_i * abs(a_i[n] dot v_i[n]) * accepted h[n] / S_E; ") +
                          "S_E=K(0)+abs(U(0)), fixed at initial state; dimensionless; initial state NaN")
            if symmetric:
                terms = symmetric_work_step_terms(accelerations, velocities, incoming_steps, masses, metadata["energy_scale"], legacy=legacy)
                arrays.update(work_step_forward_indicator=terms["forward"],
                              work_step_reverse_indicator=terms["reverse"],
                              work_step_reverse_selected=terms["reverse_selected"],
                              work_step_velocity_term=terms["velocity_term"],
                              work_step_acceleration_term=terms["acceleration_term"],
                              work_step_terms_convention="Qv and Qa come from the same winning direction; ties select forward; Qv+Qa=Q up to rounding; initial state NaN")
    return arrays


def acceleration_step_velocity_ratio(accelerations, velocities, times):
    """Signed |a[n]| * outgoing actual dt / velocity along acceleration.

    Keep the denominator's sign. Undefined directions, zero projections and
    the final record without a following interval are gaps, never infinities.
    """
    _, projection = velocity_acceleration_projections(accelerations, velocities)
    magnitude = np.hypot.reduce(np.asarray(accelerations, dtype=float), axis=-1)
    t = np.asarray(times, dtype=float)
    if (t.ndim != 1 or len(t) == 0 or projection.ndim < 1 or
            projection.shape[0] != len(t) or not np.isfinite(t).all() or
            np.any(np.diff(t) <= 0)):
        raise ValueError("Ratio requires matching, strictly increasing record times.")
    next_dt = np.r_[np.diff(t), np.nan]
    h = next_dt.reshape((len(t),) + (1,) * (projection.ndim - 1))
    ratio = np.full_like(projection, np.nan)
    np.divide(magnitude * h, projection, out=ratio,
              where=np.isfinite(projection) & (projection != 0))
    return ratio, next_dt


def relative_speed_change_ratio(accelerations, velocities, times):
    """First-order |a_parallel(t[n])| * (t[n+1]-t[n]) / |v(t[n])|.

    Use the outgoing recorded interval, not the interval ending at this state.
    Zero speed and the final state (no next interval) are undefined.
    """
    a_parallel, _ = velocity_acceleration_projections(accelerations, velocities)
    speed = np.hypot.reduce(np.asarray(velocities, dtype=float), axis=-1)
    t = np.asarray(times, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or a_parallel.shape[0] != len(t) or
            not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("Ratio requires matching, strictly increasing record times.")
    next_dt = np.r_[np.diff(t), np.nan]
    h = next_dt.reshape((len(t),) + (1,) * (a_parallel.ndim - 1))
    ratio = np.full_like(speed, np.nan)
    np.divide(np.abs(a_parallel) * h, speed, out=ratio, where=speed > 0)
    return ratio, next_dt


def acceleration_derivatives(accelerations, times):
    """Differentiate full saved states in physical time, never in step index.

    Return da/dt, |da/dt| and d|a|/dt with leading shape [time, ...].
    Nonuniform three-point differences are second order (one-sided at ends).
    Two records give a secant at both ends; one record has no derivative.
    These are postprocessing estimates, not an analytic gravitational jerk.
    """
    a = np.asarray(accelerations, dtype=float)
    t = np.asarray(times, dtype=float)
    if (a.ndim < 2 or a.shape[-1] != 3 or t.ndim != 1 or len(t) == 0
            or a.shape[0] != len(t) or not np.isfinite(a).all()
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("Derivatives require finite (..., 3) accelerations and matching increasing times.")
    magnitude = np.hypot.reduce(a, axis=-1)
    if len(t) == 1:
        derivative = np.full_like(a, np.nan)
        magnitude_rate = np.full_like(magnitude, np.nan)
    else:
        order = 2 if len(t) >= 3 else 1
        derivative = np.gradient(a, t, axis=0, edge_order=order)
        magnitude_rate = np.gradient(magnitude, t, axis=0, edge_order=order)
    return derivative, np.hypot.reduce(derivative, axis=-1), magnitude_rate


def acceleration_magnitude_derivative(accelerations, times):
    """Signed finite-difference d|a|/dt only; use actual nonuniform record times."""
    a = np.asarray(accelerations, dtype=float)
    t = np.asarray(times, dtype=float)
    if (a.ndim < 2 or a.shape[-1] != 3 or t.ndim != 1 or len(t) == 0
            or a.shape[0] != len(t) or not np.isfinite(a).all()
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("Derivatives require finite (..., 3) accelerations and matching increasing times.")
    magnitude = np.hypot.reduce(a, axis=-1)
    if len(t) == 1:
        return np.full_like(magnitude, np.nan)
    return np.gradient(magnitude, t, axis=0, edge_order=2 if len(t) >= 3 else 1)


def acceleration_derivative_export_arrays(accelerations, times):
    derivative, magnitude, magnitude_rate = acceleration_derivatives(accelerations, times)
    return dict(acceleration_derivative=derivative,
                acceleration_derivative_magnitude=magnitude,
                acceleration_magnitude_derivative=magnitude_rate,
                acceleration_derivative_convention=
                "da/dt [time, star, xyz]; |da/dt| and signed d|a|/dt [time, star]; "
                "code length/time^3; finite differences over all actual record times; "
                "nonuniform three-point centered interior and one-sided endpoints; "
                "two records use secants; one record is NaN; no smoothing; "
                "estimated from saved states, not analytic jerk or integration error")


class EnergyLogLocator(Locator):
    """Use the actual log transform for narrow ranges; decades for wide ranges."""
    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, low, high):
        low, high = sorted((float(low), float(high)))
        transform = self.axis.get_transform()
        core = transform.linthresh
        narrow = low*high > 0 and max(abs(low), abs(high))/min(abs(low), abs(high)) < 10
        if narrow or max(abs(low), abs(high)) <= core:
            bounds = transform.transform(np.array([low, high]))
            ticks = transform.inverted().transform(np.linspace(*bounds, 5))
        else:
            ticks = SymmetricalLogLocator(transform=transform, base=10).tick_values(low, high)
            ticks = ticks[(ticks >= low) & (ticks <= high)]
        # At machine-precision energy ranges, inverse-transforming adjacent
        # positions can round to the same physical floating-point value.
        return self.raise_if_exceeds(np.unique(ticks))


def energy_log_tick_label(value, position, axis):
    """Show physical energy, without an offset hiding the logarithmic labels."""
    low, high = axis.get_ylim()
    span, extent = abs(high-low), max(abs(low), abs(high))
    digits = min(17, max(4, int(np.ceil(np.log10(extent/span)))+3)) if span and extent else 4
    return f"{value:.{digits}g}"


def set_physical_log_scale(axis, values, positive_only=False):
    """Preserve signed physical values and exact zeros on logarithmic plots."""
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    if positive_only and finite.size and np.all(finite > 0):
        axis.set_yscale("log", base=10)
        scale = "log"
    else:
        extent = float(np.max(np.abs(finite))) if finite.size else 0.
        threshold = max(extent*1e-6, np.finfo(float).tiny) if extent else 1e-12
        axis.set_yscale("symlog", base=10, linthresh=threshold, linscale=1.)
        scale = "symlog"
        if extent:
            # Keep labels away from the small linear neighborhood of zero.
            powers = np.arange(int(np.ceil(np.log10(threshold)))+1,
                               int(np.floor(np.log10(extent)))+1)
            powers = powers[::max(1, int(np.ceil(len(powers)/3)))]
            ticks = np.power(10., powers)
            ticks = np.r_[-ticks[::-1], 0., ticks]
            lo, hi = float(np.min(finite)), float(np.max(finite))
            axis.yaxis.set_major_locator(FixedLocator(
                ticks[(ticks >= min(0., lo*1.05)) & (ticks <= max(0., hi*1.05))]))
            axis.yaxis.set_minor_locator(NullLocator())
    if finite.size and (np.all(finite > 0) or np.all(finite < 0)):
        smallest, largest = float(np.min(np.abs(finite))), float(np.max(np.abs(finite)))
        if largest/smallest < 10:
            # A nearly conserved negative energy may span no powers of ten.
            # The coordinate transform stays logarithmic; only tick placement
            # uses actual values so a narrow zoom still has readable labels.
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useOffset=True)
            formatter.set_powerlimits((-3, 4))
            axis.yaxis.set_major_formatter(formatter)
            axis.yaxis.set_minor_locator(NullLocator())
    return scale


def backend_summary(metadata):
    """Keep historical CPU/GPU labels; distinguish new CPU modes and actual threads."""
    code = metadata.get("compute_backend", "cpu")
    if metadata.get("method_id", metadata.get("method")) in TAYLOR_METHOD_ORDERS:
        return {"cpu_array": "CPU_ARRAY_SINGLE(float64)", "cpu_mpfr": "CPU_MPFR(269bit)",
                "cpu_mpfr_113": "CPU_MPFR(113bit)"}.get(code, "CPU_MPMATH(80dps)")
    if code == "gpu" and metadata.get("method_id", metadata.get("method")) == "3":
        return "GPU_GRAVITY + CPU_DOP853"
    if code == "cpu_array":
        return "CPU_ARRAY_SINGLE"
    if code == "cpu_parallel":
        count = metadata.get("cpu_threads")
        return f"CPU_ARRAY_MULTI({count}T)" if count is not None else "CPU_ARRAY_MULTI"
    return code.upper()


def dist(s1, s2):
    return np.linalg.norm(s2["pos"] - s1["pos"])


def dir_vector(s1, s2):
    return (s2["pos"] - s1["pos"]) / dist(s1, s2)


def acc(s1, s2):
    return G * s2["mass"] / dist(s1, s2)**2 * dir_vector(s1, s2)


def net_accelerations(s):
    result = np.zeros((len(s), 3), dtype=float)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            pair_acceleration = acc(s[i], s[j])
            result[i] += pair_acceleration
            result[j] -= pair_acceleration * s[i]["mass"] / s[j]["mass"]
    return result


def adaptive_step_size(s, max_dt, eta=0.03, previous_proposal=None):
    if not np.isfinite(max_dt) or max_dt <= 0 or not np.isfinite(eta) or eta <= 0:
        raise ValueError("The maximum time step and safety factor must be finite positive numbers.")
    proposed = float(max_dt)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            separation = np.linalg.norm(s[j]["pos"] - s[i]["pos"])
            speed = np.linalg.norm(s[j]["v"] - s[i]["v"])
            if not np.isfinite(separation) or separation <= 0 or not np.isfinite(speed):
                raise RuntimeError("Coincident particles or a non-finite state prevent adaptive time-step calculation.")
            mu = G * (s[i]["mass"] + s[j]["mass"])
            if mu > 0:
                gravity_time = separation / np.sqrt(mu / separation)
                proposed = min(proposed, eta * gravity_time)
            if speed > 0:
                flyby_time = separation / speed
                proposed = min(proposed, eta * flyby_time)
    if previous_proposal is not None:
        # Diagnostic output may clip a step; do not treat that as a physical
        # demand for small steps on the next iteration.
        proposed = min(proposed, 1.25 * previous_proposal)
    return proposed


def acceleration_power_band(numerator, denominator, trigger_ratio, max_power):
    """Require ratio > trigger_ratio for the first band.

    Higher exact powers enter their corresponding bands; for example, a ratio
    of 4 with trigger_ratio=2 belongs to band 2. Handle zeros first, then use
    binary search over integer powers to avoid logarithmic boundary errors.
    """
    if numerator == 0 or denominator == 0:
        return 0
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        ratio = float(np.float64(numerator) / np.float64(denominator))
    if ratio <= trigger_ratio:
        return 0
    low, high = 1, max_power
    while low < high:
        mid = (low + high + 1) // 2
        if ratio >= trigger_ratio ** mid:
            low = mid
        else:
            high = mid - 1
    return low


class AccelerationRatioController:
    """Track per-star reference levels and use the smallest proposed system step.

    Persistent references detect acceleration growth accumulated across steps.
    Each nonzero initial |a| sets that star's scale. Initially zero stars use
    the system's largest initial |a| as a control scale without changing their
    physical acceleration or using a tiny first nonzero value as a denominator.
    A reference becomes scale * trigger_ratio**level only when its local level
    changes; it is not reset at every step. For nonzero current acceleration,
    reference/current must exceed trigger_ratio before the level decreases.
    This hysteresis avoids repeated switching around a single threshold.
    """
    def __init__(self, initial_acceleration, max_dt, trigger_ratio=2.0, max_power=7):
        self.max_dt = float(max_dt)
        self.trigger_ratio = float(trigger_ratio)
        self.max_power = int(max_power)
        self.scales = np.linalg.norm(np.asarray(initial_acceleration, dtype=float), axis=1)
        if not np.isfinite(self.scales).all() or self.scales.size == 0:
            raise ValueError("Initial accelerations must form a non-empty finite vector.")
        positive = self.scales > 0
        if np.any(positive):
            self.scales[~positive] = np.max(self.scales[positive])
        self.levels = np.zeros(len(self.scales), dtype=int)
        self.references = self.scales.copy()
        self.global_level = 0
        self.last_event = None

    def choose(self, current_acceleration, current_time):
        current = np.linalg.norm(np.asarray(current_acceleration, dtype=float), axis=1)
        if current.shape != self.scales.shape or not np.isfinite(current).all():
            raise ValueError("The current accelerations have an incompatible size or contain non-finite values.")
        self.last_event = None
        # If all initial net accelerations are zero, the first nonzero state only initializes scales.
        uninitialized = self.scales == 0
        if np.any(uninitialized) and np.any(current > 0):
            self.scales[uninitialized] = np.max(current)
            self.references[uninitialized] = self.scales[uninitialized]
        valid = (current > 0) & (self.references > 0) & ~uninitialized
        old_levels = self.levels.copy()
        old_references = self.references.copy()
        for i in np.flatnonzero(valid):
            increase = acceleration_power_band(current[i], self.references[i],
                                               self.trigger_ratio, self.max_power)
            if increase:
                self.levels[i] = min(self.max_power, self.levels[i] + increase)
            else:
                decrease = acceleration_power_band(self.references[i], current[i],
                                                   self.trigger_ratio, self.max_power)
                self.levels[i] = max(0, self.levels[i] - decrease)
            # Integer levels set the reference; reaching the cap does not let spikes raise it further.
            if self.levels[i] != old_levels[i]:
                self.references[i] = self.scales[i] * self.trigger_ratio ** int(self.levels[i])
        next_level = int(np.max(self.levels))
        h = float(self.max_dt / self.trigger_ratio ** next_level)
        if next_level != self.global_level:
            drivers = np.flatnonzero(self.levels == next_level) + 1
            self.last_event = dict(time=float(current_time),
                                   previous_level=self.global_level, level=next_level,
                                   dt=h, controlling_stars=drivers.tolist(),
                                   star_levels=self.levels.tolist(),
                                   acceleration_magnitudes=current.tolist(),
                                   references_before=old_references.tolist(),
                                   references_after=self.references.tolist())
        self.global_level = next_level
        return h, next_level


class WorkStepController:
    """Discrete dt0/N**level bands for the normalized displacement work scale.

    Original uses end a/start v; align uses start a/start v; average uses the
    arithmetic endpoint-mean acceleration vector with start velocity.
    Predicted uses two accepted accelerations and start velocity, never trial-end a.
    The caller restores the same starting state before retrying either direction.
    """
    def __init__(self, max_dt, factor=WORK_DEFAULT_FACTOR, max_power=WORK_DEFAULT_MAX_POWER,
                 threshold=WORK_DEFAULT_THRESHOLD, *, masses, energy_scale, aligned=False, average=False, predicted=False):
        self.max_dt = float(max_dt)
        self.factor = float(factor)
        self.max_power = int(max_power)
        self.threshold = float(threshold)
        self.masses = np.asarray(masses, dtype=float)
        self.energy_scale = float(energy_scale)
        self.aligned = bool(aligned)
        self.average = bool(average)
        self.predicted = bool(predicted)
        if sum((self.aligned, self.average, self.predicted)) > 1:
            raise ValueError("Work controller must select exactly one optional variant.")
        normalized_work_power(np.zeros((len(self.masses), 3)), np.zeros((len(self.masses), 3)),
                              self.masses, self.energy_scale)
        self.level = 0

    def powers(self, acceleration, velocity):
        values = normalized_work_power(acceleration, velocity, self.masses, self.energy_scale)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise RuntimeError("Work-step control requires finite per-star acceleration and velocity.")
        return values

    def power(self, acceleration, velocity):
        return float(np.max(self.powers(acceleration, velocity)))

    def indicators(self, acceleration, initial_velocity, step_size, *, initial_acceleration, final_velocity,
                   previous_acceleration=None, previous_step_size=None):
        chosen = select_work_acceleration(initial_acceleration, acceleration, aligned=self.aligned,
                                          average=self.average, predicted=self.predicted, previous_acceleration=previous_acceleration,
                                          step_size=step_size, previous_step_size=previous_step_size)
        values = normalized_displacement_work(chosen, initial_velocity, step_size, self.masses, self.energy_scale)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise RuntimeError("Work-step control requires finite per-star displacement work indicators.")
        return values

    def step(self):
        return self.max_dt / self.factor ** self.level

    def grow(self):
        if self.level <= 0:
            return False
        self.level -= 1
        return True

    def shrink(self):
        if self.level >= self.max_power:
            return False
        self.level += 1
        return True







class WorkProbeRecoveryBudget:
    """Predicted-only protection ceiling, with accepted level-change accounting.

    A larger level is a smaller dt. During retries, can_grow may undo a
    provisional shrink, but cannot go above the entry dt. Only commit records
    net changes after acceptance; failed trials neither earn nor spend credits.
    """
    def __init__(self, entry_level):
        self.entry_level = int(entry_level)
        self.shrink_levels = 0
        self.recovery_levels = 0

    @property
    def credits(self):
        return self.shrink_levels - self.recovery_levels

    def can_grow(self, trial_level):
        return trial_level > self.entry_level

    def commit(self, previous_level, accepted_level):
        if (previous_level != self.entry_level + self.credits
                or accepted_level < self.entry_level):
            raise ValueError("Protected recovery must preserve its accepted entry-level budget.")
        change = accepted_level - previous_level
        self.shrink_levels += max(change, 0)
        self.recovery_levels += max(-change, 0)

    def record(self):
        return dict(recovery_policy="accepted_shrink_credit_v1", recovery_entry_level=self.entry_level,
                    accepted_shrink_levels=self.shrink_levels, accepted_recovery_levels=self.recovery_levels,
                    recovery_credit_remaining=self.credits)


def Ek(s1):
    return 0.5 * s1["mass"] * np.linalg.norm(s1["v"])**2


def Ep(s1, s2):
    return - G * s1["mass"] * s2["mass"] / dist(s1, s2)


def tot_E(s):
    sigma_Ep = 0
    sigma_Ek = 0
    for i in range(len(s)):
        sigma_Ek += Ek(s[i])
        for j in range(i+1, len(s)):
            sigma_Ep += Ep(s[i], s[j])
    return sigma_Ek + sigma_Ep


def Px(s):
    sigma_Px = 0
    for i in range(len(s)):
        sigma_Px += s[i]["mass"] * s[i]["v"][0]
    return sigma_Px


def Py(s):
    sigma_Py = 0
    for i in range(len(s)):
        sigma_Py += s[i]["mass"] * s[i]["v"][1]
    return sigma_Py


def Pz(s):
    sigma_Pz = 0
    for i in range(len(s)):
        sigma_Pz += s[i]["mass"] * s[i]["v"][2]
    return sigma_Pz


def tot_P(s):
    a = np.array([Px(s), Py(s), Pz(s)])
    return a


def tot_P_mag(s):
    return np.linalg.norm(tot_P(s))


def L(s):
    sigma_L = np.zeros(3)
    for i in range(len(s)):
        sigma_L += np.cross(s[i]["pos"], s[i]["mass"] * s[i]["v"])
    return sigma_L


def CoM_pos(s):
    tot_m = 0
    mr = np.zeros(3)
    for i in range(len(s)):
        tot_m += s[i]["mass"]
        mr += s[i]["mass"] * s[i]["pos"]
    return mr/tot_m


def CoM_v(s):
    tot_m = 0
    mv = np.zeros(3)
    for i in range(len(s)):
        tot_m += s[i]["mass"]
        mv += s[i]["mass"] * s[i]["v"]
    return mv/tot_m


def test(s):
    return {
        "energy": tot_E(s),
        "momentum": tot_P(s),
        "angular_momentum": L(s),
        "com_pos": CoM_pos(s),
        "com_v": CoM_v(s)
    }


def RMSE(data):
    a = 0
    for i in range(1, len(data)):
        a += (data[0] - data[i]) ** 2
    mse = a / (len(data) - 1)
    return np.sqrt(mse)


def time_weighted_energy_error(times, energies, energy_scale):
    """Integrate squared piecewise-linear E(t)-E(0) over all recorded intervals."""
    t, e = np.asarray(times, dtype=float), np.asarray(energies, dtype=float)
    if (t.ndim != 1 or e.shape != t.shape or len(t) < 2 or
            not np.isfinite(t).all() or not np.isfinite(e).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("Energy statistics require finite, strictly increasing full records.")
    offsets = e - e[0]
    amplitude = float(np.max(np.abs(offsets)))
    # Scale before squaring to protect very small errors from underflow.
    q = offsets / amplitude if amplitude else offsets
    integral = 0.
    duration = float(t[-1] - t[0])
    for start in range(0, len(t)-1, 100000):
        stop = min(start+100000, len(t)-1)
        a, b = q[start:stop], q[start+1:stop+1]
        integral += float(np.sum((np.diff(t[start:stop+1]) / duration) *
                                 (a*a + a*b + b*b) / 3.))
    rms = amplitude * np.sqrt(max(0., integral))
    scale = float(energy_scale)
    return dict(version=1, arithmetic="float64", method="full-record time-weighted RMS; exact squared piecewise-linear integral",
                rmse=float(rms), nrmse=float(rms/scale) if np.isfinite(scale) and scale > 0 else None,
                source_records=len(t), time_start=float(t[0]), time_end=float(t[-1]),
                initial_energy=float(e[0]), energy_scale=scale,
                rmse_units="code energy", nrmse_units="dimensionless")


def taylor_energy_summary(metadata):
    """Show full-run internal and float64 snapshot errors side by side when recorded."""
    if metadata.get("taylor_internal_energy_rmse") is None:
        return ""

    def number(value):
        if value is None:
            return "NA"
        decimal = Decimal(str(value))
        return "0" if decimal.is_zero() else format(decimal, ".5e")

    engine = "mpmath" if metadata.get("compute_backend", "cpu_mp") in ("cpu", "cpu_mp") else "MPFR"
    return (f"Internal {engine} energy RMSE = {number(metadata['taylor_internal_energy_rmse'])}    "
            f"NRMSE = {number(metadata.get('taylor_internal_energy_nrmse'))}    |    "
            f"float64 snapshot NRMSE = {number(metadata.get('energy_error', {}).get('nrmse'))}")


import hashlib
import html
from urllib.parse import quote
import os
import re
import tempfile
import threading
from functools import lru_cache


@lru_cache(maxsize=1)
def cpu_model_name():
    """Read the local CPU model once; never guess hardware for imported records."""
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                value = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        elif sys.platform.startswith("linux"):
            value = next(line.split(":", 1)[1].strip() for line in
                         Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
        elif sys.platform == "darwin":
            import subprocess
            value = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5).strip()
        else:
            value = "NA"
        return " ".join(str(value).split()) or "NA"
    except (OSError, ValueError, StopIteration, KeyError):
        return "NA"


RUN_HISTORY_DIRECTORY = Path(__file__).resolve().parent / "graph and data"
LEGACY_HISTORY_PATH = Path(__file__).resolve().parent / "运行记录" / "运行记录.csv"
RUN_HISTORY_COLUMNS = [
    "Date", "Time", "Method", "Initial_condition", "dt_initial",
    "Steps_set", "Steps_actual", "Adaptive_method", "Work_threshold", "Adaptive_param", "K", "E_RMSE", "E_nRMSE", "E_drift_k", "E_drift_k_norm", "Position_RMSE", "Position_NRMSE",
    "Px_RMSE", "Py_RMSE", "Pz_RMSE", "Lx_RMSE", "Ly_RMSE", "Lz_RMSE",
    "Vcmx_RMSE", "Vcmy_RMSE", "Vcmz_RMSE", "Device", "Device_model", "Compute_time_s", "Sim_T", "Trials", "Rejected", "Plot",
]
_HISTORY_FLOAT_COLUMNS = {4, 8, 9, *range(11, 26), 28, 29}
_HISTORY_INDEX_LOCK = threading.RLock()


def work_history_method(algorithm, preprobe=False):
    name = WORK_HISTORY_NAMES[algorithm]
    return name + (f" with {WORK_PREPROBE_LABEL}" if preprobe else "")


def _history_work_fields(row, report=None):
    """Backfill only recorded work settings; never assume a historical threshold."""
    row = list(row)
    method = str(row[7])
    name = method.split(" with ", 1)[0]
    names = WORK_HISTORY_ALIASES | {value: key for key, value in WORK_HISTORY_NAMES.items()}
    if name not in names:
        return row
    report = report if isinstance(report, dict) else {}
    fallback = names[name]
    algorithm = work_algorithm_from_definition(report.get("work_indicator_definition"), fallback)
    preprobe = report.get("work_preprobe_enabled", " with " in method)
    row[7] = work_history_method(algorithm, preprobe)
    if row[8] == "NA" and report.get("work_threshold") is not None:
        row[8] = report["work_threshold"]
    return row


def _history_drift_values(analysis):
    """Signed slope and energy-normalized slope; missing fits are not zero drift."""
    analysis = analysis if isinstance(analysis, dict) else {}
    values = []
    for key in ("slope", "normalized_slope"):
        value = analysis.get(key)
        values.append(float(value) if isinstance(value, (int, float, np.number))
                      and np.isfinite(value) else "NA")
    return values


def build_run_history_record(metadata, diagnostics):
    """New energy rows use all-step time-weighted statistics, never diagnostic subsampling."""
    started = datetime.fromisoformat(metadata["run_started_at"])
    error = metadata["energy_error"]["rmse"]
    scale = metadata["energy_scale"]
    adaptive_method = adaptive_param = adaptive_k = work_threshold = "NA"
    if metadata["adaptive_enabled"]:
        if metadata["adaptive_algorithm"] == "acceleration_ratio":
            adaptive_method, adaptive_param, adaptive_k = ("AccelerationRatio",
                metadata["acceleration_trigger_ratio"], metadata["acceleration_max_power"])
        elif metadata["adaptive_algorithm"] == "dop853":
            adaptive_method = (f"DOP853(rtol={metadata['dop853_rtol']:.16g},"
                               f"atol={metadata['dop853_atol']:.16g})")
        elif metadata["adaptive_algorithm"] == "step_energy_band":
            adaptive_method = "Step energy error band"
            work_threshold, adaptive_param, adaptive_k = (
                metadata["energy_drift_limit"], metadata["step_energy_lower"], metadata["step_energy_upper"])
        # Historical EnergyRatio records are retained for provenance only.
        elif metadata["adaptive_algorithm"] == "energy_ratio":
            adaptive_method, adaptive_param, adaptive_k = ("EnergyRatio", metadata["energy_relative_tolerance"],
                                                          metadata.get("energy_max_rejections", "NA"))
        elif metadata["adaptive_algorithm"] in WORK_ALGORITHM_CODES:
            adaptive_method = work_history_method(metadata["adaptive_algorithm"], metadata.get("work_preprobe_enabled", False))
            adaptive_param, adaptive_k = metadata["work_step_factor"], metadata["work_max_power"]
            work_threshold = metadata.get("work_threshold", "NA")
        else:
            adaptive_method, adaptive_param = "Timescale", metadata["adaptive_eta"]
    initial = (metadata["case_name"] if metadata["config"]["initial_mode"] == "orbit"
               else str(metadata["seed"]))
    row = [started.strftime("%Y-%m-%d"), started.strftime("%H:%M:%S"),
           _history_english(metadata["method_name"]), _history_english(initial), metadata["dt"],
           metadata["requested_steps"], metadata["steps"], adaptive_method, work_threshold, adaptive_param, adaptive_k,
           error, error / scale if scale else np.nan,
           *_history_drift_values(metadata.get("energy_drift")), "NA", "NA"]
    position_error = metadata.get("position_error")
    if position_error and position_error.get("reference_kind") == "analytic_kepler":
        row[15:17] = [position_error["rmse"], position_error["nrmse"]]
    for key in ("momentum", "angular_momentum", "com_v"):
        values = np.asarray([item[key] for item in diagnostics], dtype=float)
        row.extend(float(RMSE(values[:, axis])) for axis in range(3))
    row.extend([backend_summary(metadata), metadata.get("device_name", "NA"), metadata["elapsed_seconds"],
                metadata["simulation_end_time"]])
    if metadata["adaptive_enabled"] and metadata.get("adaptive_algorithm") == "step_energy_band":
        row.extend([metadata["step_energy_trial_evaluations"], metadata["step_energy_rejected_trials"]])
    else:
        row.extend(["NA"] * 2)
    row = ["NA" if isinstance(value, (float, np.floating)) and not np.isfinite(value)
           else value.item() if isinstance(value, np.generic) else value for value in row]
    return {"metadata": metadata, "row": row + ["NA"]}


def _history_english(value):
    text = str(value)
    for before, after in (
        ("加速度倍率法（累计参考）", "AccelerationRatio"),
        ("时间尺度法", "Timescale"), ("跳蛙法", "Leapfrog"),
        ("四阶龙格库塔", "RK4"), ("龙格-库塔", "Runge-Kutta"),
        ("自由落体", "Free-fall"), ("原算法", "original"),
        ("固定步长", "Fixed"), ("；", ";"), ("（", "("), ("）", ")"),
    ):
        text = text.replace(before, after)
    return " ".join(text.split())


def _archive_write(path, data, replace=True):
    """Publish a complete file; immutable record files never overwrite another run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


def _archive_record_path(directory, run_id):
    digest = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()
    return directory / "_records" / (digest + ".json")


def diagnostic_json_bytes(payload):
    """Strict JSON without discarding the meaning/location of undefined diagnostics.

    Non-finite numeric leaves become null, with their original NaN/Infinity kind
    and JSON-pointer location retained separately. Never mutate live results or
    replace undefined scientific measurements by zero.
    """
    invalid = []

    def clean(value, pointer=""):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, dict):
            return {key: clean(item, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item, pointer + "/" + str(i)) for i, item in enumerate(value)]
        if isinstance(value, float) and not np.isfinite(value):
            kind = "NaN" if np.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
            invalid.append(dict(pointer=pointer, value=kind))
            return None
        return value

    result = clean(payload)
    if invalid:
        if not isinstance(result, dict):
            raise TypeError("Scientific diagnostic JSON must have an object root.")
        result["serialization_nonfinite_values"] = result.get("serialization_nonfinite_values", []) + invalid
        result["serialization_nonfinite_policy"] = "Non-finite numeric fields are null; original kinds and JSON pointers retained. They are undefined/out-of-range diagnostics, not zero or valid finite measurements."
    return json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def _archive_publish_record(directory, entry):
    path = _archive_record_path(directory, entry["run_id"])
    _archive_write(path, diagnostic_json_bytes(entry), replace=False)
    return json.loads(path.read_text(encoding="utf-8"))


def _history_expand_row(row):
    """Read old schemas into shared slots; never invent unrecorded parameters."""
    row = list(row)
    if len(row) == 22:
        row.append("NA")
    if len(row) == 23:
        adaptive = _history_english(row[7])
        method = param = power = "NA"
        if "AccelerationRatio" in adaptive or "Timescale" in adaptive:
            method = "AccelerationRatio" if "AccelerationRatio" in adaptive else "Timescale"
            parameter = "N" if method == "AccelerationRatio" else "eta"
            match = re.search(r"\b" + parameter + r"\s*=\s*([^;\s]+)", adaptive)
            param = match.group(1) if match else "NA"
            match = re.search(r"\bK\s*=\s*(\d+)", adaptive)
            power = int(match.group(1)) if method == "AccelerationRatio" and match else "NA"
        elif adaptive != "NA":
            method = adaptive
        row = row[:7] + [method, param, power] + row[8:]
    if len(row) == 25:
        row = row[:12] + ["NA", "NA"] + row[12:]
    if len(row) == 27:
        row = row[:24] + ["NA"] + row[24:]
    if len(row) == 28:
        row = row[:12] + ["NA", "NA"] + row[12:]
    if len(row) == 30:
        row.insert(8, "NA")
    if len(row) == 31:
        row = row[:-1] + ["NA"] * 2 + row[-1:]
    elif len(row) == 36:
        if row[7] == "Step energy error band":
            row[8:11] = [row[32], row[30], row[31]]  # Recorded G/L/U, including G=0.
        row = row[:30] + row[33:]  # Keep Trials/Rejected/Plot; source records are immutable.
    if len(row) != 33:
        raise ValueError("Run summary must have 33 columns or a supported legacy schema.")
    return row


def _archive_display_row(row):
    row = _history_work_fields(_history_expand_row(row))
    display = []
    for i, value in enumerate(row):
        if i == 9 and row[7].split(" with ", 1)[0] in ("AccelerationRatio", "LocalAccelerationRatio", *WORK_HISTORY_NAMES.values()) and value != "NA":
            display.append(str(float(value)))
        elif i == 9 and row[7] == "Timescale" and value != "NA":
            from decimal import Decimal
            text = format(Decimal(str(value)), "f")
            display.append(text.rstrip("0").rstrip(".") if "." in text else text)
        elif ((i in _HISTORY_FLOAT_COLUMNS or i == 10 and row[7] == "Step energy error band") and value != "NA"
              and not (i in (15, 16) and str(value).startswith("reference_"))):
            display.append(f"{float(value):.6e}")
        else:
            display.append(" ".join(str(value).split()))
    return display


def rebuild_run_history(directory=None):
    """Serialize index generation while preserving independent record files."""
    with _HISTORY_INDEX_LOCK:
        return _rebuild_run_history(directory)


def _rebuild_run_history(directory=None):
    """Rebuild readable indexes from independent, full-precision run records."""
    directory = Path(directory) if directory is not None else RUN_HISTORY_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((directory / "_records").glob("*.json"))]
    entries.sort(key=lambda item: (item["row"][0], item["row"][1], item["run_id"]))
    rows = []
    for entry in entries:
        row = _history_work_fields(_history_expand_row(entry["row"]), entry.get("work_step_report"))
        # Backfill display columns from the fit saved with this run; keep records immutable.
        analysis = entry.get("energy_drift")
        if analysis is None and row[-1] != "NA":
            drift_path = directory / (Path(row[-1]).stem + "_energy_drift.json")
            if drift_path.is_file():
                saved_fit = json.loads(drift_path.read_text(encoding="utf-8"))
                if saved_fit.get("run_id") != entry["run_id"]:
                    raise ValueError("Energy drift analysis run ID mismatch.")
                analysis = saved_fit.get("analysis")
        if analysis is not None:
            row[13:15] = _history_drift_values(analysis)
        position_path = directory / "_position_comparisons" / _archive_record_path(directory, entry["run_id"]).name
        analytic_position = entry.get("position_error") or {}
        if analytic_position.get("reference_kind") == "analytic_kepler":
            row[15:17] = [analytic_position["rmse"], analytic_position["nrmse"]]
        elif position_path.exists():
            position = json.loads(position_path.read_text(encoding="utf-8"))
            if position["run_id"] != entry["run_id"]:
                raise ValueError("Position comparison run ID mismatch.")
            row[15:17] = [position["rmse"], position["nrmse"]]
        rows.append(_archive_display_row(row))
    # Keep stored filenames relative; build file URLs only for the current display location.
    display_rows = [row[:-1] + [((directory / row[-1]).resolve().as_uri()
                               if row[-1] != "NA" else "NA")] for row in rows]
    widths = [max(len(title), *(len(row[i]) for row in display_rows)) if display_rows else len(title)
              for i, title in enumerate(RUN_HISTORY_COLUMNS)]
    numeric = _HISTORY_FLOAT_COLUMNS | {5, 6, 10, 30, 31}
    def aligned(row):
        return "  ".join(value.rjust(widths[i]) if i in numeric else value.ljust(widths[i])
                         for i, value in enumerate(row)).rstrip()
    preamble = [
        "# N-body simulation run history; one completed run per row.",
        "# Code units unless specified. Compute_time_s: computation seconds; Sim_T: physical simulation time.",
        "# dt_initial is the initial trial dt for Step energy error band, otherwise initial dt / adaptive upper limit; Steps_set is the initial step count.",
        "# Conserved-quantity RMSE uses diagnostic samples after t=0, measured against the initial value.",
        "# S=initial kinetic energy+abs(initial potential energy); E_nRMSE=E_RMSE/S; E_drift_k=OLS slope of E(t)-E(0) [energy/time]; E_drift_k_norm=E_drift_k/S [1/time].",
        "# Two-body Position_RMSE / Position_NRMSE default to analytic Kepler reference. Metric v2 uses all actual records with time weighting; historical v1 used a uniform interpolation grid.",
        "# Optional numerical-reference batch comparisons use a uniform time grid including t=0; see batch_*_comparisons.dat.",
        "# reference_N marks the reference row; N counts completed other-test comparisons against it in the same batch.",
        "# _position_comparisons/*.json retain reference identity, interval, normalization and full-precision results.",
        "# Adaptive_param is eta for Timescale or N for either ratio method; K is used by both global and local ratio modes.",
        "# DOP853 native control records rtol/atol in Adaptive_method and JSON; Adaptive_param/K remain NA. Fixed DOP853 uses no error controller.",
        "# Step energy error band reuses Work_threshold/Adaptive_param/K for G/L/U, in absolute system energy units. G=0 is Off; positive G limits apply to accepted nodes only.",
        "# Trials includes accepted and rejected candidates for Step energy error band; Rejected counts rolled-back candidates. Missing legacy values remain NA.",
        "# Work_threshold is the recorded work-scale trigger threshold or band node drift limit G; other methods or unavailable historical values use NA.",
        "# Work methods append 'with Encounter Pre-probe' when enabled; 'align' identifies same-time formulas.",
        "# NA: unavailable/undefined. Device_model: recorded hardware model; legacy models use documented provenance.",
        "# Plot contains the image file URL for that row; NA means no saved figure is available.",
        "# URL clicking in this text file depends on the editor; index.html provides clickable Plot links.",
        "# Adaptive_param uses compact decimals (eta), decimal N, or scientific L. K remains an integer exponent except scientific U for band rows; JSON retains stored precision.",
        "", "",
        "# " + aligned(RUN_HISTORY_COLUMNS), "# " + "  ".join("-" * width for width in widths),
    ]
    _archive_write(directory / "runs.dat", ("\n".join(
        preamble + ["  " + aligned(row) for row in display_rows]) + "\n").encode("utf-8"))
    html_rows = []
    for row in rows:
        cells = [html.escape(value) for value in row]
        if row[-1] != "NA":
            image_href = quote(row[-1].replace("\\", "/"), safe="/")
            image_path = (directory / row[-1]).resolve()
            cells[-1] = (f'<a href="{html.escape(image_href, quote=True)}" '
                         f'target="_blank" rel="noopener" title="{html.escape(str(image_path), quote=True)}">'
                         f'{html.escape(image_path.as_uri())}</a>')
        html_rows.append("<tr>" + "".join("<td>" + value + "</td>" for value in cells) + "</tr>")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>N-body run history</title>"
            "<style>body{font:14px system-ui;margin:24px;color:#172437}table{border-collapse:collapse;"
            "font:13px monospace;white-space:nowrap}th,td{padding:9px 12px;border:1px solid #dce2eb}"
            "th{position:sticky;top:0;background:#e9eff8}tr:nth-child(even){background:#f6f8fb}"
            "th:nth-child(9),td:nth-child(9),th:nth-child(10),td:nth-child(10){text-align:right;font-variant-numeric:tabular-nums}"
            "a{color:#155db0}</style><h1>N-body run history</h1>"
            "<p><a href='runs.dat'>Aligned text table</a> &middot; "
            "<a href='README.txt'>Column definitions and precision</a></p>"
            "<p>Click the URL in a row's Plot column to open its conservation and error analysis figure in a new tab. "
            "NA means the original figure is unavailable.</p><table><thead><tr>"
            + "".join("<th>" + html.escape(title) + "</th>" for title in RUN_HISTORY_COLUMNS)
            + "</tr></thead><tbody>" + "".join(html_rows) + "</tbody></table></html>")
    _archive_write(directory / "index.html", page.encode("utf-8"))
    notes = ("N-body automatic run archive\n\n"
             "This graph and data folder is created beside the Python script, independent of the working directory.\n"
             "To share the archive, copy this entire folder. index.html uses relative image links.\n"
             "Displayed file URLs in the Plot column are refreshed when the program opens the archive.\n\n"
             "runs.dat: aligned text table, best viewed using a monospaced font. Long names are not cut.\n"
             "index.html: the same history with clickable links to the PNG figures.\n"
             "_records/: one immutable JSON summary per run; full numerical precision and run ID.\n"
             "YYYY-MM-DD_HH-MM-SS.png: conservation and error analysis figure for its recorded run.\n"
             "The filename uses the same date and time as its table row (colons become hyphens).\n"
             "If a second is already occupied, the real start-time microseconds appear in both\n"
             "the filename and Time column; existing figures are never overwritten.\n"
             "The Plot column in each DAT and HTML row displays that run's image file URL.\n"
             "Whether DAT URLs can be clicked depends on the text editor; index.html provides\n"
             "direct links in the Plot cells. HTML targets and stored filenames remain relative,\n"
             "so the links work when the entire archive folder is moved to another computer.\n\n"
             "The original adaptive field is split into method, parameter, and K; the final Plot column links the figure.\n"
             "Date and Time refer to the simulation start. started_at in JSON retains timezone\n"
             "information when available. Legacy CSV timestamps have unknown timezone.\n"
             "dt_initial is the initial trial for Step energy error band (growth allowed), otherwise the chosen initial dt / adaptive maximum. Steps_set is ceil(Sim_T/dt_initial).\n"
             "Steps_actual counts completed integration steps, including a shortened final step.\n"
             "Adaptive_method is NA, Timescale, AccelerationRatio, a Work scale method, DOP853, or Step energy error band. Work methods append 'with Encounter Pre-probe' when enabled. Historical work names are resolved from each run's saved formula definition; the original record is not rewritten. EnergyRatio rows are retained. Work_threshold precedes Adaptive_param and contains the work trigger threshold, band node drift limit G, or NA when unused/unrecorded. Adaptive_param is eta for Timescale\n"
             "or the acceleration trigger N for AccelerationRatio; K is its maximum cumulative exponent.\n"
             "Step energy error band reuses Work_threshold/Adaptive_param/K for G/L/U in absolute system energy units. G=0 is Off; positive G applies to accepted nodes only. L/U keep scientific notation. Trials counts all accepted and rejected candidates; Rejected counts rolled-back candidates.\n"
             "Old 36-column band rows map their recorded L/U/G into these shared slots on display; old 31-column rows gain only unavailable Trials/Rejected values. Original immutable JSON records are not rewritten and missing parameters remain NA.\n"
             "For all work methods, Adaptive_param is N and K is the maximum cumulative exponent; defaults: dimensionless threshold 0.01, N=10, K=3. S_E=K(0)+abs(U(0)) stays fixed. Work scale uses Q=max_i m_i*(|a_end dot v_start|*h+0.5*|a_end|^2*h^2)/S_E (constant_acceleration_displacement_v1). Work scale align uses start acceleration in both terms (start_state_displacement_work_v2). Legacy records without a formula definition used |a dot v| h, normalized only when their normalization marker is present. Recovery policy trial_end_rollback_retry_v1: Q<threshold/N rejects the trial, restores its start and retries with N times dt; enlarged trials are rechecked. A failed larger band at the same start blocks repeated growth retries. Older records may use accepted-state recovery for the next step.\n"
             "Work scale average uses a_bar=(a_start+a_end)/2 as a vector in both terms with v_start; definition endpoint_average_acceleration_displacement_work_v1. Retired historical Work scale test takes max_i max(Q_forward_i,Q_reverse_i), with end a/start v forward and start a/end v reverse (endpoint_max_displacement_work_v1). Work scale test align uses start a/start v forward and end a/end v reverse (same_time_endpoint_max_displacement_work_v2). Both retain the same N/K retry and optional pre-probe policy. Endpoint symmetry of the monitor does not guarantee reversibility of the full adaptive algorithm. Exported forward/reverse indicators and winning-direction terms identify the actual control quantity.\n"
             "Work scale predicted uses a_pred=a_current+0.5*(h/h_previous)*(a_current-a_previous) from accepted states; h is the candidate interval and h_previous the preceding accepted interval; first step uses initial a. Both terms use that vector, with v_current. Definition step_ratio_extrapolated_acceleration_displacement_work_v2. Historical previous_state_extrapolated_acceleration_displacement_work_v1 retains its unscaled half difference. During predicted pre-probe replay, accepted shrinking earns recovery credits; accepted recovery spends them and cannot exceed the entry dt.\n"
             "Historical EnergyRatio records describe a removed experimental method; it is not available for new runs.\n"
             "All four adaptive fields are NA for fixed dt. Work_threshold is NA when a historical threshold was not recorded; it is never replaced with today's default. Historical LocalAccelerationRatio rows describe a retired method.\n"
             "New energy RMSE integrates squared E(t)-E(0) over all saved step intervals, using linear interpolation in time.\n"
             "The energy_error JSON field and the E_RMSE/E_nRMSE columns use float64 snapshots. High-precision Taylor runs with internal metrics also store full-run taylor_internal_energy_rmse and taylor_internal_energy_nrmse as decimal strings in taylor_report. Historical rows retain their original definitions.\n"
             "Other conserved-quantity RMSE excludes the initial diagnostic sample and uses diagnostic records.\n"
             "Two-body Position_RMSE and Position_NRMSE are automatic using an independent analytic Kepler reference. New metric_version=2 evaluates error vectors at every actual recorded time, then integrates squared piecewise-linear error vectors over each interval and averages over both stars. NRMSE divides by the initial unweighted RMS radius about the center of mass, (1+e)/2 for the adjustable equal-mass binary. Historical records without metric_version=2 used 20001 uniform times and cubic Hermite interpolation of numerical positions; they are not rewritten. The position_error metadata retains constants and definitions. Analytic metrics take priority over optional batch-reference overlays. General non-radial two-body initial states use universal-variable Kepler propagation; singular radial states are marked unavailable. Other cases use the selected batch reference and a common uniform time grid.\n"
             "DOP853 native control stores rtol/atol in Adaptive_method and JSON. Its fixed mode uses eighth-order stages without an error controller. Stage combinations/control run on CPU; GPU selection uses GPU gravity with stage transfers.\n"
             "Disabled, pending, or unavailable position metrics are NA; reference_N marks a reference and its completed comparison count.\n"
             "_position_comparisons/ stores comparison overlays without altering immutable original run records.\n"
             "Comparison definitions and previous-test results are in the companion batch JSON and comparison DAT.\n"
             "E_nRMSE divides energy RMSE by initial kinetic energy + abs(initial potential energy).\n"
             "E_drift_k is the signed fitted slope of E(t)-E(0), in code energy/code time; a free intercept is fitted.\n"
             "E_drift_k_norm divides that slope by initial kinetic energy + abs(initial potential energy), in 1/code time.\n"
             "These are drift rates, not RMSE values. Positive/negative slopes describe upward/downward fitted trends.\n"
             "A small slope alone does not prove conservation; compare fit quality and time resolution.\n"
             "New fits (version 2) use every saved record and trapezoidal physical-time node weights (half each adjacent interval; one side at endpoints), normalized to sum to 1.\n"
             "Fit record count, weighting convention, interval and weighted residual statistics are retained in the run's energy_drift JSON. Version 1 fits used random physical-time samples.\n"
             "Old records are displayed with slopes from their archived fit, or NA if no fit was saved.\n"
             "The archived fit is retained when an interactive refit is saved separately; indexes do not silently replace it.\n"
             "P, L, and Vcm mean total momentum, total angular momentum, and center-of-mass velocity.\n"
             "Device distinguishes CPU (dictionary), CPU_ARRAY_SINGLE, CPU_ARRAY_MULTI(nT), GPU, CPU_MPMATH(80dps), and CPU_MPFR(113bit/269bit). The thread count n and MPFR mantissa length identify the execution mode.\n"
             "Device_model identifies the processor used for computation, not display rendering.\n"
             "Array CPU initialization/Numba compilation occurs before Compute_time_s; initialization_seconds is retained in NPZ metadata.\n"
             "Historical hardware provenance is retained in each migrated JSON summary; unknown models remain NA.\n"
             "Compute_time_s is elapsed real time for the simulation loop and part of result preparation, in seconds.\n"
             "It excludes subsequent plotting and file saving; Sim_T is physical simulation time in code units.\n"
             "DAT and HTML display scientific notation rounded to six decimal places; JSON preserves\n"
             "all stored digits. NA marks an unused setting or unavailable / undefined value.\n"
             "Old CSV rows are imported read-only, including repeated runs. Their missing figures\n"
             "are marked NA; the original CSV is not modified. The table and HTML can be rebuilt\n"
             "from _records. Independent records prevent concurrent runs from overwriting one another.\n")
    _archive_write(directory / "README.txt", notes.encode("utf-8"))
    return directory / "runs.dat"


def migrate_legacy_history(directory=None, legacy_path=None):
    """Read the old 22-column CSV without editing it; repeated imports are harmless."""
    directory = Path(directory) if directory is not None else RUN_HISTORY_DIRECTORY
    legacy_path = Path(legacy_path) if legacy_path is not None else LEGACY_HISTORY_PATH
    if not legacy_path.exists():
        return 0
    with legacy_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if len(header) != 22:
            raise ValueError("Legacy history must contain exactly 22 columns; source was not modified.")
        imported = 0
        for index, original in enumerate(reader):
            if not original:
                continue
            if len(original) != 22:
                raise ValueError(f"Legacy row {index + 2} does not have 22 columns.")
            identity = json.dumps([index, original], ensure_ascii=False)
            run_id = "legacy_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
            row = _history_expand_row([_history_english(value) for value in original])
            # Strings retain every original digit; the readable table formats only a display copy.
            for column in _HISTORY_FLOAT_COLUMNS | {5, 6, 10, 30, 31}:
                if str(row[column]).lower() in {"nan", "inf", "-inf", ""}:
                    row[column] = "NA"
            path = _archive_record_path(directory, run_id)
            existed = path.exists()
            _archive_publish_record(directory, {"run_id": run_id, "row": row,
                                    "started_at": "NA", "source": "legacy_csv",
                                    "legacy_source": str(legacy_path.resolve())})
            imported += not existed
    rebuild_run_history(directory)
    return imported


def _archive_save_png(directory, record, png):
    """Reserve a date/time filename without reusing another run's image."""
    pending = record.get("pending_plot_filename")
    if pending is not None:
        pending_path = directory / pending
        if pending_path.name != pending:
            raise ValueError("An image filename must not contain a directory.")
        if pending_path.is_file() and pending_path.read_bytes() == png:
            return pending
    row = _history_expand_row(record["row"])
    started = datetime.fromisoformat(record["metadata"]["run_started_at"])
    candidates = [(row[1], f"{row[0]}_{str(row[1]).replace(':', '-')}.png")]
    fractional_time = started.strftime("%H:%M:%S.%f")
    if row[1] != fractional_time:
        candidates.append((fractional_time,
                           f"{row[0]}_{fractional_time.replace(':', '-')}.png"))
    for time_label, filename in candidates:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d{1,6})?\.png", filename):
            raise ValueError("The image filename requires a valid date and time from its data row.")
        if _archive_write(directory / filename, png, replace=False):
            record["pending_plot_filename"] = filename
            record["row"][1] = time_label
            return filename
    raise FileExistsError(
        "Both the second and exact microsecond image names are occupied. "
        "Existing images were preserved; move the conflicting file before retrying.")


def save_run_history(record, directory=None):
    """Save PNG first, then its immutable summary, and rebuild both readable indexes."""
    directory = Path(directory) if directory is not None else RUN_HISTORY_DIRECTORY
    metadata = record["metadata"]
    run_id = str(metadata["run_id"])
    record_path = _archive_record_path(directory, run_id)
    with _HISTORY_INDEX_LOCK:
        if record_path.exists():
            entry = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            png = record.get("png_bytes")
            if not isinstance(png, bytes) or not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("The conservation figure must be rendered before saving this run.")
            filename = _archive_save_png(directory, record, png)
            row = _history_expand_row(record["row"])
            row[-1] = filename
            entry = _archive_publish_record(directory, {"run_id": run_id, "row": row,
                                            "started_at": metadata["run_started_at"], "source": "simulation",
                                            "device_model_source": "runtime hardware query",
                                            "schema_version": 3,
                                            "step_energy_report": metadata.get("step_energy_report"),
                                            "energy_error": metadata.get("energy_error"),
                                            "position_error": metadata.get("position_error"),
                                            "dop853_report": {key: value for key, value in metadata.items()
                                                if key.startswith("dop853_") and key != "dop853_error_norms"}
                                                if metadata.get("method_id") == "3" else None,
                                            "taylor45_report": {key: value for key, value in metadata.items()
                                                if key.startswith("taylor_")} if metadata.get("method_id") == "4" else None,
                                            "taylor_report": {key: value for key, value in metadata.items()
                                                if key.startswith("taylor_")} if metadata.get("method_id") in TAYLOR_METHOD_ORDERS else None,
                                            "energy_drift": energy_drift_summary(metadata["energy_drift"])
                                                if "energy_drift" in metadata else None,
                                            "work_step_report": {key: value for key, value in metadata.items()
                                                if key.startswith("work_")} if metadata.get("adaptive_algorithm") in WORK_ALGORITHM_CODES else None})
        if "energy_drift" in metadata:
            drift_path = directory / (Path(entry["row"][-1]).stem + "_energy_drift.json")
            _archive_write(drift_path, diagnostic_json_bytes(
                dict(run_id=run_id, analysis=metadata["energy_drift"])), replace=False)
            metadata["energy_drift_path"] = str(drift_path.resolve())
        path = rebuild_run_history(directory)
    record["row"][1] = entry["row"][1]
    record["row"][-1] = entry["row"][-1]
    metadata.update(history_saved_run_id=run_id, history_dat_path=str(path.resolve()),
                    history_csv_path=str(path.resolve()),
                    history_png_path=str((directory / entry["row"][-1]).resolve()))
    return path

# leapfrog
def leapfrog(stars, step_dt, initial_acceleration=None):
    N = len(stars)
    h = step_dt
    if initial_acceleration is None:
        accs = [np.zeros(3) for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                a = acc(stars[i], stars[j])
                accs[i] += a
                accs[j] -= a * stars[i]["mass"] / stars[j]["mass"]
    else:
        accs = np.asarray(initial_acceleration)


    # half step velocity
    for i in range(N):
        stars[i]["v"] += 0.5 * h * accs[i]

    # full step position
    for i in range(N):
        stars[i]["pos"] += stars[i]["v"] * h

    # new acceleartion
    accs = [np.zeros(3) for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            a = acc(stars[i], stars[j])
            accs[i] += a
            accs[j] -= a * stars[i]["mass"] / stars[j]["mass"]

    # another half step velocity
    for i in range(N):
        stars[i]["v"] += 0.5 * h * accs[i]


    # These accelerations were evaluated at the completed step's positions.
    return np.asarray(accs)


# Runge-Kutta 4th order method
def rk4(stars, step_dt, initial_acceleration=None):
    N = len(stars)
    h = step_dt
    if initial_acceleration is None:
        accs = [np.zeros(3) for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                a = acc(stars[i], stars[j])
                accs[i] += a
                accs[j] -= a * stars[i]["mass"] / stars[j]["mass"]
    else:
        accs = np.asarray(initial_acceleration)


    # K1
    k1 = accs
    v1 = np.array([star["v"] for star in stars])

    # K2
    v2 = v1 + 0.5 * h * np.array(k1)
    k2 = [np.zeros(3) for _ in range(N)]
    temp_stars = []
    for i in range(N):
        temp_stars.append({
            "mass": stars[i]["mass"],
            "pos": stars[i]["pos"] + 0.5 * h * v1[i],
            "v": v2[i]
        })
    for i in range(N):
        for j in range(i+1, N):
            a = acc(temp_stars[i], temp_stars[j])
            k2[i] += a
            k2[j] -= a * temp_stars[i]["mass"] / temp_stars[j]["mass"]

    # K3
    v3 = v1 + 0.5 * h * np.array(k2)
    k3 = [np.zeros(3) for _ in range(N)]
    temp_stars = []
    for i in range(N):
        temp_stars.append({
            "mass": stars[i]["mass"],
            "pos": stars[i]["pos"] + 0.5 * h * v2[i],
            "v": v3[i]
        })
    for i in range(N):
        for j in range(i+1, N):
            a = acc(temp_stars[i], temp_stars[j])
            k3[i] += a
            k3[j] -= a * temp_stars[i]["mass"] / temp_stars[j]["mass"]

    # K4
    v4 = v1 + h * np.array(k3)
    k4 = [np.zeros(3) for _ in range(N)]
    temp_stars = []
    for i in range(N):
        temp_stars.append({
            "mass": stars[i]["mass"],
            "pos": stars[i]["pos"] + h * v3[i],
            "v": v4[i]
        })
    for i in range(N):
        for j in range(i+1, N):
            a = acc(temp_stars[i], temp_stars[j])
            k4[i] += a
            k4[j] -= a * temp_stars[i]["mass"] / temp_stars[j]["mass"]

    # Update positions and velocities
    for i in range(N):
        stars[i]["pos"] += h * (v1[i] + 2*v2[i] + 2*v3[i] + v4[i]) / 6
        stars[i]["v"] += h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6




    # k4 uses a predicted position, not the final RK4 position. Record the
    # physical acceleration by evaluating gravity at the completed state.
    return net_accelerations(stars)




# GPU is optional: importing/running the CPU mode never imports CuPy.
_GPU_FIELDS_SOURCE = r'''
// NVRTC provides device math intrinsics without host C headers.
extern "C" __global__
void nbody_fields(const double* p, const double* v, const double* mass,
                  double* acceleration, double* potential, double* timescale,
                  const int n, const double gravity, const int need_times) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double ax = 0.0, ay = 0.0, az = 0.0, u = 0.0, tau = __longlong_as_double(0x7ff0000000000000LL);
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        double dx = p[3*j] - p[3*i];
        double dy = p[3*j+1] - p[3*i+1];
        double dz = p[3*j+2] - p[3*i+2];
        double r2 = dx*dx + dy*dy + dz*dz;
        if (!(r2 > 0.0) || !isfinite(r2)) {
            ax = ay = az = u = tau = __longlong_as_double(0x7ff8000000000000LL);
            break;
        }
        double r = sqrt(r2);
        double factor = gravity * mass[j] / (r2 * r);
        ax += factor * dx; ay += factor * dy; az += factor * dz;
        // Each pair appears in two rows; halve its contribution here.
        u -= 0.5 * gravity * mass[i] * mass[j] / r;
        if (need_times) {
            double mu = gravity * (mass[i] + mass[j]);
            if (mu > 0.0) tau = fmin(tau, r / sqrt(mu / r));
            double vx = v[3*j] - v[3*i];
            double vy = v[3*j+1] - v[3*i+1];
            double vz = v[3*j+2] - v[3*i+2];
            double speed = sqrt(vx*vx + vy*vy + vz*vz);
            if (speed > 0.0) tau = fmin(tau, r / speed);
        }
    }
    acceleration[3*i] = ax;
    acceleration[3*i+1] = ay;
    acceleration[3*i+2] = az;
    potential[i] = u;
    timescale[i] = tau;
}
'''


class GPUState:
    """Compute direct-summation gravity in float64 while keeping state on the GPU.

    Copy each step's recorded state to the host in a batch for plotting.
    Each CUDA thread handles one star and sums all gravitational contributions.
    Avoiding an N-by-N distance matrix keeps auxiliary device storage O(N);
    force evaluation still requires O(N**2) work. No softening or fast-math
    approximations are added. GPU summation order differs from the CPU, so
    bitwise-identical results are not guaranteed.
    """
    def __init__(self, stars, adaptive=False):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No usable NVIDIA CUDA GPU was detected.")
            self.cp = cp
            cp.cuda.Device(0).use()
            properties = cp.cuda.runtime.getDeviceProperties(0)
            name = properties["name"]
            self.device_name = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
            self.n = len(stars)
            self.adaptive = adaptive
            self.host_masses = np.array([s["mass"] for s in stars], dtype=np.float64)
            self.mass = cp.asarray(self.host_masses)
            self.pos = cp.asarray(np.array([s["pos"] for s in stars], dtype=np.float64))
            self.vel = cp.asarray(np.array([s["v"] for s in stars], dtype=np.float64))
            self.kernel = cp.RawKernel(_GPU_FIELDS_SOURCE, "nbody_fields",
                                       options=("--std=c++11", "--fmad=false"))
            self.refresh_snapshot()
        except Exception as exc:
            raise RuntimeError(
                f"GPU initialization failed: {exc}\nCurrent Python: {sys.executable}\n"
                "GPU mode requires a working NVIDIA driver and CuPy. Select CPU mode to continue."
            ) from exc

    def fields(self, pos, vel, need_times=False):
        cp = self.cp
        acceleration = cp.empty_like(pos)
        potential = cp.empty(self.n, dtype=cp.float64)
        timescale = cp.empty(self.n, dtype=cp.float64)
        self.kernel(((self.n + 127) // 128,), (128,),
                    (pos, vel, self.mass, acceleration, potential, timescale,
                     np.int32(self.n), np.float64(G), np.int32(need_times)))
        return acceleration, potential, timescale

    def refresh_snapshot(self):
        cp = self.cp
        self.acceleration, potential, timescale = self.fields(self.pos, self.vel, self.adaptive)
        kinetic = .5 * cp.sum(self.mass[:, None] * self.vel**2)
        scalars = cp.stack((kinetic, cp.sum(potential), cp.min(timescale)))
        packet = cp.asnumpy(cp.concatenate((
            self.pos.ravel(), self.vel.ravel(), self.acceleration.ravel(), scalars)))
        n3 = 3 * self.n
        if not np.isfinite(packet[:-1]).all() or np.isnan(packet[-1]):
            raise RuntimeError("The GPU encountered coincident particles or a non-finite state. Check the initial conditions or reduce dt.")
        self.host_pos = packet[:n3].reshape(self.n, 3)
        self.host_vel = packet[n3:2*n3].reshape(self.n, 3)
        self.host_acceleration = packet[2*n3:3*n3].reshape(self.n, 3)
        self.energy = float(packet[-3] + packet[-2])
        self.energy_scale = float(packet[-3] - packet[-2])
        self.fastest_timescale = float(packet[-1])

    def propose_step(self, max_dt, eta, previous_proposal):
        proposal = min(max_dt, eta * self.fastest_timescale)
        if previous_proposal is not None:
            proposal = min(proposal, 1.25 * previous_proposal)
        return proposal

    def advance(self, h, method):
        if method == "1":
            self.vel += .5 * h * self.acceleration
            self.pos += h * self.vel
            next_acceleration, _, _ = self.fields(self.pos, self.vel)
            self.vel += .5 * h * next_acceleration
        else:
            a1, v1 = self.acceleration, self.vel
            v2 = v1 + .5 * h * a1
            a2, _, _ = self.fields(self.pos + .5 * h * v1, v2)
            v3 = v1 + .5 * h * a2
            a3, _, _ = self.fields(self.pos + .5 * h * v2, v3)
            v4 = v1 + h * a3
            a4, _, _ = self.fields(self.pos + h * v3, v4)
            self.pos += h * (v1 + 2*v2 + 2*v3 + v4) / 6
            self.vel += h * (a1 + 2*a2 + 2*a3 + a4) / 6
        # Re-evaluate the completed state, including pairwise speed timescales.
        self.refresh_snapshot()
        return self.host_acceleration, self.energy

    def diagnostics(self):
        # These O(N) reductions use the already transferred snapshot.
        # The O(N²) potential energy and force have been computed on the GPU.
        momentum = self.host_masses[:, None] * self.host_vel
        total_mass = self.host_masses.sum()
        return {
            "energy": self.energy,
            "momentum": momentum.sum(axis=0),
            "angular_momentum": np.cross(self.host_pos, momentum).sum(axis=0),
            "com_pos": (self.host_masses[:, None] * self.host_pos).sum(axis=0) / total_mass,
            "com_v": momentum.sum(axis=0) / total_mass,
        }



class SimulationCancelled(Exception):
    """Signal user cancellation while preserving the previous successful view."""


class WorkEncounterWindow:
    """Track a sampled global-|a| peak and its return to the entry level."""
    def __init__(self, start_time, acceleration):
        self.start_time = self.peak_time = float(start_time)
        self.reference = self.peak = self.magnitude(acceleration)
        self.descended = False

    @staticmethod
    def magnitude(acceleration):
        return float(np.max(np.hypot.reduce(np.asarray(acceleration), axis=-1)))

    def observe(self, current_time, acceleration):
        value = self.magnitude(acceleration)
        tolerance = 64*np.finfo(float).eps*max(self.peak, self.reference, np.finfo(float).tiny)
        if value > self.peak + tolerance:
            self.peak, self.peak_time = value, float(current_time)
            self.descended = False
        elif value < self.peak - tolerance:
            self.descended = True
        return self.descended and value <= self.reference + tolerance


def probe_work_encounter(start_time, end_time, step, initial_acceleration,
                         advance, current_velocity, controller, cancel=None, progress=None,
                         prior_acceleration=None, prior_step_size=None):
    """Disposable forward pass: ignore Q rejection, locate a peak and its exit.

    The caller owns and restores the physical checkpoint in a finally block.
    No positions, velocities, energies or diagnostic samples are committed here.
    """
    window = WorkEncounterWindow(start_time, initial_acceleration)
    previous_acceleration = np.asarray(initial_acceleration).copy()
    preceding_acceleration = np.asarray(initial_acceleration if prior_acceleration is None else prior_acceleration).copy()
    previous_step_size = prior_step_size
    current_time, steps, attempts, exceeded, peak_q = float(start_time), 0, 0, 0, 0.
    status, last_report = "end_time_before_exit", time.perf_counter()
    while current_time < end_time:
        if cancel is not None and cancel.is_set():
            raise SimulationCancelled()
        h = min(float(step), end_time-current_time)
        if h <= 0 or current_time+h <= current_time:
            raise RuntimeError("Pre-probe dt is smaller than floating-point time resolution.")
        velocity = current_velocity()
        attempts += 1
        try:
            acceleration, energy = advance(h)
            indicator = float(np.max(controller.indicators(
                acceleration, velocity, h, initial_acceleration=previous_acceleration,
                final_velocity=None, previous_acceleration=preceding_acceleration,
                previous_step_size=previous_step_size)))
        except (FloatingPointError, ZeroDivisionError):
            status = "numerical_failure_replay_required"
            break
        except RuntimeError as exc:
            if not any(word in str(exc).lower() for word in ("non-finite", "nonfinite", "coincident")):
                raise
            status = "numerical_failure_replay_required"
            break
        current_time = min(current_time+h, end_time)
        steps += 1
        exceeded += indicator > controller.threshold
        peak_q = max(peak_q, indicator)
        complete = window.observe(current_time, acceleration)
        preceding_acceleration = previous_acceleration
        previous_acceleration = np.asarray(acceleration).copy()
        previous_step_size = h
        now = time.perf_counter()
        if progress is not None and now-last_report >= .15:
            progress(h)
            last_report = now
        if complete:
            status = "peak_and_return"
            break
    return dict(start_time=float(start_time), probe_end_time=current_time,
                probe_peak_time=window.peak_time, probe_peak_acceleration=window.peak,
                entry_acceleration=window.reference, probe_steps=steps,
                probe_trial_evaluations=attempts,
                probe_over_threshold_steps=int(exceeded), probe_peak_Q=peak_q,
                status=status)


def validate_config(config):
    """Validate parameters before computation to catch invalid input early."""
    result = dict(config)
    preprobe = result.get("work_preprobe_enabled", False)
    if not isinstance(preprobe, (bool, np.bool_)):
        raise ValueError("Pre-probe switch must be true or false.")
    result["work_preprobe_enabled"] = bool(preprobe)
    result.setdefault("adaptive_algorithm", "dop853" if result.get("method") == "3" else "acceleration_ratio")
    if result.get("method") == "3":
        if result["adaptive_algorithm"] == "timescale":
            raise ValueError("For DOP853 select Acceleration ratio, a Work-scale method, or DOP853 native error control.")
        if result["adaptive_algorithm"] not in WORK_ALGORITHM_CODES:
            result["work_preprobe_enabled"] = False
    elif result["adaptive_algorithm"] == "dop853":
        raise ValueError("DOP853 native error control requires the DOP853 integrator.")
    for key, default in (("dop853_rtol", DOP853_DEFAULT_RTOL), ("dop853_atol", DOP853_DEFAULT_ATOL)):
        try:
            value = float(result.get(key, default))
            if not np.isfinite(value) or value <= 0:
                raise ValueError
            if key == "dop853_rtol" and value < 100*np.finfo(float).eps:
                raise ValueError
            result[key] = value
        except (ValueError, TypeError, OverflowError):
            raise ValueError("DOP853 tolerances must be finite and positive; rtol must be >= 100 * float64 epsilon (about 2.23e-14).") from None
    if result["adaptive_algorithm"] in RETIRED_WORK_ALGORITHMS:
        raise ValueError("This Work scale test algorithm has been retired. Choose Work scale, Work scale align, Work scale average, or Work scale predicted.")
    result.pop("work_protection_before", None)
    result.pop("work_protection_after", None)
    if result["adaptive_algorithm"] not in ADAPTIVE_ALGORITHMS:
        raise ValueError("Select time-scale, acceleration-ratio, Work scale, Work scale align, Work scale average, or Work scale predicted stepping.")
    for key, default in ENERGY_BAND_DEFAULTS.items():
        if result.get("adaptive_enabled", False) and result["adaptive_algorithm"] == "step_energy_band":
            try:
                value = float(result.get(key, default))
                if not np.isfinite(value):
                    raise ValueError
            except (ValueError, TypeError, OverflowError):
                raise ValueError(f"{key} must be a finite number in absolute code energy units.") from None
            result[key] = value
        else:
            result[key] = default
    if result.get("adaptive_enabled", False) and result["adaptive_algorithm"] == "step_energy_band":
        if not 0 < result["step_energy_lower"] < result["step_energy_upper"]:
            raise ValueError("Step energy bounds must satisfy 0 < lower < upper.")
        if result["energy_drift_limit"] < 0:
            raise ValueError("Energy drift limit must be nonnegative; 0 disables this limit.")
    # Discard inactive settings carried by older plans for retained methods.
    for retired_key in ("energy_relative_tolerance", "energy_max_rejections"):
        result.pop(retired_key, None)
    result.setdefault("compute_backend", "cpu_array" if result.get("method") in TAYLOR_METHOD_ORDERS else "cpu")
    # Previously saved Taylor45 plans used cpu for the high-precision-only core.
    # Preserve their arithmetic instead of silently reinterpreting old experiments.
    if result.get("method") == "4" and result["compute_backend"] == "cpu":
        result["compute_backend"] = "cpu_mp"
    if result["compute_backend"] in TAYLOR_MP_BACKENDS and result.get("method") not in TAYLOR_METHOD_ORDERS:
        raise ValueError("High-precision CPU devices are available only for Taylor integrators.")
    result.setdefault("particle_count", 5)
    if result["compute_backend"] not in COMPUTE_BACKENDS:
        raise ValueError("Select a supported compute device for the integrator.")
    try:
        threads = result.get("cpu_threads", 0)
        if isinstance(threads, bool):
            raise ValueError
        threads = float(threads)
        if not np.isfinite(threads) or threads != int(threads) or not 0 <= threads <= 4096:
            raise ValueError
        result["cpu_threads"] = int(threads)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("CPU threads must be an integer: 0 for automatic, or a positive thread count.") from None
    for key, title in (("dt", "Time step"), ("total_time", "Simulation duration")):
        try:
            result[key] = float(result[key])
        except (ValueError, TypeError, KeyError):
            raise ValueError(f"{title} must be a number.") from None
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{title} must be finite and positive.")
    for key, title in (("sample_every", "Sampling interval"), ("seed", "Seed"), ("particle_count", "Random particle count")):
        try:
            value = float(result[key])
            if not np.isfinite(value) or value != int(value):
                raise ValueError
            result[key] = int(value)
        except (ValueError, TypeError, KeyError, OverflowError):
            raise ValueError(f"{title} must be an integer.") from None
    if not 1 <= result["particle_count"] <= 10000:
        raise ValueError("The random particle count must be an integer from 1 to 10000; complete per-step records use memory.")
    if result["sample_every"] < 1:
        raise ValueError("The sampling interval must be at least one base step.")
    if not 0 <= result["seed"] <= 2**32 - 1:
        raise ValueError("The seed must be an integer from 0 to 4294967295.")
    if result.get("method") not in METHOD_CHOICES:
        raise ValueError("Select Leapfrog, RK4, DOP853, Taylor45, Taylor100, Taylor150, or Taylor200.")
    if result.get("method") in TAYLOR_METHOD_ORDERS and result["compute_backend"] not in ("cpu_array", *TAYLOR_MP_BACKENDS):
        raise ValueError("Taylor supports cpu_array (float64), cpu_mpfr_113 (34 digits), cpu_mpfr (80 digits) or cpu_mp (mpmath, 80 digits).")
    if result.get("initial_mode") not in ("random", "manual", "orbit"):
        raise ValueError("Select an initial-condition source.")
    if result["initial_mode"] == "orbit":
        index = result.get("orbit_index")
        if not isinstance(index, int) or not 0 <= index < len(STABLE_ORBITS):
            raise ValueError("Select a known orbit.")
    result["binary_eccentricity"] = (validate_eccentricity(result.get("binary_eccentricity", DEFAULT_ECCENTRICITY))
                                      if is_kepler_config(result) else DEFAULT_ECCENTRICITY)
    result["adaptive_enabled"] = bool(result.get("adaptive_enabled", False))
    try:
        result["adaptive_eta"] = float(result.get("adaptive_eta", 0.03))
    except (ValueError, TypeError):
        raise ValueError("The safety factor eta must be a number.") from None
    if not np.isfinite(result["adaptive_eta"]) or not 0 < result["adaptive_eta"] <= 0.2:
        raise ValueError("The safety factor eta must satisfy 0 < eta <= 0.2.")
    try:
        result["acceleration_trigger_ratio"] = float(result.get("acceleration_trigger_ratio", 2.0))
    except (ValueError, TypeError):
        raise ValueError("The acceleration trigger ratio must be a number.") from None
    if (not np.isfinite(result["acceleration_trigger_ratio"])
            or result["acceleration_trigger_ratio"] <= 1):
        raise ValueError("The acceleration trigger ratio must be finite and greater than 1.")
    try:
        power = float(result.get("acceleration_max_power", 7))
        if not np.isfinite(power) or power < 1 or power != int(power):
            raise ValueError
        result["acceleration_max_power"] = int(power)
        divisor = result["acceleration_trigger_ratio"] ** result["acceleration_max_power"]
        floor_dt = result["dt"] / divisor
        if not np.isfinite(divisor) or not np.isfinite(floor_dt) or floor_dt <= 0:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError("The maximum cumulative exponent must be a positive integer, and initial dt / N^exponent must be representable as a finite positive number.") from None
    try:
        factor = float(result.get("work_step_factor", WORK_DEFAULT_FACTOR))
        power = float(result.get("work_max_power", WORK_DEFAULT_MAX_POWER))
        if (not np.isfinite(factor) or factor <= 1 or not np.isfinite(power)
                or power < 1 or power != int(power)):
            raise ValueError
        divisor = factor ** int(power)
        if not np.isfinite(divisor) or result["dt"] / divisor <= 0:
            raise ValueError
        result["work_step_factor"], result["work_max_power"] = factor, int(power)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Work-step factor N must be finite and greater than 1; K must be a positive integer with representable initial dt / N^K.") from None
    try:
        threshold = float(result.get("work_threshold", WORK_DEFAULT_THRESHOLD))
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError
        result["work_threshold"] = threshold
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Trigger Threshold must be a finite positive dimensionless fraction of S_E=K(0)+|U(0)|.") from None
    interval = result["dt"] * result["sample_every"]
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("The sampling interval is too large or too small. Adjust it.")
    return result


def make_initial_state(config):
    if is_kepler_config(config):
        return binary_stars(config.get("binary_eccentricity", DEFAULT_ECCENTRICITY))
    if config["initial_mode"] == "orbit":
        orbit = STABLE_ORBITS[config["orbit_index"]]
        masses = np.asarray(orbit.get("masses", np.ones(orbit["N"])), dtype=float)
        positions = np.asarray(orbit["positions"], dtype=float)
        velocities = np.asarray(orbit["velocities"], dtype=float)
        if (masses.shape != (orbit["N"],) or positions.shape != (orbit["N"], 3)
                or velocities.shape != positions.shape or not np.all(masses > 0)
                or not all(np.isfinite(a).all() for a in (masses, positions, velocities))):
            raise ValueError("Orbit masses, positions and velocities must have matching sizes and finite values; masses must be positive.")
        return [{"mass": float(mass), "pos": pos.copy(), "v": vel.copy()}
                for mass, pos, vel in zip(masses, positions, velocities)]
    # RandomState keeps the original seed -> initial conditions mapping,
    # without mutating NumPy's global random generator during repeat runs.
    rng = np.random.RandomState(config["seed"])
    return [{"mass": 1, "pos": rng.uniform(-50, 50, size=3),
             "v": rng.uniform(-0.1, 0.1, size=3)} for _ in range(config.get("particle_count", 5))]


def run_simulation(config, progress=None, cancel=None):
    """Run one simulation; optional CPU compilation never affects dictionary/GPU imports."""
    config = validate_config(config)
    if config["compute_backend"] in ("cpu_array", "cpu_parallel"):
        if cancel is not None and cancel.is_set():
            raise SimulationCancelled()
        try:
            from nbody_array_cpu import thread_scope
        except ImportError as exc:
            raise RuntimeError(
                f"Array CPU requires Numba in this Python: {sys.executable}\n"
                f'Install with: "{sys.executable}" -m pip install numba\n'
                "You can still select the original dictionary CPU or GPU mode.") from exc
        with thread_scope():
            return _run_simulation_impl(config, progress, cancel)
    return _run_simulation_impl(config, progress, cancel)





def _run_simulation_impl(config, progress=None, cancel=None):
    """Shared time stepping, diagnostics and recording for all four compute backends."""
    config = validate_config(config)
    stars = make_initial_state(config)
    initial_stars = copy.deepcopy(stars)
    base_dt, end_time = config["dt"], config["total_time"]
    adaptive = config["adaptive_enabled"]
    eta = config["adaptive_eta"]
    adaptive_algorithm = config["adaptive_algorithm"]
    sample_every = config["sample_every"]
    nominal_steps = int(np.ceil(end_time / base_dt))
    if cancel is not None and cancel.is_set():
        raise SimulationCancelled()
    setup_started = time.perf_counter()
    state = None
    if config["compute_backend"] == "gpu":
        state = GPUState(stars, adaptive and adaptive_algorithm == "timescale")
    elif config["compute_backend"] in ("cpu_array", "cpu_parallel"):
        from nbody_array_cpu import ArrayCPUState
        state = ArrayCPUState(stars, gravity=G,
                              parallel=config["compute_backend"] == "cpu_parallel",
                              threads=config["cpu_threads"],
                              adaptive=adaptive and adaptive_algorithm == "timescale")
        state.device_name = cpu_model_name()
    energy_scale = (state.energy_scale if state is not None else
                    sum(Ek(s) for s in stars) - sum(
                        Ep(stars[i], stars[j]) for i in range(len(stars)) for j in range(i + 1, len(stars))))
    trajectories = [[s["pos"].copy()] for s in stars]
    velocities = [np.array([s["v"].copy() for s in stars])]
    accelerations = [state.host_acceleration if state is not None else net_accelerations(stars)]
    diagnostics = [state.diagnostics() if state is not None else test(stars)]
    energies = [diagnostics[0]["energy"]]
    times, step_sizes, diagnostic_times = [0.0], [0.0], [0.0]
    minimum_dt = max(base_dt * 1e-10, np.nextafter(0.0, 1.0))
    current_time, steps = 0.0, 0
    sample_number = 1
    next_sample_time = min(sample_number * sample_every * base_dt, end_time)
    previous_proposal = None
    ratio_mode = adaptive and adaptive_algorithm == "acceleration_ratio"
    work_mode = adaptive and adaptive_algorithm in WORK_ALGORITHM_CODES
    average_work_mode = work_mode and adaptive_algorithm == "work_step_average"
    predicted_work_mode = work_mode and adaptive_algorithm == "work_step_predicted"
    aligned_work_mode = work_mode and adaptive_algorithm in WORK_ALIGNED_CODES
    dop853_mode = config["method"] == "3"
    dop853_native = dop853_mode and adaptive and adaptive_algorithm == "dop853"
    taylor_mode = config["method"] in TAYLOR_METHOD_ORDERS
    energy_band_mode = adaptive and adaptive_algorithm == "step_energy_band"
    step_sampled = ratio_mode or work_mode or dop853_mode or taylor_mode or energy_band_mode
    energy_band = None
    energy_band_h = base_dt
    energy_band_scale = energy_scale
    energy_band_trials, energy_band_rejections = [], {}
    if energy_band_mode:
        from nbody_step_energy_band import EnergyBandController
        energy_band = EnergyBandController(config["step_energy_lower"], config["step_energy_upper"],
                                           config["energy_drift_limit"], INTEGRATOR_ORDERS[config["method"]])
    work_controller = WorkStepController(base_dt, config["work_step_factor"], config["work_max_power"],
                                         config["work_threshold"], masses=[s["mass"] for s in stars],
                                         energy_scale=energy_scale, aligned=aligned_work_mode, average=average_work_mode, predicted=predicted_work_mode) if work_mode else None
    work_rejections, work_events, work_exceeded_intervals = [], [], []
    work_exceeded_steps = work_shrink_count = work_grow_count = work_max_level_used = 0
    work_grow_trial_count = work_recovery_blocked_count = 0
    work_preprobe = work_mode and config["work_preprobe_enabled"]
    work_probe_events, work_probe_lock = [], None
    work_probe_evaluations = work_probe_over_threshold_steps = work_probe_blocked_growth = 0
    ratio_shrink_count = ratio_grow_count = 0
    ratio_level = ratio_max_level_used = 0
    ratio_floor_dt = base_dt / config["acceleration_trigger_ratio"] ** config["acceleration_max_power"]
    ratio_controller = (AccelerationRatioController(
        accelerations[0], base_dt, config["acceleration_trigger_ratio"], config["acceleration_max_power"])
        if ratio_mode else None)
    ratio_events = []
    if ratio_mode:
        minimum_dt = ratio_floor_dt
    elif work_mode:
        minimum_dt = base_dt / config["work_step_factor"] ** config["work_max_power"]
    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise SimulationCancelled()

    dop853 = None
    if dop853_mode:
        from nbody_dop853 import DOP853Stepper
        dop853 = DOP853Stepper(stars, state, config, net_accelerations, tot_E, check_cancel)
    taylor = None
    retryable_taylor_error = ()
    taylor_numerical_rejections = 0
    if taylor_mode:
        from nbody_taylor45 import TaylorStepper, TaylorStepRejected
        retryable_taylor_error = (TaylorStepRejected,)
        taylor = TaylorStepper(stars, G, net_accelerations, tot_E, check_cancel,
                              backend=config["compute_backend"], state=state,
                              order=TAYLOR_METHOD_ORDERS[config["method"]])
    custom_stepper = taylor if taylor is not None else dop853
    integrator = leapfrog if config["method"] == "1" else rk4
    def advance_state(h):
        if custom_stepper is not None:
            return custom_stepper.advance(current_time, h)[2:]
        if state is not None:
            return state.advance(h, config["method"])
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            acceleration = (integrator(stars, h, initial_acceleration=accelerations[-1])
                            if ratio_mode else integrator(stars, h))
            energy = tot_E(stars)
        if (not np.isfinite(energy) or not np.isfinite(acceleration).all()
                or any(not np.isfinite(s["pos"]).all() or not np.isfinite(s["v"]).all() for s in stars)):
            raise RuntimeError("The simulation produced non-finite values. Reduce dt or check close encounters.")
        return acceleration, energy

    def physical_checkpoint():
        if state is None:
            saved = copy.deepcopy(stars)
        else:
            saved = dict(state.__dict__)
            for key in ("pos", "vel", "acceleration"):
                saved[key] = saved[key].copy()
        return (saved, custom_stepper.checkpoint()) if custom_stepper is not None else saved

    def restore_physical_checkpoint(saved):
        if custom_stepper is not None:
            saved, integrator_checkpoint = saved
            custom_stepper.restore(integrator_checkpoint)
        if state is None:
            stars[:] = copy.deepcopy(saved)
        else:
            state.__dict__.update(saved)
            for key in ("pos", "vel", "acceleration"):
                setattr(state, key, saved[key].copy())

    def current_velocity():
        return state.host_vel.copy() if state is not None else np.array([s["v"].copy() for s in stars])

    device_name = state.device_name if state is not None else cpu_model_name()
    initialization_seconds = time.perf_counter() - setup_started
    started = last_report = time.perf_counter()
    if progress:
        progress(0.0, 0, 0.0, 0.0)
    while current_time < end_time:
        if cancel is not None and cancel.is_set():
            raise SimulationCancelled()
        if dop853_native:
            new_time, h, new_acceleration, energy = dop853.advance(current_time, None)
        elif energy_band_mode:
            h = min(energy_band_h, end_time - current_time)
            new_time = end_time if h == end_time - current_time else current_time + h
        elif adaptive:
            if adaptive_algorithm == "acceleration_ratio":
                proposal, ratio_level = ratio_controller.choose(accelerations[-1], current_time)
                if ratio_controller.last_event is not None:
                    ratio_events.append(ratio_controller.last_event)
                ratio_max_level_used = max(ratio_max_level_used, ratio_level)
                if previous_proposal is not None:
                    ratio_shrink_count += proposal < previous_proposal
                    ratio_grow_count += proposal > previous_proposal
            elif work_mode:
                work_previous_level = work_controller.level
                proposal = work_controller.step()
            else:
                proposal = (state.propose_step(base_dt, eta, previous_proposal) if state is not None
                            else adaptive_step_size(stars, base_dt, eta, previous_proposal))
            if not np.isfinite(proposal) or proposal < minimum_dt:
                raise RuntimeError(
                    f"t={current_time:.12g} requires dt {proposal:.6e} "
                    f"below the safety limit {minimum_dt:.6e}; this run has been stopped.")
            previous_proposal = proposal
            # Ratio/work-step diagnostics count accepted steps; only clip at the final time.
            boundary = end_time if step_sampled else next_sample_time
            remaining = boundary - current_time
            roundoff = 8 * np.finfo(float).eps * max(boundary, base_dt)
            if proposal >= remaining - roundoff:
                h, new_time = remaining, boundary
            else:
                h, new_time = proposal, current_time + proposal
        else:
            # Full fixed steps retain the original h exactly; only the final
            # partial step is shortened to stop at the requested physical time.
            new_time = min((steps + 1) * base_dt, end_time)
            h = base_dt if (steps + 1) * base_dt <= end_time else end_time - current_time
        if h <= 0 or new_time <= current_time:
            raise RuntimeError("The time step is smaller than the floating-point resolution at the current time. Adjust the parameters.")
        if energy_band_mode:
            checkpoint = physical_checkpoint()
            energy_band.begin_step(current_time, energies[0], energies[-1], energy_band_scale)
            try:
                while True:
                    check_cancel()
                    if not np.isfinite(h) or h <= 0 or current_time + h <= current_time:
                        raise RuntimeError(f"Step energy band failed at t={current_time:.17g}: h={h!r} cannot advance time; attempted h={[x['h'] for x in energy_band_trials if x['start_step'] == steps]}")
                    at_end = h == end_time - current_time
                    trial_failure = None
                    try:
                        new_acceleration, energy = advance_state(h)
                        trial_scale = (state.energy_scale if state is not None else
                                       2 * sum(Ek(star) for star in stars) - energy)
                    except (FloatingPointError, OverflowError, RuntimeError) as exc:
                        # Retry numerical candidate failures, never swallow arbitrary programming errors.
                        numerical = isinstance(exc, (FloatingPointError, OverflowError) + retryable_taylor_error)
                        numerical = numerical or any(word in str(exc).lower() for word in
                            ("non-finite", "nonfinite", "coincident", "float64 display range"))
                        if not numerical:
                            raise
                        trial_failure = str(exc)
                        energy, trial_scale = float('inf'), float('inf')
                    decision = energy_band.assess(h, energy, trial_scale, at_end=at_end)
                    energy_band_trials.append(dict(start_step=steps, t_start=current_time, h=h,
                        try_index=decision.attempt, delta_E=decision.delta_energy,
                        abs_delta_E=abs(decision.delta_energy), trial_E_minus_E0=decision.drift,
                        roundoff_screen=decision.roundoff_screen, action=decision.action,
                        reason=decision.reason, numerical_failure=trial_failure))
                    if decision.action == "accept":
                        energy_band_h, energy_band_scale = h, trial_scale
                        new_time = end_time if at_end else current_time + h
                        break
                    energy_band_rejections[decision.reason] = energy_band_rejections.get(decision.reason, 0) + 1
                    restore_physical_checkpoint(checkpoint)
                    if decision.action == "fail":
                        raise RuntimeError(f"Step energy band failed at t={current_time:.17g}, h={h:.17g}: {decision.reason}; d={abs(decision.delta_energy)!r}, D={abs(decision.drift)!r}, L={config['step_energy_lower']}, U={config['step_energy_upper']}, G={config['energy_drift_limit']}; tried h={[x['h'] for x in energy_band_trials if x['start_step'] == steps]}")
                    h = min(decision.next_h, end_time - current_time)
            except BaseException:
                restore_physical_checkpoint(checkpoint)
                raise
        elif work_mode:
            # Every retry uses the same accepted start state and its own trial end.
            initial_velocity = velocities[-1].copy()
            checkpoint = physical_checkpoint()
            # A coarse trial can fail while the adjacent fine trial requests
            # growth. Remember failed bands for this starting state only.
            over_limit_levels = set()
            while True:
                if cancel is not None and cancel.is_set():
                    raise SimulationCancelled()
                try:
                    new_acceleration, energy = advance_state(h)
                    trial_indicators = work_controller.indicators(
                        new_acceleration, initial_velocity, h, initial_acceleration=accelerations[-1],
                        final_velocity=None, previous_acceleration=accelerations[-2] if len(accelerations)>1 else accelerations[-1],
                        previous_step_size=step_sizes[-1])
                    work_indicator = float(np.max(trial_indicators))
                    if taylor is not None and taylor.order != 45 and not np.isfinite(work_indicator):
                        raise TaylorStepRejected('Taylor WorkScale trial indicator is non-finite.')
                except retryable_taylor_error as exc:
                    # A failed candidate must never become an accepted state.
                    # Only trial evaluation failures arrive here; coefficient
                    # generation failures still stop with their specific cause.
                    restore_physical_checkpoint(checkpoint)
                    work_max_level_used = max(work_max_level_used, work_controller.level)
                    if work_controller.level >= work_controller.max_power:
                        raise RuntimeError(f'Taylor non-finite trial at maximum WorkScale level K={work_controller.max_power}; '
                                           'increase K or reduce the initial dt.') from exc
                    over_limit_levels.add(work_controller.level)
                    work_rejections.append(dict(time=float(current_time), trial_dt=float(h),
                        level=work_controller.level, action='shrink', reason='nonfinite_taylor_trial',
                        indicator=None, controlling_stars=[]))
                    taylor_numerical_rejections += 1
                    work_controller.shrink()
                    work_shrink_count += 1
                    h = min(work_controller.step(), end_time-current_time)
                    new_time = current_time+h
                    if h <= 0 or new_time <= current_time:
                        raise RuntimeError('Work-step dt is smaller than the floating-point time resolution.')
                    continue
                except RuntimeError as exc:
                    if getattr(exc, 'taylor_coefficient_failure', False):
                        context = (f'\nAt t={current_time:.12g}: WorkScale level={work_controller.level}, '
                                   f'K={work_controller.max_power}, trial dt={h:.8g}, '
                                   f'minimum dt={work_controller.max_dt/work_controller.factor**work_controller.max_power:.8g}.')
                        if work_exceeded_steps:
                            context += (f'\nEarlier accepted over-Q steps at the K limit: {work_exceeded_steps}. '
                                        'A trajectory can diverge after these forced steps, before coefficients overflow. '
                                        'Increase K or reduce the initial dt, then restart from the initial state.')
                        exc.args = (context.lstrip()+'\n'+str(exc),)
                    raise
                work_max_level_used = max(work_max_level_used, work_controller.level)
                retry_action = None
                if work_indicator > work_controller.threshold:
                    if work_controller.level < work_controller.max_power:
                        over_limit_levels.add(work_controller.level)
                        retry_action = "shrink"
                elif (work_indicator < work_controller.threshold / work_controller.factor
                      and work_controller.level > 0 and work_probe_lock is not None
                      and (not predicted_work_mode or not work_probe_lock["recovery_budget"].can_grow(work_controller.level))):
                    work_probe_blocked_growth += 1
                elif work_indicator < work_controller.threshold / work_controller.factor and work_controller.level > 0:
                    enlarged = min(work_controller.max_dt / work_controller.factor ** (work_controller.level - 1),
                                   end_time - current_time)
                    if work_controller.level - 1 in over_limit_levels:
                        work_recovery_blocked_count += 1
                    elif enlarged > h:
                        # Do not reject a final clipped trial if increasing the
                        # nominal band cannot increase its actual duration.
                        retry_action = "grow"
                if retry_action is None:
                    if (work_preprobe and work_probe_lock is None
                            and work_controller.level < work_previous_level):
                        entry_a = WorkEncounterWindow.magnitude(accelerations[-1])
                        previous_a = (WorkEncounterWindow.magnitude(accelerations[-2])
                                      if len(accelerations) > 1 else entry_a)
                        candidate_a = WorkEncounterWindow.magnitude(new_acceleration)
                        # Ordinary recovery on the declining side is unchanged.
                        # Probe a recovery inside a rising-acceleration encounter.
                        tolerance = 64*np.finfo(float).eps*max(entry_a, previous_a, np.finfo(float).tiny)
                        if entry_a > previous_a+tolerance or candidate_a > entry_a+tolerance:
                            work_rejections.append(dict(time=float(current_time), trial_dt=float(h),
                                level=work_controller.level, action="probe_replay",
                                reason="preprobe_close_encounter", indicator=work_indicator,
                                controlling_stars=(np.flatnonzero(trial_indicators == work_indicator)+1).tolist()))
                            restore_physical_checkpoint(checkpoint)
                            try:
                                event = probe_work_encounter(
                                    current_time, end_time, h, accelerations[-1], advance_state,
                                    current_velocity, work_controller, cancel,
                                    (lambda probe_h: progress(current_time/end_time, steps, probe_h,
                                                              time.perf_counter()-started)) if progress else None,
                                    prior_acceleration=accelerations[-2] if len(accelerations)>1 else accelerations[-1],
                                    prior_step_size=step_sizes[-1])
                            finally:
                                restore_physical_checkpoint(checkpoint)
                            event.update(replay_level=work_previous_level,
                                         replay_max_dt=float(base_dt/work_controller.factor**work_previous_level),
                                         replay_completed=False)
                            work_probe_events.append(event)
                            work_probe_evaluations += event["probe_trial_evaluations"]
                            work_probe_over_threshold_steps += event["probe_over_threshold_steps"]
                            work_probe_lock = dict(event=event,
                                tracker=WorkEncounterWindow(current_time, accelerations[-1]))
                            if predicted_work_mode:
                                work_probe_lock["recovery_budget"] = WorkProbeRecoveryBudget(work_previous_level)
                                event.update(work_probe_lock["recovery_budget"].record())
                            work_controller.level = work_previous_level
                            h = min(work_controller.step(), end_time-current_time)
                            new_time = current_time+h
                            continue
                    break
                work_rejections.append(dict(time=float(current_time), trial_dt=float(h),
                    level=work_controller.level,
                    action=retry_action,
                    reason="above_threshold" if retry_action == "shrink" else "below_threshold_over_N",
                    indicator=float(work_indicator) if np.isfinite(work_indicator) else None,
                    controlling_stars=(np.flatnonzero(trial_indicators == work_indicator) + 1).tolist()))
                # Restore the exact pre-trial state, including diagnostic caches.
                restore_physical_checkpoint(checkpoint)
                if retry_action == "shrink":
                    work_controller.shrink()
                    work_shrink_count += 1
                else:
                    work_controller.grow()
                    work_grow_trial_count += 1
                h = min(work_controller.step(), end_time - current_time)
                new_time = current_time + h
                if h <= 0 or new_time <= current_time:
                    raise RuntimeError("Work-step dt is smaller than the floating-point time resolution.")
            if predicted_work_mode and work_probe_lock is not None:
                budget = work_probe_lock["recovery_budget"]
                budget.commit(work_previous_level, work_controller.level)
                work_probe_lock["event"].update(budget.record())
            if work_controller.level < work_previous_level:
                work_grow_count += 1
            if work_controller.level != work_previous_level:
                work_events.append(dict(time=float(current_time), previous_level=work_previous_level,
                                        level=work_controller.level, dt=float(h)))
            if work_indicator > work_controller.threshold:
                work_exceeded_steps += 1
                drivers = (np.flatnonzero(trial_indicators > work_controller.threshold) + 1).tolist()
                peak = float(work_indicator) if np.isfinite(work_indicator) else None
                # These are accepted over-limit step intervals, not a claim that
                # every continuous instant between their endpoints exceeds the threshold.
                if work_exceeded_intervals and work_exceeded_intervals[-1]["last_step"] == steps:
                    item = work_exceeded_intervals[-1]
                    item.update(end_time=float(new_time), last_step=steps + 1,
                                steps=item["steps"] + 1,
                                stars=sorted(set(item["stars"]) | set(drivers)))
                    if peak is None or (item["peak_indicator"] is not None and peak > item["peak_indicator"]):
                        item.update(peak_indicator=peak, peak_time=float(new_time))
                else:
                    work_exceeded_intervals.append(dict(start_time=float(current_time), end_time=float(new_time),
                        first_step=steps + 1, last_step=steps + 1, steps=1,
                        level=work_controller.level, stars=drivers,
                        peak_indicator=peak, peak_time=float(new_time)))
        elif not dop853_native:
            new_acceleration, energy = advance_state(h)
        current_time = new_time
        steps += 1
        if taylor is not None and config["compute_backend"] == "cpu_mp":
            taylor.core.record_accepted_energy(h)
        if state is not None:
            for i in range(len(stars)):
                trajectories[i].append(state.host_pos[i].copy())
            velocities.append(state.host_vel)
        else:
            for i, star in enumerate(stars):
                trajectories[i].append(star["pos"].copy())
            velocities.append(np.array([s["v"].copy() for s in stars]))
        accelerations.append(new_acceleration)
        energies.append(energy)
        times.append(current_time)
        step_sizes.append(h)
        if work_probe_lock is not None:
            tracker, event = work_probe_lock["tracker"], work_probe_lock["event"]
            exited = tracker.observe(current_time, new_acceleration)
            reached_probe_end = current_time >= event["probe_end_time"]
            if (exited and reached_probe_end) or current_time == end_time:
                event.update(replay_end_time=float(current_time),
                             replay_peak_time=tracker.peak_time, replay_peak_acceleration=tracker.peak,
                             replay_completed=bool(exited and reached_probe_end),
                             replay_steps=steps-int(np.searchsorted(times, event["start_time"])))
                work_probe_lock = None
        should_sample = ((steps % sample_every == 0 or current_time == end_time)
                         if step_sampled else current_time == next_sample_time)
        if should_sample:
            diagnostics.append(state.diagnostics() if state is not None else test(stars))
            diagnostic_times.append(current_time)
            if not step_sampled:
                sample_number += 1
                next_sample_time = min(sample_number * sample_every * base_dt, end_time)
        now = time.perf_counter()
        if progress and (now - last_report >= .15 or current_time == end_time):
            progress(current_time / end_time, steps, h, now - started)
            last_report = now
    if cancel is not None and cancel.is_set():
        raise SimulationCancelled()
    trajectories = np.asarray(trajectories)
    orbit = STABLE_ORBITS[config["orbit_index"]] if config["initial_mode"] == "orbit" else None
    metadata = {
        "case_name": (f"High-eccentricity Kepler binary (e={config['binary_eccentricity']:.8g})" if is_kepler_config(config)
                      else orbit["name"] if orbit else f"Random initial conditions · seed = {config['seed']}"),
        "case_id": f"orbit_{config['orbit_index'] + 1}" if orbit else f"seed_{config['seed']}",
        "seed": orbit["seed"] if orbit else config["seed"],
        "orbit_source": orbit.get("source") if orbit else None,
        "reference_period": orbit.get("period") if orbit else None,
        "method_id": config["method"],
        "method_name": METHOD_NAMES[config["method"]],
        "compute_backend": config["compute_backend"],
        "compute_device": "GPU" if config["compute_backend"] == "gpu" else "CPU",
        "cpu_threads": state.thread_count if config["compute_backend"] in ("cpu_array", "cpu_parallel") else None,
        "cpu_threads_requested": config["cpu_threads"] if config["compute_backend"] == "cpu_parallel" else None,
        "array_cpu_runtime": "Numba; float64; fastmath=False" if config["compute_backend"] in ("cpu_array", "cpu_parallel") else None,
        "initialization_seconds": initialization_seconds,
        "device_name": device_name,
        "dtype": "float64", "particle_count": len(stars),
        "G": G, "dt": base_dt, "steps": steps, "requested_steps": nominal_steps,
        "simulation_end_time": end_time,
        "adaptive_enabled": adaptive,
        "adaptive_algorithm": adaptive_algorithm if adaptive else "fixed",
        "adaptive_eta": eta if adaptive and adaptive_algorithm == "timescale" else None,
        "zero_acceleration_policy": "current zero: skip; initial zero: use system initial nonzero acceleration scale; all-zero start: initialize once without changing dt"
                                    if adaptive and adaptive_algorithm == "acceleration_ratio" else None,
        "acceleration_trigger_ratio": config["acceleration_trigger_ratio"]
                                      if adaptive and adaptive_algorithm == "acceleration_ratio" else None,
        "acceleration_max_power": config["acceleration_max_power"] if ratio_mode else None,
        "ratio_limit_scope": "cumulative relative to initial dt" if ratio_mode else None,
        "ratio_minimum_dt": ratio_floor_dt if ratio_mode else None,
        "ratio_final_level": ratio_level if ratio_mode else None,
        "ratio_max_level_used": ratio_max_level_used if ratio_mode else None,
        "ratio_boundary_policy": "r>N to trigger; higher exact powers enter their own band" if ratio_mode else None,
        "ratio_reference_scales": ratio_controller.scales.tolist() if ratio_mode else None,
        "ratio_star_final_levels": ratio_controller.levels.tolist() if ratio_mode else None,
        "ratio_step_events": ratio_events,
        "ratio_adjustment_policy": "persistent per-star references=initial_scale*N**local_level; integer power bands; independent recovery; global maximum level capped at K"
                                   if adaptive and adaptive_algorithm == "acceleration_ratio" else None,
        "discarded_trial_steps": len(energy_band_trials)-steps if energy_band_mode else dop853.rejected_trials if dop853_native else len(work_rejections),
        "work_step_factor": config["work_step_factor"] if work_mode else None,
        "work_max_power": config["work_max_power"] if work_mode else None,
        "work_threshold": config["work_threshold"] if work_mode else None,
        "work_indicator_units": "dimensionless",
        "work_indicator_normalization": "initial_energy_scale",
        "work_indicator_definition": WORK_PREDICTED_DEFINITION if predicted_work_mode else WORK_AVERAGE_DEFINITION if average_work_mode else
                                     WORK_DISPLACEMENT_DEFINITION if aligned_work_mode or not work_mode else WORK_LEGACY_DISPLACEMENT_DEFINITION,
        "work_algorithm_name": ADAPTIVE_ALGORITHMS[adaptive_algorithm] if work_mode else None,
        "work_aligned": aligned_work_mode if work_mode else None,
        "work_average": average_work_mode if work_mode else None,
        "work_predicted": predicted_work_mode if work_mode else None,
        "work_acceleration_selection": ("accepted_history_step_ratio_half_difference" if predicted_work_mode else "endpoint_vector_average" if average_work_mode else "start" if aligned_work_mode else "end") if work_mode else None,
        "work_endpoint_symmetry": None,
        "work_displacement_convention": displacement_work_convention(legacy=work_mode and not aligned_work_mode, average=average_work_mode, predicted=predicted_work_mode),
        "work_normalization_definition": "S_E=K(0)+abs(U(0)); fixed at initial state",
        "work_energy_scale": float(energy_scale),
        "work_final_level": work_controller.level if work_mode else None,
        "work_max_level_used": work_max_level_used if work_mode else None,
        "work_shrink_count": work_shrink_count,
        "work_grow_count": work_grow_count,
        "work_grow_trial_count": work_grow_trial_count,
        "work_recovery_blocked_count": work_recovery_blocked_count,
        "work_preprobe_enabled": bool(work_preprobe),
        "work_probe_policy": ("probe recovery during rising max_i|a_i| at the proposed larger dt, ignoring Q; "
                              "stop after a sampled peak returns to entry max acceleration; restore all state; "
                              + ("replay with accepted-shrink recovery credits, never above the pre-recovery dt; " if predicted_work_mode else
                                 "replay from recovery start at the pre-recovery dt or smaller, without growth; ") +
                              "extend replay until its own peak returns to entry acceleration; end-time clips incomplete windows")
                             if work_preprobe else None,
        "work_probe_events": work_probe_events,
        "work_probe_trial_evaluations": work_probe_evaluations,
        "work_probe_over_threshold_steps": work_probe_over_threshold_steps,
        "work_probe_blocked_growth": work_probe_blocked_growth,
        "work_probe_window_count": len(work_probe_events),
        "work_trial_evaluations": steps+len(work_rejections)+work_probe_evaluations if work_mode else None,
        "work_recovery_policy": (("preprobe_accepted_shrink_credit_v2" if predicted_work_mode else "preprobe_peak_replay_v1") if work_preprobe else "trial_end_rollback_retry_v1") if work_mode else None,
        "work_retry_conflict_policy": "when a larger band already exceeded the threshold at this same starting state, accept the smaller admissible trial; no repeated band loop" if work_mode else None,
        "work_step_events": work_events,
        "work_rejection_events": work_rejections,
        "work_exceeded_steps": work_exceeded_steps,
        "work_exceeded_intervals": work_exceeded_intervals,
        "work_step_policy": work_algorithm_note(adaptive_algorithm) if work_mode else None,
        "ratio_shrink_count": int(ratio_shrink_count),
        "ratio_grow_count": int(ratio_grow_count),
        "ratio_step_reference": "current known acceleration versus persistent per-star reference; reference changes only with local level; no trials"
                                if adaptive and adaptive_algorithm == "acceleration_ratio" else None,
        "dt_min_used": float(min(step_sizes[1:])), "dt_max_used": float(max(step_sizes[1:])),
        "dt_mean_used": float(np.mean(step_sizes[1:])),
        "dt_mean_over_initial": float(np.mean(step_sizes[1:]) / base_dt),
        "minimum_allowed_dt": None if dop853_native or energy_band_mode else minimum_dt,
        "timestep_criterion": (
            "measured float64 system step energy band; same-start retries; accepted-node drift guard" if energy_band_mode else
            "persistent-reference acceleration power bands; cumulative exponent limit; global minimum"
            if adaptive and adaptive_algorithm == "acceleration_ratio"
            else f"normalized displacement work scale; {'accepted-history predicted acceleration/start velocity' if predicted_work_mode else 'mean endpoint acceleration/start velocity' if average_work_mode else 'start acceleration/start velocity' if aligned_work_mode else 'end acceleration/start velocity'}; rollback/retry both directions; K-cap exceedance marked" if work_mode
            else "DOP853 embedded 5/3 error estimator and native step control" if dop853_native
            else "pairwise gravity/flyby timescales" if adaptive else "fixed"),
        "energy_scale": float(energy_scale),
        "initial_view_extent": max(2 * float(np.max(np.abs(trajectories))), 1e-6) if orbit else 100,
        "initial_stars": [{"mass": s["mass"], "pos": s["pos"].tolist(), "v": s["v"].tolist()}
                          for s in initial_stars],
        "units": "Original program code units; no SI conversion specified",
        "positions_axes": ["star", "step", "xyz"],
        "accelerations_axes": ["step", "star", "xyz"],
        "velocities_axes": ["step", "star", "xyz"],
        "diagnostic_sampling": sample_every,
        "diagnostic_interval": None if step_sampled else sample_every * base_dt,
        "diagnostic_sampling_policy": "every N accepted steps, plus final state; no integration clipping"
                                      if step_sampled else "fixed physical-time intervals",
        "config": config, "elapsed_seconds": time.perf_counter() - started,
    }
    if energy_band_mode:
        report = dict(**{key: config[key] for key in ENERGY_BAND_DEFAULTS},
                      accepted_steps=steps, trial_evaluations=len(energy_band_trials),
                      rejected_trials=len(energy_band_trials)-steps,
                      rejection_counts=energy_band_rejections,
                      integrator_order=INTEGRATOR_ORDERS[config["method"]],
                      prediction_power=INTEGRATOR_ORDERS[config["method"]]+1,
                      energy_precision="float64_snapshot", units="absolute code energy",
                      guard_scope="accepted nodes", drift_guard_enabled=config["energy_drift_limit"] > 0,
                      dt_role="initial trial; no initial-dt upper bound", max_trials_per_start=64)
        metadata.update(step_energy_report=report, step_energy_trials=energy_band_trials,
                        step_energy_trial_evaluations=len(energy_band_trials),
                        step_energy_rejected_trials=len(energy_band_trials)-steps,
                        step_energy_rejection_counts=energy_band_rejections,
                        **{key: config[key] for key in ENERGY_BAND_DEFAULTS})
    if dop853_mode:
        metadata.update(dop853_rtol=config["dop853_rtol"] if dop853_native else None,
                        dop853_atol=config["dop853_atol"] if dop853_native else None,
                        dop853_native_control_enabled=dop853_native,
                        dop853_error_norms=dop853.accepted_error_norms,
                        dop853_rhs_evaluations=dop853.nfev,
                        dop853_trial_evaluations=dop853.trial_evaluations,
                        dop853_scipy_version=dop853.version,
                        dop853_execution=dop853.execution_description,
                        dop853_step_policy="initial trial=min(dt,T); max_step=dt; native error control" if dop853_native
                                           else f"pure eighth-order tableau; {adaptive_algorithm} supplies trial steps; native error estimation/rejection disabled" if adaptive
                                           else "fixed eighth-order tableau; no error estimation or rejection; final step clipped",
                        dop853_tolerance_scope="local position/velocity error norm; not a global or energy error bound" if dop853_native else None)
    # High-precision endpoint diagnostics are outside the compute timer too.
    if taylor_mode:
        metadata.update(taylor.core.report())
        metadata["taylor_nonfinite_rejections"] = taylor_numerical_rejections
        metadata["taylor_nonfinite_guard_enabled"] = work_mode and taylor.order != 45
    # Post-processing is deliberately outside the integration compute timer.
    metadata["energy_error"] = time_weighted_energy_error(times, energies, energy_scale)
    metadata["energy_drift"] = fit_energy_drift(times, energies, energy_scale)
    if is_kepler_config(config):
        metadata["position_error"] = preset_position_error(np.asarray(times), np.asarray(trajectories),
            config["binary_eccentricity"])
    elif len(stars) == 2:
        try:
            metadata["position_error"] = two_body_position_error(np.asarray(times), trajectories,
                np.asarray(velocities), initial_stars, G)
        except ValueError as exc:
            metadata["position_error"] = dict(reference_kind="unavailable", reason=str(exc))
    return dict(trajectories=trajectories, accelerations=np.asarray(accelerations),
                velocities=np.asarray(velocities), energies=np.asarray(energies), time_step=base_dt,
                diagnostic_times=np.asarray(diagnostic_times), diagnostics=diagnostics,
                metadata=metadata, record_times=np.asarray(times), step_sizes=np.asarray(step_sizes))



def physical_time_record_weights(times):
    """Normalized trapezoidal node weights: half each adjacent physical interval."""
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Time weighting requires at least two finite, strictly increasing time records.")
    span = float(times[-1] - times[0])
    if not np.isfinite(span) or span <= 0:
        raise ValueError("The physical-time fit interval must be finite and positive.")
    half_intervals = .5 * (np.diff(times) / span)
    weights = np.zeros(len(times), dtype=np.float64)
    weights[:-1] += half_intervals
    weights[1:] += half_intervals
    weights /= weights.sum()
    return weights


def fit_energy_drift(times, energies, energy_scale):
    """Fit all recorded energy offsets with physical-time weighted least squares.

    Each point represents half the interval on either side (endpoints: one side).
    These trapezoidal weights approximate a uniformly time-weighted squared
    residual; adding dense adaptive records does not give that interval extra
    weight merely because its record count increased. The intercept is free.
    The descriptive slope is not a conservation test or an uncertainty bound.
    """
    times = np.asarray(times, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    if (times.ndim != 1 or energies.shape != times.shape or len(times) < 2
            or not np.isfinite(times).all() or not np.isfinite(energies).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Drift analysis requires at least two finite, strictly increasing time records.")
    weights = physical_time_record_weights(times)
    span = float(times[-1] - times[0])
    # Subtract first to avoid losing small offsets when fitting a large baseline.
    offsets = energies - energies[0]
    scaled_times = (times - times[0]) / span
    mean_time = float(np.dot(weights, scaled_times))
    mean_offset = float(np.dot(weights, offsets))
    centered_times = scaled_times - mean_time
    centered_offsets = offsets - mean_offset
    denominator = float(np.dot(weights, centered_times**2))
    if denominator <= 0:
        raise ValueError("Recorded times have insufficient weighted spread for a fit.")
    delta = float(np.dot(weights * centered_times, centered_offsets) / denominator)
    slope = delta / span
    intercept = float(mean_offset - delta * mean_time)
    fitted = intercept + delta * scaled_times
    residuals = offsets - fitted
    square_scale = max(float(np.max(np.abs(residuals))), float(np.max(np.abs(centered_offsets))))
    if square_scale > np.sqrt(np.finfo(float).max) / 4:
        # A finite energy around 1e156 has an unrepresentable squared value.
        # Scale BOTH sums by the same factor, so their R-squared ratio is
        # unchanged; restore units only after taking the residual square root.
        residual_mse_unit = float(np.dot(weights, (residuals / square_scale)**2))
        sst_unit = float(np.dot(weights, (centered_offsets / square_scale)**2))
        residual_rmse = square_scale * float(np.sqrt(residual_mse_unit))
        r_squared = float(1 - residual_mse_unit / sst_unit) if sst_unit > 0 else None
    else:
        # Preserve historical floating-point results in the ordinary range.
        residual_mse = float(np.dot(weights, residuals**2))
        residual_rmse = float(np.sqrt(residual_mse))
        sst = float(np.dot(weights, centered_offsets**2))
        r_squared = float(1 - residual_mse / sst) if sst > 0 else None
    scale = float(energy_scale)
    normalized_slope = slope / scale if np.isfinite(scale) and scale > 0 else None
    normalized_delta = delta / scale if np.isfinite(scale) and scale > 0 else None
    return dict(version=2, method="physical-time weighted least squares; free intercept",
                sampling="all saved records; no random sampling",
                weighting="normalized trapezoidal node weights: half each adjacent physical-time interval; endpoints one side",
                objective="sum_i w_i * (E_i-E_0-intercept-slope*(t_i-t_0))^2",
                fit_records=int(len(times)), weight_sum=float(weights.sum()),
                source_records=int(len(times)), time_start=float(times[0]), time_end=float(times[-1]),
                initial_energy=float(energies[0]), energy_scale=scale if np.isfinite(scale) else None,
                slope=slope, intercept_offset_at_start=intercept,
                fitted_energy_change=delta, normalized_slope=normalized_slope,
                normalized_fitted_energy_change=normalized_delta,
                residual_rmse=residual_rmse, r_squared=r_squared,
                slope_units="code energy / code time", normalized_slope_units="1 / code time")


def energy_drift_summary(result):
    """Keep summaries compact, including when reading legacy sampled fits."""
    return {key: value for key, value in result.items()
            if key not in ("sample_times", "sample_energy_offsets")}


def plot_sample_indices(values, max_points=4000):
    """Keep endpoints and each display bucket's extrema; recordings stay intact."""
    count = len(values)
    if count <= max_points:
        return np.arange(count)
    width = int(np.ceil((count - 2) / ((max_points - 2) // 2)))
    keep = [0, count - 1]
    for start in range(1, count - 1, width):
        chunk = values[start:min(start + width, count - 1)]
        keep.extend((start + int(np.argmin(chunk)), start + int(np.argmax(chunk))))
    return np.unique(keep)


def energy_display_records(times, energies):
    """Add shared interpolated zero-crossing vertices for both E and delta-E views.

    These vertices are display interpolation, not additional integration states.
    """
    t, e = np.asarray(times), np.asarray(energies)
    d = e - e[0]
    crossings = np.flatnonzero(((d[:-1] < 0) & (d[1:] > 0)) |
                              ((d[:-1] > 0) & (d[1:] < 0)))
    fraction = np.abs(d[crossings]) / (np.abs(d[crossings]) + np.abs(d[crossings+1]))
    roots = t[crossings] + fraction * (t[crossings+1] - t[crossings])
    return (np.insert(t, crossings+1, roots),
            np.insert(e, crossings+1, e[0]), np.insert(d, crossings+1, 0.))


def viewport_series(times, values, limits, sampling_values=None, preserve_zero=False, steps=False):
    """Full records inside a zoomed time window, plus one neighbor on either side."""
    t, y = np.asarray(times), np.asarray(values)
    sample = y if sampling_values is None else np.asarray(sampling_values)
    lo, hi = sorted(limits)
    start = max(0, int(np.searchsorted(t, lo, side="left")) - 1)
    stop = min(len(t), int(np.searchsorted(t, hi, side="right")) + 1)
    if lo <= t[0] and hi >= t[-1]:
        finite = np.isfinite(sample)
        indices = plot_sample_indices(np.where(finite, sample, 0.))
        if steps:
            changes = np.flatnonzero(sample[1:] != sample[:-1])
            indices = np.unique(np.r_[0, len(t)-1, changes, changes+1])
        gaps = np.flatnonzero(finite[1:] != finite[:-1])
        keep = [indices, gaps, gaps+1]
        if preserve_zero:
            # Retain zero-region boundaries and their neighbors without keeping a
            # million identical zero samples in a flat full-range overview.
            zero = sample == 0
            edges = np.flatnonzero(zero[1:] != zero[:-1])
            keep.extend([edges, edges+1])
        indices = np.unique(np.concatenate(keep))
    else:
        indices = np.arange(start, stop)
    return t[indices], y[indices]


def full_time_statistics(values, times):
    """Return extrema over all recorded samples and the trapezoidal time average.

    Plot decimation and zoom do not affect these statistics. Time weighting
    avoids over-weighting densely sampled intervals.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    duration = times[-1] - times[0]
    mean = (float(np.trapezoid(values, times) / duration)
            if len(times) > 1 and duration > 0 else float(values[0]))
    return {"max": float(np.max(values)), "min": float(np.min(values)), "mean": mean}


def statistics_label(stats, quantity):
    if quantity == "ΔE":
        # Show small signed differences directly, without the large E(0) baseline.
        return (f"Max ΔE: {stats['max']:+.7e}\nMin ΔE: {stats['min']:+.7e}"
                f"\nTime mean ΔE: {stats['mean']:+.7e}")
    return (f"Max: {stats['max']:.5e}\nMin: {stats['min']:.5e}"
            f"\nTime mean: {stats['mean']:.5e}")


class SharedTimeline(ttk.Frame):
    """One control for computation progress, playback time, and GIF export progress."""
    def __init__(self, parent):
        super().__init__(parent, padding=(12, 2, 12, 5))
        self.columnconfigure(1, weight=1)
        self.viewer = None
        self.mode = "idle"
        self.updating = False
        self.mode_var = tk.StringVar(value="Progress")
        self.value = tk.DoubleVar(value=0)
        self.message = tk.StringVar(value="Choose parameters and click Start / Rerun.")
        ttk.Label(self, textvariable=self.mode_var, width=13).grid(row=0, column=0, sticky="w")
        self.scale = ttk.Scale(self, from_=0, to=100, variable=self.value,
                               orient="horizontal", command=self.seek, state="disabled")
        self.scale.grid(row=0, column=1, sticky="ew")
        ttk.Label(self, textvariable=self.message, width=1, style="Note.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="ew")
        self.scale.bind("<ButtonPress-1>", self.press)
        self.scale.bind("<ButtonRelease-1>", self.release)

    def set_value(self, value):
        self.updating = True
        try:
            self.value.set(value)
        finally:
            self.updating = False

    def begin_computation(self):
        self.mode = "computing"
        self.mode_var.set("Computing")
        self.scale.configure(to=100, state="disabled")
        self.set_value(0)
        if self.viewer is not None:
            self.viewer.save_animation_button.configure(state="disabled")

    def attach(self, viewer):
        self.viewer = viewer
        if viewer is None:
            self.mode = "idle"
            self.mode_var.set("Progress")
            self.scale.configure(to=100, state="disabled")
            self.set_value(0)
            return
        self.mode = "playback"
        self.mode_var.set("Playback time")
        self.scale.configure(to=float(viewer.times[-1]), state="normal")
        viewer.save_animation_button.configure(state="normal")
        self.update_frame(viewer)

    def detach(self, viewer):
        if self.viewer is viewer:
            self.viewer = None
            if self.mode == "playback":
                self.attach(None)

    def update_frame(self, viewer):
        if self.viewer is viewer and self.mode == "playback":
            self.set_value(float(viewer.times[viewer.step]))
            self.message.set(viewer.time_var.get())

    def show_status(self, viewer):
        if self.viewer is viewer and self.mode == "playback":
            self.message.set(viewer.status_var.get())

    def begin_export(self, viewer):
        if self.viewer is viewer and self.mode == "playback":
            self.mode = "export"
            self.mode_var.set("Saving GIF")
            self.scale.configure(to=100, state="disabled")
            self.set_value(0)

    def update_export(self, viewer, current, total):
        if self.viewer is viewer and self.mode == "export":
            self.set_value(100 * (current + 1) / total)
            self.message.set(viewer.status_var.get())

    def finish_export(self, viewer):
        if self.viewer is viewer and self.mode == "export":
            self.attach(viewer)
            self.show_status(viewer)

    def seek(self, value):
        if not self.updating and self.mode == "playback" and self.viewer is not None:
            self.viewer.on_seek(value)

    def press(self, event):
        if self.mode == "playback" and self.viewer is not None:
            self.viewer.on_slider_press(event)

    def release(self, event):
        if self.viewer is not None:
            self.viewer.on_slider_release(event)


class NBodyViewer(PositionAnalysisMixin):
    def __init__(self, trajectories, accelerations, energies, time_step,
                 diagnostic_times, diagnostics, metadata, root=None,
                 record_times=None, step_sizes=None, velocities=None, timeline=None):
        # trajectories[star, step, xyz]; accelerations[step, star, xyz].
        self.trajectories = np.asarray(trajectories)
        self.accelerations = np.asarray(accelerations)
        self.energies = np.asarray(energies)
        self.velocities = np.asarray(velocities, dtype=float)
        self.dt = time_step
        self.n, self.frame_count, _ = self.trajectories.shape
        self.last_step = self.frame_count - 1
        self.times = (np.arange(self.frame_count) * self.dt if record_times is None
                      else np.asarray(record_times, dtype=float))
        self.step_sizes = (np.r_[0.0, np.diff(self.times)] if step_sizes is None
                           else np.asarray(step_sizes, dtype=float))
        self.diagnostic_times = np.asarray(diagnostic_times)
        self.diagnostics = diagnostics
        self.metadata = metadata
        # Missing normalization identifies legacy runs with a dimensional threshold.
        self.work_normalized = metadata.get("work_indicator_normalization") == "initial_energy_scale"
        definition = metadata.get("work_indicator_definition")
        self.work_algorithm_code = work_algorithm_from_definition(definition, metadata.get("adaptive_algorithm"))
        self.work_algorithm_name = WORK_HISTORY_NAMES.get(self.work_algorithm_code, "Work scale")
        self.work_legacy_timing = definition in (WORK_LEGACY_DISPLACEMENT_DEFINITION, WORK_LEGACY_SYMMETRIC_DEFINITION, WIDE_DEFINITION)
        self.work_predicted = definition in (WORK_PREDICTED_DEFINITION, WORK_PREDICTED_LEGACY_DEFINITION)
        self.work_predicted_legacy = definition == WORK_PREDICTED_LEGACY_DEFINITION
        self.work_average = definition == WORK_AVERAGE_DEFINITION
        self.work_symmetric = definition in (WORK_SYMMETRIC_DEFINITION, WORK_LEGACY_SYMMETRIC_DEFINITION, WIDE_DEFINITION)
        self.work_displacement = definition in (WORK_PREDICTED_DEFINITION, WORK_PREDICTED_LEGACY_DEFINITION, WORK_AVERAGE_DEFINITION, WORK_DISPLACEMENT_DEFINITION, WORK_SYMMETRIC_DEFINITION,
                                               WORK_LEGACY_DISPLACEMENT_DEFINITION, WORK_LEGACY_SYMMETRIC_DEFINITION, WIDE_DEFINITION)
        self.work_timing_note = (("a_pred=a_start+0.5*(a_start-a_previous); historical unscaled prediction" if self.work_predicted_legacy else
                                 "a_pred=a_start+0.5*(dt/previous_dt)*(a_start-a_previous); actual candidate/accepted intervals") +
                                "; accepted history; first step a_start") if self.work_predicted else "a_bar=(a_start+a_end)/2; start velocity" if self.work_average else (("Qf uses end a/start v; Qr uses start a/end v" if self.work_symmetric else
                                  "VA0 = |start velocity along end acceleration|") if self.work_legacy_timing else
                                 "Qf uses start a/start v; Qr uses end a/end v" if self.work_symmetric else
                                 "A0 = start |a|; VA0 = |start velocity along start acceleration|")
        self.work_indicator_formula = "m (|a_pred dot v_start| dt + 0.5 |a_pred|² dt²) / S_E" if self.work_predicted else "m (|a_bar dot v_start| dt + 0.5 |a_bar|² dt²) / S_E" if self.work_average else ("max(Q forward, Q reverse)" if self.work_symmetric else
                                       ("m A (VA0 dt + 0.5 A dt²) / S_E" if self.work_legacy_timing else
                                        "m A0 (VA0 dt + 0.5 A0 dt²) / S_E") if self.work_displacement else
                                       "m |a dot v| dt / S_E" if self.work_normalized else "|a dot v| dt")
        self.work_indicator_units = "dimensionless" if self.work_normalized else "code L²/T²"
        self.work_masses = np.asarray([star["mass"] for star in metadata["initial_stars"]])
        self.adaptive = metadata.get("adaptive_enabled", False)
        if self.adaptive and record_times is None:
            raise ValueError("Adaptive mode requires actual cumulative times, not step count multiplied by a fixed dt.")
        if not self.adaptive:
            self.mode_description = f"Fixed dt = {self.dt:g}"
        elif metadata.get("adaptive_algorithm", "timescale") == "acceleration_ratio":
            self.mode_description = (f"Acceleration ratio (retained reference) · N = {metadata.get('acceleration_trigger_ratio', 2)}"
                                     f" · max exponent K = {metadata.get('acceleration_max_power', 3)}"
                                     f" · dt max = {self.dt:g}")
        elif metadata.get("adaptive_algorithm") == "dop853":
            self.mode_description = (f"DOP853 native control · rtol={metadata['dop853_rtol']:g}"
                                     f" · atol={metadata['dop853_atol']:g} · dt max={self.dt:g}")
        elif metadata.get("adaptive_algorithm") == "step_energy_band":
            self.mode_description = energy_band_summary(metadata) + f"; initial dt={self.dt:g}"
        elif metadata.get("adaptive_algorithm") == "energy_ratio":
            self.mode_description = (f"Energy ratio · tolerance = {100*metadata.get('energy_relative_tolerance', 0.1):.6g}% · halve and retry · dt max = {self.dt:g}"
                                     f" · max rejections/step = {metadata.get('energy_max_rejections', 'NA')}"
                                     f" · rejected trials = {metadata.get('discarded_trial_steps', 0)}")
            if metadata.get("energy_minimum_dt") is not None:
                self.mode_description += f" · global dt min = {metadata['energy_minimum_dt']:.6g}"
        elif metadata.get("adaptive_algorithm") in WORK_HISTORY_CODES:
            self.mode_description = (f"max {self.work_indicator_formula} ≤ {metadata.get('work_threshold', 1.0):g} · N = {metadata['work_step_factor']:g}"
                                     f" · K = {metadata['work_max_power']} · dt max = {self.dt:g}"
                                     f" · rejected trials = {metadata.get('discarded_trial_steps', 0)}")
        else:
            self.mode_description = f"Time-scale method · dt max = {self.dt:g} · eta = {metadata['adaptive_eta']:g}"
        precision = "float64 plots and saved states" if metadata.get("method_id") in TAYLOR_METHOD_ORDERS and metadata.get("compute_backend") != "cpu_array" else "float64"
        self.mode_description += f" · {backend_summary(metadata)} · {precision}"
        if self.adaptive and metadata.get("adaptive_algorithm") in WORK_HISTORY_CODES:
            self.mode_description = self.work_algorithm_name + " · " + self.mode_description
        if metadata.get("adaptive_algorithm") == WIDE_ALGORITHM:
            self.mode_description += (f" · protection before/after = {metadata['work_protection_before']:g} / "
                                      f"{metadata['work_protection_after']:g} code time · next-step recovery")
        if (self.times.shape != (self.frame_count,) or not np.isfinite(self.times).all()
                or self.times[0] != 0 or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Cumulative times must start at zero, increase strictly, and match the trajectory records.")
        if (self.step_sizes.shape != (self.frame_count,) or not np.isfinite(self.step_sizes).all()
                or self.step_sizes[0] != 0 or np.any(self.step_sizes[1:] <= 0)):
            raise ValueError("Actual step sizes must be positive and match trajectory records; zero marks the initial state.")
        # Every accepted step counts once; the initial zero is not an integration step.
        # Recompute from records so older saved runs receive the same statistics.
        self.metadata["dt_mean_used"] = float(np.mean(self.step_sizes[1:]))
        self.metadata["dt_mean_over_initial"] = self.metadata["dt_mean_used"] / self.dt
        self.timestep_summary = (f"Mean actual dt = {self.metadata['dt_mean_used']:.6g} | "
                                 f"mean / initial dt = {self.metadata['dt_mean_over_initial']:.6g}x "
                                 "(all completed steps)")
        if self.metadata.get("work_preprobe_enabled"):
            self.timestep_summary += (f" | Pre-probe ON: {self.metadata.get('work_probe_window_count', 0)} windows, "
                                      f"{self.metadata.get('work_probe_trial_evaluations', 0):,} extra probe trials")
        if self.work_algorithm_code == WIDE_ALGORITHM:
            self.timestep_summary += (f"\nProtection before/after = {metadata['work_protection_before']:g} / "
                                      f"{metadata['work_protection_after']:g} code time; next-step recovery")
        if self.accelerations.shape != (self.frame_count, self.n, 3):
            raise ValueError("Acceleration records must match every star and trajectory time.")
        if self.velocities.shape != (self.frame_count, self.n, 3):
            raise ValueError("Velocity records must match every star and trajectory time.")
        if self.energies.shape != (self.frame_count,):
            raise ValueError("Energy records must match trajectory times.")
        self.energy_error = time_weighted_energy_error(self.times, self.energies, metadata["energy_scale"])
        self.metadata["energy_error"] = self.energy_error
        self.energy_plot_times, self.energy_plot_values, self.energy_plot_offsets = energy_display_records(
            self.times, self.energies)
        self.energy_display_offset = None
        self.energy_display_scale = None
        self.time_series = {}
        self.viewport_axes = set()
        self.updating_viewport = False

        self.root = root if root is not None else tk.Tk()
        width = min(1600, self.root.winfo_screenwidth() - 60)
        if isinstance(self.root, tk.Tk):
            self.root.title("N-body simulation · Trajectories, energy and net acceleration")
            height = min(1000, self.root.winfo_screenheight() - 100)
            self.root.geometry(f"{width}x{height}")
            self.root.minsize(min(1000, width), min(700, height))
            self.root.protocol("WM_DELETE_WINDOW", self.close)
        # Embedded viewers use the application's shared timeline; standalone viewers own one.
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.closed = False
        self.step = 0
        self.star_index = 0
        self.fit_particles_active = False
        self.follow_view_active = False
        self.fixed_view_limits = None
        self.playing = False
        self.playback_time = 0.0
        self.playback_last_clock = None
        self.dragging = False
        self.updating_slider = False
        self.after_id = None
        self.mouse_press = None
        self.drag_axis = None
        self.hover_selection = None
        self.hover_items = {}
        self.page_readouts = {}
        self.page_readout_widgets = {}
        self.page_viewports = {}
        self.plot_headers = {}
        self.hover_idle_text = "Hover over a plot to inspect its values."
        self.selected_magnitudes = None
        self.derivative_star_index = None

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=6)
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Note.TLabel", foreground="#52606d", font=("Microsoft YaHei UI", 9))
        style.configure("Status.TLabel", foreground="#235b8c")

        header = ttk.Frame(self.root, padding=(12, 4))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        title_label = ttk.Label(header, text="N-body simulation: trajectories, energy and acceleration",
                                style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky="w")
        self.save_animation_button = ttk.Button(
            header, text="Save animation (GIF)", command=self.save_animation)
        self.save_animation_button.grid(row=0, column=1, sticky="ne", padx=(14, 0))
        description_label = ttk.Label(
            header, text=f"{metadata['case_name']}  |  {metadata['method_name']}  |  "
            f"N = {self.n}, {self.mode_description}, {self.last_step:,} steps",
            style="Note.TLabel", wraplength=width - 55)
        description_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        def resize_header(event):
            title_label.configure(wraplength=max(
                240, event.width - self.save_animation_button.winfo_reqwidth() - 50))
            description_label.configure(wraplength=max(400, event.width - 36))
        header.bind("<Configure>", resize_header)
        body = ttk.Frame(self.root, padding=(16, 0, 12, 8))
        body.grid(row=1, column=0, sticky="nsew")
        controls = self.controls = ttk.Frame(body, width=270)
        controls.pack(side="left", fill="y", padx=(0, 12))
        controls.pack_propagate(False)
        scroll = tk.Canvas(controls, width=270, highlightthickness=0,
                           background=style.lookup("TFrame", "background"))
        scrollbar = ttk.Scrollbar(controls, orient="vertical", command=scroll.yview)
        scrollbar.pack(side="right", fill="y")
        scroll.pack(side="left", fill="both", expand=True)
        scroll.configure(yscrollcommand=scrollbar.set)
        sidebar = ttk.Frame(scroll, padding=(0, 0, 8, 8))
        sidebar_window = scroll.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", lambda event: scroll.configure(scrollregion=scroll.bbox("all")))
        scroll.bind("<Configure>", lambda event: scroll.itemconfigure(sidebar_window, width=event.width))

        self.play_button = ttk.Button(sidebar, text="Play", command=self.toggle_play)
        self.play_button.pack(fill="x", pady=(4, 3))
        stepping = ttk.Frame(sidebar)
        stepping.pack(fill="x", pady=3)
        ttk.Button(stepping, text="Previous step", command=lambda: self.single_step(-1)).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(stepping, text="Next step", command=lambda: self.single_step(1)).pack(
            side="left", fill="x", expand=True)
        ttk.Button(sidebar, text="Return to start", command=self.rewind).pack(fill="x", pady=3)
        self.playback_speed_var = tk.StringVar(value="1")
        ttk.Label(sidebar, text="Playback speed (×)").pack(anchor="w", pady=(7, 2))
        ttk.Spinbox(sidebar, from_=.05, to=100, increment=.05,
                    textvariable=self.playback_speed_var).pack(fill="x")
        ttk.Label(sidebar, text="1×: full duration in 60 seconds", style="Note.TLabel").pack(anchor="w")
        self.loop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sidebar, text="Loop playback", variable=self.loop_var).pack(anchor="w", pady=4)

        ttk.Separator(sidebar).pack(fill="x", pady=10)
        ttk.Label(sidebar, text="Select a star (or click an orbit marker)").pack(anchor="w")
        self.star_var = tk.StringVar(value="Star 1")
        self.star_combo = ttk.Combobox(sidebar, textvariable=self.star_var, state="readonly",
                                      values=[f"Star {i + 1}" for i in range(self.n)])
        self.star_combo.pack(fill="x", pady=(4, 6))
        self.star_combo.bind("<<ComboboxSelected>>",
                             lambda event: self.select_star(self.star_combo.current()))
        ttk.Label(sidebar, text="Camera mode").pack(anchor="w", pady=(4, 2))
        self.camera_mode_var = tk.StringVar(value="Fixed view")
        self.camera_combo = ttk.Combobox(
            sidebar, textvariable=self.camera_mode_var, state="readonly",
            values=["Fixed view", "Follow selected star", "Fit all current particles"])
        self.camera_combo.pack(fill="x", pady=(0, 4))
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_mode_change)
        ttk.Label(sidebar, text="Follow centers the selected star. Fit all current particles adjusts the bounds at each time; rotation remains available.",
                  style="Note.TLabel", wraplength=260).pack(anchor="w", pady=(0, 5))
        self.details_var = tk.StringVar()
        ttk.Label(sidebar, textvariable=self.details_var, justify="left",
                  style="Note.TLabel", wraplength=260).pack(anchor="w", pady=4)
        self.components_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sidebar, text="Acceleration: show x, y, z components",
                        variable=self.components_var, command=self.refresh_vectors).pack(
                            anchor="w", pady=(6, 3))
        self.delta_energy_var = tk.BooleanVar(value=False)
        self.energy_scale_var = tk.StringVar(value="linear")
        ttk.Button(sidebar, text="Fit all trajectories", command=self.fit_orbits).pack(fill="x", pady=3)
        ttk.Label(sidebar,
                  text="Click a star to pause. Drag the playback slider to inspect another time.\n"
                       "Click or drag the yellow time cursor; hover to read values.\n"
                       "Time, position, energy, acceleration and velocity use code units.",
                  style="Note.TLabel", wraplength=260).pack(anchor="w", pady=(6, 3))

        ttk.Separator(sidebar).pack(fill="x", pady=10)
        ttk.Button(sidebar, text="Save current page (PNG)", command=self.save_image).pack(fill="x", pady=3)
        ttk.Button(sidebar, text="Save all-star data (NPZ)", command=self.save_data).pack(fill="x", pady=3)
        ttk.Label(sidebar, text="Full-range plots use fewer points. Zooming in time shows every record in view; hover and exports use original records.",
                  style="Note.TLabel", wraplength=260).pack(anchor="w", pady=5)

        display = self.display = ttk.Frame(body)
        display.pack(side="left", fill="both", expand=True)
        energy_controls = ttk.Frame(display)
        energy_controls.pack(fill="x", pady=(0, 5))
        ttk.Label(energy_controls, text="Energy display:").pack(side="left", padx=(0, 8))
        self.energy_value_radio = ttk.Radiobutton(
            energy_controls, text="Total energy E(t)", variable=self.delta_energy_var,
            value=False, command=self.refresh_energy)
        self.energy_value_radio.pack(side="left", padx=(0, 16))
        self.energy_delta_radio = ttk.Radiobutton(
            energy_controls, text="Offset from initial value: ΔE = E(t) − E(0)",
            variable=self.delta_energy_var, value=True, command=self.refresh_energy)
        self.energy_delta_radio.pack(side="left")
        ttk.Label(energy_controls, text="Y axis:").pack(side="left", padx=(16, 4))
        self.energy_linear_radio = ttk.Radiobutton(
            energy_controls, text="Linear", variable=self.energy_scale_var,
            value="linear", command=self.refresh_energy)
        self.energy_linear_radio.pack(side="left")
        self.energy_log_radio = ttk.Radiobutton(
            energy_controls, text="Log (symlog)", variable=self.energy_scale_var,
            value="symlog", command=self.refresh_energy)
        self.energy_log_radio.pack(side="left", padx=(4, 0))
        self.notebook = ttk.Notebook(display)
        self.notebook.pack(fill="both", expand=True)
        self.overview_tab = ttk.Frame(self.notebook)
        self.diagnostics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="Trajectory · Energy · Acceleration")
        self.notebook.add(self.diagnostics_tab, text="Conservation & Errors")
        self.build_overview()
        self.build_diagnostics()
        self.build_energy_drift()
        self.build_acceleration_derivative()
        self.build_position_analysis()
        self.mark_work_limit_intervals()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.owns_timeline = timeline is None
        self.timeline = SharedTimeline(self.root) if timeline is None else timeline
        if self.owns_timeline:
            self.timeline.grid(row=2, column=0, sticky="ew")
        self.slider = self.timeline.scale
        self.time_var = tk.StringVar(master=self.root)
        self.status_var = tk.StringVar(master=self.root, value="Computation complete.")
        self.status_trace = self.status_var.trace_add(
            "write", lambda *args: self.timeline.show_status(self))
        self.refresh_energy()
        self.refresh_vectors()
        self.draw_frame(0)
        if self.owns_timeline:
            self.timeline.attach(self)
        # Initialize the 3D projection before constrained layout measures its labels.
        self.orbit_axes.draw(self.canvas.get_renderer())
        self.canvas.draw()
        self.diagnostic_canvas.draw()
        self.after_id = self.root.after(40, self.tick)




    def mark_work_limit_intervals(self):
        intervals = self.metadata.get("work_exceeded_intervals", [])
        if not intervals:
            return
        self.work_limit_spans = {}
        for axis in self.hover_items:
            spans = [axis.axvspan(item["start_time"], item["end_time"], facecolor="#ef4444",
                                 edgecolor="#b91c1c", linewidth=.5, alpha=.18, zorder=.4,
                                 label="_nolegend_")
                     for item in intervals]
            self.work_limit_spans[axis] = spans
        # One bottom legend per figure explains the marked time intervals.
        for figure in {axis.figure for axis in self.work_limit_spans}:
            handle = next(spans[0] for axis, spans in self.work_limit_spans.items()
                          if axis.figure is figure)
            figure.legend([handle], ["Red shading: threshold exceeded at minimum step (K limit)"],
                          loc="outside lower center", frameon=False, fontsize=8)

    def embed_figure(self, figure, parent):
        # The plot uses the available viewport exactly; no oversized scrollable canvas.
        variable = tk.StringVar(value=self.hover_idle_text)
        self.page_readouts[figure] = variable
        area = ttk.Frame(parent)
        area.pack(side="top", fill="both", expand=True)
        area.rowconfigure(0, weight=1)
        area.columnconfigure(0, weight=1)
        viewport = tk.Canvas(area, highlightthickness=0, background="white")
        viewport.grid(row=0, column=0, sticky="nsew")
        canvas = FigureCanvasTkAgg(figure, master=viewport)
        widget = canvas.get_tk_widget()
        window = viewport.create_window(0, 0, window=widget, anchor="nw")
        self.page_viewports[figure] = viewport

        def resize(event):
            # Hidden notebook pages briefly report 1 px; retain their last usable layout.
            if event.width <= 1 or event.height <= 1:
                return
            self.apply_figure_layout(figure, event.width, event.height)
            viewport.itemconfigure(window, width=event.width, height=event.height)
            viewport.configure(scrollregion=(0, 0, event.width, event.height))
            self.update_page_readout(figure)
        viewport.bind("<Configure>", resize)
        inspector = tk.Label(viewport, textvariable=variable, font=("Consolas", 9),
                             anchor="ne", justify="right", padx=7, pady=5,
                             background="#f3f6fa", foreground="#172435",
                             borderwidth=1, relief="solid")
        self.page_readout_widgets[figure] = inspector
        variable.trace_add("write", lambda *args: self.update_page_readout(figure))
        inspector.bind("<Leave>", lambda event: self.hide_page_readout(figure))
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.pack(side="bottom", fill="x", before=area)
        canvas.mpl_connect("motion_notify_event", self.on_plot_motion)
        canvas.mpl_connect("figure_leave_event", self.clear_hover)
        canvas.mpl_connect("button_press_event", self.on_plot_press)
        canvas.mpl_connect("button_release_event", self.on_plot_release)
        return canvas, toolbar

    def set_time_series(self, line, times, values, sampling_values=None, preserve_zero=False):
        axis = line.axes
        if axis.get_autoscalex_on():
            axis.set_xlim(times[0], times[-1])
        steps = line.get_drawstyle().startswith("steps")
        self.time_series[line] = (np.asarray(times), np.asarray(values), sampling_values, preserve_zero, steps)
        if axis not in self.viewport_axes:
            axis.callbacks.connect("xlim_changed", self.on_time_view_changed)
            self.viewport_axes.add(axis)
        line.set_data(*viewport_series(times, values, axis.get_xlim(), sampling_values, preserve_zero, steps))
        line.get_path().should_simplify = False

    def on_time_view_changed(self, changed_axis):
        if self.updating_viewport:
            return
        self.updating_viewport = True
        try:
            siblings = changed_axis.get_shared_x_axes().get_siblings(changed_axis)
            # Shared axes callbacks can run before siblings receive their limits.
            # Use the emitting axis's limits for every curve in that shared group.
            for line, (times, values, sample, zero, steps) in self.time_series.items():
                if line.axes in siblings:
                    line.set_data(*viewport_series(times, values, changed_axis.get_xlim(), sample, zero, steps))
                    line.get_path().should_simplify = False
            changed_axis.figure.canvas.draw_idle()
        finally:
            self.updating_viewport = False

    def apply_figure_layout(self, figure, width, height, for_export=False):
        """Use compact screen text while keeping the original font sizes for export."""
        diagnostic = figure is getattr(self, "diagnostic_figure", None)
        base_width, base_height = (1220, 780) if diagnostic else (1000, 650)
        scale = (1.0 if for_export else
                 max(.82, min(1.0, width / base_width, height / base_height)))
        text_items = list(figure.texts)
        for legend in figure.legends:
            text_items.extend(legend.get_texts())
            text_items.append(legend.get_title())
        for axis in figure.axes:
            text_items.extend(axis.texts)
            text_items.append(axis.title)
            for axis_name in ("xaxis", "yaxis", "zaxis"):
                coordinate = getattr(axis, axis_name, None)
                if coordinate is None:
                    continue
                text_items.append(coordinate.label)
                if not hasattr(coordinate, "_nbody_full_tick_size"):
                    labels = coordinate.get_ticklabels()
                    coordinate._nbody_full_tick_size = labels[0].get_fontsize() if labels else 8.0
                size = coordinate._nbody_full_tick_size * scale
                coordinate.set_tick_params(labelsize=max(6.2, size))
                coordinate.offsetText.set_fontsize(max(6.2, size))
            legend = axis.get_legend()
            if legend is not None:
                text_items.extend(legend.get_texts())
                text_items.append(legend.get_title())
        for item in text_items:
            if not hasattr(item, "_nbody_full_fontsize"):
                item._nbody_full_fontsize = item.get_fontsize()
            item.set_fontsize(max(6.2, item._nbody_full_fontsize * scale))
        engine = figure.get_layout_engine()
        if engine is not None:
            engine.set(w_pad=.035 * scale, h_pad=.035 * scale, wspace=.035, hspace=.045)

    def build_overview(self):
        self.figure = Figure(figsize=(12, 7.5), dpi=100, facecolor="white", constrained_layout=True)
        if not self.adaptive:
            compact_mode = f"dt={self.dt:g}"
        elif self.metadata["adaptive_algorithm"] == "acceleration_ratio":
            compact_mode = (f"ratio N={self.metadata['acceleration_trigger_ratio']}, "
                            f"K={self.metadata['acceleration_max_power']}; dt_max={self.dt:g}")
        elif self.metadata["adaptive_algorithm"] == "dop853":
            compact_mode = (f"rtol={self.metadata['dop853_rtol']:g}, atol={self.metadata['dop853_atol']:g}; dt_max={self.dt:g}")
        elif self.metadata["adaptive_algorithm"] == "step_energy_band":
            compact_mode = energy_band_summary(self.metadata)
        elif self.metadata["adaptive_algorithm"] == "energy_ratio":
            compact_mode = (f"energy bound={100*self.metadata.get('energy_relative_tolerance', 0.1):.6g}%; "
                            f"max rejects/step={self.metadata.get('energy_max_rejections', 'NA')}; "
                            f"rejected={self.metadata.get('discarded_trial_steps', 0)}; dt_max={self.dt:g}")
        elif self.metadata["adaptive_algorithm"] in WORK_HISTORY_CODES:
            compact_mode = (f"max {self.work_indicator_formula} ≤ {self.metadata.get('work_threshold', 1.0):g}; N={self.metadata['work_step_factor']:g}, "
                            f"K={self.metadata['work_max_power']}; dt_max={self.dt:g}")
        else:
            compact_mode = f"eta={self.metadata['adaptive_eta']:g}; dt_max={self.dt:g}"
        self.figure_caption = (f"{self.metadata['case_name']}\n"
                               f"{self.metadata['method_name']} | N={self.n} | {compact_mode} | "
                               f"{backend_summary(self.metadata)} | steps={self.last_step:,}")
        self.figure_caption += "\n" + self.timestep_summary
        internal_energy = taylor_energy_summary(self.metadata)
        if internal_energy:
            self.figure_caption += "\n" + internal_energy
        if self.adaptive and self.metadata.get("adaptive_algorithm") in WORK_HISTORY_CODES:
            self.figure_caption = self.work_algorithm_name + " | " + self.figure_caption
        self.figure.suptitle(self.figure_caption, fontsize=9)
        grid = self.figure.add_gridspec(2, 2, width_ratios=(1.0, 1.45))
        self.orbit_axes = self.figure.add_subplot(grid[:, 0], projection="3d")
        energy_grid = grid[0, 1].subgridspec(2, 1, height_ratios=(.55, 1), hspace=.02)
        acceleration_grid = grid[1, 1].subgridspec(2, 1, height_ratios=(.55, 1), hspace=.02)
        self.energy_info_axes = self.figure.add_subplot(energy_grid[0])
        self.acceleration_info_axes = self.figure.add_subplot(acceleration_grid[0])
        self.energy_axes = self.figure.add_subplot(energy_grid[1])
        self.acceleration_axes = self.figure.add_subplot(acceleration_grid[1], sharex=self.energy_axes)
        for axis, info in ((self.energy_axes, self.energy_info_axes),
                           (self.acceleration_axes, self.acceleration_info_axes)):
            info.set_axis_off()
            self.plot_headers[axis] = info
        self.energy_title_text = self.energy_info_axes.text(0, .98, "Energy", va="top", fontsize=10)
        self.acceleration_title_text = self.acceleration_info_axes.text(0, .98, "Net acceleration", va="top", fontsize=10)
        self.orbit_axes.set(xlabel="x", ylabel="y", zlabel="z")
        self.orbit_axes.set_title("Trajectories", fontsize=10)
        self.orbit_axes.set_box_aspect((1, 1, 1))
        self.orbit_axes.set_facecolor("#f6f8fb")
        colors = plt.get_cmap("tab10")
        self.lines = []
        self.points = []
        for i in range(self.n):
            color = colors(i % 10)
            self.lines.append(self.orbit_axes.plot([], [], [], color=color, linewidth=1)[0])
            self.points.append(self.orbit_axes.plot([], [], [], "o", color=color,
                                                    markersize=6, label=f"Star {i + 1}")[0])
        self.highlight = self.orbit_axes.plot([], [], [], "o", markersize=12,
                                              markerfacecolor="none", markeredgecolor="#172435",
                                              markeredgewidth=1.8)[0]
        self.time_text = self.orbit_axes.text2D(.98, .98, "", transform=self.orbit_axes.transAxes,
                                               ha="right", va="top", fontsize=10)
        extent = self.metadata["initial_view_extent"]
        for setter in (self.orbit_axes.set_xlim, self.orbit_axes.set_ylim, self.orbit_axes.set_zlim):
            setter(-extent, extent)
        self.energy_line, = self.energy_axes.plot([], [], color="#172435", linewidth=1.3, label="E")
        self.energy_reference = self.energy_axes.axhline(self.energies[0], color="#8a94a3",
                                                         linestyle="--", linewidth=.8, label="Initial value")
        self.acceleration_lines = {}
        for key, color, label in (("magnitude", "#7c3aed", "|a|"),
                                   ("x", "#d62728", "a_x"), ("y", "#2ca02c", "a_y"),
                                   ("z", "#1f77b4", "a_z")):
            line, = self.acceleration_axes.plot([], [], color=color,
                                                linewidth=1.4 if key == "magnitude" else .8, label=label)
            self.acceleration_lines[key] = line
        self.energy_cursor = self.energy_axes.axvline(0, color="#de8a28", linewidth=1, alpha=.7)
        self.acceleration_cursor = self.acceleration_axes.axvline(0, color="#de8a28", linewidth=1, alpha=.7)
        for axis in (self.energy_axes, self.acceleration_axes):
            axis.set_xlabel("Time (code units)", fontsize=9)
            axis.tick_params(labelsize=8)
            axis.grid(alpha=.22)
            axis.set_xlim(0, self.times[-1])
        self.build_step_size_axes()
        # Place scientific-notation and offset labels at the right end of each y axis.
        self.energy_axes.yaxis.set_offset_position("right")
        self.acceleration_axes.yaxis.set_offset_position("right")
        self.acceleration_axes.set_ylabel("Net acceleration (code units)", fontsize=9)
        self.energy_statistics_text = self.energy_info_axes.text(
            1, .98, "", ha="right", va="top", fontsize=8, linespacing=1.2)
        self.acceleration_statistics_text = self.acceleration_info_axes.text(
            1, .98, "", ha="right", va="top", fontsize=8, linespacing=1.2)
        self.update_overview_legend(self.energy_axes)
        self.register_hover(self.energy_axes, self.times, {"E": self.energies})
        self.register_hover(self.acceleration_axes, self.times, {})
        self.canvas, self.toolbar = self.embed_figure(self.figure, self.overview_tab)

    def step_plot_data(self, axis=None):
        """Map h[n] to (t[n-1], t[n]]; omit the initial zero marker from the log axis."""
        return self.times, np.r_[self.step_sizes[1], self.step_sizes[1:]]

    def add_step_size_axis(self, axis, base, show_ratio_levels=False):
        times, steps = self.step_plot_data(axis)
        color = "#b45309"
        low, high = float(np.min(steps)), float(np.max(steps))
        padding = max((np.log(high) - np.log(low)) * .08, np.log(1.5))
        y_low = max(np.nextafter(0., 1.), np.exp(max(np.log(np.nextafter(0., 1.)), np.log(low) - padding)))
        y_high = np.exp(min(np.log(np.finfo(float).max), np.log(high) + padding))
        right = axis.twinx()
        line, = right.plot(times, steps, color=color, linestyle="--",
                           linewidth=1.15, alpha=.85, drawstyle="steps-pre",
                           label="Actual dt")
        self.set_time_series(line, times, steps)
        right.set_yscale("log", base=base)
        right.set_ylabel(f"log DT", color=color, fontsize=9, labelpad=7)
        right.tick_params(axis="y", colors=color, labelsize=8)
        right.spines["right"].set_color(color)
        right.set_ylim(y_low, y_high)
        if show_ratio_levels:
            # Align major ticks with dt0/N**k; show at most six levels even for N near 1 or large K.
            last_level = max(0, int(np.floor(np.log(self.dt / low) / np.log(base))))
            first_level = max(0, int(np.ceil(np.log(self.dt / high) / np.log(base))))
            levels = np.unique(np.rint(np.linspace(first_level, max(first_level, last_level),
                                        min(6, max(1, last_level - first_level + 1)))).astype(int))
            ticks = self.dt / np.power(base, levels.astype(float))
            ticks = ticks[(ticks >= y_low) & (ticks <= y_high)]
            right.yaxis.set_major_locator(FixedLocator(ticks))
        else:
            right.yaxis.set_major_locator(LogLocator(base=base, numticks=6))
        right.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.4g}"))
        right.yaxis.set_minor_locator(NullLocator())
        right.grid(False)
        # Put labels, the yellow cursor and physical curves above dt.
        # Make the upper axis transparent so the secondary line is visible.
        right.set_zorder(axis.get_zorder() - 1)
        axis.patch.set_visible(False)
        self.step_axes[axis] = right
        self.step_lines[axis] = line
        self.time_axis_for[right] = axis

    def build_step_size_axes(self):
        self.step_axes = {}
        self.step_lines = {}
        self.time_axis_for = {}
        self.step_log_base = float(self.metadata.get("acceleration_trigger_ratio") or
                                   self.metadata.get("work_step_factor") or 10.0)
        ratio_mode = self.adaptive and self.metadata.get("adaptive_algorithm") in ("acceleration_ratio", *WORK_HISTORY_CODES)
        for axis in (self.energy_axes, self.acceleration_axes):
            self.add_step_size_axis(axis, self.step_log_base, show_ratio_levels=ratio_mode)

    def update_overview_legend(self, axis, handles=None, ncol=1):
        if handles is None:
            handles, _ = axis.get_legend_handles_labels()
        handles = list(handles)
        handles += [self.step_lines[axis]]
        diagnostic = axis is getattr(self, "diagnostic_energy_axes", None)
        if axis.figure is getattr(self, "derivative_figure", None):
            self.plot_headers[axis].legend(
                handles=handles, fontsize=6.5, ncol=len(handles), loc="upper right",
                bbox_to_anchor=(1, 1.05), borderaxespad=0, frameon=False,
                handlelength=1.3, columnspacing=.8)
            return
        self.plot_headers[axis].legend(
            handles=handles, fontsize=7, ncol=min(3, len(handles)),
            loc="center" if diagnostic else "lower left",
            bbox_to_anchor=(.63, .68) if diagnostic else (0, -.05), borderaxespad=0,
            frameon=False, handlelength=1.5, columnspacing=.9)

    def build_diagnostics(self):
        self.diagnostic_figure = Figure(figsize=(15, 10), dpi=100, constrained_layout=True)
        has_position_error = self.metadata.get("position_error", {}).get("reference_kind") == "analytic_kepler"
        grid = self.diagnostic_figure.add_gridspec(5, 4, height_ratios=(.85 if has_position_error else .6, 1.25, 1, 1, 1))
        self.diagnostic_figure.suptitle("Conservation and error analysis\n" + self.figure_caption, fontsize=9)
        self.diagnostic_cursors = []
        self.diagnostic_info_axes = self.diagnostic_figure.add_subplot(grid[0, :])
        self.diagnostic_info_axes.set_axis_off()
        energy_axis = self.diagnostic_figure.add_subplot(grid[1, :])
        self.plot_headers[energy_axis] = self.diagnostic_info_axes
        self.diagnostic_title_text = self.diagnostic_info_axes.text(0, .98, "Energy", va="top", fontsize=10)
        self.diagnostic_error_text = self.diagnostic_info_axes.text(0, .62, "", va="top", fontsize=8)
        self.diagnostic_drift_text = self.diagnostic_info_axes.text(0, .28, "", va="top", fontsize=8)
        if has_position_error:
            self.diagnostic_error_text.set_y(.72)
            self.diagnostic_drift_text.set_y(.46)
            error = self.metadata["position_error"]
            self.diagnostic_position_text = self.diagnostic_info_axes.text(0, .17,
                f"Kepler position RMSE = {error['rmse']:.5e}    NRMSE = {error['nrmse']:.5e} "
                f"| e = {error['eccentricity']:g}", va="top", fontsize=8)
        self.diagnostic_statistics_text = self.diagnostic_info_axes.text(
            1, .98, "", ha="right", va="top", fontsize=8, linespacing=1.2)

        values = self.energies
        self.diagnostic_energy_axes = energy_axis
        self.diagnostic_energy_values = values
        self.diagnostic_energy_line, = energy_axis.plot(
            self.times, values, color="#172435", linewidth=1, label="E")
        self.diagnostic_energy_reference = energy_axis.axhline(
            self.energies[0], color="#8a94a3", linestyle="--", linewidth=.9, label="Initial energy E(0)")
        self.add_step_size_axis(energy_axis, base=10.0)
        self.update_overview_legend(energy_axis, ncol=3)
        error = self.energy_error["rmse"]
        scale = self.metadata["energy_scale"]
        normalized = error / scale if scale else np.nan
        self.diagnostic_energy_error_label = (
            f"float64 time-weighted RMSE = {error:.5e}    NRMSE = {normalized:.5e} | all {len(self.times):,} records")

        energy_axis.set_ylabel("E")
        energy_axis.grid(alpha=.22)
        self.register_hover(energy_axis, self.times, {"E": values})
        self.diagnostic_cursors.append(energy_axis.axvline(0, color="#de8a28", linewidth=.8))
        vector_info = [
            ("momentum", "Total momentum", "P"),
            ("angular_momentum", "Total angular momentum", "L"),
            ("com_v", "COM velocity", "V_com"),
        ]
        for row, (key, title, symbol) in enumerate(vector_info, start=2):
            values = np.array([item[key] for item in self.diagnostics])
            for col, (suffix, color) in enumerate((("x", "tab:red"), ("y", "tab:green"),
                                                    ("z", "tab:blue"), ("Magnitude", "tab:purple"))):
                series = values[:, col] if col < 3 else np.linalg.norm(values, axis=1)
                axis = self.diagnostic_figure.add_subplot(grid[row, col], sharex=energy_axis)
                line, = axis.plot(self.diagnostic_times, series, color=color, linewidth=1)
                self.set_time_series(line, self.diagnostic_times, series)
                axis.set_title(f"{title}：{suffix}\nRMSE = {float(RMSE(series)):.5e}", fontsize=8)
                # Shared time coordinates need labels only on the bottom row.
                axis.set_xlabel("Time (code units)" if row == 4 else "", fontsize=8)
                axis.tick_params(axis="x", labelbottom=(row == 4))
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
                axis.set_ylabel(f"{symbol}_{suffix}" if col < 3 else f"|{symbol}|", fontsize=8)
                axis.tick_params(labelsize=7)
                axis.grid(alpha=.22)
                self.register_hover(axis, self.diagnostic_times, {f"{symbol} {suffix}": series})
                self.diagnostic_cursors.append(axis.axvline(0, color="#de8a28", linewidth=.8))
        energy_axis.set_xlabel("")
        energy_axis.tick_params(labelsize=7, axis="both")
        energy_axis.tick_params(axis="x", labelbottom=False)
        energy_axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        set_physical_log_scale(energy_axis, self.diagnostic_energy_values)
        self.diagnostic_canvas, self.diagnostic_toolbar = self.embed_figure(
            self.diagnostic_figure, self.diagnostics_tab)

    def build_acceleration_derivative(self):
        self.derivative_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.derivative_tab, text="Work-step indicator")
        self.derivative_readout_var = tk.StringVar(value=f"Select this page to calculate Q = {self.work_indicator_formula} ({self.work_indicator_units}).")
        note = ttk.Label(self.derivative_tab, style="Note.TLabel", justify="left",
                         textvariable=self.derivative_readout_var, wraplength=900)
        note.pack(fill="x", padx=8, pady=(0, 2))
        self.derivative_tab.bind("<Configure>", lambda event: note.configure(
            wraplength=max(220, event.width - 20)))
        figure = self.derivative_figure = Figure(
            figsize=(10.6, 10.3), dpi=100, facecolor="white", constrained_layout=True)
        figure.suptitle((f"{self.work_algorithm_name}: energy and endpoint-symmetric work monitor\n" if self.work_symmetric else
                        f"{self.work_algorithm_name}: energy and work-step indicator\n")
                       + f"Q = {self.work_indicator_formula}; "
                       + (self.work_timing_note + "; " if self.work_displacement else "end-of-step a and v; ")
                       + f"incoming actual dt; {self.work_indicator_units}"
                       + ("\nS_E = K(0) + |U(0)|, fixed at initial state" if self.work_normalized else "")
                       + (f"\nPre-probe ON: {self.metadata.get('work_probe_window_count', 0)} replay windows; "
                          f"{self.metadata.get('work_probe_trial_evaluations', 0):,} extra probe trials"
                          if self.metadata.get("work_preprobe_enabled") else ""), fontsize=9)
        grid = figure.add_gridspec(10, 1, height_ratios=(.23, 1) * 5)
        axes, titles = [], []
        for index, title in enumerate(("System energy", f"Work-step indicator {self.work_indicator_formula}",
                                       "Separate contributions to Q",
                                       "Acceleration magnitude and velocity along acceleration",
                                       "Acceleration magnitude times speed along acceleration")):
            info = figure.add_subplot(grid[2*index]); info.set_axis_off()
            axis = figure.add_subplot(grid[2*index+1], sharex=axes[0] if axes else None)
            self.plot_headers[axis] = info
            titles.append(info.text(0, .98, title, va="top", fontsize=9))
            axis.grid(alpha=.22)
            axis.tick_params(labelsize=8)
            axis.yaxis.set_offset_position("right")
            axis.set_xlim(self.times[0], self.times[-1])
            if index == 4:
                axis.set_xlabel("Time (code units)", fontsize=9)
            else:
                axis.tick_params(labelbottom=False)
            axes.append(axis)
        (self.derivative_energy_axes, self.work_indicator_axes, self.work_terms_axes, self.work_components_axes,
         self.work_power_axes) = axes
        for axis in axes[1:]:
            axis.callbacks.connect("ylim_changed", self.update_derivative_ticks)
        self.work_velocity_axes = self.work_components_axes.twinx()
        self.time_axis_for[self.work_velocity_axes] = self.work_components_axes
        self.work_acceleration_line, = self.work_components_axes.plot(
            [], [], color="#0891b2", linewidth=1.2, label="|a| (left)")
        self.work_velocity_line, = self.work_velocity_axes.plot(
            [], [], color="#ea580c", linewidth=1.2, label="v along a (right; signed)")
        self.work_velocity_axes.axhline(0, color="#8a94a3", linestyle=":", linewidth=.7)
        self.work_components_axes.set_ylabel("|a| (code L/T²)", color="#0891b2", fontsize=8)
        self.work_components_axes.tick_params(axis="y", colors="#0891b2")
        self.work_velocity_axes.set_ylabel("v along a (code L/T)", color="#ea580c", fontsize=8)
        self.work_velocity_axes.tick_params(axis="y", colors="#ea580c", labelsize=8)
        self.work_velocity_axes.set_yscale("linear")
        self.work_velocity_axes.yaxis.set_offset_position("right")
        self.work_power_line, = self.work_power_axes.plot(
            [], [], color="#16a34a", linewidth=1.25, label="A × |VA| = |a dot v|")
        self.work_power_axes.set_ylabel("A × |VA|\n(code L²/T³)", fontsize=8)
        self.work_velocity_term_line, = self.work_terms_axes.plot(
            [], [], color="#2563eb", linewidth=1.25, label="Qv = m A VA0 dt / S_E")
        self.work_acceleration_term_line, = self.work_terms_axes.plot(
            [], [], color="#dc2626", linewidth=1.25, label="Qa = m A² dt² / (2 S_E)")
        self.work_terms_axes.set_ylabel("Q contributions\n(dimensionless)", fontsize=8)
        self.derivative_titles = titles
        self.derivative_energy_line, = axes[0].plot([], [], color="#172435", linewidth=1.2, label="E")
        self.derivative_energy_reference = axes[0].axhline(
            self.energies[0], color="#8a94a3", linestyle="--", linewidth=.7, label="Initial energy")
        self.work_indicator_line, = axes[1].plot(
            [], [], color="#7c3aed", linewidth=1.3, label="Selected star Q")
        self.work_indicator_max_line, = axes[1].plot(
            [], [], color="#087f8c", linewidth=1.0, linestyle="--", alpha=.8, label="All-star maximum Q")
        if self.work_symmetric:
            self.work_forward_line, = axes[1].plot([], [], color="#2563eb", linewidth=.85, linestyle=":", label="Q forward")
            self.work_reverse_line, = axes[1].plot([], [], color="#ea580c", linewidth=.85, linestyle="-.", label="Q reverse")
        self.work_plot_threshold = float(self.metadata.get("work_threshold") or
                                        self.metadata.get("config", {}).get("work_threshold",
                                            WORK_DEFAULT_THRESHOLD if self.work_displacement else 1.0))
        active = self.adaptive and self.metadata.get("adaptive_algorithm") in WORK_HISTORY_CODES
        threshold_label = "Trigger Threshold" if active else "Reference threshold (inactive)"
        self.work_indicator_threshold_line = axes[1].axhline(
            self.work_plot_threshold, color="#b91c1c", linestyle=":", linewidth=1.1,
            label=f"{threshold_label} = {self.work_plot_threshold:g}")
        self.derivative_cursors = []
        ratio_mode = self.adaptive and self.metadata.get("adaptive_algorithm") in ("acceleration_ratio", *WORK_HISTORY_CODES)
        for axis in axes:
            for number, window in enumerate(self.metadata.get("work_protection_windows") or []):
                axis.axvspan(window["start"], window["end"], color="#0d9488", alpha=.09,
                             label="Protection window" if number == 0 and axis is axes[0] else "_nolegend_", zorder=0)
                axis.axvspan(window["core_start"], window["core_end"], color="#0d9488", alpha=.10,
                             label="Locator core" if number == 0 and axis is axes[0] else "_nolegend_", zorder=0)
            if axis is not self.work_components_axes:
                self.add_step_size_axis(axis, self.step_log_base, show_ratio_levels=ratio_mode)
            self.derivative_cursors.append(axis.axvline(
                self.times[self.step], color="#de8a28", linewidth=1, alpha=.7))
            self.register_hover(axis, self.times, {})
        self.derivative_canvas, self.derivative_toolbar = self.embed_figure(figure, self.derivative_tab)

    def refresh_derivative_energy(self):
        offset = self.delta_energy_var.get()
        values = self.energies - self.energies[0] if offset else self.energies
        plot_values = self.energy_plot_offsets if offset else self.energy_plot_values
        axis = self.derivative_energy_axes
        old_limits = None
        if self.energy_display_offset is not None and not axis.get_autoscaley_on():
            shift = (float(self.energy_display_offset) - float(offset)) * self.energies[0]
            old_limits = np.asarray(axis.get_ylim()) + shift
        self.set_time_series(self.derivative_energy_line, self.energy_plot_times, plot_values,
                             self.energy_plot_offsets, preserve_zero=True)
        label = "ΔE" if offset else "E"
        self.derivative_energy_line.set_label(label)
        self.derivative_energy_reference.set_ydata([0 if offset else self.energies[0]] * 2)
        self.derivative_energy_reference.set_label("ΔE = 0" if offset else "E(0)")
        self.derivative_titles[0].set_text("System energy offset ΔE" if offset else "System total energy E")
        self.apply_energy_scale(axis, values)
        self.update_energy_drift_bounds(axis, offset)
        axis.set_ylabel(f"{label} (code energy; {self.energy_scale_var.get()})", fontsize=8)
        axis.relim(); axis.autoscale_view(scalex=False)
        if old_limits is not None:
            axis.set_ylim(old_limits)
        self.update_overview_legend(axis, ncol=3)
        self.register_hover(axis, self.times,
                            {"E": self.energies, "ΔE": self.energies - self.energies[0]},
                            "System energy at saved records")
        self.derivative_canvas.draw_idle()

    def refresh_acceleration_derivative(self):
        # Lazy O(time) work arrays; avoid an extra [time, star, xyz] product allocation.
        if not hasattr(self, "global_work_step_indicator"):
            maximum = np.zeros(self.frame_count)
            for star in range(self.n):
                values = self.star_work_indicator(star)
                np.maximum(maximum, values, out=maximum)
            self.global_work_step_indicator = maximum
        if self.derivative_star_index != self.star_index:
            self.selected_work_step_indicator = self.star_work_indicator(self.star_index)
            if self.work_symmetric:
                terms = symmetric_work_step_terms(self.accelerations[:, self.star_index, :],
                    self.velocities[:, self.star_index, :], self.step_sizes,
                    self.work_masses[self.star_index], self.metadata["energy_scale"], legacy=self.work_legacy_timing)
                self.selected_work_forward, self.selected_work_reverse = terms["forward"], terms["reverse"]
                self.selected_work_reverse_selected = terms["reverse_selected"]
                self.selected_work_velocity_term, self.selected_work_acceleration_term = terms["velocity_term"], terms["acceleration_term"]
            elif self.work_displacement:
                self.selected_work_velocity_term, self.selected_work_acceleration_term = displacement_work_step_terms(
                    self.accelerations[:, self.star_index, :], self.velocities[:, self.star_index, :],
                    self.step_sizes, self.work_masses[self.star_index], self.metadata["energy_scale"], legacy=self.work_legacy_timing, average=self.work_average, predicted=self.work_predicted, predicted_legacy=self.work_predicted_legacy)
            else:
                # Historical Q had only its endpoint-velocity term; do not
                # retrospectively add the new quadratic contribution to it.
                self.selected_work_velocity_term = self.selected_work_step_indicator.copy()
                self.selected_work_acceleration_term = np.zeros_like(self.selected_work_velocity_term)
                self.selected_work_acceleration_term[0] = np.nan
            self.selected_work_acceleration = np.hypot.reduce(self.accelerations[:, self.star_index, :], axis=-1)
            _, self.selected_work_velocity_along_a = velocity_acceleration_projections(
                self.accelerations[:, self.star_index, :], self.velocities[:, self.star_index, :])
            # Same recorded-time vectors, without mass, dt or energy normalization.
            # Direct dot product also defines the product as zero when |a|=0.
            self.selected_work_power = acceleration_velocity_projection_product(
                self.accelerations[:, self.star_index, :], self.velocities[:, self.star_index, :])
            self.derivative_star_index = self.star_index
        axis = self.work_indicator_axes
        self.work_indicator_line.set_label(f"Star {self.star_index + 1}: Q")
        self.set_time_series(self.work_indicator_line, self.times, self.selected_work_step_indicator, preserve_zero=True)
        self.set_time_series(self.work_indicator_max_line, self.times, self.global_work_step_indicator, preserve_zero=True)
        if self.work_symmetric:
            self.set_time_series(self.work_forward_line, self.times, self.selected_work_forward, preserve_zero=True)
            self.set_time_series(self.work_reverse_line, self.times, self.selected_work_reverse, preserve_zero=True)
        scale = set_physical_log_scale(axis, np.r_[self.selected_work_step_indicator,
                                                   self.global_work_step_indicator, self.work_plot_threshold], positive_only=True)
        axis.set_ylabel(f"{'Q' if self.work_displacement else self.work_indicator_formula}\n({self.work_indicator_units})", fontsize=8)
        axis.set_autoscaley_on(True)
        axis.relim(); axis.autoscale_view(scalex=False)
        self.update_derivative_ticks(axis)
        self.derivative_titles[1].set_text(f"Work-step Q: selected Star {self.star_index + 1} and all-star maximum [{scale}]")
        self.update_overview_legend(axis, ncol=2)
        self.register_hover(axis, self.times, {f"Star {self.star_index + 1} Q": self.selected_work_step_indicator,
                                             "All-star maximum Q": self.global_work_step_indicator},
                            f"Work-step indicator ({self.work_indicator_units}); threshold = {self.work_plot_threshold:g}; initial state undefined")
        if self.work_symmetric:
            short_name = "Wide Test" if self.work_algorithm_code == WIDE_ALGORITHM else self.work_algorithm_name
            self.derivative_titles[1].set_text(f"{short_name} Q: Star {self.star_index + 1} and all-star maximum [{scale}]")
            self.register_hover(axis, self.times,
                {"Q forward": self.selected_work_forward, "Q reverse": self.selected_work_reverse,
                 "Selected max Q": self.selected_work_step_indicator, "All-star maximum Q": self.global_work_step_indicator,
                 "Reverse direction selected (0/1)": self.selected_work_reverse_selected},
                f"{self.work_algorithm_name}; maximum is taken per star and then over all stars")
        terms_axis = self.work_terms_axes
        self.set_time_series(self.work_velocity_term_line, self.times, self.selected_work_velocity_term, preserve_zero=True)
        self.set_time_series(self.work_acceleration_term_line, self.times, self.selected_work_acceleration_term, preserve_zero=True)
        terms_scale = set_physical_log_scale(terms_axis, np.r_[self.selected_work_velocity_term,
                                                               self.selected_work_acceleration_term], positive_only=True)
        terms_axis.set_autoscaley_on(True)
        terms_axis.relim(); terms_axis.autoscale_view(scalex=False)
        self.update_derivative_ticks(terms_axis)
        terms_axis.set_ylabel(f"Q contributions\n({self.work_indicator_units})", fontsize=8)
        self.work_velocity_term_line.set_label(("Qv = m A VA0 dt / S_E" if self.work_legacy_timing else "Qv = m A0 VA0 dt / S_E") if self.work_displacement else "Legacy Q (end velocity)")
        self.work_acceleration_term_line.set_label(("Qa = m A² dt² / (2 S_E)" if self.work_legacy_timing else "Qa = m A0² dt² / (2 S_E)") if self.work_displacement else "Quadratic term (not used)")
        self.derivative_titles[2].set_text(
            f"Star {self.star_index + 1}: Qv + Qa = Q; {'mixed-time' if self.work_legacy_timing else 'start-state a0 and v0'} [{terms_scale}]"
            if self.work_displacement else f"Star {self.star_index + 1}: legacy Q has no quadratic term [{terms_scale}]")
        if self.work_average:
            self.work_velocity_term_line.set_label("Qv = m |a_bar dot v_start| dt / S_E")
            self.work_acceleration_term_line.set_label("Qa = m |a_bar|² dt² / (2 S_E)")
            self.derivative_titles[2].set_text(
                f"Star {self.star_index + 1}: Qv + Qa = Q; endpoint-mean a and start v [{terms_scale}]")
        if self.work_predicted:
            self.work_velocity_term_line.set_label("Qv = m |a_pred dot v_start| dt / S_E")
            self.work_acceleration_term_line.set_label("Qa = m |a_pred|² dt² / (2 S_E)")
            self.derivative_titles[2].set_text(
                f"Star {self.star_index + 1}: Qv + Qa = Q; accepted-history predicted a and start v [{terms_scale}]")
        if self.work_symmetric:
            self.work_velocity_term_line.set_label("Qv (winning direction)")
            self.work_acceleration_term_line.set_label("Qa (winning direction)")
            self.derivative_titles[2].set_text(f"Star {self.star_index + 1}: Qv + Qa from winning direction = Q [{terms_scale}]")
        self.update_overview_legend(terms_axis, ncol=3)
        self.register_hover(terms_axis, self.times,
                            {"Qv (velocity term)": self.selected_work_velocity_term,
                             "Qa (acceleration term)": self.selected_work_acceleration_term,
                             "Qv + Qa": self.selected_work_velocity_term+self.selected_work_acceleration_term},
                            "Recorded accepted-step contributions to selected-star Q")
        self.set_time_series(self.work_acceleration_line, self.times, self.selected_work_acceleration, preserve_zero=True)
        self.set_time_series(self.work_velocity_line, self.times, self.selected_work_velocity_along_a, preserve_zero=True)
        component_scale = set_physical_log_scale(self.work_components_axes, self.selected_work_acceleration,
                                                positive_only=True)
        for component_axis in (self.work_components_axes, self.work_velocity_axes):
            component_axis.set_autoscaley_on(True)
            component_axis.relim(); component_axis.autoscale_view(scalex=False)
        self.update_derivative_ticks(self.work_components_axes)
        self.work_acceleration_line.set_label(f"|a| (left; {component_scale})")
        self.plot_headers[self.work_components_axes].legend(
            handles=[self.work_acceleration_line, self.work_velocity_line],
            fontsize=6.5, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.05),
            borderaxespad=0, frameon=False, handlelength=1.3, columnspacing=.8)
        self.derivative_titles[3].set_text(f"Star {self.star_index + 1}: |a| and velocity along a (signed)")
        self.register_hover(self.work_components_axes, self.times,
                            {"|a| (code L/T²)": self.selected_work_acceleration,
                             "v along a (code L/T; signed)": self.selected_work_velocity_along_a},
                            f"Star {self.star_index + 1}: acceleration and velocity projection; |a|=0 gives undefined direction")
        self.set_time_series(self.work_power_line, self.times, self.selected_work_power, preserve_zero=True)
        power_scale = set_physical_log_scale(self.work_power_axes, self.selected_work_power,
                                              positive_only=True)
        self.work_power_axes.set_autoscaley_on(True)
        self.work_power_axes.relim(); self.work_power_axes.autoscale_view(scalex=False)
        self.update_derivative_ticks(self.work_power_axes)
        self.derivative_titles[4].set_text(
            f"Star {self.star_index + 1}: A × |VA| = |a dot v| (same-time vectors) [{power_scale}]")
        self.update_overview_legend(self.work_power_axes, ncol=2)
        self.register_hover(self.work_power_axes, self.times,
                            {"A × |VA| (code L²/T³)": self.selected_work_power},
                            f"Star {self.star_index + 1}: specific power magnitude; no mass, dt or normalization")
        self.update_derivative_readout()
        self.apply_figure_layout(self.derivative_figure, *self.derivative_figure.bbox.size)
        self.derivative_canvas.draw_idle()

    def star_work_indicator(self, star):
        arguments = (self.accelerations[:, star, :], self.velocities[:, star, :], self.step_sizes)
        if self.work_normalized:
            compute = symmetric_work_step_indicator if self.work_symmetric else displacement_work_step_indicator if self.work_displacement else normalized_work_step_indicator
            return compute(*arguments, self.work_masses[star], self.metadata["energy_scale"],
                           **({"legacy": self.work_legacy_timing, **({"average": True} if self.work_average else {}), **({"predicted": True, "predicted_legacy": self.work_predicted_legacy} if self.work_predicted else {})} if self.work_displacement else {}))
        return work_step_indicator(*arguments)

    @staticmethod
    def update_derivative_ticks(axis):
        """Keep numerical labels readable when a log/symlog plot is zoomed narrowly."""
        if axis.get_yscale() not in ("log", "symlog"):
            return
        low, high = sorted(axis.get_ylim())
        threshold = (axis.yaxis.get_transform().linthresh if axis.get_yscale() == "symlog" else 0.)
        narrow = low * high > 0 and max(abs(low), abs(high)) / min(abs(low), abs(high)) < 10
        if narrow or max(abs(low), abs(high)) <= 10 * threshold:
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useOffset=True)
            formatter.set_powerlimits((-3, 4))
        else:
            if axis.get_yscale() == "log":
                locator = LogLocator(base=10, numticks=5)
            else:
                # Label decades outside the linear core, rather than crowded +/-1 ticks.
                powers = np.arange(int(np.ceil(np.log10(threshold)))+1,
                                   int(np.floor(np.log10(max(abs(low), abs(high)))))+1)
                powers = powers[::max(1, int(np.ceil(len(powers)/3)))]
                ticks = np.power(10., powers)
                ticks = np.r_[-ticks[::-1], 0., ticks]
                locator = FixedLocator(ticks[(ticks >= low) & (ticks <= high)])
            axis.yaxis.set_major_locator(locator)
            formatter = LogFormatterSciNotation(base=10)
        axis.yaxis.set_major_formatter(formatter)
        axis.yaxis.set_minor_locator(NullLocator())

    def update_derivative_readout(self):
        if self.derivative_star_index != self.star_index:
            return
        self.derivative_readout_var.set(
            f"Star {self.star_index + 1} | t = {self.times[self.step]:.8g} | "
            f"Q = {self.selected_work_step_indicator[self.step]:.7e} | "
            f"Qv = {self.selected_work_velocity_term[self.step]:.7e} | "
            f"Qa = {self.selected_work_acceleration_term[self.step]:.7e} | "
            f"max Q = {self.global_work_step_indicator[self.step]:.7e} | "
            f"|a| = {self.selected_work_acceleration[self.step]:.7e} | "
            f"v along a = {self.selected_work_velocity_along_a[self.step]:+.7e} | "
            f"A × |VA| = {self.selected_work_power[self.step]:.7e} | "
            f"dt = {self.step_sizes[self.step]:.7e} | threshold = {self.work_plot_threshold:g} "
            f"(Q: {self.work_indicator_units}; initial state has no completed step)")

    def build_energy_drift(self):
        self.drift_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.drift_tab, text="Energy drift")
        self.drift_result = self.metadata.get("energy_drift")
        if self.drift_result is None or self.drift_result.get("version") != 2:
            self.drift_result = fit_energy_drift(self.times, self.energies, self.metadata["energy_scale"])
        self.metadata["energy_drift"] = self.drift_result
        bar = ttk.Frame(self.drift_tab, padding=(6, 4))
        bar.pack(fill="x")
        ttk.Label(bar, text=f"All {self.drift_result['fit_records']:,} records · weighted by physical time").grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(bar, text="Save fit JSON", command=self.save_energy_drift).grid(row=1, column=0, sticky="w")
        ttk.Button(bar, text="Save plot PNG", command=self.save_image).grid(row=1, column=1, sticky="w", padx=8)
        ttk.Label(bar, text=self.timestep_summary).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        note = ttk.Label(self.drift_tab, style="Note.TLabel", justify="left",
                        text="All saved records are fitted. Each weight is half the physical-time interval on each side; endpoints use one side.\n"
                             "The fit uses E(t) - E(0), with a free intercept; both axes are linear.\n"
                             "Positive slope: energy increases; negative: decreases. Near-zero slope alone does not imply conservation.",
                        wraplength=900)
        note.pack(fill="x", padx=8, pady=(0, 4))
        self.drift_tab.bind("<Configure>", lambda event: note.configure(wraplength=max(300, event.width - 20)))
        self.drift_figure = Figure(figsize=(12, 7), dpi=100, constrained_layout=True)
        grid = self.drift_figure.add_gridspec(2, 1, height_ratios=(.34, 1))
        info = self.drift_figure.add_subplot(grid[0]); info.set_axis_off()
        self.drift_summary_text = info.text(0, .97, "", va="top", fontsize=9, linespacing=1.5)
        self.drift_axes = self.drift_figure.add_subplot(grid[1])
        recorded_line, = self.drift_axes.plot([], [], color="#8a94a3",
                                               linewidth=1, label="Recorded energy offset")
        self.set_time_series(recorded_line, self.energy_plot_times, self.energy_plot_offsets,
                             preserve_zero=True)
        self.drift_line, = self.drift_axes.plot([], [], color="#bb3040", linewidth=1.7, label="Time-weighted linear fit")
        self.drift_axes.axhline(0, color="#687485", linestyle=":", linewidth=.7, label="Initial energy offset")
        self.drift_axes.set_xlabel("Time (code units)")
        self.drift_axes.set_ylabel("E(t) - E(0) (code energy)")
        self.drift_axes.grid(alpha=.22)
        self.drift_axes.set_yscale("linear")
        info.legend(*self.drift_axes.get_legend_handles_labels(), loc="lower left",
                    ncol=2, fontsize=8, frameon=False)
        self.drift_canvas, self.drift_toolbar = self.embed_figure(self.drift_figure, self.drift_tab)
        self.update_drift_plot()

    def update_drift_plot(self):
        result = self.drift_result
        times = self.times
        offsets = self.energies - self.energies[0]
        ends = np.array([result["time_start"], result["time_end"]])
        fit_times = np.linspace(ends[0], ends[1], 257)
        fitted = result["intercept_offset_at_start"] + result["slope"] * (fit_times - ends[0])
        self.drift_line.set_data(fit_times, fitted)
        self.drift_axes.set_yscale("linear")
        def number(value):
            return "NA" if value is None else f"{value:.6e}"
        self.drift_summary_text.set_text(
            f"Time-weighted energy drift fit | all {result['fit_records']:,} records\n"
            f"Slope k = {number(result['slope'])} energy/time | normalized k = {number(result['normalized_slope'])} /time\n"
            f"Fitted change k*T = {number(result['fitted_energy_change'])} | normalized change = {number(result['normalized_fitted_energy_change'])}\n"
            f"Offset intercept = {number(result['intercept_offset_at_start'])} | weighted residual RMSE = {number(result['residual_rmse'])} | weighted R-squared = {number(result['r_squared'])}")
        self.update_energy_drift_bounds(self.drift_axes, True)
        self.drift_axes.relim(); self.drift_axes.autoscale_view()
        predicted = result["intercept_offset_at_start"] + result["slope"] * (times - ends[0])
        self.register_hover(self.drift_axes, times, {"Recorded ΔE": offsets, "Fit ΔE": predicted}, "All-record energy drift fit")
        self.drift_canvas.draw_idle()

    def save_energy_drift(self):
        path = self.choose_save_path(".json", "Energy drift analysis", "_energy_drift")
        if not path:
            return
        try:
            payload = dict(run_id=self.metadata.get("run_id"), case_name=self.metadata.get("case_name"),
                           method=self.metadata.get("method_name"), analysis=self.drift_result,
                           fit_data=dict(times=self.times.tolist(),
                               energy_offsets=(self.energies-self.energies[0]).tolist(),
                               normalized_time_weights=physical_time_record_weights(self.times).tolist()))
            _archive_write(Path(path), diagnostic_json_bytes(payload))
            self.status_var.set(f"Time-weighted drift fit, all records and weights saved: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)

    def register_hover(self, axis, times, series, title=None):
        if axis in getattr(self, "step_axes", {}):
            series = dict(series)
            indices = np.clip(np.searchsorted(self.times, np.asarray(times)), 0, self.frame_count - 1)
            actual_steps = self.step_sizes[indices].copy()
            actual_steps[indices == 0] = np.nan
            series["dt"] = actual_steps
        if title is None:
            title = axis.get_title().split("\n")[0] or "Energy"
        self.hover_items[axis] = dict(times=np.asarray(times), series=series, title=title)
        if axis.figure in self.page_readouts:
            self.page_readouts[axis.figure].set(self.hover_idle_text)
        self.hover_selection = None

    @staticmethod
    def format_readout(title, fields):
        rows = [title]
        rows.extend("   ".join(fields[i:i + 2]) for i in range(0, len(fields), 2))
        return "\n".join(rows)

    def update_page_readout(self, figure):
        inspector = self.page_readout_widgets.get(figure)
        if inspector is None:
            return
        text = self.page_readouts[figure].get()
        if not text or text == self.hover_idle_text:
            inspector.place_forget()
            return
        viewport = self.page_viewports[figure]
        inspector.configure(wraplength=max(180, min(470, viewport.winfo_width() - 18)))
        inspector.place(relx=1.0, x=-7, y=7, anchor="ne")
        inspector.lift()

    def hide_page_readout(self, figure):
        self.page_readouts[figure].set(self.hover_idle_text)
        self.hover_selection = None

    def clear_hover(self, event):
        figure = event.canvas.figure
        inspector = self.page_readout_widgets.get(figure)
        if inspector is not None and inspector.winfo_ismapped():
            # Entering the inspector itself is not leaving the plot page.
            x, y = inspector.winfo_pointerxy()
            if (inspector.winfo_rootx() <= x < inspector.winfo_rootx() + inspector.winfo_width()
                    and inspector.winfo_rooty() <= y < inspector.winfo_rooty() + inspector.winfo_height()):
                return
        variable = self.page_readouts.get(figure)
        if variable is not None:
            variable.set(self.hover_idle_text)
        self.hover_selection = None

    @staticmethod
    def nearest_index(times, value):
        index = min(int(np.searchsorted(times, value)), len(times) - 1)
        if index > 0 and abs(value - times[index - 1]) <= abs(value - times[index]):
            index -= 1
        return index

    def apply_energy_scale(self, axis, values):
        if self.energy_scale_var.get() == "symlog":
            set_physical_log_scale(axis, values)
            axis.yaxis.set_major_locator(EnergyLogLocator())
            axis.yaxis.set_major_formatter(FuncFormatter(
                lambda value, position: energy_log_tick_label(value, position, axis)))
            axis.yaxis.set_minor_locator(NullLocator())
        else:
            axis.set_yscale("linear")
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useOffset=True)
            formatter.set_powerlimits((-3, 4))
            axis.yaxis.set_major_formatter(formatter)
            axis.yaxis.set_minor_locator(NullLocator())

    def update_energy_drift_bounds(self, axis, offset):
        """Red dashed cumulative drift bounds of the step energy error band (display only, 2026-11-11).

        With step_energy_band and energy_drift_limit D > 0, accepted nodes satisfy |E - E(0)| <= D
        (nbody_step_energy_band rule 2), i.e. E(0) - D <= E <= E(0) + D, or -D <= E - E(0) <= D.
        The two lines are one LineCollection in axes-x / data-y coordinates added with autolim=False:
        relim()/autoscale_view() never include them, so they do not enlarge the y range; when the data
        stay far from D they are simply outside (clipped) the current view. D = 0 shows nothing.
        """
        config = self.metadata.get("config", self.metadata)
        try:
            limit = float(config.get("energy_drift_limit") or 0.0)
        except (TypeError, ValueError):
            limit = 0.0
        enabled = (bool(config.get("adaptive_enabled")) and config.get("adaptive_algorithm") == "step_energy_band"
                   and np.isfinite(limit) and limit > 0)
        artists = self.__dict__.setdefault("_energy_drift_bound_artists", {})
        artist = artists.get(axis)
        if not enabled:
            if artist is not None:
                artist.set_visible(False)
            return
        reference = 0.0 if offset else float(self.energies[0])
        if not (np.isfinite(reference - limit) and np.isfinite(reference + limit)):
            if artist is not None:
                artist.set_visible(False)
            return
        segments = [[(0, reference - limit), (1, reference - limit)], [(0, reference + limit), (1, reference + limit)]]
        if artist is None or artist.axes is not axis:
            artist = LineCollection(segments, colors="#dc2626", linestyles="--", linewidths=1.0,
                                    transform=axis.get_yaxis_transform(), zorder=4, label="_nolegend_")
            axis.add_collection(artist, autolim=False)
            artists[axis] = artist
        else:
            artist.set_segments(segments)
        artist.set_visible(True)

    def refresh_energy(self):
        offset = self.delta_energy_var.get()
        scale = self.energy_scale_var.get()
        old_scale = self.energy_display_scale
        values = self.energies - self.energies[0] if offset else self.energies
        old_offset = self.energy_display_offset
        plot_values = self.energy_plot_offsets if offset else self.energy_plot_values
        saved_limits = {}
        for axis in (self.energy_axes, self.diagnostic_energy_axes):
            if old_offset is not None and not axis.get_autoscaley_on():
                shift = (float(old_offset) - float(offset)) * self.energies[0]
                saved_limits[axis] = np.asarray(axis.get_ylim()) + shift
        self.set_time_series(self.energy_line, self.energy_plot_times, plot_values,
                             self.energy_plot_offsets, preserve_zero=True)
        label = "ΔE" if offset else "E"
        self.energy_line.set_label(label)
        reference = 0 if offset else self.energies[0]
        self.energy_reference.set_ydata([reference, reference])
        self.energy_reference.set_label("ΔE = 0" if offset else "E(0)")
        self.energy_title_text.set_text("Energy offset ΔE" if offset else "Total energy E")
        # Always report signed offsets from E(0), including in absolute-energy view.
        if not hasattr(self, "energy_statistics"):
            self.energy_statistics = full_time_statistics(self.energies - self.energies[0], self.times)
        self.energy_statistics_text.set_text(statistics_label(self.energy_statistics, "ΔE"))
        self.apply_energy_scale(self.energy_axes, values)
        self.update_energy_drift_bounds(self.energy_axes, offset)
        self.energy_axes.set_ylabel(f"{label} (code units; {scale})", fontsize=9)
        self.energy_axes.relim(); self.energy_axes.autoscale_view(scalex=False)
        if self.energy_axes in saved_limits:
            self.energy_axes.set_ylim(saved_limits[self.energy_axes])
        self.update_overview_legend(self.energy_axes)
        hover = {"E": self.energies, "E(0)": np.broadcast_to(self.energies[0], self.energies.shape),
                 "ΔE": self.energies - self.energies[0]}
        self.register_hover(self.energy_axes, self.times, hover,
                            "Recorded energy (crossing vertices are interpolated)")
        if hasattr(self, "diagnostic_energy_line"):
            diagnostic_values = self.diagnostic_energy_values - self.energies[0] if offset else self.diagnostic_energy_values
            self.set_time_series(self.diagnostic_energy_line, self.energy_plot_times, plot_values,
                                 self.energy_plot_offsets, preserve_zero=True)
            self.diagnostic_energy_line.set_label("Energy offset" if offset else "Energy")
            self.diagnostic_energy_reference.set_ydata([reference, reference])
            self.diagnostic_energy_reference.set_label("Initial offset = 0" if offset else "Initial energy")
            axis = self.diagnostic_energy_axes
            self.apply_energy_scale(axis, diagnostic_values)
            self.update_energy_drift_bounds(axis, offset)
            axis.set_ylabel(f"{label} ({scale})")
            self.diagnostic_title_text.set_text("Energy offset ΔE = E(t) - E(0)" if offset else "Total energy E(t)")
            self.diagnostic_error_text.set_text(self.diagnostic_energy_error_label)
            drift = getattr(self, "drift_result", None) or self.metadata.get("energy_drift") or {}
            def drift_number(value):
                return "NA" if value is None else f"{value:+.5e}"
            self.diagnostic_drift_text.set_text(
                f"Energy drift k = {drift_number(drift.get('slope'))} (energy/time)    "
                f"Normalized k = {drift_number(drift.get('normalized_slope'))} (1/time)")
            self.diagnostic_statistics_text.set_text(statistics_label(self.energy_statistics, "ΔE"))
            axis.relim(); axis.autoscale_view(scalex=False)
            if axis in saved_limits:
                axis.set_ylim(saved_limits[axis])
            self.update_overview_legend(axis, ncol=3)
            self.register_hover(axis, self.times, hover, "Recorded energy (crossing vertices are interpolated)")
            self.apply_figure_layout(self.diagnostic_figure, *self.diagnostic_figure.bbox.size)
            self.diagnostic_canvas.draw_idle()
        if hasattr(self, "derivative_energy_line"):
            self.refresh_derivative_energy()
        self.apply_figure_layout(self.figure, *self.figure.bbox.size)
        self.canvas.draw_idle()
        self.energy_display_offset = bool(offset)
        self.energy_display_scale = scale
        # Old toolbar history contains y limits in the other energy coordinates.
        # Rebase Home/Back/Forward on the current view after a mode change.
        if old_offset is not None and (bool(old_offset) != bool(offset) or old_scale != scale):
            for toolbar in (self.toolbar, self.diagnostic_toolbar, self.derivative_toolbar):
                toolbar.update()
                toolbar.push_current()
            self.active_canvas().draw()

    def refresh_acceleration(self):
        vectors = self.accelerations[:, self.star_index, :]
        self.selected_magnitudes = np.linalg.norm(vectors, axis=1)
        series = {"|a|": self.selected_magnitudes}
        for col, key in enumerate(("x", "y", "z")):
            values = vectors[:, col]
            line = self.acceleration_lines[key]
            self.set_time_series(line, self.times, values)
            line.set_visible(self.components_var.get())
            if self.components_var.get():
                series[f"a_{key}"] = values
        self.set_time_series(self.acceleration_lines["magnitude"], self.times, self.selected_magnitudes)
        self.acceleration_title_text.set_text(f"Star {self.star_index + 1}: |a|")
        self.acceleration_statistics = full_time_statistics(self.selected_magnitudes, self.times)
        self.acceleration_statistics_text.set_text(statistics_label(self.acceleration_statistics, "|a|"))
        scale = set_physical_log_scale(
            self.acceleration_axes,
            np.concatenate(list(series.values())),
            positive_only=not self.components_var.get())
        self.acceleration_axes.set_ylabel(f"Net acceleration (code units; {scale})", fontsize=9)
        self.acceleration_axes.relim(visible_only=True)
        self.acceleration_axes.autoscale_view(scalex=False)
        visible_lines = [line for line in self.acceleration_lines.values() if line.get_visible()]
        self.update_overview_legend(self.acceleration_axes, visible_lines, ncol=3)
        self.register_hover(self.acceleration_axes, self.times, series, f"Star {self.star_index + 1}: net acceleration")
        self.apply_figure_layout(self.figure, *self.figure.bbox.size)
        self.canvas.draw_idle()

    def refresh_vectors(self):
        self.refresh_acceleration()
        self.refresh_position_analysis()
        if self.notebook.select() == str(self.derivative_tab):
            self.refresh_acceleration_derivative()

    def select_star(self, index):
        if not 0 <= index < self.n:
            return
        self.pause()
        self.star_index = int(index)
        self.star_var.set(f"Star {self.star_index + 1}")
        self.refresh_vectors()
        self.draw_frame(self.step)
        follow_note = "The view now follows this star. " if self.follow_view_active else ""
        self.status_var.set(f"Selected Star {self.star_index + 1}. {follow_note}Drag the playback slider to inspect its state at another time.")

    def orbit_limits(self):
        return np.array([self.orbit_axes.get_xlim3d(), self.orbit_axes.get_ylim3d(),
                         self.orbit_axes.get_zlim3d()], dtype=float)

    def set_orbit_limits(self, limits):
        for setter, bounds in zip(
                (self.orbit_axes.set_xlim3d, self.orbit_axes.set_ylim3d, self.orbit_axes.set_zlim3d),
                limits):
            setter(*bounds)

    def on_camera_mode_change(self, event=None):
        mode = self.camera_mode_var.get()
        follow = mode == "Follow selected star"
        fit = mode == "Fit all current particles"
        was_dynamic = self.follow_view_active or self.fit_particles_active
        if (follow or fit) and not was_dynamic:
            self.fixed_view_limits = self.orbit_limits().copy()
        elif not (follow or fit) and was_dynamic and self.fixed_view_limits is not None:
            self.set_orbit_limits(self.fixed_view_limits)
        self.follow_view_active = follow
        self.fit_particles_active = fit
        self.draw_frame(self.step)
        self.status_var.set("Fitting all current particles with a margin at each time."
                            if fit else f"Following Star {self.star_index + 1}." if follow else
                            "Fixed view restored to the bounds used before automatic camera mode.")

    def update_follow_view(self, position):
        if self.fit_particles_active:
            current = self.trajectories[:, self.step, :]
            current = current[np.all(np.isfinite(current), axis=1)]
            if len(current):
                low, high = current.min(axis=0), current.max(axis=0)
                center = low / 2 + high / 2
                # Equal axis spans preserve geometry; padding includes isolated or planar states.
                extent = max(float(np.max(high - low)) * 0.6, 1e-6,
                             float(np.max(np.abs(center))) * 1e-12)
                self.set_orbit_limits(np.column_stack((center - extent, center + extent)))
            self.orbit_axes.set_title("All current particles", fontsize=10)
        elif self.follow_view_active:
            # Translate the viewing volume only. Read the current spans so
            # manual zoom is retained; leave elevation/azimuth/roll unchanged.
            limits = self.orbit_limits()
            half_spans = (limits[:, 1] - limits[:, 0]) / 2
            self.set_orbit_limits(np.column_stack((position - half_spans, position + half_spans)))
            self.orbit_axes.set_title(f"Following Star {self.star_index + 1}", fontsize=10)
        else:
            self.orbit_axes.set_title("Trajectories", fontsize=10)

    def draw_frame(self, step, redraw=True, playback_time=None):
        self.step = int(np.clip(step, 0, self.last_step))
        # Keep the continuous playhead separate from the nearest stored state.
        # Rounding to a record must not accumulate a playback-time error.
        self.playback_time = float(self.times[self.step] if playback_time is None else playback_time)
        # Cap only the displayed trail, always keeping its current endpoint.
        stride = max(1, int(np.ceil((self.step + 1) / 4000)))
        indices = np.arange(0, self.step + 1, stride)
        if indices[-1] != self.step:
            indices = np.append(indices, self.step)
        for i in range(self.n):
            path = self.trajectories[i, indices]
            self.lines[i].set_data_3d(path[:, 0], path[:, 1], path[:, 2])
            position = self.trajectories[i, self.step]
            self.points[i].set_data_3d([position[0]], [position[1]], [position[2]])
        position = self.trajectories[self.star_index, self.step]
        self.highlight.set_data_3d([position[0]], [position[1]], [position[2]])
        self.update_follow_view(position)
        current_time = self.times[self.step]
        self.selected_step_distance = (float(np.linalg.norm(
            position - self.trajectories[self.star_index, self.step - 1]))
            if self.step > 0 else 0.0)
        distance_text = (f"Step displacement = {self.selected_step_distance:.6g}"
                         if self.step > 0 else "Initial state; no step taken")
        self.time_text.set_text(
            f"t = {current_time:.6g}\n{self.step:,} / {self.last_step:,} steps\n{distance_text}")
        self.energy_cursor.set_xdata([current_time, current_time])
        self.acceleration_cursor.set_xdata([current_time, current_time])
        for line in self.diagnostic_cursors:
            line.set_xdata([current_time, current_time])
        for line in self.derivative_cursors:
            line.set_xdata([current_time, current_time])
        for line in getattr(self, "position_cursors", []):
            line.set_xdata([current_time, current_time])
        self.update_derivative_readout()
        acceleration = self.accelerations[self.step, self.star_index]
        velocity = self.velocities[self.step, self.star_index]
        a_along_v, v_along_a = velocity_acceleration_projections(acceleration, velocity)
        a_projection_text = f"{a_along_v:.8e}" if np.isfinite(a_along_v) else "undefined (|v| = 0)"
        v_projection_text = f"{v_along_a:.8e}" if np.isfinite(v_along_a) else "undefined (|a| = 0)"
        self.details_var.set(
            f"Star {self.star_index + 1} · t = {current_time:.8g}\n"
            + (f"Step dt = {self.step_sizes[self.step]:.8e}\n" if self.step else "Initial state\n") +
            f"a_x = {acceleration[0]:.8e}\n"
            f"a_y = {acceleration[1]:.8e}\n"
            f"a_z = {acceleration[2]:.8e}\n"
            f"|a| = {np.linalg.norm(acceleration):.8e}\n\n"
            f"v_x = {velocity[0]:.8e}\n"
            f"v_y = {velocity[1]:.8e}\n"
            f"v_z = {velocity[2]:.8e}\n"
            f"|v| = {np.linalg.norm(velocity):.8e}\n\n"
            f"a along v = {a_projection_text}\n"
            f"v along a = {v_projection_text}\n(signed; code units)\n\n"
            f"{distance_text}\n(code length units)\n\n"
            f"E = {self.energies[self.step]:.10e}\n"
            f"ΔE = {self.energies[self.step] - self.energies[0]:.8e}")
        self.time_var.set(f"Time t = {current_time:.10g} / {self.times[-1]:.10g}    "
                          f"Step {self.step:,} / {self.last_step:,} steps    "
                          f"Step dt = {self.step_sizes[self.step]:.6g}")
        self.updating_slider = True
        try:
            self.timeline.update_frame(self)
        finally:
            self.updating_slider = False
        if redraw:
            self.active_canvas().draw_idle()
        return self.lines + self.points + [self.highlight, self.time_text,
                                          self.energy_cursor, self.acceleration_cursor]

    def active_canvas(self):
        selected = self.notebook.select()
        if getattr(self, "position_tab", None) is not None and selected == str(self.position_tab):
            return self.position_canvas
        if hasattr(self, "derivative_tab") and selected == str(self.derivative_tab):
            return self.derivative_canvas
        if hasattr(self, "drift_tab") and selected == str(self.drift_tab):
            return self.drift_canvas
        return self.diagnostic_canvas if selected == str(self.diagnostics_tab) else self.canvas

    def pause(self):
        self.playing = False
        self.playback_last_clock = None
        self.play_button.configure(text="Play")

    def toggle_play(self):
        self.playing = not self.playing
        self.playback_last_clock = time.perf_counter() if self.playing else None
        self.play_button.configure(text="Pause" if self.playing else "Play")

    def single_step(self, direction):
        self.pause()
        self.draw_frame(self.step + direction)

    def rewind(self):
        self.pause()
        self.draw_frame(0)

    def tick(self):
        if self.closed:
            return
        if self.playing and not self.dragging:
            now = time.perf_counter()
            elapsed = max(0., now - self.playback_last_clock) if self.playback_last_clock is not None else 0.
            self.playback_last_clock = now
            try:
                speed = float(self.playback_speed_var.get())
                if not np.isfinite(speed) or speed <= 0:
                    speed = 1.
            except (ValueError, OverflowError):
                speed = 1.
            duration = float(self.times[-1] - self.times[0])
            target = self.playback_time + elapsed * speed * duration / 60.
            if target >= self.times[-1]:
                if self.loop_var.get():
                    target = self.times[0] + (target - self.times[0]) % duration
                else:
                    target = float(self.times[-1])
                    self.pause()
            self.draw_frame(self.nearest_index(self.times, target), playback_time=target)
        else:
            self.playback_last_clock = None
        self.after_id = self.root.after(40, self.tick)

    def on_seek(self, value):
        if not self.updating_slider:
            self.draw_frame(self.nearest_index(self.times, float(value)))

    def on_slider_press(self, event):
        self.pause()
        self.dragging = True

    def on_slider_release(self, event):
        self.dragging = False

    def on_tab_change(self, event):
        if hasattr(self, "diagnostic_canvas"):
            if self.notebook.select() == str(self.derivative_tab):
                self.refresh_acceleration_derivative()
            if self.notebook.select() in (str(self.diagnostics_tab), str(getattr(self, "drift_tab", ""))):
                self.controls.pack_forget()
            elif not self.controls.winfo_manager():
                self.controls.pack(side="left", fill="y", padx=(0, 12), before=self.display)
            self.active_canvas().draw_idle()

    def on_hover(self, event):
        axis = self.time_axis_for.get(event.inaxes, event.inaxes)
        if axis is self.orbit_axes:
            index = self.star_index
            position = self.trajectories[index, self.step]
            fields = [f"t = {self.times[self.step]:.9g}", f"x = {position[0]:.8e}",
                      f"y = {position[1]:.8e}", f"z = {position[2]:.8e}",
                      f"|v| = {np.linalg.norm(self.velocities[self.step, index]):.8e}",
                      f"|a| = {np.linalg.norm(self.accelerations[self.step, index]):.8e}"]
            self.page_readouts[self.figure].set(self.format_readout(f"Orbit | selected Star {index + 1}", fields))
            self.hover_selection = (axis, self.step)
            return
        item = self.hover_items.get(axis)
        if item is None or event.xdata is None:
            self.clear_hover(event)
            return
        index = self.nearest_index(item["times"], event.xdata)
        selection = (axis, index)
        if selection == self.hover_selection:
            return
        self.hover_selection = selection
        fields = [f"t = {item['times'][index]:.9g}"]
        for key, values in item["series"].items():
            fields.append("dt = NA (initial state)" if key == "dt" and index == 0
                          else f"{key} = {values[index]:.8e}")
        self.page_readouts[axis.figure].set(self.format_readout(item["title"], fields))

    def seek_plot_event(self, event):
        if self.drag_axis is None or event.x is None:
            return
        # Transform pixels through the axis where dragging started, even outside it.
        time_value = self.drag_axis.transData.inverted().transform((event.x, event.y or 0))[0]
        time_value = float(np.clip(time_value, self.times[0], self.times[-1]))
        index = self.nearest_index(self.times, time_value)
        if index != self.step:
            self.draw_frame(index)

    def on_plot_motion(self, event):
        if self.drag_axis is not None:
            self.seek_plot_event(event)
        else:
            self.on_hover(event)

    def on_plot_press(self, event):
        toolbar = (self.toolbar if event.canvas is self.canvas else
                   self.position_toolbar if event.canvas is getattr(self, "position_canvas", None) else
                   self.drift_toolbar if event.canvas is self.drift_canvas else
                   self.derivative_toolbar if event.canvas is self.derivative_canvas else self.diagnostic_toolbar)
        self.mouse_press = None
        self.drag_axis = None
        if event.button != 1 or toolbar.mode or event.inaxes is None:
            return
        axis = self.time_axis_for.get(event.inaxes, event.inaxes)
        if axis in self.hover_items:
            self.pause()
            self.dragging = True
            self.drag_axis = axis
            self.seek_plot_event(event)
        elif event.inaxes is self.orbit_axes:
            self.mouse_press = (event.canvas, event.inaxes, event.x, event.y)

    def on_plot_release(self, event):
        if self.drag_axis is not None:
            self.seek_plot_event(event)
            self.drag_axis = None
            self.dragging = False
            return
        press, self.mouse_press = self.mouse_press, None
        if press is None or event.button != 1 or event.inaxes is not press[1]:
            return
        if event.x is None or event.y is None or np.hypot(event.x - press[2], event.y - press[3]) > 6:
            return
        positions = self.trajectories[:, self.step, :]
        x, y, _ = proj3d.proj_transform(positions[:, 0], positions[:, 1], positions[:, 2],
                                       self.orbit_axes.get_proj())
        pixels = self.orbit_axes.transData.transform(np.column_stack((x, y)))
        distances = np.linalg.norm(pixels - [event.x, event.y], axis=1)
        index = int(np.argmin(distances))
        if distances[index] <= 14:
            self.select_star(index)

    def fit_orbits(self):
        center = (self.trajectories[self.star_index, self.step] if self.follow_view_active
                  else np.zeros(3))
        # Fit the complete trajectory at the current frame without changing
        # the chosen follow target or allocating a second full trajectory.
        extent = 1e-6
        for dim in range(3):
            values = self.trajectories[:, :, dim]
            finite = values[np.isfinite(values)]
            if finite.size:
                extent = max(extent, 1.15 * abs(float(finite.min()) - center[dim]),
                             1.15 * abs(float(finite.max()) - center[dim]))
        self.set_orbit_limits(np.column_stack((center - extent, center + extent)))
        self.canvas.draw_idle()

    def choose_save_path(self, suffix, label, extra=""):
        if extra == "_energy_drift":
            original = self.metadata.get("history_png_path")
            output = Path(original).parent if original else RUN_HISTORY_DIRECTORY
            output.mkdir(parents=True, exist_ok=True)
            stamp = (Path(original).stem if original else
                     datetime.fromisoformat(self.metadata["run_started_at"]).strftime("%Y-%m-%d_%H-%M-%S"))
            result = self.drift_result
            name = f"{stamp}_energy_drift_all{result['fit_records']}_timeweighted{suffix}"
            return filedialog.asksaveasfilename(parent=self.root, title=label, initialdir=str(output),
                                                initialfile=name, defaultextension=suffix,
                                                filetypes=[(label, "*" + suffix)])
        if suffix == ".gif":
            output_path = self.animation_archive_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return filedialog.asksaveasfilename(
                parent=self.root, title=label, initialdir=str(output_path.parent),
                initialfile=output_path.name, defaultextension=suffix,
                filetypes=[(label, "*" + suffix)])
        output = Path(__file__).resolve().parents[1] / "Codex" / "N体加速度交互版"
        output.mkdir(parents=True, exist_ok=True)
        base = (f"{self.metadata['case_id']}_method_{self.metadata['method_id']}"
                f"_{self.metadata.get('compute_backend', 'cpu')}"
                f"_{self.metadata.get('adaptive_algorithm', 'timescale') if self.adaptive else 'fixed'}"
                f"_dtmax_{self.dt:g}_steps_{self.last_step}{extra}")
        if self.adaptive and self.metadata.get("adaptive_algorithm") == "acceleration_ratio":
            base += (f"_trigger_{self.metadata.get('acceleration_trigger_ratio', 2):g}"
                     f"_maxpower_{self.metadata.get('acceleration_max_power', 3)}_reference")
        elif self.adaptive and self.metadata.get("adaptive_algorithm") in WORK_HISTORY_CODES:
            base += (f"_N_{self.metadata['work_step_factor']:g}_K_{self.metadata['work_max_power']}"
                     f"_bound_{self.metadata.get('work_threshold', 1.0):g}")
            if self.metadata.get("work_preprobe_enabled"):
                base += "_preprobe"
            if self.work_symmetric:
                base += "_work_skill_test"
            if self.work_algorithm_code in WORK_ALIGNED_CODES and self.metadata.get("adaptive_algorithm") not in WORK_ALIGNED_CODES:
                # Older aligned files used the original algorithm code; keep exports distinct.
                base += "_align"
        return filedialog.asksaveasfilename(parent=self.root, title=label,
                                            initialdir=str(output), initialfile=base + suffix,
                                            defaultextension=suffix, filetypes=[(label, "*" + suffix)])


    def diagnostics_png(self):
        figure = self.diagnostic_figure
        original_size = figure.get_size_inches().copy()
        visibility = [line.get_visible() for line in self.diagnostic_cursors]
        try:
            for line in self.diagnostic_cursors:
                line.set_visible(False)
            # Archives use a fixed pixel size independent of the window dimensions.
            figure.set_size_inches(10.6, 9, forward=False)
            self.apply_figure_layout(figure, 1060, 900, for_export=True)
            output = BytesIO()
            figure.savefig(output, format="png", dpi=100, facecolor="white", bbox_inches=None)
            return output.getvalue()
        finally:
            figure.set_size_inches(original_size, forward=False)
            self.apply_figure_layout(figure, *(original_size * figure.dpi))
            for line, visible in zip(self.diagnostic_cursors, visibility):
                line.set_visible(visible)
            self.diagnostic_canvas.draw_idle()


    def save_image(self):
        self.pause()
        diagnostic = self.notebook.select() == str(self.diagnostics_tab)
        drift = self.notebook.select() == str(self.drift_tab)
        derivative = self.notebook.select() == str(self.derivative_tab)
        position = getattr(self, "position_tab", None) is not None and self.notebook.select() == str(self.position_tab)
        path = self.choose_save_path(
            ".png", "PNG image", "_position_error" if position else "_energy_drift" if drift else "_work_step_indicator" if derivative else
            "_diagnostics" if diagnostic else "_overview")
        if not path:
            return
        figure = self.active_canvas().figure
        original_size = figure.get_size_inches().copy()
        try:
            figure.set_size_inches(10.6, 9, forward=False)
            self.apply_figure_layout(figure, 1060, 900, for_export=True)
            figure.savefig(path, dpi=100, facecolor="white", bbox_inches=None)
            self.status_var.set(f"Image saved: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
        finally:
            figure.set_size_inches(original_size, forward=False)
            self.apply_figure_layout(figure, *(original_size * figure.dpi))
            self.active_canvas().draw_idle()

    def save_data(self):
        self.pause()
        path = self.choose_save_path(".npz", "NumPy data")
        if not path:
            return
        try:
            np.savez_compressed(
                path, times=self.times, step_sizes=self.step_sizes, positions=self.trajectories,
                accelerations=self.accelerations, velocities=self.velocities, energies=self.energies,
                diagnostic_times=self.diagnostic_times,
                diagnostic_energy=np.array([item["energy"] for item in self.diagnostics]),
                momentum=np.array([item["momentum"] for item in self.diagnostics]),
                angular_momentum=np.array([item["angular_momentum"] for item in self.diagnostics]),
                com_pos=np.array([item["com_pos"] for item in self.diagnostics]),
                com_v=np.array([item["com_v"] for item in self.diagnostics]),
                metadata=json.dumps(self.metadata, ensure_ascii=False),
                **projection_export_arrays(self.accelerations, self.velocities, self.times, self.step_sizes, self.metadata),
                **acceleration_derivative_export_arrays(self.accelerations, self.times),
                **getattr(self, "position_analysis_arrays", {}),
            )
            self.status_var.set(f"Saved every star and time step: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)

    def animation_archive_path(self):
        """Reuse the persisted PNG name, including any same-second timestamp suffix."""
        png_path = self.metadata.get("history_png_path")
        saved_id = self.metadata.get("history_saved_run_id")
        if not png_path or not saved_id or saved_id != str(self.metadata.get("run_id")):
            raise ValueError(
                "Save this run's history first using Retry saving run history. "
                "The animation will then use exactly the same timestamp as its data and image.")
        return Path(png_path).with_suffix(".gif")

    def save_animation(self):
        self.pause()
        try:
            path = self.choose_save_path(".gif", "Save animation (GIF)")
        except Exception as exc:
            messagebox.showerror("Animation unavailable", str(exc), parent=self.root)
            return
        if not path:
            return
        frame_count = simpledialog.askinteger(
            "Animation frames", "How many frames should be exported? More frames take longer and produce larger files.\n"
                        "The animation covers the full simulation interval without changing the original records.",
            parent=self.root, initialvalue=min(300, self.frame_count),
            minvalue=2, maxvalue=min(2000, self.frame_count))
        if frame_count is None:
            return
        original_step = self.step
        hover_texts = [(variable, variable.get()) for variable in self.page_readouts.values()]
        for variable, _ in hover_texts:
            variable.set("")
        # Sample GIF frames in physical time, not adaptive step number.
        frame_times = np.linspace(self.times[0], self.times[-1], frame_count)
        frames = np.array([self.nearest_index(self.times, t) for t in frame_times])
        temporary_path = None
        original_figure_size = self.figure.get_size_inches().copy()
        self.save_animation_button.configure(state="disabled")
        self.timeline.begin_export(self)
        try:
            self.status_var.set(f"Saving animation ({len(frames)} frames). Please wait...")
            self.root.update_idletasks()
            destination = Path(path)
            handle, temporary_name = tempfile.mkstemp(
                prefix="." + destination.stem + ".", suffix=".gif", dir=destination.parent)
            os.close(handle)
            temporary_path = Path(temporary_name)
            self.figure.set_size_inches(10.6, 9, forward=False)
            self.apply_figure_layout(self.figure, 1060, 900, for_export=True)
            writer = PillowWriter(fps=20)
            with writer.saving(self.figure, temporary_path, dpi=100):
                for index, frame in enumerate(frames):
                    self.draw_frame(frame, redraw=False)
                    writer.grab_frame()
                    self.export_progress(index, len(frames))
            # Publish only the completed GIF; a failed export preserves any existing animation.
            os.replace(temporary_path, destination)
            self.status_var.set(f"Animation saved: {path}")
        except Exception as exc:
            self.status_var.set("Animation export failed; existing files were preserved.")
            messagebox.showerror("Save failed", str(exc), parent=self.root)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self.figure.set_size_inches(original_figure_size, forward=False)
            self.apply_figure_layout(self.figure, *(original_figure_size * self.figure.dpi))
            self.save_animation_button.configure(state="normal")
            for variable, text in hover_texts:
                variable.set(text)
            self.draw_frame(original_step)
            self.timeline.finish_export(self)

    def export_progress(self, current, total):
        self.status_var.set(f"Saving animation: {current + 1} / {total} frames")
        self.timeline.update_export(self, current, total)
        self.root.update_idletasks()

    def close(self):
        self.closed = True
        self.timeline.detach(self)
        self.status_var.trace_remove("write", self.status_trace)
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        canvases = [self.canvas, self.diagnostic_canvas, self.drift_canvas, self.derivative_canvas]
        if hasattr(self, "position_canvas"):
            canvases.append(self.position_canvas)
        for canvas in canvases:
            idle_id = getattr(canvas, "_idle_draw_id", None)
            if idle_id:
                # Cancel through the widget that registered the Tk command,
                # so its callback registry is cleaned before window destruction.
                canvas.get_tk_widget().after_cancel(idle_id)
                canvas._idle_draw_id = None
        self.figure.clear()
        self.diagnostic_figure.clear()
        self.drift_figure.clear()
        self.derivative_figure.clear()
        if hasattr(self, "position_figure"):
            self.position_figure.clear()
        self.root.destroy()

    def run(self):
        self.root.mainloop()




def prepare_batch_plan(plan):
    """Resolve every override before starting; freeze the seed across the batch."""
    if not isinstance(plan, dict) or not {"base", "runs"} <= set(plan) or set(plan) - {"base", "runs", "comparison"}:
        raise ValueError('The plan must contain "base", "runs", and optionally "comparison".')
    base, runs = plan["base"], plan["runs"]
    if not isinstance(base, dict) or not isinstance(runs, list) or not runs:
        raise ValueError("base must be an object; runs must be a nonempty list.")
    allowed = {"initial_mode", "seed", "orbit_index", "dt", "total_time", "method",
               "adaptive_enabled", "adaptive_algorithm", "adaptive_eta",
               "acceleration_trigger_ratio", "acceleration_max_power", "work_step_factor", "work_max_power", "work_threshold",
               "work_preprobe_enabled", "work_protection_before", "work_protection_after",
               "binary_eccentricity", "dop853_rtol", "dop853_atol",
               "step_energy_lower", "step_energy_upper", "energy_drift_limit",
               "sample_every", "compute_backend", "cpu_threads", "particle_count"}
    resolved = []
    for index, override in enumerate(runs, 1):
        if not isinstance(override, dict):
            raise ValueError(f"Run {index}: parameters must be an object.")
        config = dict(base)
        config.update(override)
        for retired_key in ("energy_relative_tolerance", "energy_max_rejections"):
            config.pop(retired_key, None)
        unknown = set(config) - allowed - {"label"}
        if unknown:
            raise ValueError(f"Run {index}: unknown parameters: {sorted(unknown)}")
        label = str(config.pop("label", f"Run {index}"))
        if "adaptive_enabled" in config and not isinstance(config["adaptive_enabled"], bool):
            raise ValueError(f"Run {index}: adaptive_enabled must be true or false.")
        config["method"] = {label: code for code, label in METHOD_CHOICES.items()}.get(config.get("method"), config.get("method"))
        try:
            config = validate_config(config)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Run {index}: {exc}") from exc
        if config["initial_mode"] == "random":
            config["initial_mode"] = "manual"
        resolved.append({"label": label, "config": config})
    return resolved


def save_batch_arrays(result):
    """Atomically archive all retained arrays beside this run's timestamped PNG."""
    metadata = result["metadata"]
    path = Path(metadata["history_png_path"]).with_suffix(".npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["metadata"]))["run_id"] == metadata["run_id"]:
                return path
        raise FileExistsError(f"Refusing to overwrite another run: {path}")
    diagnostics = result["diagnostics"]
    arrays = dict(times=result["record_times"], step_sizes=result["step_sizes"],
                  positions=result["trajectories"], accelerations=result["accelerations"],
                  velocities=result["velocities"], energies=result["energies"],
                  diagnostic_times=result["diagnostic_times"],
                  metadata=json.dumps(metadata, ensure_ascii=False))
    arrays.update(projection_export_arrays(result["accelerations"], result["velocities"], result["record_times"], result["step_sizes"], metadata))
    arrays.update(acceleration_derivative_export_arrays(result["accelerations"], result["record_times"]))
    arrays.update(position_export_arrays(result["trajectories"], result["record_times"], metadata))
    for key in ("energy", "momentum", "angular_momentum", "com_pos", "com_v"):
        arrays["diagnostic_energy" if key == "energy" else key] = np.asarray([d[key] for d in diagnostics])
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        # Windows rename refuses to overwrite an existing destination.
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def publish_batch_position_columns(manifest, directory):
    """Overlay reference metrics on history without modifying immutable run summaries."""
    options = manifest.get("comparison", {})
    if not (options.get("rmse") or options.get("nrmse")):
        return
    ref = options["reference"]
    jobs = manifest["jobs"]
    for i, job in enumerate(jobs, 1):
        if job["status"] != "saved":
            continue
        if is_kepler_config(job.get("config", {})):
            # The analytic Kepler metric is automatic; keep batch comparisons separate.
            continue
        config = job.get("config", {})
        if config.get("particle_count") == 2 and config.get("initial_mode") != "orbit":
            continue
        row = next((r for r in manifest.get("comparisons", []) if r["test"] == i and r["against"] == ref), {})
        count = sum(r["against"] == ref and r["test"] != ref and r["status"] == "saved"
                    for r in manifest.get("comparisons", []))
        marker = f"reference_{count}"
        rmse = (marker if i == ref else row.get("rmse")) if options["rmse"] else None
        nrmse = (marker if i == ref else row.get("nrmse")) if options["nrmse"] else None
        entry = dict(run_id=job["run_id"], reference_test=ref, reference_run_id=jobs[ref-1].get("run_id"),
                     reference_comparison_count=count if i == ref else None,
                     rmse="NA" if rmse is None else rmse, nrmse="NA" if nrmse is None else nrmse,
                     status="saved" if i == ref else row.get("status", "pending"), settings=options)
        path = directory / "_position_comparisons" / _archive_record_path(directory, job["run_id"]).name
        _archive_write(path, diagnostic_json_bytes(entry))
    rebuild_run_history(directory)


def format_acceleration_ratio(variable):
    """Normalize the ratio without rounding away valid float64 input digits."""
    try:
        value = float(variable.get())
        if np.isfinite(value):
            variable.set(str(value))
    except (ValueError, TypeError):
        pass


def scientific_dt_text(value):
    """Format dt without rounding or fixing the number of mantissa digits."""
    from decimal import Decimal
    return format(Decimal(str(value)), "e")


def adjust_time_step(variable, factor, parent, scientific=False, quantity="Time step"):
    """Change the visible dt without starting a run or changing adaptive settings."""
    from decimal import Decimal, InvalidOperation, localcontext
    try:
        entered = Decimal(variable.get())
        multiplier = Decimal(str(factor))
        with localcontext() as context:
            context.prec = max(28, len(entered.as_tuple().digits) + len(multiplier.as_tuple().digits) + 2)
            value = entered * multiplier
        if not value.is_finite() or value <= 0 or not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{quantity} must remain finite and positive.")
        variable.set(scientific_dt_text(value) if scientific else str(value.normalize()))
        return True
    except (ValueError, InvalidOperation, OverflowError) as exc:
        messagebox.showerror(f"Invalid {quantity.lower()}", str(exc), parent=parent)
        return False


def time_step_controls(parent, variable, dialog_parent, scientific=False, quantity="Time step"):
    frame = ttk.Frame(parent)
    buttons = []
    for text, factor in (("÷2", .5), ("×2", 2), ("÷5", .2), ("×5", 5), ("÷10", .1), ("×10", 10)):
        button = ttk.Button(frame, text=text, width=3, padding=1,
                            command=lambda f=factor: adjust_time_step(variable, f, dialog_parent, scientific=scientific, quantity=quantity))
        button.pack(side="left", padx=(0, 2))
        buttons.append(button)
    return frame, buttons


def validate_comparison_settings(settings, jobs):
    """Use one reference, normalization length and physical time grid for the batch."""
    settings = {} if settings is None else settings
    if not isinstance(settings, dict):
        raise ValueError("comparison must be an object.")
    allowed = {"rmse", "nrmse", "previous", "reference", "samples", "auto_reference"}
    if set(settings) - allowed:
        raise ValueError("Unknown comparison settings.")
    result = dict(rmse=False, nrmse=False, previous=False, reference=1, samples=1001, auto_reference=True)
    result.update(settings)
    for key in ("rmse", "nrmse", "previous", "auto_reference"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be true or false.")
    for key, low, high in (("reference", 1, len(jobs)), ("samples", 2, 100000)):
        value = float(result[key])
        if not np.isfinite(value) or value != int(value) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}.")
        result[key] = int(value)
    if result["auto_reference"]:
        result["reference"] = min(range(len(jobs)), key=lambda i: jobs[i]["config"]["dt"]) + 1
    if result["rmse"] or result["nrmse"]:
        reference = make_initial_state(jobs[result["reference"] - 1]["config"])
        def state_array(stars):
            return np.asarray([[star["mass"], *star["pos"], *star["v"]] for star in stars])
        state = state_array(reference)
        for i, job in enumerate(jobs, 1):
            if not np.array_equal(state, state_array(make_initial_state(job["config"]))):
                raise ValueError(f"Test {i}: trajectory comparison requires identical initial masses, positions, velocities and particle order. Disable comparison to run unrelated cases.")
        masses, positions = state[:, 0], state[:, 1:4]
        center = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
        scale = float(np.sqrt(np.mean(np.sum((positions - center)**2, axis=1))))
        if result["nrmse"] and (not np.isfinite(scale) or scale <= 0):
            raise ValueError("NRMSE needs a positive initial RMS radius about the center of mass. Use RMSE only for this initial state.")
        result.update(length_scale=scale, time_start=0.0,
                      time_end=min(job["config"]["total_time"] for job in jobs),
                      alignment="cubic Hermite interpolation using saved positions and velocities; exact saved nodes retained",
                      normalization="unweighted initial RMS radius about the mass-weighted center of mass of the reference test",
                      metric="sqrt(mean over stars and uniform time samples of squared 3D position distance); endpoints included")
    return result


def batch_execution_order(jobs, reference):
    """Run the reference first, then stably sort all remaining jobs by configured dt."""
    return [reference] + sorted((i for i in range(len(jobs)) if i != reference),
                                key=lambda i: jobs[i]["config"]["dt"])


def ordered_batch_plan(jobs, comparison=None):
    """Resolve the reference before ordering, and preserve its identity in saved plans."""
    jobs = prepare_batch_plan(dict(base={}, runs=[dict(**job["config"], label=job["label"]) for job in jobs]))
    options = validate_comparison_settings(comparison, jobs)
    order = batch_execution_order(jobs, options["reference"] - 1)
    options["reference"] = order.index(options["reference"] - 1) + 1
    jobs = [jobs[i] for i in order]
    settings = {key: options[key] for key in ("rmse", "nrmse", "previous", "reference", "samples", "auto_reference")}
    plan = dict(base={}, runs=[dict(**job["config"], label=job["label"]) for job in jobs], comparison=settings)
    return jobs, options, plan


def positions_on_grid(path, grid):
    """Align each saved trajectory in physical time, without extrapolation."""
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=float)
        positions = np.asarray(data["positions"], dtype=float).transpose(1, 0, 2)
        velocities = np.asarray(data["velocities"], dtype=float)
        if (times.ndim != 1 or len(times) < 2 or positions.shape != velocities.shape
                or positions.shape[0] != len(times) or positions.shape[-1] != 3
                or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
                or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities))):
            raise ValueError(f"Invalid saved trajectory: {path}")
        if grid[0] < times[0] or grid[-1] > times[-1]:
            raise ValueError("Comparison times must lie inside both trajectories; extrapolation is disabled.")
        right = np.clip(np.searchsorted(times, grid, side="right"), 1, len(times)-1)
        left = right-1
        width = (times[right]-times[left])[:, None, None]
        u = ((grid-times[left])/(times[right]-times[left]))[:, None, None]
        aligned = ((2*u**3-3*u**2+1)*positions[left] + (u**3-2*u**2+u)*width*velocities[left]
                   + (-2*u**3+3*u**2)*positions[right] + (u**3-u**2)*width*velocities[right])
        # Avoid interpolation at exactly recorded sample times.
        nodes = np.searchsorted(times, grid)
        matched = times[np.minimum(nodes, len(times)-1)] == grid
        aligned[matched] = positions[nodes[matched]]
        if not np.all(np.isfinite(aligned)):
            raise ValueError("Non-finite interpolated positions.")
        return aligned


def update_batch_comparisons(manifest, directory):
    """Compute newly available pairs; a later reference is resolved automatically."""
    options = manifest.get("comparison", {})
    if not (options.get("rmse") or options.get("nrmse")):
        return
    jobs = manifest["jobs"]
    reference = options["reference"]
    old = {(row["test"], row["against"]): row for row in manifest.get("comparisons", [])}
    pairs = {}
    for test in range(1, len(jobs)+1):
        if test != reference:
            pairs.setdefault((test, reference), []).append("reference")
        if options["previous"] and test > 1:
            pairs.setdefault((test, test-1), []).append("previous")
    grid = np.linspace(options["time_start"], options["time_end"], options["samples"])
    rows = []
    for (test, against), kinds in pairs.items():
        row = old.get((test, against), dict(test=test, against=against, status="pending"))
        row["kind"] = "+".join(kinds)
        if row["status"] != "saved" and jobs[test-1]["status"] == jobs[against-1]["status"] == "saved":
            row.update(status="pending", message="")
            try:
                left = positions_on_grid(directory / jobs[test-1]["data"], grid)
                right = positions_on_grid(directory / jobs[against-1]["data"], grid)
                if left.shape != right.shape:
                    raise ValueError("Particle counts do not match.")
                squared = np.sum((left-right)**2, axis=2)
                per_star = np.sqrt(np.mean(squared, axis=0))
                rms = float(np.sqrt(np.mean(squared)))
                if not np.isfinite(rms):
                    raise ValueError("Non-finite position RMS; inspect trajectory scales.")
                row.update(status="saved", rmse=rms if options["rmse"] else None,
                           nrmse=rms/options["length_scale"] if options["nrmse"] else None,
                           per_star_rmse=per_star.tolist() if options["rmse"] else None,
                           per_star_nrmse=(per_star/options["length_scale"]).tolist() if options["nrmse"] else None)
            except Exception as exc:
                row.update(status="error", message=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    manifest["comparisons"] = rows


def comparison_dat(manifest):
    """Keep a readable comparison table with the definitions needed to reproduce it."""
    options = manifest["comparison"]
    lines = ["# Position comparisons: relative trajectory differences, not exact-solution errors.",
             f"# Reference: Test {options['reference']}; previous means the preceding executed run (reference first, then ascending dt).",
             f"# Common time interval: [{options['time_start']:.16g}, {options['time_end']:.16g}]; uniform samples: {options['samples']} (including endpoints).",
             f"# Length scale: {options['length_scale']:.16g}; {options['normalization']}.",
             f"# Alignment: {options['alignment']}.", f"# Metric: {options['metric']}.",
             "# Per-star values and full settings are stored in the companion batch JSON.", "", "",
             f"{'Test':>6} {'Against':>8} {'Kind':<20} {'Star':>6} {'RMSE':>23} {'NRMSE':>23} {'Status':<10}"]
    def number(value):
        return "NA" if value is None else f"{value:.16e}"
    for row in manifest.get("comparisons", []):
        def add(star, rms, nrms):
            lines.append(f"{row['test']:>6} {row['against']:>8} {row['kind']:<20} {str(star):>6} {number(rms):>23} {number(nrms):>23} {row['status']:<10}")
        add("All", row.get("rmse"), row.get("nrmse"))
        per = row.get("per_star_rmse") or row.get("per_star_nrmse") or []
        for i in range(len(per)):
            add(i+1, row['per_star_rmse'][i] if row.get('per_star_rmse') else None,
                row['per_star_nrmse'][i] if row.get('per_star_nrmse') else None)
        if row.get("message"):
            lines.append(f"# Test {row['test']} vs {row['against']}: {row['message']}")
    return "\n".join(lines) + "\n"


class BatchPlanEditor:
    """Edit numbered test cases with the same parameter choices as the main page."""
    def __init__(self, app, base):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Test mode · Configure runs, then start the batch")
        self.window.geometry("1120x800")
        self.window.minsize(920, 730)
        self.window.transient(app.root)
        self.window.grab_set()
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(2, weight=1)
        self.jobs = [dict(label="Test 1", config=copy.deepcopy(base))]
        self.current = 0
        self.vars = {}
        self.widgets = {}
        # Keep each algorithm's values while one parameter/limit pair is visible.
        for key, default in (("adaptive_eta", "0.03"), ("acceleration_trigger_ratio", "2.0"),
                             ("acceleration_max_power", "7"), ("work_step_factor", str(WORK_DEFAULT_FACTOR)),
                             ("work_max_power", str(WORK_DEFAULT_MAX_POWER)),
                             ("dop853_rtol", str(DOP853_DEFAULT_RTOL)), ("dop853_atol", str(DOP853_DEFAULT_ATOL)),
                             *ENERGY_BAND_DEFAULTS.items()):
            self.vars[key] = tk.StringVar(value=default)
        top = ttk.Frame(self.window, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Select test").pack(side="left")
        self.selector = ttk.Combobox(top, state="readonly", width=32)
        self.selector.pack(side="left", padx=8)
        self.selector.bind("<<ComboboxSelected>>", self.select_test)
        ttk.Button(top, text="Save this test", command=self.save_current).pack(side="left", padx=4)
        ttk.Button(top, text="Add next test", command=self.add_test).pack(side="left", padx=4)
        ttk.Button(top, text="Delete this test", command=self.delete_test).pack(side="left", padx=4)
        form = ttk.LabelFrame(self.window, text="Parameters for the selected test", padding=12)
        form.grid(row=1, column=0, sticky="ew", padx=12)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)
        fields = [
            ("label", "Test label", None),
            ("initial_mode", "Initial conditions", ["Specified seed", "Known orbit"]),
            ("seed", "Seed", None),
            ("orbit_index", "Known orbit", [f"{i+1}. {o['name']}" for i, o in enumerate(STABLE_ORBITS)]),
            ("dt", "dt / upper limit", None), ("total_time", "Simulation duration T", None),
            ("method", "Integrator", list(METHOD_CHOICES.values())),
            ("sample_every", "Diagnostic interval (steps)", None),
            ("adaptive_enabled", "Enable adaptive time step", ["Off", "On"]),
            ("adaptive_algorithm", "Adaptive algorithm", list(ADAPTIVE_ALGORITHMS.values())),
            ("adaptive_parameter", "Adaptive parameter", None),
            ("adaptive_limit", "Maximum exponent K", None),
            ("compute_backend", "Compute device", STANDARD_BACKEND_LABELS),
            ("cpu_threads", "CPU threads (0 = auto)", None),
            ("particle_count", "Random particle count", None),
            ("work_threshold", "Trigger Threshold (dimensionless)", None),
            ("binary_eccentricity", "Binary eccentricity e", None),
        ]
        for i, (key, title, choices) in enumerate(fields):
            row, col = i // 2, 2 * (i % 2)
            field_label = ttk.Label(form, text=title)
            field_label.grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
            if key == "dt":
                self.dt_label = field_label
            if key == "adaptive_parameter":
                self.adaptive_parameter_label = field_label
            elif key == "adaptive_limit":
                self.adaptive_limit_label = field_label
            elif key == "work_threshold":
                self.work_threshold_label = field_label
            variable = tk.StringVar()
            self.vars[key] = variable
            scaled_time = key in ("dt", "total_time")
            host = ttk.Frame(form) if scaled_time else form
            if scaled_time:
                host.grid(row=row, column=col+1, sticky="ew", padx=(0, 12), pady=4)
                host.columnconfigure(0, weight=1)
            widget = (ttk.Combobox(host, textvariable=variable, values=choices, state="readonly",
                                   width=40 if key == "adaptive_algorithm" else 28)
                      if choices else ttk.Entry(host, textvariable=variable, width=28))
            if scaled_time:
                widget.grid(row=0, column=0, sticky="ew")
                controls, buttons = time_step_controls(host, variable, self.window, scientific=key == "dt",
                                                       quantity="Time step" if key == "dt" else "Simulation duration")
                if key == "dt":
                    self.dt_buttons = buttons
                    widget.bind("<FocusOut>", self.format_dt)
                    widget.bind("<Return>", self.format_dt)
                else:
                    self.duration_buttons = buttons
                controls.grid(row=1, column=0, sticky="w", pady=(3, 0))
            else:
                widget.grid(row=row, column=col+1, sticky="ew", padx=(0, 12), pady=4)
            self.widgets[key] = widget
            if key == "adaptive_parameter":
                widget.bind("<FocusOut>", self.format_adaptive_parameter)
                widget.bind("<Return>", self.format_adaptive_parameter)
            if choices:
                widget.bind("<<ComboboxSelected>>", self.sync_fields)
        self.widgets["orbit_index"].bind("<<ComboboxSelected>>", self.orbit_defaults)
        self.vars["work_preprobe_enabled"] = tk.BooleanVar(value=False)
        self.widgets["work_preprobe_enabled"] = ttk.Checkbutton(
            form, text=WORK_PREPROBE_LABEL, variable=self.vars["work_preprobe_enabled"])
        self.widgets["work_preprobe_enabled"].grid(row=(len(fields)+1)//2, column=0,
                                                  columnspan=2, sticky="w")
        ttk.Button(form, text="New random seed", command=lambda: self.vars["seed"].set(str(secrets.randbits(32)))).grid(
            row=(len(fields)+1)//2, column=3, sticky="w")
        self.auto_reference_var = tk.BooleanVar(value=True)
        self.rmse_var = tk.BooleanVar(value=False)
        self.nrmse_var = tk.BooleanVar(value=False)
        self.previous_var = tk.BooleanVar(value=False)
        self.samples_var = tk.StringVar(value="1001")
        comparison = ttk.LabelFrame(self.window, text="Position comparison · Shared by all tests", padding=8)
        comparison.grid(row=3, column=0, sticky="ew", padx=12)
        ttk.Checkbutton(comparison, text="Position RMSE", variable=self.rmse_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(comparison, text="Position NRMSE", variable=self.nrmse_var).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(comparison, text="Also compare with previous test", variable=self.previous_var).grid(row=0, column=2, columnspan=2, sticky="w", padx=12)
        ttk.Label(comparison, text="Reference test").grid(row=1, column=0, sticky="w", pady=5)
        self.reference_combo = ttk.Combobox(comparison, state="readonly", width=30)
        self.reference_combo.grid(row=1, column=1, sticky="ew")
        self.reference_combo.bind("<<ComboboxSelected>>", lambda event: self.refresh())
        ttk.Button(comparison, text="Use selected test as reference", command=self.use_selected_reference).grid(row=1, column=2, padx=12)
        ttk.Label(comparison, text="Uniform time samples").grid(row=1, column=3)
        ttk.Entry(comparison, textvariable=self.samples_var, width=8).grid(row=1, column=4, padx=8)
        ttk.Label(comparison, text="Same initial state required. Shared interval: 0 to shortest duration. NRMSE uses one fixed initial RMS radius.").grid(row=2, column=0, columnspan=5, sticky="w")
        ttk.Checkbutton(comparison, text="Automatic reference: smallest configured dt", variable=self.auto_reference_var,
                        command=self.refresh).grid(row=3, column=0, columnspan=3, sticky="w")
        ttk.Label(comparison, text="Reference runs first; remaining tests run from smallest to largest dt.").grid(row=3, column=3, columnspan=2, sticky="w")
        preview = ttk.Frame(self.window)
        preview.grid(row=2, column=0, sticky="nsew", padx=12, pady=8)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        columns = (
            ("order", "Order", 55, "center"),
            ("role", "Role", 85, "w"),
            ("test", "Test / Label", 125, "w"),
            ("source", "Initial conditions", 300, "w"),
            ("method", "Method", 85, "w"),
            ("dt", "DT", 185, "e"),
            ("duration", "Duration T", 100, "e"),
            ("adaptive", "Adaptive method", 340, "w"),
            ("device", "Device", 200, "center"),
        )
        self.summary = ttk.Treeview(
            preview, columns=[item[0] for item in columns],
            show="headings", height=5, selectmode="none",
        )
        for key, title, width, anchor in columns:
            self.summary.heading(key, text=title, anchor=anchor)
            self.summary.column(key, width=width, minwidth=width,
                                anchor=anchor, stretch=(key == "source"))
        self.summary.tag_configure("reference", background="#e6f0ff")
        self.summary.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(preview, orient="vertical", command=self.summary.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(preview, orient="horizontal", command=self.summary.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.summary.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.status = tk.StringVar(value="Edit Test 1, save it, then add the next test. New tests copy the selected test's parameters.")
        ttk.Label(self.window, textvariable=self.status, wraplength=1000, padding=(12, 2)).grid(row=4, column=0, sticky="ew")
        bar = ttk.Frame(self.window, padding=12)
        bar.grid(row=5, column=0, sticky="ew")
        ttk.Button(bar, text="Load plan...", command=self.load_plan).pack(side="left")
        ttk.Button(bar, text="Save plan...", command=self.save_plan).pack(side="left", padx=8)
        ttk.Label(bar, text="Output: graph and data (PNG + NPZ + run summary)").pack(side="left", padx=8)
        ttk.Button(bar, text="Start all tests", command=self.start).pack(side="right")
        self.refresh()
        self.load_current()

    def format_adaptive_parameter(self, event=None):
        if self.vars["adaptive_algorithm"].get() == ADAPTIVE_ALGORITHMS["acceleration_ratio"]:
            format_acceleration_ratio(self.vars["acceleration_trigger_ratio"])
        elif self.vars["adaptive_algorithm"].get() in [ADAPTIVE_ALGORITHMS[key] for key in WORK_ALGORITHM_CODES]:
            format_acceleration_ratio(self.vars["work_step_factor"])

    def sync_fields(self, event=None):
        taylor45 = self.vars["method"].get() in [METHOD_CHOICES[key] for key in TAYLOR_METHOD_ORDERS]
        sync_integrator_backend(self, self.vars["compute_backend"], self.widgets["compute_backend"], taylor45)
        dop853 = self.vars["method"].get() == "DOP853"
        choices = ([ADAPTIVE_ALGORITHMS[k] for k in ("dop853", "step_energy_band", "acceleration_ratio", *WORK_ALGORITHM_CODES)] if dop853 else
                   [v for k, v in ADAPTIVE_ALGORITHMS.items() if k != "dop853"])
        self.widgets["adaptive_algorithm"].configure(values=choices)
        if self.vars["adaptive_algorithm"].get() not in choices:
            self.vars["adaptive_algorithm"].set(choices[0] if dop853 else ADAPTIVE_ALGORITHMS["acceleration_ratio"])
        orbit = self.vars["initial_mode"].get() == "Known orbit"
        enabled = self.vars["adaptive_enabled"].get() == "On"
        band = self.vars["adaptive_algorithm"].get() == ADAPTIVE_ALGORITHMS["step_energy_band"]
        self.dt_label.configure(text="Initial trial dt" if band and enabled else "dt / upper limit")
        native = dop853 and self.vars["adaptive_algorithm"].get() == ADAPTIVE_ALGORITHMS["dop853"]
        timescale = self.vars["adaptive_algorithm"].get() == "Time-scale method"
        ratio = self.vars["adaptive_algorithm"].get() == ADAPTIVE_ALGORITHMS["acceleration_ratio"]
        work = self.vars["adaptive_algorithm"].get() in [ADAPTIVE_ALGORITHMS[key] for key in WORK_ALGORITHM_CODES]
        parameter_key, parameter_label = (
            ("step_energy_lower", "Step energy lower L") if band else
            ("dop853_rtol", "Relative tolerance rtol") if native else
            ("adaptive_eta", "Safety factor") if timescale else
            ("work_step_factor", "Shrink / recovery factor N") if work else
            ("acceleration_trigger_ratio", "Acceleration ratio N"))
        self.adaptive_parameter_label.configure(text=parameter_label)
        self.widgets["adaptive_parameter"].configure(textvariable=self.vars[parameter_key])
        self.adaptive_limit_label.configure(text="Step energy upper U" if band else "Absolute tolerance atol" if native else "Maximum exponent K")
        self.widgets["adaptive_limit"].configure(textvariable=self.vars[
            "step_energy_upper" if band else "dop853_atol" if native else "work_max_power" if work else "acceleration_max_power"])
        self.work_threshold_label.configure(text="Node drift limit (0 = Off)" if band else "Trigger Threshold (dimensionless)")
        self.widgets["work_threshold"].configure(textvariable=self.vars["energy_drift_limit" if band else "work_threshold"])
        for key, active in (("seed", not orbit), ("particle_count", not orbit), ("orbit_index", orbit),
                            ("binary_eccentricity", orbit and STABLE_ORBITS[self.widgets["orbit_index"].current()].get("kepler_binary", False)),
                            ("adaptive_algorithm", enabled), ("adaptive_parameter", enabled),
                            ("adaptive_limit", enabled and (band or native or ratio or work)),
                            ("work_threshold", enabled and (band or work)),
                            ("work_preprobe_enabled", enabled and work),
                            ("cpu_threads", BACKEND_CODES.get(self.vars["compute_backend"].get()) == "cpu_parallel")):
            widget = self.widgets[key]
            widget.configure(state=("readonly" if isinstance(widget, ttk.Combobox) else "normal") if active else "disabled")

    def orbit_defaults(self, event=None):
        orbit = STABLE_ORBITS[self.widgets["orbit_index"].current()]
        self.vars["dt"].set(scientific_dt_text(orbit["dt"]))
        self.vars["total_time"].set(str(orbit["dt"] * orbit["tot_time"]))
        self.vars["particle_count"].set(str(orbit["N"]))
        if orbit.get("recommended_method"):
            self.vars["method"].set("Leapfrog" if str(orbit["recommended_method"]) == "1" else "RK4")
        self.sync_fields()

    def format_dt(self, event=None):
        # Wait until editing is complete, so intermediate text such as "1e-" is allowed.
        from decimal import InvalidOperation
        try:
            value = scientific_dt_text(self.vars["dt"].get())
        except (InvalidOperation, ValueError):
            return
        self.vars["dt"].set(value)

    def load_current(self):
        job = self.jobs[self.current]
        config = job["config"]
        for key, variable in self.vars.items():
            if key != "work_preprobe_enabled":
                variable.set(str(config.get(key, "")))
        self.vars["work_preprobe_enabled"].set(bool(config.get("work_preprobe_enabled", False)))
        self.vars["work_step_factor"].set(str(config.get("work_step_factor", WORK_DEFAULT_FACTOR)))
        self.vars["work_max_power"].set(str(config.get("work_max_power", WORK_DEFAULT_MAX_POWER)))
        self.vars["work_threshold"].set(str(config.get("work_threshold", WORK_DEFAULT_THRESHOLD)))
        self.vars["binary_eccentricity"].set(str(config.get("binary_eccentricity", DEFAULT_ECCENTRICITY)))
        self.vars["dop853_rtol"].set(str(config.get("dop853_rtol", DOP853_DEFAULT_RTOL)))
        self.vars["dop853_atol"].set(str(config.get("dop853_atol", DOP853_DEFAULT_ATOL)))
        for key, default in ENERGY_BAND_DEFAULTS.items():
            self.vars[key].set(str(config.get(key, default)))
        self.vars["dt"].set(scientific_dt_text(config["dt"]))
        format_acceleration_ratio(self.vars["acceleration_trigger_ratio"])
        self.vars["label"].set(job["label"])
        self.vars["initial_mode"].set("Known orbit" if config["initial_mode"] == "orbit" else "Specified seed")
        self.widgets["orbit_index"].current(max(0, config.get("orbit_index", 0)))
        self.vars["method"].set(METHOD_CHOICES[config["method"]])
        self.vars["adaptive_enabled"].set("On" if config["adaptive_enabled"] else "Off")
        self.vars["adaptive_algorithm"].set(ADAPTIVE_ALGORITHMS[config["adaptive_algorithm"]])
        backend_label = TAYLOR_FLOAT_LABEL if config["method"] in TAYLOR_METHOD_ORDERS and config["compute_backend"] == "cpu_array" else COMPUTE_BACKENDS[config["compute_backend"]]
        self.vars["compute_backend"].set(backend_label)
        self.vars["cpu_threads"].set(str(config.get("cpu_threads", 0)))
        self.sync_fields()

    def read_current(self):
        self.format_dt()
        config = {key: variable.get() for key, variable in self.vars.items()}
        config.pop("adaptive_parameter")
        config.pop("adaptive_limit")
        config["initial_mode"] = "orbit" if config["initial_mode"] == "Known orbit" else "manual"
        config["orbit_index"] = self.widgets["orbit_index"].current()
        config["adaptive_enabled"] = config["adaptive_enabled"] == "On"
        config["adaptive_algorithm"] = ADAPTIVE_CODES[config["adaptive_algorithm"]]
        config["compute_backend"] = BACKEND_CODES[config["compute_backend"]]
        if config["compute_backend"] != "cpu_parallel":
            config["cpu_threads"] = 0
        if config["initial_mode"] == "orbit":
            orbit = STABLE_ORBITS[config["orbit_index"]]
            config.update(seed=orbit["seed"], particle_count=orbit["N"])
        return prepare_batch_plan(dict(base={}, runs=[config]))[0]

    def save_current(self):
        try:
            self.jobs[self.current] = self.read_current()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid test parameters", str(exc), parent=self.window)
            return False
        self.refresh()
        self.status.set(f"Test {self.current + 1} settings saved. Save plan exports all settings for reuse.")
        return True

    def use_selected_reference(self):
        self.auto_reference_var.set(False)
        self.reference_combo.configure(state="readonly")
        self.reference_combo.current(self.current)
        self.refresh()

    def refresh(self):
        reference = self.reference_combo.current()
        if self.auto_reference_var.get():
            reference = min(range(len(self.jobs)), key=lambda i: self.jobs[i]["config"]["dt"])
        self.reference_combo.configure(state="disabled" if self.auto_reference_var.get() else "readonly")
        self.reference_combo.configure(values=[f"{i+1}. {job['label']}" for i, job in enumerate(self.jobs)])
        self.reference_combo.current(min(max(reference, 0), len(self.jobs)-1))
        self.selector.configure(values=[f"{i+1}. {job['label']}" for i, job in enumerate(self.jobs)])
        self.selector.current(self.current)
        self.summary.delete(*self.summary.get_children())
        ordered = [(i+1, self.jobs[i]) for i in batch_execution_order(self.jobs, self.reference_combo.current())]
        for rank, (i, job) in enumerate(ordered, 1):
            c = job["config"]
            source = STABLE_ORBITS[c["orbit_index"]]["name"] if c["initial_mode"] == "orbit" else f"seed {c['seed']}"
            method = METHOD_CHOICES[c["method"]]
            adaptive = c["adaptive_algorithm"] if c["adaptive_enabled"] else "Off"
            if c["adaptive_enabled"] and c["adaptive_algorithm"] in WORK_ALGORITHM_CODES:
                adaptive = work_history_method(c["adaptive_algorithm"], c.get("work_preprobe_enabled", False))
            self.summary.insert(
                "", "end",
                values=("★" if rank == 1 else str(rank),
                        "Reference" if rank == 1 else "",
                        f"{i}: {job['label']}", source, method,
                        scientific_dt_text(c["dt"]), f"{c['total_time']:g}",
                        adaptive, backend_summary(dict(c, cpu_threads=c["cpu_threads"] or "auto"))),
                tags=("reference",) if rank == 1 else (),
            )

    def select_test(self, event=None):
        selected = self.selector.current()
        if self.save_current():
            self.current = selected
            self.refresh()
            self.load_current()

    def add_test(self):
        if not self.save_current():
            return
        job = copy.deepcopy(self.jobs[self.current])
        job["label"] = f"Test {len(self.jobs) + 1}"
        self.jobs.append(job)
        self.current = len(self.jobs) - 1
        self.refresh()
        self.load_current()

    def delete_test(self):
        if len(self.jobs) == 1:
            self.status.set("Keep at least one test in the plan.")
            return
        reference = self.reference_combo.current()
        removed = self.current
        self.jobs.pop(removed)
        # Preserve the referenced job when deleting an earlier test.
        self.reference_combo.current(0 if removed == reference else reference - (removed < reference))
        self.current = min(self.current, len(self.jobs) - 1)
        self.refresh()
        self.load_current()

    def comparison_settings(self):
        return dict(rmse=self.rmse_var.get(), nrmse=self.nrmse_var.get(), previous=self.previous_var.get(),
                    reference=self.reference_combo.current()+1, samples=self.samples_var.get(), auto_reference=self.auto_reference_var.get())

    def plan(self):
        return dict(base={}, runs=[dict(**job["config"], label=job["label"]) for job in self.jobs],
                    comparison=self.comparison_settings())

    def save_plan(self):
        if not self.save_current():
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".json", filetypes=[("Test plan", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps(self.plan(), indent=2), encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Plan not saved", str(exc), parent=self.window)

    def load_plan(self):
        path = filedialog.askopenfilename(parent=self.window, filetypes=[("Test plan", "*.json")])
        if not path:
            return
        try:
            plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            if "plan" in plan:
                plan = plan["plan"]
            jobs = prepare_batch_plan(plan)
            settings = validate_comparison_settings(plan.get("comparison"), jobs)
        except (ValueError, TypeError, OSError) as exc:
            messagebox.showerror("Invalid plan", str(exc), parent=self.window)
            return
        self.jobs, self.current = jobs, 0
        self.auto_reference_var.set(settings["auto_reference"])
        self.refresh()
        self.reference_combo.current(settings["reference"]-1)
        self.rmse_var.set(settings["rmse"])
        self.nrmse_var.set(settings["nrmse"])
        self.previous_var.set(settings["previous"])
        self.samples_var.set(str(settings["samples"]))
        self.refresh()
        self.load_current()

    def start(self):
        if not self.save_current():
            return
        try:
            self.app.start_batch(copy.deepcopy(self.jobs), self.comparison_settings())
        except Exception as exc:
            messagebox.showerror("Batch not started", str(exc), parent=self.window)
            return
        if self.app.running:
            self.window.destroy()


class NBodyApp:
    """Combine a parameter panel, background simulation, and replaceable result view."""
    def __init__(self, root=None):
        self.cpu_preparation = _CPU_STARTUP if _CPU_STARTUP is not None else CPUStartupPreparation()
        self.root = root if root is not None else tk.Tk()
        self.root.title("N-body simulation · Settings and interactive playback")
        width = min(1640, self.root.winfo_screenwidth() - 60)
        height = min(1080, self.root.winfo_screenheight() - 100)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(1100, width), min(740, height))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.viewer = None
        self.worker = None
        self.cancel_event = threading.Event()
        self.messages = self.cpu_preparation.messages
        self.cpu_warmup_done = self.cpu_preparation.done
        self.cpu_warmup_errors = self.cpu_preparation.errors
        self.cpu_warmup_worker = self.cpu_preparation.worker
        self.cpu_warmup_status = tk.StringVar(value="CPU single/multi: preparing at startup...")
        self.pending_history = []
        self.history_migration_pending = True
        self.closed = False
        self.poll_id = None
        self.running = False
        self.batch_jobs = []
        self.batch_index = 0
        self.batch_manifest = None
        self.batch_path = None
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei UI", 9))
        style.configure("TButton", font=("Microsoft YaHei UI", 9), padding=4)
        style.configure("Note.TLabel", foreground="#52606d", font=("Microsoft YaHei UI", 9))
        self.configuration_area = ttk.Frame(self.root)
        self.configuration_area.grid(row=0, column=0, sticky="ew")
        self.configuration_area.columnconfigure(0, weight=1)
        self.settings_visible = True
        self.settings_toggle = ttk.Button(
            self.configuration_area, text="Hide settings", command=self.toggle_settings)
        self.settings_toggle.grid(row=0, column=0, sticky="w", padx=12, pady=(3, 0))
        self.batch_button = ttk.Button(self.configuration_area, text="Test mode...", command=self.open_batch_editor)
        self.batch_button.grid(row=0, column=0, sticky="e", padx=12, pady=(3, 0))
        self.comparison_button = ttk.Button(self.configuration_area, text="Comparison results...", command=self.show_comparisons)
        self.comparison_button.grid(row=0, column=0, sticky="e", padx=(0, 125), pady=(3, 0))
        self.settings = ttk.LabelFrame(self.configuration_area,
                                      text="Simulation settings · Edit, then click Start / Rerun", padding=(12, 6))
        self.settings.grid(row=1, column=0, sticky="ew", padx=12, pady=(3, 0))
        for col in (1, 3, 5, 7):
            self.settings.columnconfigure(col, weight=1)
        self.mode_var = tk.StringVar(value="Known orbit")
        self.seed_var = tk.StringVar(value=str(secrets.randbits(32)))
        self.orbit_var = tk.StringVar()
        self.dt_var = tk.StringVar(value="1")
        self.duration_var = tk.StringVar(value="10000")
        self.method_var = tk.StringVar(value="Leapfrog (KDK)")
        self.dop853_rtol_var = tk.StringVar(value=str(DOP853_DEFAULT_RTOL))
        self.dop853_atol_var = tk.StringVar(value=str(DOP853_DEFAULT_ATOL))
        self.sample_var = tk.StringVar(value="10")
        self.adaptive_var = tk.BooleanVar(value=False)
        self.adaptive_algorithm_var = tk.StringVar(value=ADAPTIVE_ALGORITHMS["acceleration_ratio"])
        self.eta_var = tk.StringVar(value="0.03")
        self.trigger_ratio_var = tk.StringVar(value="2.0")
        self.work_factor_var = tk.StringVar(value=str(WORK_DEFAULT_FACTOR))
        self.work_max_power_var = tk.StringVar(value=str(WORK_DEFAULT_MAX_POWER))
        self.work_threshold_var = tk.StringVar(value=str(WORK_DEFAULT_THRESHOLD))
        self.work_preprobe_var = tk.BooleanVar(value=False)
        self.energy_band_vars = {key: tk.StringVar(value=str(value)) for key, value in ENERGY_BAND_DEFAULTS.items()}
        self.max_power_var = tk.StringVar(value="7")
        self.backend_var = tk.StringVar(value=COMPUTE_BACKENDS["cpu_parallel"])
        self.cpu_threads_var = tk.StringVar(value="0")
        self.particle_count_var = tk.StringVar(value="5")
        self.binary_eccentricity_var = tk.StringVar(value=str(DEFAULT_ECCENTRICITY))
        self.record_run_var = tk.BooleanVar(value=True)
        self.form_widgets = []

        def label(text, row, col):
            ttk.Label(self.settings, text=text).grid(row=row, column=col, sticky="w", padx=(0, 5), pady=3)

        def entry(variable, row, col, width=11):
            scaled_time = variable is self.dt_var or variable is self.duration_var
            host = ttk.Frame(self.settings) if scaled_time else self.settings
            if scaled_time:
                host.grid(row=row, column=col, sticky="ew", padx=(0, 12), pady=3)
                host.columnconfigure(0, weight=1)
            widget = ttk.Entry(host, textvariable=variable, width=width)
            if scaled_time:
                widget.grid(row=0, column=0, sticky="ew")
                controls, buttons = time_step_controls(host, variable, self.root, quantity="Time step" if variable is self.dt_var else "Simulation duration")
                controls.grid(row=1, column=0, sticky="w", pady=(3, 0))
                if variable is self.dt_var:
                    self.dt_buttons = buttons
                else:
                    self.duration_buttons = buttons
                self.form_widgets.extend(buttons)
            else:
                widget.grid(row=row, column=col, sticky="ew", padx=(0, 12), pady=3)
            self.form_widgets.append(widget)
            return widget

        label("Initial conditions", 0, 0)
        self.mode_combo = ttk.Combobox(self.settings, textvariable=self.mode_var, state="readonly",
                                      values=["Random seed", "Specified seed", "Known orbit"], width=17)
        self.mode_combo.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=3)
        self.form_widgets.append(self.mode_combo)
        label("Seed", 0, 2)
        self.seed_entry = entry(self.seed_var, 0, 3)
        label("Known orbit", 0, 4)
        self.orbit_combo = ttk.Combobox(
            self.settings, textvariable=self.orbit_var, state="readonly",
            values=[f"{i + 1}. {orbit['name']}" for i, orbit in enumerate(STABLE_ORBITS)], width=34)
        self.orbit_combo.grid(row=0, column=5, columnspan=3, sticky="ew", pady=3)
        self.orbit_combo.current(0)
        self.form_widgets.append(self.orbit_combo)
        self.dt_label = ttk.Label(self.settings, text="dt / upper limit")
        self.dt_label.grid(row=1, column=0, sticky="w", padx=(0, 5), pady=3)
        entry(self.dt_var, 1, 1)
        label("Simulation duration T", 1, 2)
        entry(self.duration_var, 1, 3)
        label("Integrator", 1, 4)
        self.method_combo = ttk.Combobox(self.settings, textvariable=self.method_var, state="readonly",
                                        values=["Leapfrog (KDK)", "RK4 (fourth order)", "DOP853 (eighth order)",
                                                *[METHOD_NAMES[key] for key in TAYLOR_METHOD_ORDERS]], width=22)
        self.method_combo.grid(row=1, column=5, sticky="ew", padx=(0, 12), pady=3)
        self.form_widgets.append(self.method_combo)
        self.method_combo.bind("<<ComboboxSelected>>", lambda event: self.sync_fields())
        label("Diagnostic interval (steps)", 1, 6)
        entry(self.sample_var, 1, 7, width=8)
        self.adaptive_check = ttk.Checkbutton(
            self.settings, text="Enable adaptive time step", variable=self.adaptive_var, command=self.sync_fields)
        self.adaptive_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        self.form_widgets.append(self.adaptive_check)
        self.work_threshold_label = ttk.Label(self.settings, text="Trigger Threshold")
        self.work_threshold_label.grid(row=2, column=2, sticky="w", padx=(0, 5), pady=3)
        self.work_threshold_host = ttk.Frame(self.settings)
        self.work_threshold_host.grid(row=2, column=3, sticky="ew", padx=(0, 12), pady=3)
        self.work_threshold_entry = ttk.Entry(self.work_threshold_host, textvariable=self.work_threshold_var, width=9)
        self.work_threshold_entry.pack(side="left", fill="x", expand=True)
        self.work_preprobe_check = ttk.Checkbutton(
            self.work_threshold_host, text=WORK_PREPROBE_LABEL, variable=self.work_preprobe_var, command=self.sync_fields)
        self.work_preprobe_check.pack(side="left", padx=(5, 0))
        self.form_widgets.extend((self.work_threshold_entry, self.work_preprobe_check))
        self.run_button = ttk.Button(self.settings, text="Start / Rerun", command=self.start_run)
        self.run_button.grid(row=2, column=4, columnspan=2, sticky="ew", padx=(0, 12), pady=3)
        self.cancel_button = ttk.Button(self.settings, text="Cancel run", command=self.cancel_run, state="disabled")
        self.cancel_button.grid(row=2, column=6, columnspan=2, sticky="ew", pady=3)
        label("Compute device", 3, 0)
        self.backend_combo = ttk.Combobox(
            self.settings, textvariable=self.backend_var, state="readonly",
            values=STANDARD_BACKEND_LABELS, width=28)
        self.backend_combo.grid(row=3, column=1, sticky="ew", padx=(0, 12), pady=3)
        self.form_widgets.append(self.backend_combo)
        self.backend_combo.bind("<<ComboboxSelected>>", lambda event: self.sync_fields())
        label("CPU threads (0 = auto)", 3, 4)
        self.cpu_threads_entry = entry(self.cpu_threads_var, 3, 5)
        self.particle_count_label = ttk.Label(self.settings, text="Random particle count")
        self.particle_count_label.grid(row=3, column=2, sticky="w", padx=(0, 5), pady=3)
        self.particle_count_entry = entry(self.particle_count_var, 3, 3)
        ttk.Label(self.settings, textvariable=self.cpu_warmup_status, wraplength=270,
                  style="Note.TLabel").grid(row=3, column=6, columnspan=2, sticky="w")
        label("Adaptive algorithm", 4, 0)
        self.adaptive_algorithm_combo = ttk.Combobox(
            self.settings, textvariable=self.adaptive_algorithm_var, state="disabled",
            values=list(ADAPTIVE_ALGORITHMS.values()), width=40)
        self.adaptive_algorithm_combo.grid(row=4, column=1, sticky="ew", padx=(0, 12), pady=3)
        self.adaptive_algorithm_combo.bind("<<ComboboxSelected>>", lambda event: self.sync_fields())
        self.form_widgets.append(self.adaptive_algorithm_combo)
        self.adaptive_parameter_label = ttk.Label(self.settings, text="Acceleration ratio N")
        self.adaptive_parameter_label.grid(row=4, column=2, sticky="w", padx=(0, 5), pady=3)
        self.adaptive_parameter_entry = entry(self.trigger_ratio_var, 4, 3)
        self.adaptive_parameter_entry.bind("<FocusOut>", self.format_adaptive_parameter)
        self.adaptive_parameter_entry.bind("<Return>", self.format_adaptive_parameter)
        self.adaptive_limit_label = ttk.Label(self.settings, text="Maximum exponent K")
        self.adaptive_limit_label.grid(row=4, column=4, sticky="w", padx=(0, 5), pady=3)
        self.max_power_entry = entry(self.max_power_var, 4, 5)
        self.adaptive_limit_note_var = tk.StringVar(value="Minimum suggested dt = initial dt / N^K")
        ttk.Label(self.settings, textvariable=self.adaptive_limit_note_var,
                  style="Note.TLabel").grid(row=4, column=6, columnspan=2, sticky="w")
        self.algorithm_note_var = tk.StringVar()
        ttk.Label(self.settings, textvariable=self.algorithm_note_var,
                  style="Note.TLabel", wraplength=width - 70).grid(
                      row=5, column=0, columnspan=8, sticky="w")
        self.hint_var = tk.StringVar()
        ttk.Label(self.settings, textvariable=self.hint_var, style="Note.TLabel",
                  wraplength=width - 70).grid(row=6, column=0, columnspan=8, sticky="w", pady=(4, 0))
        self.record_check = ttk.Checkbutton(self.settings, text="Record this run (single-run mode)", variable=self.record_run_var)
        self.record_check.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.form_widgets.append(self.record_check)
        ttk.Label(self.settings, text="Off: view results without automatic history or PNG. Test mode keeps its batch archives.",
                  style="Note.TLabel").grid(row=7, column=3, columnspan=5, sticky="w")
        label("Binary eccentricity e", 8, 0)
        self.binary_eccentricity_entry = entry(self.binary_eccentricity_var, 8, 1)
        ttk.Label(self.settings, text="Kepler binary only: 0 <= e < 1; a=1, masses=1+1; automatic analytic position RMSE.",
                  style="Note.TLabel").grid(row=8, column=2, columnspan=6, sticky="w")
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_source_change)
        self.orbit_combo.bind("<<ComboboxSelected>>", self.use_orbit_defaults)
        self.use_orbit_defaults()

        self.settings.bind("<Configure>", lambda event: [
            child.configure(wraplength=max(450, event.width - 28))
            for child in self.settings.winfo_children()
            if isinstance(child, ttk.Label) and int(child.cget("wraplength") or 0) > 0])
        self.result_host = ttk.Frame(self.root)
        self.result_host.grid(row=1, column=0, sticky="nsew")
        self.result_host.columnconfigure(0, weight=1)
        self.result_host.rowconfigure(0, weight=1)
        self.placeholder = ttk.Label(
            self.result_host, text="Select initial conditions and parameters above, then click Start / Rerun.\n\n"
            "Completed runs display trajectories, energy and per-star acceleration. Velocity values remain available for inspection and export.\n"
            "Drag the playback slider at the bottom or the yellow time cursor in a plot.",
            anchor="center", justify="center", font=("Microsoft YaHei UI", 13))
        self.placeholder.grid(row=0, column=0, sticky="nsew")
        self.timeline = SharedTimeline(self.root)
        self.progress_footer = self.timeline
        self.progress_footer.grid(row=2, column=0, sticky="ew")
        self.progress_var = self.timeline.value
        self.progress_bar = self.timeline.scale
        self.progress_text = self.timeline.message
        self.history_status = tk.StringVar()
        # Successful automatic saves reserve no footer space; show Retry only on a failure.
        self.retry_history_button = ttk.Button(
            self.progress_footer, text="Retry saving run history", command=self.save_pending_history,
            state="disabled")
        self.retry_history_button.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.retry_history_button.grid_remove()
        self.save_pending_history()
        self.poll_id = self.root.after(100, self.poll_worker)

    def show_comparisons(self):
        manifest = self.batch_manifest
        if not manifest or not manifest.get("comparisons"):
            messagebox.showinfo("Position comparisons", "Enable Position RMSE or NRMSE in Test mode and configure at least two tests.", parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("Position comparison results")
        window.geometry("980x460")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        options = manifest["comparison"]
        ttk.Label(window, text=f"Reference: Test {options['reference']} | Common time: 0 to {options['time_end']:.8g} | Samples: {options['samples']} | Length scale: {options['length_scale']:.8g}\n"
                  "Relative differences, not exact-solution errors. Test numbers follow execution order (reference first, then ascending dt). Previous = preceding run. Reopen to refresh.", padding=10).grid(row=0, column=0, sticky="ew")
        columns = ("test", "against", "kind", "star", "rmse", "nrmse", "status")
        tree = ttk.Treeview(window, columns=columns, show="headings")
        for key in columns:
            tree.heading(key, text=key.upper())
            tree.column(key, width=90 if key in ("test", "against", "star") else 145, anchor="center")
        tree.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(window, command=tree.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)
        for row in manifest["comparisons"]:
            def add(star, rms, nrms):
                tree.insert("", "end", values=(row['test'], row['against'], row['kind'], star,
                            "NA" if rms is None else f"{rms:.8e}", "NA" if nrms is None else f"{nrms:.8e}", row['status']))
            add("All", row.get("rmse"), row.get("nrmse"))
            per = row.get("per_star_rmse") or row.get("per_star_nrmse") or []
            for i in range(len(per)):
                add(i+1, row['per_star_rmse'][i] if row.get('per_star_rmse') else None,
                    row['per_star_nrmse'][i] if row.get('per_star_nrmse') else None)
        errors = [f"Test {r['test']} vs {r['against']}: {r.get('message', '')}" for r in manifest['comparisons'] if r['status'] == 'error']
        ttk.Label(window, text="\n".join(errors) if errors else f"Saved table: {self.batch_manifest.get('comparison_file', '')}", wraplength=930, padding=10).grid(row=2, column=0, sticky="ew")

    def open_batch_editor(self):
        if self.running:
            return
        try:
            base = self.read_config()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid base parameters", str(exc), parent=self.root)
            return
        self.seed_var.set(str(base["seed"]))
        BatchPlanEditor(self, base)

    def start_batch(self, jobs, comparison=None):
        if self.running:
            return
        if self.pending_history:
            messagebox.showerror("Archive pending", "Retry saving the previous run before starting a batch.", parent=self.root)
            return
        jobs, options, plan = ordered_batch_plan(jobs, comparison)
        stamp = datetime.now(gettz("America/Toronto")).strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.batch_path = RUN_HISTORY_DIRECTORY / f"batch_{stamp}.json"
        self.batch_manifest = dict(plan=plan, jobs=[dict(**job, status="pending") for job in jobs], comparison=options)
        update_batch_comparisons(self.batch_manifest, self.batch_path.parent)
        self.write_batch_manifest()
        self.batch_jobs = jobs
        self.batch_index = 0
        self.start_run(config=jobs[0]["config"])

    def write_batch_manifest(self):
        if self.batch_manifest.get("comparison", {}).get("rmse") or self.batch_manifest.get("comparison", {}).get("nrmse"):
            dat_path = self.batch_path.with_name(self.batch_path.stem + "_comparisons.dat")
            _archive_write(dat_path, comparison_dat(self.batch_manifest).encode("utf-8"))
            self.batch_manifest["comparison_file"] = dat_path.name
        _archive_write(self.batch_path, json.dumps(self.batch_manifest, indent=2, ensure_ascii=False).encode("utf-8"))
        publish_batch_position_columns(self.batch_manifest, self.batch_path.parent)

    def batch_finished_run(self, success, message=""):
        if not self.batch_jobs:
            return
        entry = self.batch_manifest["jobs"][self.batch_index]
        entry["status"] = "saved" if success else "stopped"
        entry["message"] = message
        if success:
            entry["plot"] = Path(self.viewer.metadata["history_png_path"]).name
            entry["data"] = Path(entry["plot"]).with_suffix(".npz").name
            entry["run_id"] = self.viewer.metadata["run_id"]
        try:
            update_batch_comparisons(self.batch_manifest, self.batch_path.parent)
            self.write_batch_manifest()
        except Exception as exc:
            success = False
            messagebox.showerror("Batch manifest not saved", str(exc), parent=self.root)
        total = len(self.batch_jobs)
        self.batch_index += 1
        if success and self.batch_index < total:
            self.start_run(config=self.batch_jobs[self.batch_index]["config"])
        else:
            self.batch_jobs = []
            comparison_errors = sum(row["status"] == "error" for row in self.batch_manifest.get("comparisons", []))
            if comparison_errors:
                messagebox.showerror("Position comparison failed", f"{comparison_errors} comparison(s) failed. Simulation files are saved. Open Comparison results for details.", parent=self.root)
            self.progress_text.set(f"Batch complete: {total}/{total} runs saved." if success else
                                   f"Batch stopped at run {self.batch_index}/{total}. Completed archives are retained. {message}")

    def toggle_settings(self, visible=None):
        self.settings_visible = not self.settings_visible if visible is None else bool(visible)
        if self.settings_visible:
            self.settings.grid()
        else:
            self.settings.grid_remove()
        self.settings_toggle.configure(text="Hide settings" if self.settings_visible else "Show settings")

    def save_pending_history(self, viewer=None):
        try:
            if self.history_migration_pending:
                imported = migrate_legacy_history()
                rebuild_run_history()
                self.history_migration_pending = False
                if imported:
                    self.history_status.set(f"Imported {imported} previous runs. Their missing plots are marked NA.")
            while self.pending_history:
                record = self.pending_history[0]
                if "png_bytes" not in record:
                    if viewer is not None and record["metadata"] is viewer.metadata:
                        record["png_bytes"] = viewer.diagnostics_png()
                    else:
                        # Render from retained arrays if an earlier export failed.
                        # A closed viewer's figure is cleared, so it must not be queued by reference.
                        temporary_root = tk.Toplevel(self.root)
                        temporary_root.withdraw()
                        temporary_viewer = None
                        try:
                            temporary_viewer = NBodyViewer(root=temporary_root, **record["result"])
                            record["png_bytes"] = temporary_viewer.diagnostics_png()
                        finally:
                            if temporary_viewer is not None:
                                temporary_viewer.close()
                            else:
                                temporary_root.destroy()
                path = save_run_history(record)
                if record.get("save_full_data"):
                    save_batch_arrays(record["result"])
                record.pop("result", None)
                self.pending_history.pop(0)
                self.history_status.set(f"Saved data + conservation plot. Clickable image index: {path.parent / 'index.html'}")
        except Exception as exc:
            self.history_status.set(
                f"Archive pending ({len(self.pending_history)} new runs): {exc}. "
                "Resolve the issue and click Retry before closing.")
            self.retry_history_button.configure(state="normal")
            self.retry_history_button.grid()
            messagebox.showerror("Run archive not saved", self.history_status.get(), parent=self.root)
            return False
        self.retry_history_button.configure(state="disabled")
        self.retry_history_button.grid_remove()
        return True

    def format_adaptive_parameter(self, event=None):
        if self.adaptive_algorithm_var.get() == ADAPTIVE_ALGORITHMS["acceleration_ratio"]:
            format_acceleration_ratio(self.trigger_ratio_var)
        elif self.adaptive_algorithm_var.get() in [ADAPTIVE_ALGORITHMS[key] for key in WORK_ALGORITHM_CODES]:
            format_acceleration_ratio(self.work_factor_var)

    def sync_fields(self):
        if self.running:
            return
        mode = self.mode_var.get()
        use_taylor45 = str(self.method_combo.current()+1) in TAYLOR_METHOD_ORDERS
        sync_integrator_backend(self, self.backend_var, self.backend_combo, use_taylor45)
        use_dop853 = self.method_combo.current() == 2
        choices = ([ADAPTIVE_ALGORITHMS[k] for k in ("dop853", "step_energy_band", "acceleration_ratio", *WORK_ALGORITHM_CODES)] if use_dop853 else
                   [v for k, v in ADAPTIVE_ALGORITHMS.items() if k != "dop853"])
        self.adaptive_algorithm_combo.configure(values=choices)
        if self.adaptive_algorithm_var.get() not in choices:
            self.adaptive_algorithm_var.set(choices[0] if use_dop853 else ADAPTIVE_ALGORITHMS["acceleration_ratio"])
        self.adaptive_check.configure(text="Enable adaptive time step")
        use_native = use_dop853 and self.adaptive_algorithm_var.get() == ADAPTIVE_ALGORITHMS["dop853"]
        self.cpu_threads_entry.configure(state="normal" if BACKEND_CODES.get(self.backend_var.get()) == "cpu_parallel" else "disabled")
        self.seed_entry.configure(state="normal" if mode == "Specified seed" else "disabled")
        self.particle_count_entry.configure(state="disabled" if mode == "Known orbit" else "normal")
        self.orbit_combo.configure(state="readonly" if mode == "Known orbit" else "disabled")
        self.binary_eccentricity_entry.configure(state="normal" if mode == "Known orbit"
            and STABLE_ORBITS[self.orbit_combo.current()].get("kepler_binary", False) else "disabled")
        use_timescale = self.adaptive_algorithm_var.get() == "Time-scale method"
        use_ratio = self.adaptive_algorithm_var.get() == ADAPTIVE_ALGORITHMS["acceleration_ratio"]
        use_work = self.adaptive_algorithm_var.get() in [ADAPTIVE_ALGORITHMS[key] for key in WORK_ALGORITHM_CODES]
        self.adaptive_limit_note_var.set("Minimum dt = initial dt / N^K" if use_work
                                         else "Minimum suggested dt = initial dt / N^K" if use_ratio
                                         else "Time-scale safety factor: eta")
        enabled = self.adaptive_var.get()
        use_band = self.adaptive_algorithm_var.get() == ADAPTIVE_ALGORITHMS["step_energy_band"]
        self.dt_label.configure(text="Initial trial dt" if use_band and enabled else "dt / upper limit")
        self.work_threshold_label.configure(text="Node drift limit (0 = Off)" if use_band else "Trigger Threshold")
        self.work_threshold_entry.configure(textvariable=self.energy_band_vars["energy_drift_limit"] if use_band else self.work_threshold_var,
                                            state="normal" if enabled and (use_band or use_work) else "disabled")
        band_layout = enabled and use_band
        self.work_threshold_label.grid_configure(row=3 if band_layout else 2)
        self.work_threshold_host.grid_configure(row=3 if band_layout else 2)
        self.particle_count_label.grid_configure(row=2 if band_layout else 3)
        self.particle_count_entry.grid_configure(row=2 if band_layout else 3)
        self.work_preprobe_check.configure(state="normal" if enabled and use_work else "disabled")
        self.adaptive_algorithm_combo.configure(state="readonly" if enabled else "disabled")
        parameter_label, parameter_var = (
            ("Step energy lower L", self.energy_band_vars["step_energy_lower"]) if use_band else
            ("Relative tolerance rtol", self.dop853_rtol_var) if use_native else
            ("Safety factor", self.eta_var) if use_timescale else
            ("Shrink / recovery factor N", self.work_factor_var) if use_work else
            ("Acceleration ratio N", self.trigger_ratio_var))
        self.adaptive_parameter_label.configure(text=parameter_label)
        self.adaptive_parameter_entry.configure(textvariable=parameter_var, state="normal" if enabled else "disabled")
        self.adaptive_limit_label.configure(text="Step energy upper U" if use_band else "Absolute tolerance atol" if use_native else "Maximum exponent K")
        self.max_power_entry.configure(textvariable=self.energy_band_vars["step_energy_upper"] if use_band else self.dop853_atol_var if use_native else self.work_max_power_var if use_work else self.max_power_var,
                                       state="normal" if enabled and (use_band or use_native or use_ratio or use_work) else "disabled")
        self.algorithm_note_var.set(
            work_algorithm_note(ADAPTIVE_CODES[self.adaptive_algorithm_var.get()])
            if use_work else
            "Selects dt from pairwise gravitational and encounter time scales; smaller eta gives more conservative steps."
            if use_timescale else
            "Each star is compared with its retained reference, so accumulated growth also triggers adjustment. The largest level controls dt, capped at K. Zero values do not automatically reduce dt.")
        if use_band:
            self.adaptive_limit_note_var.set("Absolute code energy: 0 < L < U; node drift 0 = Off")
            self.algorithm_note_var.set("Float64 total-system energy: reject and retry from the same start on both sides; "
                "prediction power = integrator order + 1. Drift limit applies only to accepted nodes; 0 disables it. dt is the initial trial, not a maximum.")
        if use_work and self.work_preprobe_var.get():
            self.algorithm_note_var.set(self.algorithm_note_var.get() +
                " Pre-probe ON: if recovery occurs while max |a| rises, probe through its peak and return to entry |a|, "
                "discard the probe, then replay at the original smaller dt or below with recovery disabled. "
                "Probe evaluations are included in elapsed time and counted separately from accepted steps.")
        if use_native:
            self.adaptive_limit_note_var.set("Local error scale: atol + rtol * |state|")
            self.algorithm_note_var.set("DOP853 native: embedded local-error rejection and dt control. dt is the upper limit; first trial = min(dt,T). Array CPU uses Numba. Tolerances do not bound total energy error. Adaptive OFF = fixed eighth-order steps.")
        elif use_dop853:
            self.algorithm_note_var.set("DOP853 eighth-order stages; native error estimator/control OFF. " + self.algorithm_note_var.get())
        if use_taylor45:
            device = BACKEND_CODES[self.backend_var.get()]
            arithmetic = {
                "cpu_array": "float64, single-thread Numba. ",
                "cpu_mpfr_113": "Compiled MPFR, 113-bit mantissa (about 34 digits). ",
                "cpu_mpfr": "Compiled MPFR, 269-bit mantissa (about 80 digits). ",
                "cpu_mp": "Python mpmath, 80 decimal digits. ",
            }[device]
            if device in TAYLOR_MP_BACKENDS:
                arithmetic += "Internal energy RMSE/NRMSE use solver precision. Saved states and plots use float64. "
            order = TAYLOR_METHOD_ORDERS[str(self.method_combo.current()+1)]
            scaling = "Time-scaled float64 coefficients. " if order != 45 and device == "cpu_array" else ""
            self.algorithm_note_var.set(f"Taylor{order}: true order {order}; " + arithmetic + scaling + "No native error controller; large dt can fail near encounters. " + (self.algorithm_note_var.get() if enabled else "Fixed steps."))
        seed_hint = ("Generates a new seed for each run; use Specified seed to reproduce it."
                     if mode == "Random seed" else
                     "Known orbits use preset positions and velocities, independent of random seeds."
                     if mode == "Known orbit" else "The same seed recreates the same initial positions and velocities.")
        sampling_hint = ("Diagnostics are sampled every specified number of accepted steps, without shortening dt."
                         if use_dop853 or use_taylor45 or (enabled and not use_timescale) else "Diagnostic sampling interval = entered value × dt.")
        self.hint_var.set(seed_hint + "  " + sampling_hint + "Full trajectories, velocities and accelerations are retained at every step.")

    def on_source_change(self, event=None):
        if self.mode_var.get() == "Known orbit":
            self.use_orbit_defaults()
        else:
            self.dt_var.set("1")
            self.duration_var.set("10000")
            self.particle_count_var.set("5")
        self.sync_fields()

    def use_orbit_defaults(self, event=None):
        index = self.orbit_combo.current()
        if index < 0:
            return
        orbit = STABLE_ORBITS[index]
        self.dt_var.set(f"{orbit['dt']:.16g}")
        self.duration_var.set(f"{orbit['dt'] * orbit['tot_time']:.16g}")
        if orbit.get("recommended_method"):
            self.method_combo.current(int(orbit["recommended_method"]) - 1)
        self.sync_fields()
        if orbit.get("note"):
            self.hint_var.set(self.hint_var.get() + "  Orbit note: " + orbit["note"])

    def read_config(self):
        mode = {"Random seed": "random", "Specified seed": "manual", "Known orbit": "orbit"}[self.mode_var.get()]
        # Generate at the start of each random run, and retain the actual seed on screen.
        seed = secrets.randbits(32) if mode == "random" else self.seed_var.get()
        if mode == "orbit":
            seed = STABLE_ORBITS[self.orbit_combo.current()]["seed"]
        return validate_config(dict(
            **{key: variable.get() for key, variable in self.energy_band_vars.items()},
            initial_mode=mode, seed=seed, orbit_index=self.orbit_combo.current(),
            binary_eccentricity=self.binary_eccentricity_var.get(),
            dt=self.dt_var.get(), total_time=self.duration_var.get(),
            method=str(self.method_combo.current() + 1),
            dop853_rtol=self.dop853_rtol_var.get(), dop853_atol=self.dop853_atol_var.get(),
            adaptive_enabled=self.adaptive_var.get(),
            adaptive_algorithm=ADAPTIVE_CODES[self.adaptive_algorithm_var.get()],
            work_step_factor=(self.work_factor_var.get() if self.adaptive_var.get()
                              and self.adaptive_algorithm_var.get() in [ADAPTIVE_ALGORITHMS[key] for key in WORK_ALGORITHM_CODES] else WORK_DEFAULT_FACTOR),
            work_max_power=(self.work_max_power_var.get() if self.adaptive_var.get()
                            and self.adaptive_algorithm_var.get() in [ADAPTIVE_ALGORITHMS[key] for key in WORK_ALGORITHM_CODES] else WORK_DEFAULT_MAX_POWER),
            work_threshold=(self.work_threshold_var.get() if self.adaptive_var.get()
                            and self.adaptive_algorithm_var.get() in [ADAPTIVE_ALGORITHMS[key] for key in WORK_ALGORITHM_CODES] else WORK_DEFAULT_THRESHOLD),
            work_preprobe_enabled=self.work_preprobe_var.get(),
            adaptive_eta=(self.eta_var.get() if self.adaptive_var.get()
                          and self.adaptive_algorithm_combo.current() == 0 else .03),
            acceleration_trigger_ratio=(self.trigger_ratio_var.get() if self.adaptive_var.get()
                                        and self.adaptive_algorithm_var.get() == ADAPTIVE_ALGORITHMS["acceleration_ratio"] else 2.0),
            acceleration_max_power=(self.max_power_var.get() if self.adaptive_var.get()
                                    and self.adaptive_algorithm_var.get() == ADAPTIVE_ALGORITHMS["acceleration_ratio"] else 7),
            sample_every=self.sample_var.get(),
            compute_backend=BACKEND_CODES[self.backend_var.get()],
            cpu_threads=(self.cpu_threads_var.get() if BACKEND_CODES[self.backend_var.get()] == "cpu_parallel" else 0),
            particle_count=(STABLE_ORBITS[self.orbit_combo.current()]["N"]
                            if mode == "orbit" else self.particle_count_var.get())))

    def set_running(self, running):
        self.running = running
        if running:
            self.timeline.begin_computation()
        else:
            self.timeline.attach(self.viewer)
        for widget in self.form_widgets:
            widget.configure(state="disabled" if running else
                             "readonly" if isinstance(widget, ttk.Combobox) else "normal")
        self.batch_button.configure(state="disabled" if running else "normal")
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        if not running:
            self.sync_fields()

    def start_run(self, config=None):
        if self.running:
            return
        try:
            config = self.read_config() if config is None else validate_config(config)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid parameters", str(exc), parent=self.root)
            return
        if config["initial_mode"] != "orbit":
            self.seed_var.set(str(config["seed"]))
        if self.viewer:
            self.viewer.pause()
        self.cancel_event = threading.Event()
        self.set_running(True)
        self.progress_var.set(0)
        self.progress_text.set(
            f"Starting {COMPUTE_BACKENDS[config['compute_backend']]} computation... " +
            ("Waiting for startup CPU preparation; the run will start automatically."
             if config["compute_backend"] in ("cpu_array", "cpu_parallel") and not self.cpu_warmup_done.is_set()
             else "GPU kernels may compile on first use. Existing plots show the previous result."
             if config["compute_backend"] == "gpu" else "Existing plots show the previous result."))

        # Record the Toronto start time and UTC offset; saving later does not replace this timestamp.
        run_started_at = datetime.now(gettz("America/Toronto")).isoformat(timespec="microseconds")
        run_id = secrets.token_hex(16)
        record_enabled = bool(self.batch_jobs) or self.record_run_var.get()

        def worker():
            try:
                if config["compute_backend"] in ("cpu_array", "cpu_parallel"):
                    # Avoid concurrent compilation and keep startup work outside run timing.
                    while not self.cpu_warmup_done.wait(0.05):
                        if self.cancel_event.is_set():
                            raise SimulationCancelled()
                    if self.cancel_event.is_set():
                        raise SimulationCancelled()
                    error = self.cpu_warmup_errors.get(config["compute_backend"])
                    if error:
                        raise RuntimeError("Startup CPU preparation failed: " + error)
                result = run_simulation(config, progress=lambda *data: self.messages.put(("progress", data)),
                                        cancel=self.cancel_event)
                result["metadata"].update(run_started_at=run_started_at, run_id=run_id, record_enabled=record_enabled)
                if self.batch_jobs:
                    result["metadata"].update(batch_file=self.batch_path.name, batch_index=self.batch_index + 1,
                                              batch_label=self.batch_jobs[self.batch_index]["label"])
                self.messages.put(("complete", result))
            except SimulationCancelled:
                self.messages.put(("cancelled", None))
            except Exception as exc:
                self.messages.put(("error", f"{type(exc).__name__}: {exc}"))

        self.worker = threading.Thread(target=worker, name="NBodySimulation", daemon=True)
        self.worker.start()

    def cancel_run(self):
        if self.running:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.progress_text.set("Cancelling; keeping the previous successful result.")

    def poll_worker(self):
        if self.closed:
            return
        try:
            while True:
                kind, data = self.messages.get_nowait()
                if kind == "cpu_warmup":
                    self.cpu_warmup_status.set(data)
                elif kind == "progress":
                    fraction, steps, h, elapsed = data
                    self.progress_var.set(100 * fraction)
                    if not self.cancel_event.is_set():
                        eta = elapsed * (1 - fraction) / fraction if fraction > 0 else 0
                        self.progress_text.set(
                            (f"Batch {self.batch_index + 1}/{len(self.batch_jobs)} · " if self.batch_jobs else "") +
                            f"Progress {fraction:.1%} · {steps:,} steps · current dt = {h:.6g} · "
                            f"Elapsed {elapsed:.1f} s · ETA {eta:.1f} s")
                elif kind == "complete":
                    # A cancel arriving just after the final step should still preserve the old view.
                    if self.cancel_event.is_set():
                        self.set_running(False)
                        self.progress_text.set("Run cancelled; the previous successful result is retained.")
                        self.batch_finished_run(False, "Cancelled.")
                        continue
                    record_enabled = data["metadata"].get("record_enabled", True)
                    if record_enabled:
                        record = build_run_history_record(data["metadata"], data["diagnostics"])
                        record["result"] = data
                        record["save_full_data"] = bool(self.batch_jobs)
                        self.pending_history.append(record)
                    self.progress_text.set("Computation complete. Drawing results...")
                    self.root.update_idletasks()
                    container = ttk.Frame(self.result_host)
                    try:
                        new_viewer = NBodyViewer(root=container, timeline=self.timeline, **data)
                    except Exception as exc:
                        container.destroy()
                        self.set_running(False)
                        self.progress_text.set(f"Plotting failed: {exc}; the previous result is retained.")
                        if record_enabled:
                            self.history_status.set("The run awaits plotting and archiving. Click Retry to try again.")
                            self.retry_history_button.configure(state="normal")
                            self.retry_history_button.grid()
                        self.batch_finished_run(False, f"Plotting failed: {exc}")
                        messagebox.showerror("Plotting failed", str(exc), parent=self.root)
                        continue
                    saved = True
                    if record_enabled:
                        self.progress_text.set("Saving the run summary and conservation figure...")
                        saved = self.save_pending_history(viewer=new_viewer)
                    if self.viewer:
                        self.viewer.close()
                    self.placeholder.grid_remove()
                    container.grid(row=0, column=0, sticky="nsew")
                    self.viewer = new_viewer
                    self.set_running(False)
                    self.toggle_settings(False)
                    if not record_enabled:
                        self.progress_text.set("Simulation complete. Recording disabled; no history or image saved.")
                    self.batch_finished_run(saved, "Archive failed; use Retry saving run history before closing." if not saved else "")
                    if data["metadata"].get("adaptive_algorithm") == "step_energy_band":
                        self.progress_text.set(self.progress_text.get() + " | " + energy_band_summary(data["metadata"]))
                    if data["metadata"].get("adaptive_algorithm") in WORK_ALGORITHM_CODES:
                        count = data["metadata"].get("work_exceeded_steps", 0)
                        intervals = len(data["metadata"].get("work_exceeded_intervals", []))
                        self.progress_text.set(self.progress_text.get() +
                            f" | K-cap over-limit: {count} steps / {intervals} intervals; see red shading and legend below the graphs.")
                        if data["metadata"].get("work_preprobe_enabled"):
                            self.progress_text.set(self.progress_text.get() +
                                f" | Pre-probe: {data['metadata']['work_probe_window_count']} replay windows, "
                                f"{data['metadata']['work_probe_trial_evaluations']:,} extra probe trials.")
                    if data["metadata"].get("position_error", {}).get("reference_kind") == "analytic_kepler":
                        error = data["metadata"]["position_error"]
                        self.progress_text.set(self.progress_text.get() + f" | Kepler position RMSE = {error['rmse']:.6e}")
                    elif data["metadata"].get("position_error", {}).get("reference_kind") == "unavailable":
                        self.progress_text.set(self.progress_text.get() + " | Analytic position reference unavailable: "
                                               + data["metadata"]["position_error"]["reason"])
                elif kind == "cancelled":
                    self.set_running(False)
                    self.batch_finished_run(False, "Cancelled.")
                    self.progress_text.set("Run cancelled; the previous successful result is retained.")
                elif kind == "error":
                    self.set_running(False)
                    self.progress_text.set("Run failed; the previous successful result is retained.")
                    self.batch_finished_run(False, data)
                    messagebox.showerror("Computation failed", data, parent=self.root)
        except queue.Empty:
            pass
        self.poll_id = self.root.after(100, self.poll_worker)

    def close(self):
        self.closed = True
        self.cpu_preparation.stop.set()
        self.cancel_event.set()
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None
        if self.viewer:
            self.viewer.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    NBodyApp().run()
