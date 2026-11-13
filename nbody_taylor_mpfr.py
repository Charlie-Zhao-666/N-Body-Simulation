"""Python interface to the compiled, single-thread arbitrary-precision Taylor core.

The solver owns its MPFR state. pos and vel are binary64 snapshots for the Lab
controllers and plots. report() exports decimal strings directly from MPFR.
"""
import ctypes as ct
import math
import operator
import sys
import threading
import weakref
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np


_Callback = ct.CFUNCTYPE(ct.c_int)
_Strings = ct.POINTER(ct.c_char_p)
_Doubles = ct.POINTER(ct.c_double)


@lru_cache(maxsize=1)
def _load_library():
    filename = {"win32": "nbody_taylor_mpfr.dll", "darwin": "libnbody_taylor_mpfr.dylib"}.get(
        sys.platform, "libnbody_taylor_mpfr.so")
    path = Path(__file__).resolve().with_name(filename)
    hint = (f"Cannot load the compiled Taylor MPFR library at {path}. "
            "On Windows, run build_taylor_mpfr.ps1 in the Lab code folder, "
            "or select the Python/mpmath Taylor backend.")
    try:
        lib = ct.CDLL(str(path))
        lib.nb_abi.argtypes, lib.nb_abi.restype = [], ct.c_int
        if lib.nb_abi() != 1:
            raise RuntimeError(f"{hint} Incompatible library ABI.")
        handle, integer = ct.c_void_p, ct.c_int
        signatures = {
            "nb_version": ([], ct.c_char_p),
            "nb_error": ([], ct.c_char_p),
            "nb_create": ([integer, integer, integer, ct.c_char_p, _Strings, _Strings, _Strings], handle),
            "nb_destroy": ([handle], None),
            "nb_checkpoint": ([handle], handle),
            "nb_free_checkpoint": ([handle], None),
            "nb_restore": ([handle, handle], integer),
            "nb_advance": ([handle, ct.c_double, _Callback], integer),
            "nb_state": ([handle, _Doubles, _Doubles], integer),
            "nb_text": ([handle, integer, integer, ct.POINTER(ct.c_char), integer], integer),
            "nb_coefficient_builds": ([handle], ct.c_ulonglong),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, result
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"{hint} {exc}") from exc
    return lib


def _message(lib):
    return (lib.nb_error() or b"MPFR integration failed").decode("utf-8", errors="replace")


def _number(value):
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("Taylor MPFR requires finite input.")
        numerator, denominator = value.as_integer_ratio()
        return f"{numerator:#x}p-{denominator.bit_length()-1}".encode("ascii")
    if isinstance(value, bool):
        raise ValueError("Taylor MPFR expects numbers, not booleans.")
    text = str(value)
    if "\0" in text:
        raise ValueError("Embedded null in numeric input.")
    return text.encode("ascii")


def _strings(array):
    return (ct.c_char_p * array.size)(*(_number(value) for value in array.flat))


class _Checkpoint:
    # Copies share one immutable, reference-counted native snapshot.
    __slots__ = ("_owner", "_ptr", "__weakref__")

    def __init__(self, owner, ptr, lib):
        self._owner, self._ptr = owner, ptr
        weakref.finalize(self, lib.nb_free_checkpoint, ptr)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class MPFRTaylorNBody:
    """Taylor integration using all three spatial coordinates and arbitrary masses."""

    def __init__(self, masses, positions, velocities, gravity=1, dps=80, cancel_check=None,
                 order=45, *, precision_bits=None):
        self.order = operator.index(order)
        if self.order not in (45, 100, 150, 200):
            raise ValueError("Taylor order must be 45, 100, 150 or 200.")
        if precision_bits is None:
            self.dps = operator.index(dps)
            if not 1 <= self.dps <= 2465:
                raise ValueError("MPFR decimal precision must be between 1 and 2465.")
            # The default exactly reproduces mp.dps=80 (269 significand bits).
            self.bits = round((self.dps + 1)*math.log2(10))
        else:
            self.bits = operator.index(precision_bits)
            if not 2 <= self.bits <= 8192:
                raise ValueError("MPFR binary precision must be between 2 and 8192 bits.")
            self.dps = max(1, int(self.bits * math.log10(2)))
        mass = np.asarray(masses, dtype=object)
        pos = np.asarray(positions, dtype=object)
        vel = np.asarray(velocities, dtype=object)
        if mass.ndim != 1 or len(mass) == 0 or pos.shape != (len(mass), 3) or vel.shape != pos.shape:
            raise ValueError(f"Taylor{self.order} requires N masses and matching N x 3 positions and velocities.")
        self.n = len(mass)
        self._lib = _load_library()
        ptr = self._lib.nb_create(self.n, self.order, self.bits, _number(gravity),
                                  _strings(mass), _strings(pos), _strings(vel))
        if not ptr:
            raise ValueError(_message(self._lib))
        self._ptr = ptr
        weakref.finalize(self, self._lib.nb_destroy, ptr)
        self._owner = object()
        self._lock = threading.Lock()
        self._text_size = 80 + math.ceil(self.bits*math.log10(2))
        self.cancel_check = cancel_check
        self.trials = 0
        self._read_snapshots()

    def __copy__(self):
        raise TypeError("Use checkpoint() to save a Taylor MPFR state.")

    def __deepcopy__(self, memo):
        raise TypeError("Use checkpoint() to save a Taylor MPFR state.")

    def _read_snapshots(self):
        pos, vel = np.empty((self.n, 3)), np.empty((self.n, 3))
        if self._lib.nb_state(self._ptr, pos.ctypes.data_as(_Doubles), vel.ctypes.data_as(_Doubles)):
            raise RuntimeError(_message(self._lib))
        self.pos, self.vel = pos, vel

    def checkpoint(self):
        with self._lock:
            ptr = self._lib.nb_checkpoint(self._ptr)
            if not ptr:
                raise RuntimeError(_message(self._lib))
            return _Checkpoint(self._owner, ptr, self._lib)

    def restore(self, checkpoint):
        if not isinstance(checkpoint, _Checkpoint) or checkpoint._owner is not self._owner:
            raise ValueError("Taylor MPFR checkpoint belongs to a different integration.")
        with self._lock:
            if self._lib.nb_restore(self._ptr, checkpoint._ptr):
                raise RuntimeError(_message(self._lib))
            self._read_snapshots()

    def advance(self, h):
        h = float(h)
        if not math.isfinite(h) or h <= 0:
            raise ValueError(f"Taylor{self.order} requires a finite positive step.")
        # ctypes callbacks cannot propagate a Python exception through a native
        # stack. Have C abort before commit, then re-raise the original exception.
        errors = []
        def poll():
            try:
                self.cancel_check()
                return 0
            except BaseException as exc:
                errors.append(exc)
                return 1
        callback = _Callback(poll) if self.cancel_check is not None else _Callback()
        with self._lock:
            if self.cancel_check is not None:
                self.cancel_check()
            status = self._lib.nb_advance(self._ptr, h, callback)
            if errors:
                raise errors[0]
            if status:
                message = _message(self._lib)
                if status == 3 and self.order != 45:
                    from nbody_taylor45 import TaylorStepRejected
                    raise TaylorStepRejected(message)
                raise RuntimeError(message)
            self.trials += 1
            self._read_snapshots()

    def _text(self, field, index=0):
        buffer = ct.create_string_buffer(self._text_size)
        if self._lib.nb_text(self._ptr, field, index, buffer, len(buffer)):
            raise RuntimeError(_message(self._lib))
        return buffer.value.decode("ascii")

    def energy(self):
        with self._lock:
            return Decimal(self._text(7))

    @property
    def time(self):
        with self._lock:
            return Decimal(self._text(2))

    def report(self):
        with self._lock:
            def vector(field):
                return [[self._text(field, 3*i+d) for d in range(3)] for i in range(self.n)]
            nrmse = self._text(6)
            return dict(taylor_order=self.order, taylor_precision="mpfr", taylor_decimal_precision=self.dps,
                        taylor_mantissa_bits=self.bits, taylor_mpfr_version=self._lib.nb_version().decode("ascii"),
                        taylor_trial_evaluations=self.trials,
                        taylor_coefficient_generations=self._lib.nb_coefficient_builds(self._ptr),
                        taylor_final_positions=vector(0), taylor_final_velocities=vector(1),
                        taylor_internal_time=self._text(2), taylor_initial_energy=self._text(3),
                        taylor_final_energy_difference=self._text(4), taylor_internal_energy_rmse=self._text(5),
                        taylor_internal_energy_nrmse=nrmse if Decimal(nrmse).is_finite() else None,
                        taylor_precision_scope=f"MPFR {self.bits}-bit internal working precision. "
                            "Internal energy RMS integrates squared, piecewise-linear accepted-node energy offsets "
                            "over physical time. Lab snapshots, controller inputs and ordinary RMS use float64.",
                        taylor_execution=f"CPU single-thread compiled MPFR; degree-{self.order} Taylor polynomials "
                            "for position and velocity; external step control")
