"""Lightweight startup entry: no scientific imports on the launching thread."""
import queue
import threading
import time


class CPUStartupPreparation:
    """Prepare both CPU backends in this process, independently of the UI/run button."""

    def __init__(self):
        self.messages = queue.Queue()
        self.done = threading.Event()
        self.stop = threading.Event()
        self.errors = {}
        self.started_at = time.perf_counter()
        self.finished_at = None
        self.worker = threading.Thread(target=self._prepare, name="NBodyCPUWarmup", daemon=True)
        self.worker.start()

    def _prepare(self):
        try:
            # The default backend is multi-thread CPU: prepare it first.
            for backend, parallel, label in (("cpu_parallel", True, "multi"),
                                              ("cpu_array", False, "single")):
                if self.stop.is_set():
                    return
                self.messages.put(("cpu_warmup", f"Startup CPU preparation: compiling {label}..."))
                try:
                    from nbody_array_cpu import warmup_cpu_backend
                    warmup_cpu_backend(parallel=parallel)
                except Exception as exc:
                    self.errors[backend] = f"{type(exc).__name__}: {exc}"
                if backend not in self.errors:
                    try:
                        from nbody_dop853_cpu import warmup_dop853_backend
                        self.messages.put(("cpu_warmup", f"Startup CPU preparation: compiling DOP853 {label}..."))
                        warmup_dop853_backend(parallel=parallel)
                    except Exception as exc:
                        # Optional SciPy/DOP853 failure must not block KDK or RK4.
                        self.errors['dop853_'+backend] = f"{type(exc).__name__}: {exc}"
            if self.stop.is_set():
                return
            try:
                from nbody_taylor45_cpu import warmup_taylor45_backend
                self.messages.put(("cpu_warmup", "Startup CPU preparation: compiling Taylor45 float64 single..."))
                warmup_taylor45_backend()
            except Exception as exc:
                self.errors['taylor45_cpu_array'] = f"{type(exc).__name__}: {exc}"
            for order in (100, 150, 200):
                if self.stop.is_set():
                    return
                try:
                    from nbody_taylor_cpu import warmup_taylor_backend
                    self.messages.put(("cpu_warmup", f"Startup CPU preparation: compiling Taylor{order} float64 single..."))
                    warmup_taylor_backend(order)
                except Exception as exc:
                    self.errors[f'taylor{order}_cpu_array'] = f"{type(exc).__name__}: {exc}"
            if self.errors:
                failed = "; ".join(f"{backend}: {error}" for backend, error in self.errors.items())
                prefix = ("Optional integrator preparation unavailable; KDK/RK4 ready: "
                          if all(key.startswith(('dop853_', 'taylor')) for key in self.errors)
                          else "CPU preparation failed: ")
                self.messages.put(("cpu_warmup", prefix + failed))
            else:
                elapsed = time.perf_counter() - self.started_at
                self.messages.put(("cpu_warmup", f"CPU ready (startup {elapsed:.1f}s; KDK + RK4 + DOP853 + Taylor45/100/150/200 float64 single)"))
        finally:
            self.finished_at = time.perf_counter()
            self.done.set()
