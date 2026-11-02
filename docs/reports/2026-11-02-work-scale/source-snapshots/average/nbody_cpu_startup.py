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
            if self.errors:
                failed = "; ".join(f"{backend}: {error}" for backend, error in self.errors.items())
                self.messages.put(("cpu_warmup", "CPU preparation failed: " + failed))
            else:
                elapsed = time.perf_counter() - self.started_at
                self.messages.put(("cpu_warmup", f"CPU single/multi: ready (startup {elapsed:.1f}s; KDK + RK4)"))
        finally:
            self.finished_at = time.perf_counter()
            self.done.set()
