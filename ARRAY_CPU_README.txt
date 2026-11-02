Array CPU modes
===============
Open the acceleration-interactive Python program as before. Both the main
settings and Test mode now offer these Compute device choices:

  CPU (dictionary / original)   Existing implementation; still the default.
  CPU (array / single thread)   Compiled float64 force calculation on one thread.
  CPU (array / multi-thread)    Compiled force calculation across worker threads.
  GPU (NVIDIA / CuPy)           Existing GPU implementation.

Keep nbody_array_cpu.py in the same folder as the interactive Python program.
The array modes need Numba. It is already installed in the local Python used
for this implementation. In another environment, use that environment's Python:

  python -m pip install -r requirements-array-cpu.txt

This optional requirements file does not install the application's other
existing dependencies. Dictionary CPU and GPU do not require Numba.

CPU threads
-----------
The field is enabled only for array multi-thread mode. Zero selects the smaller
of the particle count and Numba's available thread limit. A positive integer
selects that many threads, up to the limit reported by this Python process.
The tested local process has a limit of 32. Threads are not physical cores:
the operating system schedules workers, and no core affinity is forced.
For small particle counts, array single-thread can beat multi-thread because
worker coordination has a cost. More threads do not guarantee a faster run.

Data and physics
----------------
Initial dictionaries are copied to mass[N], position[N,3], velocity[N,3] arrays
once. Time integration then stays in arrays. Each parallel worker owns one
target particle's output; it reads all source particles at the same stage.
The two array modes use the same summation order and no fast-math or softening.
KDK, RK4, the current adaptive-step controllers, diagnostics, playback, cancellation,
recording controls, batch plans and position comparisons use existing paths.
Compared with dictionary CPU, the force summation implementation differs, so
floating-point results need not be bit-for-bit identical. Long chaotic
trajectories can amplify initially tiny differences.

Timing and records
------------------
First use in a new Python process compiles the selected array kernels. This
may take a few seconds; it is outside Compute_time_s. Subsequent runs in the
same process reuse the compiled kernels. Initialization time is recorded as
initialization_seconds in result metadata and batch NPZ metadata.
Compute_time_s still includes the simulation loop and its recording work;
it is not a force-kernel-only benchmark. Plotting and file saving are excluded.

The existing Device column distinguishes CPU, CPU_ARRAY_SINGLE,
CPU_ARRAY_MULTI(nT), and GPU. n is the actual worker-thread count.
Device_model continues to identify the processor. NPZ metadata also retains
cpu_threads, cpu_threads_requested, compute_backend and the full configuration.
Existing archive rows and legacy plans are not rewritten.

Plan codes: cpu, cpu_array, cpu_parallel, gpu. Set cpu_threads to 0 or a positive
integer for cpu_parallel. Old plans without cpu_threads remain valid.

Verification (2026-10-19)
-------------------------
On this computer: original CPU/GPU regression; analytic unequal-mass force;
KDK/RK4 convergence on a circular binary; fixed and both adaptive-step modes;
serial/parallel agreement; independent history snapshots; cancellation/thread
restoration; GUI worker delivery; all four backend exports; reference and
previous-run position comparisons. Development results are in the project's
Codex/2026-10-19-array-cpu folder, separate from scientific run history.

For individual local stepping, see LOCAL_STEPS_README.md. Its predictor/scheduler
and diagnostic costs differ from the global array solver described above.
