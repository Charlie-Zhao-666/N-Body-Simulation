Published N-body run archive

README.md: compact clickable table on GitHub.
runs.md: complete table with every column and clickable relative Plot links.
runs.dat: aligned plain-text copy; relative paths are not clickable in GitHub code view.
index.html: clickable full table after downloading this entire directory.
images/: lossless WebP figures, verified pixel-for-pixel against the original PNGs.
_records/: full-precision run summaries; image paths are relative to this archive root.
_position_comparisons/: full-precision comparison overlays.
NPZ names in batch metadata are provenance only; raw trajectories are not included.
This export does not update the live local archive or the simulation source code.

The original adaptive field is split into method, parameter, and K; the final Plot column links the figure.
Date and Time refer to the simulation start. started_at in JSON retains timezone
information when available. Legacy CSV timestamps have unknown timezone.
dt_initial is the chosen initial dt / adaptive maximum. Steps_set is ceil(Sim_T/dt_initial).
Steps_actual counts completed integration steps, including a shortened final step.
Adaptive_method is NA, Timescale, AccelerationRatio (global), or LocalAccelerationRatio. Adaptive_param is eta for Timescale
or the acceleration trigger N for either ratio method. K is the maximum cumulative
exponent, used by either ratio mode. Local Steps_actual counts update events; per-star counts and masks are in NPZ metadata/arrays. All three fields are NA for fixed dt.
Conserved-quantity RMSE excludes the initial sample and compares diagnostics against the initial value.
Position_RMSE and Position_NRMSE use the selected batch reference and a common uniform time grid.
Disabled, pending, or unavailable position metrics are NA; reference_N marks a reference and its completed comparison count.
_position_comparisons/ stores comparison overlays without altering immutable original run records.
Comparison definitions and previous-test results are in the companion batch JSON and comparison DAT.
E_nRMSE divides energy RMSE by initial kinetic energy + abs(initial potential energy).
E_drift_k is the signed OLS slope of E(t)-E(0), in code energy/code time; a free intercept is fitted.
E_drift_k_norm divides that slope by initial kinetic energy + abs(initial potential energy), in 1/code time.
These are drift rates, not RMSE values. Positive/negative slopes describe upward/downward fitted trends.
A small slope alone does not prove conservation; compare fit quality and sensitivity to sampling.
Fits use uniform random physical times and interpolate the full saved energy trajectory.
Sampling count, seed, fit interval and residual statistics are retained in the run's energy_drift JSON.
Old records are displayed with slopes from their archived fit, or NA if no fit was saved.
The archived fit is retained when an interactive refit is saved separately; indexes do not silently replace it.
P, L, and Vcm mean total momentum, total angular momentum, and center-of-mass velocity.
Device distinguishes CPU (dictionary), CPU_ARRAY_SINGLE, CPU_ARRAY_MULTI(nT), and GPU; n is the actual thread count.
Device_model identifies the processor used for computation, not display rendering.
Array CPU initialization/Numba compilation occurs before Compute_time_s; initialization_seconds is retained in NPZ metadata.
Historical hardware provenance is retained in each migrated JSON summary; unknown models remain NA.
Compute_time_s is elapsed real time for the simulation loop and part of result preparation, in seconds.
It excludes subsequent plotting and file saving; Sim_T is physical simulation time in code units.
DAT and HTML display scientific notation rounded to six decimal places; JSON preserves
all stored digits. NA marks an unused setting or unavailable / undefined value.
Old CSV rows are imported read-only, including repeated runs. Their missing figures
are marked NA; the original CSV is not modified. The table and HTML can be rebuilt
from _records. Independent records prevent concurrent runs from overwriting one another.
