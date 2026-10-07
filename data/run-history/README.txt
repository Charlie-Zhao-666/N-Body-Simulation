N-body automatic run archive

This directory is a published snapshot. New local runs are saved to graph and data beside interactive_nbody_lab.py.
To share the archive, copy this entire folder. index.html uses relative image links.
The published Plot column uses relative filenames. README.md provides links rendered by GitHub.

runs.dat: aligned text table, best viewed using a monospaced font. Long names are not cut.
index.html: the same history with clickable links to the PNG figures.
_records/: one immutable JSON summary per run; full numerical precision and run ID.
YYYY-MM-DD_HH-MM-SS.png: conservation and error analysis figure for its recorded run.
The filename uses the same date and time as its table row (colons become hyphens).
If a second is already occupied, the real start-time microseconds appear in both
the filename and Time column; existing figures are never overwritten.
The Plot column in each DAT row contains the relative image filename; the Markdown and HTML tables link to it.
Whether DAT paths can be clicked depends on the text editor; index.html provides
direct links in the Plot cells. HTML targets and stored filenames remain relative,
so the links work when the entire archive folder is moved to another computer.

The original adaptive field is split into method, parameter, and K; the final Plot column links the figure.
Date and Time refer to the simulation start. started_at in JSON retains timezone
information when available. Legacy CSV timestamps have unknown timezone.
dt_initial is the chosen initial dt / adaptive maximum. Steps_set is ceil(Sim_T/dt_initial).
Steps_actual counts completed integration steps, including a shortened final step.
Adaptive_method is NA, Timescale, or AccelerationRatio. Adaptive_param is eta for Timescale
or the acceleration trigger N for AccelerationRatio. K is the maximum cumulative
exponent, used only by AccelerationRatio. All three fields are NA for fixed dt.
Conserved-quantity RMSE excludes the initial sample and compares diagnostics against the initial value.
Position_RMSE and Position_NRMSE use the selected batch reference and a common uniform time grid.
Disabled, pending, or unavailable position metrics are NA; reference_N marks a reference and its completed comparison count.
_position_comparisons/ stores comparison overlays without altering immutable original run records.
Comparison definitions and previous-test results are in the companion batch JSON and comparison DAT.
E_nRMSE divides energy RMSE by initial kinetic energy + abs(initial potential energy).
P, L, and Vcm mean total momentum, total angular momentum, and center-of-mass velocity.
Compute_time_s is elapsed real time for the simulation loop and part of result preparation, in seconds.
It excludes subsequent plotting and file saving; Sim_T is physical simulation time in code units.
DAT and HTML display scientific notation rounded to six decimal places; JSON preserves
all stored digits. NA marks an unused setting or unavailable / undefined value.
Old CSV rows are imported read-only, including repeated runs. Their missing figures
are marked NA; the original CSV is not modified. The table and HTML can be rebuilt
from _records. Independent records prevent concurrent runs from overwriting one another.

The full per-step NPZ trajectory archives are deliberately not tracked in Git. References to NPZ names in historical batch manifests identify local outputs, not files included here.
