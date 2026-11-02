# Project roadmap

## Implemented

Development order; the [README](../README.md#completed-features) describes each stage and its scope.

- [x] Build the 3D direct Newtonian point-mass model with the original leapfrog update.
- [x] Add energy and conservation diagnostics, RMSE and normalized energy error.
- [x] Add random/specified seed control, followed by known periodic orbit initial conditions.
- [x] Add RK4 as an alternative to KDK, with sampling and end-of-run saving options.
- [x] Build the interactive acceleration/energy view and per-star records; extend motion inspection and interface controls over subsequent updates.
- [x] Add timescale-based adaptive stepping.
- [x] Add optional CuPy/CUDA computation using float64.
- [x] Add acceleration-ratio-based adaptive stepping with adjustable N and K.
- [x] Expand run history, figure saving and animation export.
- [x] Add batch Test mode, JSON plans and automatic per-run archives.
- [x] Add numerical-reference position RMSE/NRMSE, previous-run comparisons and reference-ordering refinements.
- [x] Complete the original Butterfly I step, method, adaptive and proposed cost-matching experiments; publish four main and two supplementary [importable historical plans](../plans/historical/README.md).
- [x] Complete Figure-8 reference/cost/long-time studies and Butterfly I fixed/adaptive comparisons (45 additional CPU runs).

- [x] Complete the historical dictionary CPU/GPU crossover study (262 runs).
- [x] Implement sampled energy-drift fitting and saved slope diagnostics.
- [x] Add Numba array single-thread and multi-thread CPU modes, preserving dictionary CPU.
- [x] Complete the latest 240-run four-backend retest with background-load telemetry and published crossover brackets.
- [x] Expand to 22 interactive orbit choices, including planar stars/triangles, a spatial trefoil and a nonsymmetric six-body orbit; record stability limitations.

- [x] Implement and compare Original, Align, Average and Predicted normalized Work scale controllers with KDK/RK4 and optional Encounter Pre-probe.
- [x] Preserve rollback/history isolation, capped-step exception reports and all-record time-weighted energy statistics.
- [x] Add a configurable eccentric Kepler binary with an analytic position reference.
- [x] Publish four-variant energy/drift/cost evidence, including counterexamples and Goggles ten-period cap tests.

## Next: validation and performance

- [ ] Scan fixed dt and adaptive threshold/bands to compare achieved error at matched measured cost; current same-initial-dt comparisons do not establish equal-accuracy speedup.
- [ ] Compare encounter Pre-probe on/off across deterministic encounter cases and longer durations; quantify drift, reversibility and probe overhead.
- [ ] Investigate cancellation in Average and unequal-step extrapolation in Predicted without assuming either variant is generally better.

- [ ] Refine the measured [CPU/GPU crossover brackets](reports/2026-10-20-backends/README.md) with more seeds, N values and thread counts.
- [ ] Repeat representative GPU and adaptive timing measurements, reporting medians and spread.
- [ ] Instrument gravity-evaluation counts and separate stepping, transfer, diagnostics and export costs.
- [ ] Further tune compiled CPU and GPU execution; the first optimized-CPU/CUDA comparison is complete.
- [ ] Keep GPU state resident between steps where feasible; benchmark different recording cadences without changing integration or diagnostics definitions silently.
- [ ] Bounded-memory/chunked recording, restart files and practical large-N limits.
- [ ] Automated regression suite around two-body analytic cases, conservation, interpolation and reference ordering.
- [ ] Wider initial-condition/reference convergence studies and more long-time validation.

Completion means measured, reproducible data and documented limitations, not only an implemented code path.

## Planned: long-term energy drift and recovery analysis

Status: **sampled linear trend fitting is implemented**. Repeated-sampling robustness and the flat-interval recovery method remain open; the two methods need not be mutually exclusive.

Goal: distinguish bounded energy oscillations and temporary excursions that recover from persistent offsets or cumulative one-direction drift. Compare KDK and RK4, with fixed-step and adaptive-step results assessed separately.

- [x] **Candidate A - sampling and trend fitting (initial implementation):** sample energy deviation from its initial value across physical time, then fit its long-term trend. Evaluate sampling density, time coverage and sensitivity to periodic oscillations and short excursions. Account for nonuniform recording times under adaptive stepping. A near-zero fitted slope alone does not establish small energy error or recovery to the initial energy.
- [ ] **Candidate B - relatively flat interval analysis:** identify sustained, relatively flat energy intervals using local variability and change over physical time. Estimate each interval's representative energy, compare successive baselines and compare them with the initial energy. Assess residual offsets after excursions and, where measurable, recovery time. Do not select intervals merely because they are close to the initial energy.
- [ ] Compare both candidates on representative existing runs before deciding whether to adopt one or combine them. Document sensitivity to sampling/window choices, thresholds and minimum interval durations; allow for trajectories with no suitable flat intervals.
- [ ] Define a recovery tolerance and required persistence: a momentary crossing of the initial energy is not sufficient. Distinguish recovery to the previous baseline from recovery to the initial energy. If recovery is not observed before the run ends, report only that finite-time finding.
- [ ] Preserve the complete energy record and retain excursion amplitude/duration diagnostics. Excluding jump intervals from baseline estimation must not remove them from the overall error assessment. Drift and recovery metrics supplement energy RMSE and trajectory errors rather than replace them.

The existing fitted slope is a finite-interval diagnostic. Recovery thresholds and flat-interval classification are not finalized.

## Numerical-method research

- [ ] Assess stronger close-encounter criteria and compare accuracy at equal cost.
- [ ] Test higher-order alternatives and error-controlled integration beyond classical RK4.
- [ ] Evaluate softening/regularization with explicit force and potential consistency; distinguish physical model changes from integration improvements.
- [ ] Investigate method switching only with a specified trigger and validation of transition errors.

## Later physical extensions

- [ ] Add a specified dark-matter halo model; distinguish a fixed external potential from a live particle halo and adapt conservation diagnostics accordingly.
- [ ] Construct and validate equilibrium galaxy initial conditions before merger experiments.
- [ ] Add conservation/error analysis to the galaxy-collision track and test inclination/orbit parameters.
- [ ] Write a formal project report and maintain reproducible figure/data releases.

GPU rendering, parallel independent batches, collision/merger physics, dark matter and a validated production galaxy model are not completed features of the current application.
