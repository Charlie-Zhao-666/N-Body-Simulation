# Project roadmap

## Implemented

- [x] 3D direct Newtonian point-mass model and reproducible initial states.
- [x] KDK leapfrog and classical RK4 integration.
- [x] Energy/momentum/angular-momentum/COM-velocity diagnostics, RMSE and normalized energy error.
- [x] Per-star acceleration and velocity records; actual step sizes; interactive trajectory exploration.
- [x] Timescale-based and acceleration-ratio-based adaptive stepping.
- [x] Optional CuPy/CUDA computation using float64.
- [x] Batch plans, selectable numerical reference, position RMSE/NRMSE and optional previous-run comparisons.
- [x] Automatic local data/figure saving and lightweight published results.
- [x] Figure-8 reference/cost/long-time studies and Butterfly I fixed/adaptive comparisons (45 additional CPU runs).

## Next: validation and performance

- [ ] CPU/GPU crossover sweep, with identical initial arrays and numerical-consistency checks; [protocol](cpu_gpu_benchmark_plan.md).
- [ ] Repeat representative GPU and adaptive timing measurements, reporting medians and spread.
- [ ] Instrument gravity-evaluation counts and separate stepping, transfer, diagnostics and export costs.
- [ ] Reduce redundant force evaluations and compare an optimized CPU baseline to CUDA.
- [ ] Keep GPU state resident between steps where feasible; benchmark different recording cadences without changing integration or diagnostics definitions silently.
- [ ] Bounded-memory/chunked recording, restart files and practical large-N limits.
- [ ] Automated regression suite around two-body analytic cases, conservation, interpolation and reference ordering.
- [ ] Wider initial-condition/reference convergence studies and more long-time validation.

Completion means measured, reproducible data and documented limitations, not only an implemented code path.

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
