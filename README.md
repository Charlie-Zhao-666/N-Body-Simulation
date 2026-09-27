# N-Body-Simulation

A 3D Newtonian-gravity N-body simulation in Python, developed step by
step as an ongoing learning and research project. The current version
integrates orbits with the leapfrog (kick-drift-kick) scheme and
quantifies how well the conserved quantities (total energy, total
momentum, total angular momentum, center-of-mass motion) are preserved
by the integration.

Sister project: [Neutron-Star-Modeling](https://github.com/Charlie-Zhao-666/Neutron-Star-Modeling)

## What the main program does

- Pairwise Newtonian gravity in 3D (computational units with G = 1,
  equal particle masses m = 1 by default).
- Leapfrog (kick-drift-kick) time integration.
- Three initial-condition modes, selected by number at runtime:
  1. Random seed (a new seed is generated each run),
  2. Imported seed (reproduce a previous run exactly),
  3. Known stable periodic solutions (exact initial conditions):
     the figure-8 three-body choreography (Chenciner-Montgomery),
     the Lagrange equilateral-triangle circular orbit, a circular
     two-body orbit, and an elliptic two-body orbit (e = 0.5).
- Adjustable time step and number of integration steps.
- Conservation diagnostics after the run: RMSE of total energy, total
  momentum, total angular momentum and center-of-mass velocity relative
  to their initial values, plus a normalized energy RMSE.
- Interactive 3D orbit animation with a draggable time slider, and a
  diagnostic figure (energy, momentum, angular momentum, CoM velocity
  versus time) with mouse-hover value readout.

## Repository contents

| File | Description |
|---|---|
| `N body simulation.py` | Main simulation, latest version. Improvement over the previous upload: the "Known seeds" menu (option 3) was a placeholder that printed an empty menu and crashed when selected; it now loads four exact stable periodic solutions, each setting N and a suggested dt / step count of about one period. Earlier history: the first upload introduced leapfrog integration with RMSE error analysis; the second added random / imported seed control and interactive parameters. |
| `time_step_tester.py` | Time-step scan tool, kept as a separate column (not an update of the main program). Runs the same initial conditions through a list of time steps (total physical time held fixed) and plots energy RMSE and normalized energy RMSE versus dt on log-log scales. Difference from the earlier local draft: the particle state and all record lists are now reset before each dt run, so every time step starts from identical initial conditions and results are not mixed across runs. Note: its seed menu still has the old placeholder option 3 - use option 1 or 2. |
| `.gitignore` | Standard Python ignore rules. |
| `README.md` | This file. |

## Completed work so far

- 3D pairwise Newtonian gravity with leapfrog integration.
- Conservation diagnostics (energy, momentum, angular momentum,
  center-of-mass motion) with RMSE-based error quantification and a
  normalized energy error.
- Reproducible random initial conditions via seeds, with interesting
  seeds recorded for reuse.
- Four exact periodic reference solutions loaded from the seed menu.
  Smoke tests over about one period (dt = 0.001) give normalized energy
  RMSE of 1.1e-07 (figure-8), 4.6e-13 (Lagrange circle), 2.0e-13
  (circular binary) and 7.4e-07 (elliptic binary) - consistent with
  second-order leapfrog behavior and a first check that the integrator
  reproduces known periodic orbits.
- Time-step scan tooling. A documented case (seed 221024875, N = 5,
  1000 time units) shows that a smaller dt does not always give a
  smaller error: normalized energy RMSE is 4.3e-04 at dt = 0.01 but
  2.84 at dt = 0.1 and 1.60 at dt = 1, because chaotic amplification
  of close encounters can dominate over discretization error. This
  motivates the planned adaptive time stepping.
- 3D orbit animation with a playback slider; diagnostic plots with
  hover readout.

## Planned / future work

- An RK4 integrator side by side with leapfrog, with a controlled
  comparison (same initial conditions and dt; accuracy, conservation
  and computational cost).
- Adaptive time stepping; gravitational softening; special handling of
  close encounters.
- Saving / loading initial conditions and a library of recorded seeds.
- More benchmark and convergence studies using the periodic reference
  solutions.
- GPU acceleration (an early CuPy prototype exists locally).
- Galaxy and galaxy-collision demos built on the same integrator.
- Result images / animations and a written project report.

## How to run

```
pip install numpy matplotlib mplcursors
python "N body simulation.py"
```

Follow the on-screen menus to choose the seed mode, the time step and
the number of steps. (The current version imports `mplcursors`; the
hover readout itself is implemented with matplotlib events.)

## Units and conventions

The simulation uses dimensionless computational units with G = 1 and
equal masses m = 1. The parameter called "total time" in the current
interface is the number of integration steps, so the physical simulated
time is steps x dt.

## Known issues and notes

- "total time" is a step count, not a physical time (naming to be
  unified in a future cleanup).
- In `time_step_tester.py`, the seed menu option 3 is still a
  placeholder (use 1 or 2); the two smallest dt values in its default
  list imply millions of integration steps and several GB of trajectory
  records, so a full scan is a long run.
- Verification so far is based on conservation diagnostics and periodic
  reference solutions; systematic benchmark and convergence studies are
  part of the planned work.
