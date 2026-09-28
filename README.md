# N-Body-Simulation

> **This branch: `time_step_tester.py`** — the time-step scan tool.
> The main line (simulation program, stable-orbit data, galaxy-collision
> demo) lives on [`main`](https://github.com/Charlie-Zhao-666/N-Body-Simulation/tree/main);
> this tool was moved here from main, and its leapfrog-only version is
> preserved in this branch's history. The general project README is
> appended below after the branch-specific documentation.

## Time-step tester (this branch)

### What it does

`time_step_tester.py` runs one set of initial conditions through the
list `dt_test_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1,
0.5, 1]`. Every run covers the same 1000 time units (`tot_time =
int(1000 / dt)`), so only the step size differs between runs. The
particle state and all record lists are reset before each run, so no
results are mixed. The integrator is selected at runtime (1 = leapfrog
KDK, 2 = 4th-order Runge-Kutta), the diagnostics sampling rate is
selectable (every 10 steps / 1% / 0.1% / 0.01% of total steps), and
after the run the two figures - energy RMSE vs dt and normalized energy
RMSE vs dt, both log-log - can be saved as
`seed_<seed>_method_<method>_dt_sweep_time_1000_{energy,normalized}.png`.
Note: the seed menu's option 3 is still a placeholder here; use option
1 or 2.

### Scan results and what they show

The figures below live on the [`images`](https://github.com/Charlie-Zhao-666/N-Body-Simulation/tree/images)
branch.

#### 1. A high-order method alone does not fix an under-resolved encounter

Same setup integrated with both methods at dt = 5 (seed 3347704169,
N = 5, 50000 steps):

![Conservation dashboard, leapfrog, seed 3347704169, dt = 5](https://raw.githubusercontent.com/Charlie-Zhao-666/N-Body-Simulation/images/dashboard_seed3347704169_dt5_leapfrog_kdk.png)

![Conservation dashboard, RK4, seed 3347704169, dt = 5](https://raw.githubusercontent.com/Charlie-Zhao-666/N-Body-Simulation/images/dashboard_seed3347704169_dt5_rk4.png)

- Total momentum is conserved to machine precision by both methods
  (RMSE ~1e-15, as expected: the pairwise forces are equal and
  opposite by construction).
- Energy: both methods take the same large jump at t ~ 2000 (an
  unresolved close encounter). RK4's energy RMSE is 0.86 vs leapfrog's
  1.61 (normalized: 5.09 vs 9.47) - better by less than a factor of 2,
  nowhere near the orders-of-magnitude gain one might expect from a
  4th-order scheme.
- Angular momentum: here RK4 is far *worse*. Leapfrog keeps all
  components at RMSE ~1e-10; RK4's components jump at the encounter
  and never come back (|L| RMSE 9.3e-2, L_z 2.8e-1).

Reading: at dt = 5 neither integrator resolves the close encounter, so
RK4's formal accuracy advantage never comes into play, while its
non-symplectic update damages angular momentum conservation exactly at
the encounter. A higher-order method is not a substitute for resolving
the dynamics.

#### 2. Shrinking dt alone does not keep buying accuracy

Leapfrog scan (seed 221024875, N = 5, 1000 time units per run):

![Energy RMSE vs dt, leapfrog, seed 221024875](https://raw.githubusercontent.com/Charlie-Zhao-666/N-Body-Simulation/images/dt_scan_leapfrog_seed221024875_energy_rmse.png)

![Normalized energy RMSE vs dt, leapfrog, seed 221024875](https://raw.githubusercontent.com/Charlie-Zhao-666/N-Body-Simulation/images/dt_scan_leapfrog_seed221024875_normalized_rmse.png)

- From dt = 1e-4 up to about 1e-2 the error follows the expected
  ~dt^2 scaling of a 2nd-order method (2e-8 at 1e-4, ~1e-4 at 1e-2).
- For dt >= 0.1 the error stops scaling: 0.7 at dt = 0.1, ~0.4 at
  dt = 0.5 and 1 (normalized 2.8 -> 1.6 -> 1.6). Chaotic amplification
  of close encounters saturates the error at the scale of the system
  energy itself, so changing dt no longer changes accuracy
  proportionally - a direct counterexample to "accuracy is inversely
  proportional to step size".

RK4 scans (two separate runs, N = 5, 1000 time units per run; the seeds
were not recorded in the saved figure names):

![Energy RMSE vs dt, RK4, run 1](https://raw.githubusercontent.com/Charlie-Zhao-666/N-Body-Simulation/images/dt_scan_rk4_energy_rmse_run1.png)

![Normalized energy RMSE vs dt, RK4, run 1](https://raw.githubusercontent.com/Charlie-Zhao-666/N-Body-Simulation/images/dt_scan_rk4_normalized_rmse_run1.png)

![Energy RMSE vs dt, RK4, run 2](https://raw.githubusercontent.com/Charlie-Zhao-666/N-Body-Simulation/images/dt_scan_rk4_energy_rmse_run2.png)

![Normalized energy RMSE vs dt, RK4, run 2](https://raw.githubusercontent.com/Charlie-Zhao-666/N-Body-Simulation/images/dt_scan_rk4_normalized_rmse_run2.png)

- For large dt the error rises steeply, roughly ~dt^4 as expected of a
  4th-order scheme (run 1: ~1e-2 at dt = 1 down to ~1e-13 at 1e-2).
- Below dt ~ 4e-3 (run 1) / ~1e-2 (run 2) the curve flattens onto a
  floor at ~1e-14 / ~1e-15 and even wiggles non-monotonically: the
  truncation error is already gone and round-off dominates, so making
  dt smaller buys nothing while the cost grows like 1/dt. The minimum
  sits at dt = 4e-3 (6e-15) in run 1 and dt = 1e-2 (5.5e-16) in run 2.

Conclusion: for RK4 there is a most cost-effective dt window - on these
N = 5 setups roughly dt = 4e-3..1e-2, where the error already sits on
the floor at the lowest step count. Going to dt = 1e-4 costs 40-100x
more steps for no gain.

#### 3. Strategy this suggests: match the method to the regime

Both scans together say that shrinking dt and switching to a
higher-order method are two levers that buy the same thing (truncation
accuracy per unit cost), and each lever stops paying at both ends of
the range. A practical strategy for future versions:

- (a) Large-scale / long evolution, where no close encounter dominates:
  leapfrog. One force evaluation per step, symplectic, bounded energy
  oscillation - the cheapest way to cover physical time.
- (b) Close encounters, where the dynamics change fast: switch to a
  higher-order method *and* a smaller dt together, so the encounter is
  actually resolved. Section 1 shows that the higher order alone does
  nothing there, and section 2 shows that shrinking dt everywhere
  wastes the budget.

Assessment notes: the strategy is consistent with the measurements
above. Two refinements worth keeping in mind when implementing it:
"close" should be defined by the local dynamical timescale (how fast
accelerations change), not by distance alone; and since RK4 is not
symplectic, the high-order phase should be used in bursts and switched
back afterwards (or replaced later by a high-order symplectic or
adaptive scheme). In effect this is a hand-made prototype of adaptive
time stepping, which is already on the planned-work list.

---

# Project README (from main)

A 3D Newtonian-gravity N-body simulation in Python, developed step by
step as an ongoing learning and research project. The current version
integrates orbits with the leapfrog (kick-drift-kick) scheme or with
4th-order Runge-Kutta (selected at runtime) and quantifies how well
the conserved quantities (total energy, total momentum, total angular
momentum, center-of-mass motion) are preserved by the integration.

Sister project: [Neutron-Star-Modeling](https://github.com/Charlie-Zhao-666/Neutron-Star-Modeling)

## What the main program does

- Pairwise Newtonian gravity in 3D (computational units with G = 1,
  equal particle masses m = 1 by default).
- Leapfrog (kick-drift-kick) or 4th-order Runge-Kutta time
  integration, selected at runtime.
- Three initial-condition modes, selected by number at runtime:
  1. Random seed (a new seed is generated each run),
  2. Imported seed (reproduce a previous run exactly),
  3. Known stable periodic solutions (exact initial conditions, loaded
     from `stable_orbits.py`): the figure-8 three-body choreography
     (Chenciner-Montgomery), the Lagrange equilateral-triangle
     circular orbit, a circular two-body orbit, an elliptic two-body
     orbit (e = 0.5), plus nine periodic three-body orbits discovered
     by Suvakov & Dmitrasinovic (2013): butterfly I-III, moth I-III,
     goggles, dragonfly and yin-yang I. Each option sets N and a
     suggested dt / step count of about one period.
- Adjustable time step and number of integration steps.
- Conservation diagnostics after the run: RMSE of total energy, total
  momentum, total angular momentum and center-of-mass velocity relative
  to their initial values, plus a normalized energy RMSE. For runs
  longer than 1000 steps the diagnostics sampling rate is selectable
  (every 10 steps / 1% / 0.1% / 0.01% of total steps).
- Interactive 3D orbit animation with a draggable time slider, and a
  diagnostic figure (energy, momentum, angular momentum, CoM velocity
  versus time) with mouse-hover value readout. Both figures state the
  seed, dt and method used. For the known stable solutions the
  animation view is sized automatically from the actual trajectory
  extent (same factor-2 rule as the random-mode view), so orbits of
  scale ~1 are no longer rendered as a point in a +/-100 box.
- After the plot windows close, results can be saved on request: the
  animation as a gif (capped at 2000 frames, longer runs are
  sub-sampled) and/or the diagnostic figure as a png, named
  `seed_<seed>_method_<method>_dt_<dt>_time_<steps>` next to the
  script.

## Repository contents

| File / branch | Description |
|---|---|
| `N body simulation.py` | Main simulation, latest version. Latest update: the integrator is selectable at runtime (leapfrog KDK or RK4), the diagnostics sampling rate is selectable for long runs, both figures state seed / dt / method, and results can be saved after the run (gif / png, named by seed, method, dt and step count). Earlier history: the first upload introduced leapfrog integration with RMSE error analysis; the second added random / imported seed control and interactive parameters; the third turned the placeholder "Known seeds" option into four exact stable periodic solutions; the fourth grew the stable-solution menu from 4 to 13 entries and moved all orbit data into `stable_orbits.py`. |
| `stable_orbits.py` | Data module with the initial conditions of the known stable / periodic solutions, imported by the main program (must sit in the same folder). Each entry carries N, a recommended dt and step count (exactly one period for the three-body choreographies), zero-momentum initial positions and velocities, the period, a literature source, and an optional warning note. |
| `Galaxies collision simulation.py` | Galaxy-collision demo, version 1 of the galaxy-collision track, kept as a separate file parallel to the programs above (not an update of any of them). Two identical galaxies - each a central AGN of mass 5.5e6 plus a 24-star disk (star masses 1-10, orbital radii 2-25, initially circular Keplerian speeds around its own center) - start 60 length units apart and drift toward each other with small transverse kicks, the disk stars sharing 1.2x their center's drift velocity. Same pairwise leapfrog scheme as the main program (G = 1e-11, dt = 100, 1000 steps), animated in 3D with matplotlib (gray trails and blue points for disk stars, red/orange curves for the two AGNs, fixed +/-60 view box). No conservation diagnostics - it is a visual demo. Needs only numpy and matplotlib. |
| branch [`time-step-tester`](https://github.com/Charlie-Zhao-666/N-Body-Simulation/tree/time-step-tester) | Time-step scan tool, moved off main onto its own branch (this branch). Sweeps a list of time steps (each run covers the same 1000 time units) with either leapfrog or RK4 and plots energy RMSE and normalized energy RMSE vs dt on log-log scales, with selectable diagnostics sampling and optional figure saving. The branch README documents the scan results and what they imply for choosing dt and method. The leapfrog-only version is preserved in the branch history. |
| branch [`method-tester`](https://github.com/Charlie-Zhao-666/N-Body-Simulation/tree/method-tester) | `Method_tester.py`: runs one set of initial conditions through leapfrog and RK4 back to back (same dt and step count, state reset between runs) and produces the full conservation report and dashboard for each, saved automatically as png. Quantifies how much each method actually conserves on identical setups. |
| branch [`no-z-component-test`](https://github.com/Charlie-Zhao-666/N-Body-Simulation/tree/no-z-component-test) | `no_z_component_test.py`: verifies that planar initial conditions (like the figure-8 and the other planar stable solutions) stay exactly in their initial plane under the 3D integrator - z, vz and az tracked every step, all exactly zero in the reference run. |
| branch [`images`](https://github.com/Charlie-Zhao-666/N-Body-Simulation/tree/images) | Result figures (dt scans and method-comparison dashboards), referenced by the branch READMEs. |
| `.gitignore` | Standard Python ignore rules. |
| `README.md` | This file. |

## Completed work so far

- 3D pairwise Newtonian gravity with leapfrog integration, plus a
  selectable 4th-order Runge-Kutta integrator (main program and both
  test tools).
- Conservation diagnostics (energy, momentum, angular momentum,
  center-of-mass motion) with RMSE-based error quantification and a
  normalized energy error.
- Reproducible random initial conditions via seeds, with interesting
  seeds recorded for reuse.
- Thirteen exact periodic reference solutions in the seed menu: four
  classical ones plus nine Suvakov-Dmitrasinovic orbits. The classical
  four are unchanged - smoke tests over about one period (dt = 0.001)
  give normalized energy RMSE of 1.1e-07 (figure-8), 4.6e-13
  (Lagrange circle), 2.0e-13 (circular binary) and 7.4e-07 (elliptic
  binary). Each of the nine new orbits was verified with the program's
  own leapfrog scheme: integrating one full period with its
  recommended step count (80k-640k steps), the trajectory returns to
  its starting point with a drift below 8e-3 (best 2e-4 for moth I).
  Butterfly II is included with its published 5-digit initial
  velocities, so a slightly larger deviation is expected and is
  flagged in its menu note.
- Honest limit of the current integrator: not every published periodic
  orbit can be reproduced in double-precision leapfrog. The bumblebee,
  butterfly IV, yarn and yin-yang II orbits are linearly unstable, and
  in verification runs their end-of-period drift stayed of order
  1-1000 even at 640k steps. They are documented in `stable_orbits.py`
  but intentionally not offered in the menu until a higher-order or
  high-precision integrator is available.
- Time-step scan tooling for both integrators, and a controlled
  leapfrog-vs-RK4 comparison tool. The scans (documented on this
  branch) show that accuracy does not scale with dt forever in either
  direction: leapfrog follows ~dt^2 only until chaotic close
  encounters saturate the error, and RK4 hits a round-off floor below
  dt ~ 4e-3 - so there is a most cost-effective dt window rather than
  "smaller is always better". A documented case (seed 221024875,
  N = 5, 1000 time units): normalized energy RMSE is 4.3e-04 at
  dt = 0.01 but 2.84 at dt = 0.1 and 1.60 at dt = 1.
- 3D orbit animation with a playback slider and automatic view sizing
  for stable solutions; diagnostic plots with hover readout; optional
  gif / png result saving named by seed, method, dt and step count.
- Galaxy-collision demo, first version: two AGN-centered 24-star disks
  set on a collision course, integrated with the same leapfrog scheme
  and animated in 3D with matplotlib.
- Planarity check: planar initial conditions stay exactly in their
  initial plane under the 3D integrator (no-z-component-test branch).

## Planned / future work

- Higher-order / high-precision integration beyond RK4, to make the
  four linearly unstable Suvakov-Dmitrasinovic orbits (bumblebee,
  butterfly IV, yarn, yin-yang II) reproducible and complete the menu.
- Regime-adaptive integration, guided by the dt-scan results: cheap
  leapfrog for the large-scale evolution, a higher-order method with a
  smaller dt during close encounters (see the strategy discussion
  above).
- Gravitational softening; special handling of close encounters.
- Saving / loading initial conditions and a library of recorded seeds.
- More benchmark and convergence studies using the periodic reference
  solutions.
- GPU acceleration (an early CuPy prototype exists locally).
- Galaxy track: conservation diagnostics for the galaxy-collision runs,
  more realistic collision setups (inclined / parabolic encounters,
  tidal tails), and a single-galaxy vispy demo (a local draft exists).
- Result images / animations and a written project report.

## How to run

```
pip install numpy matplotlib mplcursors
python time_step_tester.py
```

For the main program, switch to the
[`main`](https://github.com/Charlie-Zhao-666/N-Body-Simulation/tree/main)
branch and run `python "N body simulation.py"` with `stable_orbits.py`
in the same folder. The tester itself needs numpy, matplotlib and
mplcursors and has no other project dependencies (its seed menu option
3 is a placeholder; use 1 or 2).

## Units and conventions

The simulation uses dimensionless computational units with G = 1 and
equal masses m = 1. In this tester every dt run covers the same 1000
time units, so the step count is int(1000 / dt).

## Known issues and notes

- "total time" is a step count, not a physical time (naming to be
  unified in a future cleanup).
- The seed menu option 3 in this tester is still a placeholder (use
  1 or 2); the smallest dt values in the default list imply up to 10M
  steps per run, so a full sweep is a long run (the coarse sampling
  options exist for exactly this case).
- Two menu orbits of the main program (moth III, dragonfly) need 640k
  steps for a clean closed loop; their integrations take a few minutes
  and print progress every 10 steps.
- Verification so far is based on conservation diagnostics, periodic
  reference solutions and dt scans of both integrators; systematic
  benchmark studies are part of the planned work.

## References

- C. Moore, Phys. Rev. Lett. 70, 3675 (1993) - the figure-8 braid.
- A. Chenciner & R. Montgomery, Ann. of Math. 152, 881 (2000) -
  existence proof of the figure-8 solution.
- M. Suvakov & V. Dmitrasinovic, Phys. Rev. Lett. 110, 114301 (2013),
  arXiv:1303.0181 - 13 new families of periodic three-body orbits;
  butterfly II initial velocities are taken from its Table I.
- X. Li & S. Liao, Sci. China Phys. Mech. Astron. 60, 129511 (2017) -
  SJTU catalogue of 695 periodic three-body orbits; source of the
  high-precision initial velocities and periods for the other eight
  Suvakov-Dmitrasinovic orbits (catalogue IDs appear in the menu
  names).
