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
  to their initial values, plus a normalized energy RMSE.
- Interactive 3D orbit animation with a draggable time slider, and a
  diagnostic figure (energy, momentum, angular momentum, CoM velocity
  versus time) with mouse-hover value readout. For the known stable
  solutions the animation view is sized automatically from the actual
  trajectory extent (same factor-2 rule as the random-mode view), so
  orbits of scale ~1 are no longer rendered as a point in a +/-100
  box.

## Repository contents

| File | Description |
|---|---|
| `N body simulation.py` | Main simulation, latest version. Improvement over the previous upload: the stable-solution menu grew from 4 to 13 entries - nine Suvakov-Dmitrasinovic periodic three-body orbits were added, all stable-solution initial conditions moved into the new data module `stable_orbits.py` (the menu is generated from the data), and the animation view is now sized automatically for stable solutions. Earlier history: the first upload introduced leapfrog integration with RMSE error analysis; the second added random / imported seed control and interactive parameters; the third turned the placeholder "Known seeds" option into four exact stable periodic solutions. |
| `stable_orbits.py` | Data module with the initial conditions of the known stable / periodic solutions, imported by the main program (must sit in the same folder). Each entry carries N, a recommended dt and step count (exactly one period for the three-body choreographies), zero-momentum initial positions and velocities, the period, a literature source, and an optional warning note. |
| `time_step_tester.py` | Time-step scan tool, kept as a separate column (not an update of the main program). Runs the same initial conditions through a list of time steps (total physical time held fixed) and plots energy RMSE and normalized energy RMSE versus dt on log-log scales. Difference from the earlier local draft: the particle state and all record lists are now reset before each dt run, so every time step starts from identical initial conditions and results are not mixed across runs. Note: its seed menu still has the old placeholder option 3 - use option 1 or 2. |
| `Galaxies collision simulation.py` | Galaxy-collision demo, version 1 of the galaxy-collision track, kept as a separate file parallel to the programs above (not an update of any of them). Two identical galaxies - each a central AGN of mass 5.5e6 plus a 24-star disk (star masses 1-10, orbital radii 2-25, initially circular Keplerian speeds around its own center) - start 60 length units apart and drift toward each other with small transverse kicks, the disk stars sharing 1.2x their center's drift velocity. Same pairwise leapfrog scheme as the main program (G = 1e-11, dt = 100, 1000 steps), animated in 3D with matplotlib (gray trails and blue points for disk stars, red/orange curves for the two AGNs, fixed +/-60 view box). No conservation diagnostics - it is a visual demo. Needs only numpy and matplotlib. |
| `.gitignore` | Standard Python ignore rules. |
| `README.md` | This file. |

## Completed work so far

- 3D pairwise Newtonian gravity with leapfrog integration.
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
- Time-step scan tooling. A documented case (seed 221024875, N = 5,
  1000 time units) shows that a smaller dt does not always give a
  smaller error: normalized energy RMSE is 4.3e-04 at dt = 0.01 but
  2.84 at dt = 0.1 and 1.60 at dt = 1, because chaotic amplification
  of close encounters can dominate over discretization error. This
  motivates the planned adaptive time stepping.
- 3D orbit animation with a playback slider and automatic view sizing
  for stable solutions; diagnostic plots with hover readout.
- Galaxy-collision demo, first version: two AGN-centered 24-star disks
  set on a collision course, integrated with the same leapfrog scheme
  and animated in 3D with matplotlib.

## Planned / future work

- An RK4 integrator side by side with leapfrog, with a controlled
  comparison (same initial conditions and dt; accuracy, conservation
  and computational cost). A higher-order (and eventually
  high-precision) integrator should also make the four linearly
  unstable Suvakov-Dmitrasinovic orbits - bumblebee, butterfly IV,
  yarn, yin-yang II - reproducible, which would complete the menu.
- Adaptive time stepping; gravitational softening; special handling of
  close encounters.
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
python "N body simulation.py"
```

Follow the on-screen menus to choose the seed mode, the time step and
the number of steps. Keep `stable_orbits.py` in the same folder as the
main program - the "Known seeds" menu imports it. (The current version
imports `mplcursors`; the hover readout itself is implemented with
matplotlib events.) `Galaxies collision simulation.py` needs only
numpy and matplotlib.

## Units and conventions

The simulation uses dimensionless computational units with G = 1 and
equal masses m = 1. The parameter called "total time" in the current
interface is the number of integration steps, so the physical simulated
time is steps x dt. (`Galaxies collision simulation.py` uses G = 1e-11
and unequal masses - see its header.)

## Known issues and notes

- "total time" is a step count, not a physical time (naming to be
  unified in a future cleanup).
- In `time_step_tester.py`, the seed menu option 3 is still a
  placeholder (use 1 or 2); the two smallest dt values in its default
  list imply millions of integration steps and several GB of trajectory
  records, so a full scan is a long run.
- Two menu orbits (moth III, dragonfly) need 640k steps for a clean
  closed loop; their integrations take a few minutes and print progress
  every 10 steps.
- Verification so far is based on conservation diagnostics and periodic
  reference solutions; systematic benchmark and convergence studies are
  part of the planned work.

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
