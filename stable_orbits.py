"""Initial conditions for the known stable / periodic solutions.

Data module for "N body simulation.py". Keeping the orbit data here lets the
main program stay unchanged in structure: it imports STABLE_ORBITS and builds
its menu directly from the entries below.

Each entry is a dict with:

    name       : display name shown in the menu
    N          : number of bodies
    seed       : seed label used by the main program (equals the menu number)
    dt         : recommended time step
    tot_time   : recommended number of integration steps; for the periodic
                 three-body orbits this is exactly one period, so that
                 dt = period / tot_time
    positions  : initial positions  [[x, y, z], ...], length N
    velocities : initial velocities [[vx, vy, vz], ...], length N
    period     : orbital period in code time units (None where not applicable)
    source     : literature reference for the initial conditions
    note       : optional warning shown when the orbit is loaded

Units: G = 1, all masses = 1. Every entry has zero total momentum and its
center of mass at the origin.

The step counts of the three-body choreographies were verified with the same
leapfrog (KDK) integrator used by the main program: over one full period the
trajectory returns to its starting point with a drift below 8e-3 in code
length units (see README.md). Orbits from the literature that are linearly
unstable and therefore cannot be reproduced with a double-precision leapfrog
integrator (bumblebee, butterfly IV, yarn, yin-yang II) are intentionally not
included.
"""

import numpy as np


def _sd_orbit(name, seed, v1, v2, period, steps, source, note=None):
    """Build one Suvakov-Dmitrasinovic type initial condition.

    The three bodies start at (-1, 0), (1, 0) and (0, 0) with velocities
    (v1, v2), (v1, v2) and (-2*v1, -2*v2) in the z = 0 plane.
    """
    return {
        "name": name,
        "N": 3,
        "seed": seed,
        "dt": period / steps,
        "tot_time": steps,
        "positions": [[-1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]],
        "velocities": [[v1, v2, 0.0],
                       [v1, v2, 0.0],
                       [-2.0 * v1, -2.0 * v2, 0.0]],
        "period": period,
        "source": source,
        "note": note,
    }


# --- Lagrange equilateral triangle: radius and phase construction ----------
_rc = 1.0 / np.sqrt(3.0)
_lagrange_pos = []
_lagrange_vel = []
for _k in range(3):
    _theta = 2.0 * np.pi * _k / 3.0
    _lagrange_pos.append([_rc * np.cos(_theta), _rc * np.sin(_theta), 0.0])
    _lagrange_vel.append([-np.sin(_theta), np.cos(_theta), 0.0])


STABLE_ORBITS = [
    {
        "name": "Figure-8 three-body choreography (Chenciner-Montgomery)",
        "N": 3,
        "seed": 1,
        "dt": 0.001,
        "tot_time": 6500,
        "positions": [[0.97000436, -0.24308753, 0.0],
                      [-0.97000436, 0.24308753, 0.0],
                      [0.0, 0.0, 0.0]],
        "velocities": [[-0.466203685, -0.43236573, 0.0],
                       [-0.466203685, -0.43236573, 0.0],
                       [0.93240737, 0.86473146, 0.0]],
        "period": 6.32591398,
        "source": "Moore (1993); Chenciner & Montgomery, Ann. of Math. 152 "
                  "(2000) 881-902",
        "note": None,
    },
    {
        "name": "Lagrange equilateral triangle (circular three-body orbit)",
        "N": 3,
        "seed": 2,
        "dt": 0.001,
        "tot_time": 3650,
        "positions": _lagrange_pos,
        "velocities": _lagrange_vel,
        "period": 2.0 * np.pi / np.sqrt(3.0),
        "source": "Lagrange (1772); angular frequency omega = sqrt(3)",
        "note": None,
    },
    {
        "name": "Circular two-body orbit",
        "N": 2,
        "seed": 3,
        "dt": 0.001,
        "tot_time": 4500,
        "positions": [[0.5, 0.0, 0.0],
                      [-0.5, 0.0, 0.0]],
        "velocities": [[0.0, np.sqrt(2.0) / 2.0, 0.0],
                       [0.0, -np.sqrt(2.0) / 2.0, 0.0]],
        "period": 2.0 * np.pi / np.sqrt(2.0),
        "source": "Classical two-body; separation 1, period T approx 4.4429",
        "note": None,
    },
    {
        "name": "Elliptic two-body orbit (e = 0.5)",
        "N": 2,
        "seed": 4,
        "dt": 0.001,
        "tot_time": 4500,
        "positions": [[0.75, 0.0, 0.0],
                      [-0.75, 0.0, 0.0]],
        "velocities": [[0.0, np.sqrt(1.0 / 6.0), 0.0],
                       [0.0, -np.sqrt(1.0 / 6.0), 0.0]],
        "period": 2.0 * np.pi / np.sqrt(2.0),
        "source": "Classical two-body; semi-major axis 1, apocenter start",
        "note": None,
    },
    # --- Suvakov-Dmitrasinovic periodic orbits (equal mass, zero angular
    #     momentum), discovered in Phys. Rev. Lett. 110, 114301 (2013),
    #     arXiv:1303.0181. High-precision velocities and periods from the
    #     Li-Liao SJTU catalogue (Science China PMA 60, 129511 (2017)),
    #     catalogue IDs given in the names. -------------------------------
    _sd_orbit(
        "Butterfly I (Suvakov-Dmitrasinovic I.A-2)",
        5, 0.3068934205, 0.1255065670, 6.2346748391, 80000,
        "Suvakov & Dmitrasinovic, PRL 110, 114301 (2013); SJTU catalogue "
        "I.A-2"),
    _sd_orbit(
        "Butterfly II (Suvakov-Dmitrasinovic 2013)",
        6, 0.39295, 0.09758, 7.0039, 320000,
        "Suvakov & Dmitrasinovic, PRL 110, 114301 (2013), Table I",
        note="initial velocities known to 5 significant digits, so the "
             "loop may drift slightly before closing"),
    _sd_orbit(
        "Butterfly III (Suvakov-Dmitrasinovic I.B-2)",
        7, 0.4059155671, 0.2301631260, 13.8671234361, 320000,
        "Suvakov & Dmitrasinovic, PRL 110, 114301 (2013); SJTU catalogue "
        "I.B-2"),
    _sd_orbit(
        "Moth I (Suvakov-Dmitrasinovic I.B-1)",
        8, 0.4644451728, 0.3960600146, 14.8943051743, 80000,
        "Suvakov & Dmitrasinovic, PRL 110, 114301 (2013); SJTU catalogue "
        "I.B-1"),
    _sd_orbit(
        "Moth II (Suvakov-Dmitrasinovic I.B-5)",
        9, 0.4391659182, 0.4529676431, 28.6692709402, 80000,
        "Suvakov & Dmitrasinovic, PRL 110, 114301 (2013); SJTU catalogue "
        "I.B-5"),
    _sd_orbit(
        "Moth III (Suvakov-Dmitrasinovic I.B-6)",
        10, 0.3834435199, 0.3773636946, 25.8392363356, 640000,
        "Suvakov & Dmitrasinovic, PRL 110, 114301 (2013); SJTU catalogue "
        "I.B-6",
        note="640000 steps, the integration takes a few minutes"),
    _sd_orbit(
        "Goggles (Suvakov-Dmitrasinovic I.B-3)",
        11, 0.0833000718, 0.1278892555, 10.4648495256, 80000,
        "Suvakov & Dmitrasinovic, PRL 110, 114301 (2013); SJTU catalogue "
        "I.B-3"),
    _sd_orbit(
        "Dragonfly (Suvakov-Dmitrasinovic I.B-4)",
        12, 0.0805842255, 0.5888360898, 21.2723373956, 640000,
        "Suvakov & Dmitrasinovic, PRL 110, 114301 (2013); SJTU catalogue "
        "I.B-4",
        note="640000 steps, the integration takes a few minutes"),
    _sd_orbit(
        "Yin-yang I (Suvakov-Dmitrasinovic II.C-1)",
        13, 0.2827020949, 0.3272089716, 10.9633031497, 320000,
        "Suvakov & Dmitrasinovic, PRL 110, 114301 (2013); SJTU catalogue "
        "II.C-1"),
]
