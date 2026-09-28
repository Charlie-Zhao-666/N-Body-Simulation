import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

print("simulation start!")

# ==================== parameters ====================

dt = 100
tot_time = 1000       # simulation steps
G = 1e-11

N = 50               # includes the two centers; must be an even number >= 4
seed = 42
rng = np.random.default_rng(seed)

if N < 4 or N % 2 != 0:
    raise ValueError("N must be an even number >= 4")

num_per_galaxy = N // 2

r_min, r_max = 2, 25
galaxy_distance = 60
mass_range = (1, 10)

mass_center = 10**6 * np.mean(mass_range)

v_center_mag = 0.0003
bias_strength = v_center_mag

stars = []
positions = []

# ==================== two central particles ====================

pos1 = np.array(
    [-galaxy_distance / 2, 0.0, 0.0],
    dtype=float,
)

pos2 = np.array(
    [galaxy_distance / 2, 0.0, 0.0],
    dtype=float,
)

# in-plane motion only: vertical velocity is exactly zero
vel1 = np.array(
    [v_center_mag, -bias_strength, 0.0],
    dtype=float,
)

vel2 = np.array(
    [-v_center_mag, bias_strength, 0.0],
    dtype=float,
)

for pos, vel in [(pos1, vel1), (pos2, vel2)]:
    stars.append({
        "mass": mass_center,
        "pos": pos.copy(),
        "v": vel.copy(),
    })
    positions.append([pos.copy()])

# ==================== galaxy disks ====================

for g, pos_center in enumerate([pos1, pos2]):
    v_center = stars[g]["v"]

    for _ in range(num_per_galaxy - 1):
        r = rng.uniform(r_min, r_max)
        theta = rng.uniform(0, 2 * np.pi)

        x = pos_center[0] + r * np.cos(theta)
        y = pos_center[1] + r * np.sin(theta)

        # all particles lie exactly in the xy plane
        pos = np.array([x, y, 0.0], dtype=float)

        mass = rng.uniform(*mass_range)

        v_mag = np.sqrt(G * mass_center / r)

        vx = -v_mag * np.sin(theta)
        vy = v_mag * np.cos(theta)

        # orbital velocity has no z component
        vel = np.array([vx, vy, 0.0], dtype=float)

        # keep the original 1.2x center velocity
        # center velocity has zero z, so this adds no vertical motion
        vel += 1.2 * v_center

        stars.append({
            "mass": mass,
            "pos": pos,
            "v": vel,
        })
        positions.append([pos.copy()])

# ==================== 3d gravity ====================

def accelerations(particles):
    count = len(particles)
    accs = np.zeros((count, 3), dtype=float)

    for i in range(count):
        for j in range(i + 1, count):
            diff = particles[j]["pos"] - particles[i]["pos"]
            distance = np.linalg.norm(diff)

            if distance == 0:
                raise ValueError(
                    f"particles {i} and {j} coincide exactly; point gravity undefined"
                )

            # full 3d acceleration is still computed
            pair_acc = (
                G * particles[j]["mass"]
                / distance**3
                * diff
            )

            accs[i] += pair_acc
            accs[j] -= (
                pair_acc
                * particles[i]["mass"]
                / particles[j]["mass"]
            )

    if not np.all(np.isfinite(accs)):
        raise FloatingPointError("accelerations became NaN or inf")

    return accs


# ==================== leapfrog ====================

def leapfrog(particles):
    count = len(particles)

    old_accs = accelerations(particles)

    # half step velocity
    for i in range(count):
        particles[i]["v"] += 0.5 * dt * old_accs[i]

    # full step position
    for i in range(count):
        particles[i]["pos"] += dt * particles[i]["v"]

    new_accs = accelerations(particles)

    # second half step velocity
    for i in range(count):
        particles[i]["v"] += 0.5 * dt * new_accs[i]

    for i in range(count):
        positions[i].append(particles[i]["pos"].copy())

    # largest vertical acceleration seen in both evaluations
    return max(
        np.max(np.abs(old_accs[:, 2])),
        np.max(np.abs(new_accs[:, 2])),
    )


# ==================== planarity diagnostics ====================

def plane_error(particles):
    pos = np.array([star["pos"] for star in particles])
    vel = np.array([star["v"] for star in particles])

    if not np.all(np.isfinite(pos)):
        raise FloatingPointError("positions became NaN or inf")

    if not np.all(np.isfinite(vel)):
        raise FloatingPointError("velocities became NaN or inf")

    return (
        np.max(np.abs(pos[:, 2])),
        np.max(np.abs(vel[:, 2])),
    )


initial_z, initial_vz = plane_error(stars)

max_z_seen = initial_z
max_vz_seen = initial_vz
max_az_seen = 0.0
first_departure = None

print(f"seed = {seed}")
print(f"initial max |z|  = {initial_z:.17g}")
print(f"initial max |vz| = {initial_vz:.17g}")

# ==================== main loop ====================

start_time = time.perf_counter()
progress_interval = max(1, tot_time // 10)

for step in range(tot_time):
    step_max_az = leapfrog(stars)
    current_z, current_vz = plane_error(stars)

    max_z_seen = max(max_z_seen, current_z)
    max_vz_seen = max(max_vz_seen, current_vz)
    max_az_seen = max(max_az_seen, step_max_az)

    if first_departure is None:
        if current_z != 0.0 or current_vz != 0.0 or step_max_az != 0.0:
            first_departure = step + 1

    if (step + 1) % progress_interval == 0:
        print(f"progress: {(step + 1) / tot_time:.0%}")

end_time = time.perf_counter()

print(f"\nsimulation cost {end_time - start_time:.2f} s")
print(f"whole run max |z|  = {max_z_seen:.17g}")
print(f"whole run max |vz| = {max_vz_seen:.17g}")
print(f"whole run max |az| = {max_az_seen:.17g}")

if first_departure is None:
    print("PASS: z, vz and az stayed exactly zero.")
else:
    print(f"Nonzero z component first detected during step {first_departure}.")

# ==================== 3d animation ====================

trajectories = np.asarray(positions)
last_step = trajectories.shape[1] - 1

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.set_title("Galaxy collision: planar initial conditions")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

b = galaxy_distance
ax.set_xlim(-b, b)
ax.set_ylim(-b, b)
ax.set_zlim(-b, b)
ax.set_box_aspect((1, 1, 1))

lines = []
points = []

for i in range(len(stars)):
    if i == 0:
        color = "red"
        linewidth = 1.5
        markersize = 6
    elif i == 1:
        color = "orange"
        linewidth = 1.5
        markersize = 6
    else:
        color = "tab:blue"
        linewidth = 0.5
        markersize = 2

    line, = ax.plot(
        [], [], [],
        color=color,
        linewidth=linewidth,
        alpha=0.6,
    )

    point, = ax.plot(
        [], [], [],
        "o",
        color=color,
        markersize=markersize,
    )

    lines.append(line)
    points.append(point)

time_text = ax.text2D(
    0.98, 0.98, "",
    transform=ax.transAxes,
    ha="right",
    va="top",
)


def update(step):
    for i in range(len(stars)):
        traj = trajectories[i, :step + 1]

        lines[i].set_data(traj[:, 0], traj[:, 1])
        lines[i].set_3d_properties(traj[:, 2])

        points[i].set_data([traj[-1, 0]], [traj[-1, 1]])
        points[i].set_3d_properties([traj[-1, 2]])

    frame_max_z = np.max(np.abs(trajectories[:, step, 2]))

    time_text.set_text(
        f"t = {step * dt:.2f}\n"
        f"max |z| = {frame_max_z:.3e}"
    )

    return lines + points + [time_text]


ani = FuncAnimation(
    fig,
    update,
    frames=range(last_step + 1),
    init_func=lambda: update(0),
    interval=20,
    blit=False,
    cache_frame_data=False,
)

plt.show()
