import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import mplcursors
import sys
import copy
import json
import secrets


# global var
dt = 5
tot_time = 10000
G = 1
N = 5

# setting initial information

p = 50      # position factor
v = 0.1     # initial velocity range
test_result = []
stars = []
positions = []

# seeds selecting
print("select the seed that you want")
print("1. random seed.\n2. import seed.\n3. Known seeds")

select_seeds = 4
while select_seeds not in (1, 2, 3):
    select_seeds = int(input())

    if select_seeds == 1:
        seed = secrets.randbits(32)

    elif select_seeds == 2:
        seed = int(input("please type the seed:"))

    elif select_seeds == 3:
        # Known stable periodic solutions (G = 1, equal masses m = 1).
        # The orbit data lives in stable_orbits.py, imported locally here
        # so this block stays self-contained. Each option sets N, dt,
        # tot_time and loads the exact initial conditions directly, so the
        # random builder below is skipped. All solutions have zero total
        # momentum and their center of mass at the origin.
        from stable_orbits import STABLE_ORBITS

        print("which stable solution do you want to run?")
        for i, orbit in enumerate(STABLE_ORBITS, start=1):
            print(f"{i}. {orbit['name']}")

        solution = 0
        while solution not in range(1, len(STABLE_ORBITS) + 1):
            solution = int(input())

            if solution not in range(1, len(STABLE_ORBITS) + 1):
                print("ERROR!")

        orbit = STABLE_ORBITS[solution - 1]
        N = orbit["N"]
        dt = orbit["dt"]
        tot_time = orbit["tot_time"]
        seed = orbit["seed"]
        solution_stars = [
            {"mass": 1,
             "pos": np.array(pos, dtype=float),
             "v": np.array(vel, dtype=float)}
            for pos, vel in zip(orbit["positions"], orbit["velocities"])
        ]

        for star in solution_stars:
            stars.append(star)
            positions.append([star["pos"].copy()])

        print(f"loaded stable solution {solution}: {orbit['name']}")
        print(f"N = {N}, "
              f"suggested dt = {dt}, suggested steps = {tot_time}")
        if orbit.get("note"):
            print(f"note: {orbit['note']}")

    else:
        print("ERROR!")


# build stars
np.random.seed(seed)
# option 3 loads exact stable solutions above; skip the random builder for it
if select_seeds != 3:
    for i in range(N):
        star = {
            "mass": 1,
            "pos": np.random.uniform(-p, p, size=3),
            "v": np.random.uniform(-v, v, size=3),
        }
        stars.append(star)
        positions.append([star["pos"].copy()])

initial_stars = copy.deepcopy(stars)


print("chose time step")
print("1. defult value/determined by seed.\n2. Customized value.")
dtselection = 0
while dtselection not in (1, 2):
    dtselection = int(input())
    if dtselection == 1:
        pass
    elif dtselection == 2:
        dt = float(input())
    else:
        print("ERROR")

print("chose total time")
print("1. defult value/determined by seed.\n2. Customized value.")
ttselection = 0
while ttselection not in (1, 2):
    ttselection = int(input())
    if ttselection == 1:
        pass
    elif ttselection == 2:
        tot_time = int(input())
    else:
        print("ERROR")


# function

def dist(s1, s2):
    return np.linalg.norm(s2["pos"] - s1["pos"])


def dir_vector(s1, s2):
    return (s2["pos"] - s1["pos"]) / dist(s1, s2)


def acc(s1, s2):
    return G * s2["mass"] / dist(s1, s2)**2 * dir_vector(s1, s2)


def Ek(s1):
    return 0.5 * s1["mass"] * np.linalg.norm(s1["v"])**2


def Ep(s1, s2):
    return - G * s1["mass"] * s2["mass"] / dist(s1, s2)


def tot_E(s):
    sigma_Ep = 0
    sigma_Ek = 0
    for i in range(len(s)):
        sigma_Ek += Ek(s[i])
        for j in range(i+1, N):
            sigma_Ep += Ep(s[i], s[j])
    return sigma_Ek + sigma_Ep


def Px(s):
    sigma_Px = 0
    for i in range(len(s)):
        sigma_Px += s[i]["mass"] * s[i]["v"][0]
    return sigma_Px


def Py(s):
    sigma_Py = 0
    for i in range(len(s)):
        sigma_Py += s[i]["mass"] * s[i]["v"][1]
    return sigma_Py


def Pz(s):
    sigma_Pz = 0
    for i in range(len(s)):
        sigma_Pz += s[i]["mass"] * s[i]["v"][2]
    return sigma_Pz


def tot_P(s):
    a = np.array([Px(s), Py(s), Pz(s)])
    return a


def tot_P_mag(s):
    return np.linalg.norm(tot_P(s))


def L(s):
    sigma_L = np.zeros(3)
    for i in range(len(s)):
        sigma_L += np.cross(s[i]["pos"], s[i]["mass"] * s[i]["v"])
    return sigma_L


def CoM_pos(s):
    tot_m = 0
    mr = np.zeros(3)
    for i in range(len(s)):
        tot_m += s[i]["mass"]
        mr += s[i]["mass"] * s[i]["pos"]
    return mr/tot_m


def CoM_v(s):
    tot_m = 0
    mv = np.zeros(3)
    for i in range(len(s)):
        tot_m += s[i]["mass"]
        mv += s[i]["mass"] * s[i]["v"]
    return mv/tot_m


def test(s):
    return {
        "energy": tot_E(s),
        "momentum": tot_P(s),
        "angular_momentum": L(s),
        "com_pos": CoM_pos(s),
        "com_v": CoM_v(s)
    }


def RMSE(data):
    a = 0
    for i in range(1, len(data)):
        a += (data[0] - data[i]) ** 2
    mse = a / (len(data) - 1)
    return np.sqrt(mse)


# leapfrog
def leapfrog(stars):
    accs = [np.zeros(3) for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            a = acc(stars[i], stars[j])
            accs[i] += a
            accs[j] -= a * stars[i]["mass"] / stars[j]["mass"]

    # half step velocity
    for i in range(N):
        stars[i]["v"] += 0.5 * dt * accs[i]

    # full step position
    for i in range(N):
        stars[i]["pos"] += stars[i]["v"] * dt

    # new acceleartion
    accs = [np.zeros(3) for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            a = acc(stars[i], stars[j])
            accs[i] += a
            accs[j] -= a * stars[i]["mass"] / stars[j]["mass"]

    # another half step velocity
    for i in range(N):
        stars[i]["v"] += 0.5 * dt * accs[i]

    # record
    for i in range(N):
        positions[i].append(stars[i]["pos"].copy())


# Runge-Kutta 4th order method
def rk4(stars):
    accs = [np.zeros(3) for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            a = acc(stars[i], stars[j])
            accs[i] += a
            accs[j] -= a * stars[i]["mass"] / stars[j]["mass"]

    # K1
    k1 = accs
    v1 = np.array([star["v"] for star in stars])

    # K2
    v2 = v1 + 0.5 * dt * np.array(k1)
    k2 = [np.zeros(3) for _ in range(N)]
    temp_stars = []
    for i in range(N):            
        temp_stars.append({
            "mass": stars[i]["mass"],
            "pos": stars[i]["pos"] + 0.5 * dt * v1[i],
            "v": v2[i]
        })
    for i in range(N):            
        for j in range(i+1, N):
            a = acc(temp_stars[i], temp_stars[j])
            k2[i] += a
            k2[j] -= a * temp_stars[i]["mass"] / temp_stars[j]["mass"]

    # K3
    v3 = v1 + 0.5 * dt * np.array(k2)
    k3 = [np.zeros(3) for _ in range(N)]
    temp_stars = []
    for i in range(N):            
        temp_stars.append({
            "mass": stars[i]["mass"],
            "pos": stars[i]["pos"] + 0.5 * dt * v2[i],
            "v": v3[i]
        })
    for i in range(N):            
        for j in range(i+1, N):
            a = acc(temp_stars[i], temp_stars[j])
            k3[i] += a
            k3[j] -= a * temp_stars[i]["mass"] / temp_stars[j]["mass"]

    # K4
    v4 = v1 + dt * np.array(k3)
    k4 = [np.zeros(3) for _ in range(N)]
    temp_stars = []
    for i in range(N):            
        temp_stars.append({
            "mass": stars[i]["mass"],
            "pos": stars[i]["pos"] + dt * v3[i],
            "v": v4[i]
        })
    for i in range(N):            
        for j in range(i+1, N):
            a = acc(temp_stars[i], temp_stars[j])
            k4[i] += a
            k4[j] -= a * temp_stars[i]["mass"] / temp_stars[j]["mass"]

    # Update positions and velocities
    for i in range(N):
        stars[i]["pos"] += dt * (v1[i] + 2*v2[i] + 2*v3[i] + v4[i]) / 6
        stars[i]["v"] += dt * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6

        # record
    for i in range(N):
        positions[i].append(stars[i]["pos"].copy())







# main loop
#======================================================================================================================================================================
#======================================================================================================================================================================
#======================================================================================================================================================================
#======================================================================================================================================================================
energy_eigenvalue = 0
sigma_Ep = 0
sigma_Ek = 0
for i in range(N):
    sigma_Ek += Ek(stars[i])
    for j in range(i+1, N):
        sigma_Ep += Ep(stars[i], stars[j])
energy_eigenvalue = sigma_Ek - sigma_Ep

print("which method do you want to use?")
print("1. Leapfrog")
print("2. Runge-Kutta 4th order")

method = input("Enter your choice (1 or 2): ")


simpling_rate = 0
if tot_time > 1000:
    print("chose your simpling rate")
    print("1. Every 10 steps.\n2. 1% of total steps.\n3. 0.1% of total steps.\n4. 0.01% of total steps.")
    simpling_rate = int(input("Enter your choice (1, 2 or 3): "))


start_time = time.time()
test_time = [0]
test_result.append(test(stars))


for step in range(tot_time):
    if method == "1":
        leapfrog(stars)
    elif method == "2":
        rk4(stars)
    else:
        print("Invalid choice. Using Leapfrog by default.")
        leapfrog(stars)

    if simpling_rate in (0, 1):
        if (step + 1) % 10 == 0:
            print(step/tot_time, "/", 1)
            test_result.append(test(stars))
            test_time += [(step + 1) * dt]
        if step/tot_time == 0.01:
            pridict_time = time.time()
            print(f"remaining {(pridict_time-start_time)*99} s")
        if step/tot_time == 0.05:
            pridict_time = time.time()
            print(f"remaining {(pridict_time-start_time)*19} s")

    elif simpling_rate == 2:
        if (step + 1) % (tot_time // 100) == 0:
            print(step/tot_time, "/", 1)
            test_result.append(test(stars))
            test_time += [(step + 1) * dt]
        if step/tot_time == 0.01:
            pridict_time = time.time()
            print(f"remaining {(pridict_time-start_time)*99} s")
        if step/tot_time == 0.05:
            pridict_time = time.time()
            print(f"remaining {(pridict_time-start_time)*19} s")

    elif simpling_rate == 3:
        if (step + 1) % (tot_time // 1000) == 0:
            print(step/tot_time, "/", 1)
            test_result.append(test(stars))
            test_time += [(step + 1) * dt]
        if step/tot_time == 0.01:
            pridict_time = time.time()
            print(f"remaining {(pridict_time-start_time)*99} s")
        if step/tot_time == 0.05:
            pridict_time = time.time()
            print(f"remaining {(pridict_time-start_time)*19} s")

end_time = time.time()
print("\nreport: simulation done!")
print(f"\nsimulation cost {end_time - start_time:.2f} s\n")


# error analysis
energy_result = np.array([
    result["energy"]
    for result in test_result
])

energy_rmse = float(RMSE(energy_result))

rmse_results = {
    "energy": energy_rmse
}

normalized_energy_RMSE = energy_rmse / energy_eigenvalue
print("\n=============== RMSE ===============")
quantity_info = [
    ("momentum", "Total Momentum"),
    ("angular_momentum", "Total Angular Momentum"),
    ("com_v", "Center Of Mass Velocity"),
]

for key, title in quantity_info:
    values = np.array([
        result[key]
        for result in test_result
    ])

    component_rmse = RMSE(values)

    magnitudes = np.linalg.norm(values, axis=1)
    magnitude_rmse = RMSE(magnitudes)

    rmse_results[key] = {
        "x": float(component_rmse[0]),
        "y": float(component_rmse[1]),
        "z": float(component_rmse[2]),
        "magnitude": float(magnitude_rmse),
    }

    print(f"\n{title}\uFF1A")
    print(f"  x component\uFF1A{component_rmse[0]:.6e}")
    print(f"  y component\uFF1A{component_rmse[1]:.6e}")
    print(f"  z component\uFF1A{component_rmse[2]:.6e}")
    print(f"  Mag\uFF1A  {magnitude_rmse:.6e}\n")

print(f"Total Energy\uFF1A{energy_rmse:.6e}")
print(f"Normalized Energy RMSE:{normalized_energy_RMSE:.6e}")

print("====================================")

print(f"seed: {seed}, N = {N}, dt = {dt}, step = {tot_time}\n")








# animation & plot
# ============================================================================================================================================
# ============================================================================================================================================
# ============================================================================================================================================
# ============================================================================================================================================

fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(bottom=0.18)

ax = fig.add_subplot(111, projection="3d")
method_name = {"1": "Leapfrog (KDK)", "2": "Runge-Kutta 4th order"}.get(method, method)
ax.set_title(f"N-body simulation\nseed = {seed}, dt = {dt}, method = {method_name}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

colors = ["r", "g", "b", "y", "c", "m"]

lines = [
    ax.plot(
        [], [], [],
        color=colors[i % len(colors)],
        linewidth=1,
    )[0]
    for i in range(N)
]

points = [
    ax.plot(
        [], [], [],
        "o",
        color=colors[i % len(colors)],
        markersize=6,
    )[0]
    for i in range(N)
]

time_text = ax.text2D(
    0.98, 0.98, "",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=12,
)

b = 2 * p
ax.set_xlim(-b, b)
ax.set_ylim(-b, b)
ax.set_zlim(-b, b)

# positions[star][step][coordinate], converted to an array once
trajectories = np.array(positions)

# for the known stable solutions the orbit scale is ~1 instead of p, so the
# view is resized from the actual trajectory extent, following the same
# factor-2 rule as b = 2 * p above
if select_seeds == 3:
    b = 2 * np.max(np.abs(trajectories))
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.set_zlim(-b, b)

# positions include the initial state, so the last index is count - 1
last_step = trajectories.shape[1] - 1

# playback state
playback = {
    "step": 0,
    "dragging": False,
}


def update(step):
    """show step; step=0 is the initial state"""
    step = int(step)

    for i in range(N):
        # include the initial position up to the current step
        traj = trajectories[i, :step + 1]

        lines[i].set_data(traj[:, 0], traj[:, 1])
        lines[i].set_3d_properties(traj[:, 2])

        points[i].set_data(
            [traj[-1, 0]],
            [traj[-1, 1]],
        )
        points[i].set_3d_properties([traj[-1, 2]])

    time_text.set_text(
        f"t = {step * dt:.2f}\n"
        f"Step: {step} / {last_step}"
    )

    return lines + points + [time_text]


# ==================== draggable progress slider ====================

slider_ax = fig.add_axes([0.20, 0.07, 0.60, 0.03])


def on_press(event):
    # pause auto-advance while the slider is pressed
    if event.inaxes == slider_ax and event.button == 1:
        playback["dragging"] = True


def on_release(event):
    # resume auto-advance on release
    if event.button == 1:
        playback["dragging"] = False


# register mouse events before creating the Slider
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)

time_slider = Slider(
    ax=slider_ax,
    label="Step",
    valmin=0,
    valmax=last_step,
    valinit=0,
    valstep=1,
    valfmt="%0.0f",
)


def on_slider_change(value):
    """both slider drags and autoplay update through here"""
    playback["step"] = int(value)
    update(playback["step"])
    fig.canvas.draw_idle()


time_slider.on_changed(on_slider_change)


# ==================== autoplay ====================

def init_animation():
    # show current progress on init, do not advance
    return update(playback["step"])


def animate(_):
    if playback["dragging"]:
        return lines + points + [time_text]

    next_step = playback["step"] + 1

    # loop back to the first frame after the last one
    if next_step > last_step:
        next_step = 0

    # set_val triggers on_slider_change
    time_slider.set_val(next_step)

    return lines + points + [time_text]


ani = FuncAnimation(
    fig,
    animate,
    init_func=init_animation,
    interval=20,
    blit=False,
    cache_frame_data=False,
)


# ==================== conservation dashboard ====================

times = np.asarray(test_time)

energies = np.array([
    result["energy"]
    for result in test_result
])

# vector quantities are stored as (num_records, 3)
vector_info = [
    ("momentum", "Total momentum", "P"),
    ("angular_momentum", "Total angular momentum", "L"),
    ("com_v", "Center-of-mass velocity", "V_com"),
]

fig_test = plt.figure(
    figsize=(17, 12),
    constrained_layout=True,
)

# 4x4 grid: energy on row 0, then one row per vector quantity (x, y, z, magnitude)
grid = fig_test.add_gridspec(4, 4)
fig_test.suptitle(f"seed = {seed}, dt = {dt}, method = {method_name}")

# -------------------- total energy --------------------

energy_ax = fig_test.add_subplot(grid[0, :])

energy_ax.plot(
    times,
    energies,
    color="black",
)

energy_rmse_final = RMSE(energies)
energy_title = f"Total energy (RMSE: {energy_rmse_final:.6e}, normalized: {energy_rmse_final / energy_eigenvalue:.6e})"
energy_ax.set_title(energy_title)
energy_ax.set_xlabel("Time")
energy_ax.set_ylabel("E")
energy_ax.grid(True, alpha=0.3)

# -------------------- vector quantities --------------------

component_names = ["x", "y", "z"]
component_colors = ["tab:red", "tab:green", "tab:blue"]

for row, (key, title, symbol) in enumerate(vector_info, start=1):
    values = np.array([
        result[key]
        for result in test_result
    ])

    # x, y, z components are plotted in separate subplots
    for component in range(3):
        axis = fig_test.add_subplot(
            grid[row, component],
            sharex=energy_ax,
        )

        name = component_names[component]

        axis.plot(
            times,
            values[:, component],
            color=component_colors[component],
        )

        # RMSE of this component
        component_rmse = RMSE(values[:, component])
        component_title = f"{title}: {name}\nRMSE: {component_rmse:.6e}"
        axis.set_title(component_title, fontsize=9)
        axis.set_xlabel("Time")
        axis.set_ylabel(f"{symbol}_{name}")
        axis.grid(True, alpha=0.3)

    # magnitude of the three components
    magnitudes = np.linalg.norm(values, axis=1)

    magnitude_ax = fig_test.add_subplot(
        grid[row, 3],
        sharex=energy_ax,
    )

    magnitude_ax.plot(
        times,
        magnitudes,
        color="tab:purple",
    )

    magnitude_rmse = RMSE(magnitudes)
    magnitude_title = f"{title}: magnitude\nRMSE: {magnitude_rmse:.6e}"
    magnitude_ax.set_title(magnitude_title, fontsize=9)
    magnitude_ax.set_xlabel("Time")
    magnitude_ax.set_ylabel(f"|{symbol}|")
    magnitude_ax.grid(True, alpha=0.3)

# one hover tooltip per subplot
hover_items = {}

for axis in fig_test.axes:
    if not axis.lines:
        continue

    line = axis.lines[0]

    label = axis.text(
        0.02, 0.98, "",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            edgecolor="0.8",
            alpha=0.9,
        ),
    )

    hover_items[axis] = {
        "x": np.asarray(line.get_xdata()),
        "y": np.asarray(line.get_ydata()),
        "label": label,
    }

hover_state = {"selection": None}


def show_nearest_value(event):
    axis = event.inaxes

    if axis not in hover_items or event.xdata is None:
        return

    item = hover_items[axis]
    xs = item["x"]

    if len(xs) == 0:
        return

    # times are sorted: binary search for the nearest record
    index = int(np.searchsorted(xs, event.xdata))
    index = min(index, len(xs) - 1)

    if index > 0:
        left_distance = abs(event.xdata - xs[index - 1])
        right_distance = abs(event.xdata - xs[index])

        if left_distance <= right_distance:
            index -= 1

    # skip redraw when still pointing at the same record
    selection = (axis, index)

    if selection == hover_state["selection"]:
        return

    hover_state["selection"] = selection

    item["label"].set_text(
        f"Time = {xs[index]:.6g}\n"
        f"Value = {item['y'][index]:.17g}"
    )

    fig_test.canvas.draw_idle()


fig_test.canvas.mpl_connect(
    "motion_notify_event",
    show_nearest_value,
)




plt.show()

# ask what to save after the windows are closed:
# 1 = animation (gif), 2 = image (png), 3 = both, anything else = nothing
print("save results?  1 = animation (gif)   2 = image (png)   3 = both   anything else = no")
try:
    save_choice = input().strip()
except EOFError:
    save_choice = ""

base_name = f"seed_{seed}_method_{method_name}_dt_{dt}_time_{tot_time}"

if save_choice in ("1", "3"):
    # the interactive animation loops forever, so build a separate finite one
    # for saving; the gif is capped at max_gif_frames frames (long runs get
    # sub-sampled) -- change this number to raise/lower the cap
    max_gif_frames = 2000
    frame_stride = max(1, (last_step + 1) // max_gif_frames)
    if frame_stride > 1:
        print(f"saving every {frame_stride}th frame to keep the gif under {max_gif_frames} frames ...")
    print("saving animation (this can take a while) ...")
    save_ani = FuncAnimation(fig, update, frames=range(0, last_step + 1, frame_stride),
                             interval=50, blit=False)
    save_ani.save(f"{base_name}.gif", writer="pillow")
    print(f"saved: {base_name}.gif")

if save_choice in ("2", "3"):
    fig_test.savefig(f"{base_name}.png", dpi=300)
    print(f"saved: {base_name}.png")
