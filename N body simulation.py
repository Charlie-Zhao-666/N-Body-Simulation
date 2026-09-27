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
dt = 1
tot_time = 1000
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


# main loop
start_time = time.time()
test_time = [0]
test_result.append(test(stars))

energy_eigenvalue = 0
sigma_Ep = 0
sigma_Ek = 0
for i in range(N):
    sigma_Ek += Ek(stars[i])
    for j in range(i+1, N):
        sigma_Ep += Ep(stars[i], stars[j])
energy_eigenvalue = sigma_Ek - sigma_Ep

for step in range(tot_time):
    leapfrog(stars)
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
end_time = time.time()
print("report: simulation done!")
print(f"simulation cost {end_time - start_time:.2f} s\n")

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
ax.set_title("N-body simulation")
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

# positions \u7684\u7ed3\u6784\u662f\uFF1A
# positions[\u6052\u661f\u7f16\u53f7][\u6b65\u6570][\u5750\u6807\u5206\u91cf]
# \u63d0\u524d\u8f6c\u6210\u6570\u7ec4\uff0c\u907f\u514d\u6bcf\u6b21\u5237\u65b0\u90fd\u91cd\u65b0\u8f6c\u6362
trajectories = np.array(positions)

# for the known stable solutions the orbit scale is ~1 instead of p, so the
# view is resized from the actual trajectory extent, following the same
# factor-2 rule as b = 2 * p above
if select_seeds == 3:
    b = 2 * np.max(np.abs(trajectories))
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.set_zlim(-b, b)

# \u5305\u542b\u521d\u59cb\u72b6\u6001\uff0c\u6240\u4ee5\u6700\u540e\u4e00\u6b65\u662f\u8bb0\u5f55\u6570\u91cf\u51cf 1
last_step = trajectories.shape[1] - 1

# \u5f53\u524d\u64ad\u653e\u72b6\u6001
playback = {
    "step": 0,
    "dragging": False,
}


def update(step):
    """\u663e\u793a\u7b2c step \u6b65\uff1bstep=0 \u8868\u793a\u521d\u59cb\u72b6\u6001\u3002"""
    step = int(step)

    for i in range(N):
        # \u5305\u542b\u521d\u59cb\u4f4d\u7f6e\u548c\u5f53\u524d\u6b65\u7684\u4f4d\u7f6e
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


# ==================== \u53ef\u62d6\u52a8\u8fdb\u5ea6\u6761 ====================

slider_ax = fig.add_axes([0.20, 0.07, 0.60, 0.03])


def on_press(event):
    # \u5728\u8fdb\u5ea6\u6761\u4e0a\u6309\u4e0b\u9f20\u6807\u5de6\u952e\u65f6\uff0c\u6682\u505c\u81ea\u52a8\u63a8\u8fdb
    if event.inaxes == slider_ax and event.button == 1:
        playback["dragging"] = True


def on_release(event):
    # \u677e\u5f00\u9f20\u6807\u5de6\u952e\u540e\uff0c\u6062\u590d\u81ea\u52a8\u63a8\u8fdb
    if event.button == 1:
        playback["dragging"] = False


# \u5148\u6ce8\u518c\u9f20\u6807\u4e8b\u4ef6\uff0c\u518d\u521b\u5efa Slider
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
    """\u624b\u52a8\u62d6\u52a8\u548c\u81ea\u52a8\u64ad\u653e\uff0c\u90fd\u901a\u8fc7\u8fd9\u91cc\u66f4\u65b0\u753b\u9762\u3002"""
    playback["step"] = int(value)
    update(playback["step"])
    fig.canvas.draw_idle()


time_slider.on_changed(on_slider_change)


# ==================== \u81ea\u52a8\u64ad\u653e ====================

def init_animation():
    # \u521d\u59cb\u5316\u65f6\u663e\u793a\u5f53\u524d\u8fdb\u5ea6\uff0c\u4e0d\u81ea\u52a8\u5411\u524d\u8df3\u4e00\u6b65
    return update(playback["step"])


def animate(_):
    if playback["dragging"]:
        return lines + points + [time_text]

    next_step = playback["step"] + 1

    # \u5230\u8fbe\u6700\u540e\u4e00\u5e27\u540e\uff0c\u5faa\u73af\u64ad\u653e
    if next_step > last_step:
        next_step = 0

    # set_val \u4f1a\u89e6\u53d1 on_slider_change
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


# ==================== \u7269\u7406\u91cf\u9a8c\u8bc1\u56fe ====================

times = np.asarray(test_time)

energies = np.array([
    result["energy"]
    for result in test_result
])

# \u6bcf\u79cd\u5411\u91cf\u7269\u7406\u91cf\u7684\u6570\u636e\u5f62\u72b6\u90fd\u662f\uFF1A
# (\u8bb0\u5f55\u65f6\u523b\u6570\u91cf, 3)
vector_info = [
    ("momentum", "Total momentum", "P"),
    ("angular_momentum", "Total angular momentum", "L"),
    ("com_v", "Center-of-mass velocity", "V_com"),
]

fig_test = plt.figure(
    figsize=(17, 12),
    constrained_layout=True,
)

# \u5171\u56db\u884c\u3001\u56db\u5217
# \u7b2c\u4e00\u884c\uff1a\u603b\u80fd\u91cf\uff0c\u5360\u6574\u884c
# \u540e\u4e09\u884c\uff1a\u6bcf\u884c\u4e00\u4e2a\u5411\u91cf\u7269\u7406\u91cf\uff0c\u5206\u522b\u753b x\u3001y\u3001z\u3001\u6a21\u957f
grid = fig_test.add_gridspec(4, 4)

# -------------------- \u603b\u80fd\u91cf --------------------

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

# -------------------- \u5411\u91cf\u7269\u7406\u91cf --------------------

component_names = ["x", "y", "z"]
component_colors = ["tab:red", "tab:green", "tab:blue"]

for row, (key, title, symbol) in enumerate(vector_info, start=1):
    values = np.array([
        result[key]
        for result in test_result
    ])

    # x\u3001y\u3001z \u5206\u91cf\u5206\u522b\u753b\u5728\u4e09\u4e2a\u72ec\u7acb\u5b50\u56fe\u4e2d
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
        
        # \u8ba1\u7b97\u8be5\u5206\u91cf\u7684RMSE
        component_rmse = RMSE(values[:, component])
        component_title = f"{title}: {name}\nRMSE: {component_rmse:.6e}"
        axis.set_title(component_title, fontsize=9)
        axis.set_xlabel("Time")
        axis.set_ylabel(f"{symbol}_{name}")
        axis.grid(True, alpha=0.3)

    # axis=1\uFF1A\u5bf9\u6bcf\u4e2a\u65f6\u523b\u7684\u4e09\u4e2a\u5206\u91cf\u8ba1\u7b97\u6a21\u957f
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

# \u6bcf\u4e2a\u5b50\u56fe\u51c6\u5907\u4e00\u4e2a\u6570\u503c\u63d0\u793a\u6846
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

# ==================== \u6dfb\u52a0RMSE\u6c47\u603b\u4fe1\u606f ====================

# \u8ba1\u7b97\u6240\u6709RMSE\u503c
# all_rmse_info = f"RMSE Summary:\nEnergy: {energy_rmse_final:.6e}"

# for key, title, symbol in vector_info:
#     values = np.array([result[key] for result in test_result])
#     magnitudes = np.linalg.norm(values, axis=1)
#     magnitude_rmse = RMSE(magnitudes)
#     all_rmse_info += f"\n{title} (mag): {magnitude_rmse:.6e}"

# # \u5728\u56fe\u8868\u53f3\u4e0a\u89d2\u6dfb\u52a0RMSE\u4fe1\u606f\u6846
# fig_test.text(
#     0.98, 0.98, all_rmse_info,
#     transform=fig_test.transFigure,
#     fontsize=9,
#     verticalalignment='top',
#     horizontalalignment='right',
#     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.8)
# )

hover_state = {"selection": None}


def show_nearest_value(event):
    axis = event.inaxes

    if axis not in hover_items or event.xdata is None:
        return

    item = hover_items[axis]
    xs = item["x"]

    if len(xs) == 0:
        return

    # \u65f6\u95f4\u6309\u5347\u5e8f\u8bb0\u5f55\uff0c\u7528\u4e8c\u5206\u67e5\u627e\u627e\u5230\u6700\u8fd1\u7684\u6570\u636e\u70b9
    index = int(np.searchsorted(xs, event.xdata))
    index = min(index, len(xs) - 1)

    if index > 0:
        left_distance = abs(event.xdata - xs[index - 1])
        right_distance = abs(event.xdata - xs[index])

        if left_distance <= right_distance:
            index -= 1

    # \u4ecd\u7136\u6307\u5411\u540c\u4e00\u6761\u8bb0\u5f55\u65f6\uff0c\u4e0d\u91cd\u590d\u5237\u65b0
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
