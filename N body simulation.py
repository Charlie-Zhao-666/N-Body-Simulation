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
        print("which model do you want to run?")
        print("1. \n")

    else:
        print("ERROR!")


# build stars
np.random.seed(seed)
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

    print(f"\n{title}：")
    print(f"  x component：{component_rmse[0]:.6e}")
    print(f"  y component：{component_rmse[1]:.6e}")
    print(f"  z component：{component_rmse[2]:.6e}")
    print(f"  Mag：  {magnitude_rmse:.6e}\n")

print(f"Total Energy：{energy_rmse:.6e}")
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

# positions 的结构是：
# positions[恒星编号][步数][坐标分量]
# 提前转成数组，避免每次刷新都重新转换
trajectories = np.array(positions)

# 包含初始状态，所以最后一步是记录数量减 1
last_step = trajectories.shape[1] - 1

# 当前播放状态
playback = {
    "step": 0,
    "dragging": False,
}


def update(step):
    """显示第 step 步；step=0 表示初始状态。"""
    step = int(step)

    for i in range(N):
        # 包含初始位置和当前步的位置
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


# ==================== 可拖动进度条 ====================

slider_ax = fig.add_axes([0.20, 0.07, 0.60, 0.03])


def on_press(event):
    # 在进度条上按下鼠标左键时，暂停自动推进
    if event.inaxes == slider_ax and event.button == 1:
        playback["dragging"] = True


def on_release(event):
    # 松开鼠标左键后，恢复自动推进
    if event.button == 1:
        playback["dragging"] = False


# 先注册鼠标事件，再创建 Slider
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
    """手动拖动和自动播放，都通过这里更新画面。"""
    playback["step"] = int(value)
    update(playback["step"])
    fig.canvas.draw_idle()


time_slider.on_changed(on_slider_change)


# ==================== 自动播放 ====================

def init_animation():
    # 初始化时显示当前进度，不自动向前跳一步
    return update(playback["step"])


def animate(_):
    if playback["dragging"]:
        return lines + points + [time_text]

    next_step = playback["step"] + 1

    # 到达最后一帧后，循环播放
    if next_step > last_step:
        next_step = 0

    # set_val 会触发 on_slider_change
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


# ==================== 物理量验证图 ====================

times = np.asarray(test_time)

energies = np.array([
    result["energy"]
    for result in test_result
])

# 每种向量物理量的数据形状都是：
# (记录时刻数量, 3)
vector_info = [
    ("momentum", "Total momentum", "P"),
    ("angular_momentum", "Total angular momentum", "L"),
    ("com_v", "Center-of-mass velocity", "V_com"),
]

fig_test = plt.figure(
    figsize=(17, 12),
    constrained_layout=True,
)

# 共四行、四列
# 第一行：总能量，占整行
# 后三行：每行一个向量物理量，分别画 x、y、z、模长
grid = fig_test.add_gridspec(4, 4)

# -------------------- 总能量 --------------------

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

# -------------------- 向量物理量 --------------------

component_names = ["x", "y", "z"]
component_colors = ["tab:red", "tab:green", "tab:blue"]

for row, (key, title, symbol) in enumerate(vector_info, start=1):
    values = np.array([
        result[key]
        for result in test_result
    ])

    # x、y、z 分量分别画在三个独立子图中
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
        
        # 计算该分量的RMSE
        component_rmse = RMSE(values[:, component])
        component_title = f"{title}: {name}\nRMSE: {component_rmse:.6e}"
        axis.set_title(component_title, fontsize=9)
        axis.set_xlabel("Time")
        axis.set_ylabel(f"{symbol}_{name}")
        axis.grid(True, alpha=0.3)

    # axis=1：对每个时刻的三个分量计算模长
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

# 每个子图准备一个数值提示框
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

# ==================== 添加RMSE汇总信息 ====================

# 计算所有RMSE值
# all_rmse_info = f"RMSE Summary:\nEnergy: {energy_rmse_final:.6e}"

# for key, title, symbol in vector_info:
#     values = np.array([result[key] for result in test_result])
#     magnitudes = np.linalg.norm(values, axis=1)
#     magnitude_rmse = RMSE(magnitudes)
#     all_rmse_info += f"\n{title} (mag): {magnitude_rmse:.6e}"

# # 在图表右上角添加RMSE信息框
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

    # 时间按升序记录，用二分查找找到最近的数据点
    index = int(np.searchsorted(xs, event.xdata))
    index = min(index, len(xs) - 1)

    if index > 0:
        left_distance = abs(event.xdata - xs[index - 1])
        right_distance = abs(event.xdata - xs[index])

        if left_distance <= right_distance:
            index -= 1

    # 仍然指向同一条记录时，不重复刷新
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
