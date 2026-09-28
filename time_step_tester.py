import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import copy
import secrets
import mplcursors   

# global var
dt = 1
tot_time = int(1000 / dt)
G = 1
N = 5
dt_test_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

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


# note: no per-run dt / tot_time menus here -- this tester sweeps dt_test_list,
# and each run uses tot_time = int(1000 / dt) so every run covers the same
# physical time
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
method_name = {"1": "Leapfrog (KDK)", "2": "Runge-Kutta 4th order"}.get(method)
if method_name is None:
    print("Invalid choice. Using Leapfrog by default.")
    method = "1"
    method_name = "Leapfrog (KDK)"
integrator = leapfrog if method == "1" else rk4

# the sweep always includes long runs, so always ask for the sampling rate
print("chose your simpling rate")
print("1. Every 10 steps.\n2. 1% of total steps.\n3. 0.1% of total steps.\n4. 0.01% of total steps.")
simpling_rate = int(input("Enter your choice (1, 2, 3 or 4): "))

energy_rmse_list = []
normalized_energy_rmse_list = []

# sweep the dt list: every run starts from the same initial conditions and
# covers the same physical time (1000), only the step size differs
for dt in dt_test_list:
    tot_time = int(1000 / dt)
    # reset state and records, so every time step starts from the same
    # initial conditions and the RMSE of each run is not mixed together
    stars = copy.deepcopy(initial_stars)
    test_result = [test(stars)]
    test_time = [0]

    if simpling_rate in (0, 1):
        sample_every = 10
    elif simpling_rate == 2:
        sample_every = max(1, tot_time // 100)
    elif simpling_rate == 3:
        sample_every = max(1, tot_time // 1000)
    else:
        sample_every = max(1, tot_time // 10000)

    start_time = time.time()
    for step in range(tot_time):
        integrator(stars)
        if (step + 1) % sample_every == 0:
            print(f"procesing: {step / tot_time}, dt = {dt}")
            test_result.append(test(stars))
            test_time.append((step + 1) * dt)
        if (step + 1) % max(1, tot_time // 10) == 0:
            print(f"dt = {dt}: {(step + 1) / tot_time:.0%}")

    end_time = time.time()
    print(f"dt = {dt}, steps = {tot_time}, "
          f"simulation cost {end_time - start_time:.2f} s")

    # error analysis
    energy_result = np.array([
        result["energy"]
        for result in test_result
    ])

    energy_rmse = float(RMSE(energy_result))
    normalized_energy_RMSE = energy_rmse / energy_eigenvalue
    print(f"energy RMSE: {energy_rmse:.6e}, "
          f"normalized: {normalized_energy_RMSE:.6e}\n")

    energy_rmse_list.append(energy_rmse)
    normalized_energy_rmse_list.append(abs(normalized_energy_RMSE))

print(f"dt_test_list: {dt_test_list}")
print(f"energy_rmse_list: {energy_rmse_list}")
print(f"normalized_energy_rmse_list: {normalized_energy_rmse_list}")

fig_energy = plt.figure(figsize=(10, 6))
plt.yscale("log")
plt.xscale("log")
plt.plot(dt_test_list, energy_rmse_list, color="blue", marker="o", label="Energy RMSE")
plt.xlabel("Time Step (dt)")
plt.ylabel("RMSE")
plt.title(f"Energy RMSE vs Time Step (N={N}, {method_name})")
plt.legend()
plt.grid()


fig_normalized = plt.figure(figsize=(10, 6))
plt.yscale("log")
plt.xscale("log")
plt.plot(dt_test_list, normalized_energy_rmse_list, color="red", marker="o", label="Normalized Energy RMSE")
plt.xlabel("Time Step (dt)")
plt.ylabel("Normalized RMSE")
plt.title(f"Normalized Energy RMSE vs Time Step (N={N}, {method_name})")


plt.legend()
plt.grid()

mplcursors.cursor(hover=True)         
plt.show()

# ask whether to save the two figures after the windows are closed:
# 1 = save (png), anything else = no
print("save the two RMSE figures?  1 = save (png)   anything else = no")
try:
    save_choice = input().strip()
except EOFError:
    save_choice = ""

if save_choice == "1":
    base_name = f"seed_{seed}_method_{method_name}_dt_sweep_time_1000"
    fig_energy.savefig(f"{base_name}_energy.png", dpi=300)
    fig_normalized.savefig(f"{base_name}_normalized.png", dpi=300)
    print(f"saved: {base_name}_energy.png / {base_name}_normalized.png")
