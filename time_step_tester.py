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

energy_rmse_list = []
normalized_energy_rmse_list = []

for dt in dt_test_list:
    tot_time = int(1000 / dt)
    # reset state and records, so every time step starts from the same
    # initial conditions and the RMSE of each run is not mixed together
    stars = copy.deepcopy(initial_stars)
    positions = [[star["pos"].copy()] for star in stars]
    test_result = []
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
    energy_rmse_list.append(energy_rmse)
    normalized_energy_rmse_list.append(abs(normalized_energy_RMSE))

print(f"dt_test_list: {dt_test_list}")
print(f"energy_rmse_list: {energy_rmse_list}")
print(f"normalized_energy_rmse_list: {normalized_energy_rmse_list}")

plt.figure(figsize=(10, 6))
plt.yscale("log")
plt.xscale("log")
plt.plot(dt_test_list, energy_rmse_list, color="blue", marker="o", label="Energy RMSE")
plt.xlabel("Time Step (dt)")
plt.ylabel("RMSE")
plt.title(f"Energy RMSE vs Time Step (N={N})")
plt.legend()
plt.grid()


plt.figure(figsize=(10, 6))
plt.yscale("log")
plt.xscale("log")
plt.plot(dt_test_list, normalized_energy_rmse_list, color="red", marker="o", label="Normalized Energy RMSE")
plt.xlabel("Time Step (dt)")
plt.ylabel("Normalized RMSE")
plt.title(f"Normalized Energy RMSE vs Time Step (N={N})")


plt.legend()
plt.grid()

mplcursors.cursor(hover=True)         
plt.show()
