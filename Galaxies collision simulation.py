import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
print("simulation start!")

# global var
dt = 100
tot_time = 1000
G = 1e-12
N = 50

# setting initial information
num_per_galaxy = N // 2
r_min, r_max = 2, 25
galaxy_distance = 60
mass_range = (1, 10)
G = 1e-11

stars = []
positions = []

# galaxy center
mass_center = 10**6 * np.mean(mass_range)
pos1 = np.array([-galaxy_distance/2, 0, 0], dtype=float)
pos2 = np.array([galaxy_distance/2, 0, 0], dtype=float)

# AGN velocity & bias
v_center_mag = 0.0003
bias_strength = 1*v_center_mag  
vel1 = np.array([v_center_mag, -np.random.uniform(bias_strength, 1*bias_strength), np.random.uniform(-bias_strength, bias_strength)])
vel2 = np.array([-v_center_mag, np.random.uniform(bias_strength, 1*bias_strength), -np.random.uniform(-bias_strength, bias_strength)])

# vel1 = np.array([v_center_mag, -bias_strength, np.random.uniform(-bias_strength, bias_strength)])
# vel2 = np.array([-v_center_mag, bias_strength, np.random.uniform(-bias_strength, bias_strength)])

# stars
stars.append({"mass": mass_center, "pos": pos1, "v": vel1})
positions.append([pos1.copy()])
stars.append({"mass": mass_center, "pos": pos2, "v": vel2})
positions.append([pos2.copy()])

# disk
for g, pos_center in enumerate([pos1, pos2]):
    v_center = stars[g*1]["v"] 
    for i in range(num_per_galaxy - 1):
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(0, 2*np.pi)
        x = pos_center[0] + r*np.cos(theta)
        y = pos_center[1] + r*np.sin(theta)
        z = np.random.normal(0, 1)
        pos = np.array([x, y, z], dtype=float)

        mass = np.random.uniform(*mass_range)
        v_mag = np.sqrt(G * mass_center / r)
        vx = -v_mag * np.sin(theta)
        vy = v_mag * np.cos(theta)
        vz = np.random.uniform(-v_mag * 0.01,v_mag * 0.01)
        vel = np.array([vx, vy, vz], dtype=float)

        vel += 1.2*v_center

        stars.append({"mass": mass, "pos": pos, "v": vel})
        positions.append([pos.copy()])

# function
def dist(s1, s2):
    return np.linalg.norm(s2["pos"] - s1["pos"])

def dir_vector(s1, s2):
    return (s2["pos"] - s1["pos"]) / dist(s1, s2)

def acc(s1, s2):
    return G * s2["mass"] / dist(s1, s2)**2 * dir_vector(s1, s2)

# leapfrog
def leapfrog(stars):
    N = len(stars)
    accs = [np.zeros(3) for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            diff = stars[j]["pos"] - stars[i]["pos"]
            r = np.linalg.norm(diff)
            a = G * stars[j]["mass"] / r**3 * diff
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
            diff = stars[j]["pos"] - stars[i]["pos"]
            r = np.linalg.norm(diff)
            a = G * stars[j]["mass"] / r**3 * diff
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
for step in range(tot_time):
    leapfrog(stars)
    # print(stars[1]['pos'])
    # print(step)
    if step%10 == 0:
        print(step/tot_time,"/",1)
    if step/tot_time == 0.01:
        pridict_time = time.time()
        print(f"remaining {(pridict_time-start_time)*99} s")
    if step/tot_time == 0.05:
        pridict_time = time.time()
        print(f"remaining {(pridict_time-start_time)*19} s")
end_time = time.time()
print(f"simulation cost {end_time - start_time:.2f} s")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

b = galaxy_distance
ax.set_xlim(-b,b)
ax.set_ylim(-b,b)
ax.set_zlim(-b,b)

lines = [ax.plot([],[],[], color='gray', alpha=0.5)[0] for _ in range(N-2)]
points = [ax.plot([],[],[], 'o', color='blue', markersize=2)[0] for _ in range(N-2)]

agn_lines = [
    ax.plot([],[],[], color='red', linewidth=1.5)[0],
    ax.plot([],[],[], color='orange', linewidth=1.5)[0]
]

agn_points = [
    ax.plot([],[],[], 'o', color='red', markersize=6)[0],
    ax.plot([],[],[], 'o', color='orange', markersize=6)[0]
]

def update(frame):
    for i in range(2, N):
        if len(positions[i][:frame]) == 0:
            continue
        traj = np.array(positions[i][:frame])
        lines[i-2].set_data(traj[:,0], traj[:,1])
        lines[i-2].set_3d_properties(traj[:,2])

        points[i-2].set_data([traj[-1,0]], [traj[-1,1]])
        points[i-2].set_3d_properties([traj[-1,2]])

    for i in range(2):
        traj = np.array(positions[i][:frame])
        agn_lines[i].set_data(traj[:,0], traj[:,1])
        agn_lines[i].set_3d_properties(traj[:,2])

        agn_points[i].set_data([traj[-1,0]], [traj[-1,1]])
        agn_points[i].set_3d_properties([traj[-1,2]])

    return lines + points + agn_lines + agn_points

ani = FuncAnimation(fig, update, frames=range(1, tot_time), interval=20, blit=True)
plt.show()