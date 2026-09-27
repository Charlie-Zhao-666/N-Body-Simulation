import numpy as np
from vispy import scene, app, visuals
import time

# ------------------ 参数 ------------------
dt = 3
tot_time = 50000
G = 1
N = 100  # 恒星数量
p = 3000  # 盘半径
disk_mass = 1
N_big = int(N*0.05)  # 大质量恒星数量
big_mass_range = (100, 1000)
tail_len = 30  # 拖尾长度
# ----------------------------------------

# ------------------ 初始化星系 ------------------
stars = []
positions = []

# 中心 AGN
stars.append({"mass": 1e6, "pos": np.array([0,0,0], dtype=float), "v": np.array([0,0,0], dtype=float)})
positions.append([stars[0]["pos"].copy()])

# 盘内恒星
for i in range(1, N):
    r = np.random.uniform(200, p)
    theta = np.random.uniform(0, 2*np.pi)
    x, y = r*np.cos(theta), r*np.sin(theta)
    z = np.random.normal(0, r*0.05)
    pos = np.array([x, y, z], dtype=float)
    
    v_mag = np.sqrt(G*stars[0]["mass"]/r)
    vx = -v_mag*np.sin(theta) + np.random.uniform(-0.05*v_mag,0.05*v_mag)
    vy = v_mag*np.cos(theta) + np.random.uniform(-0.05*v_mag,0.05*v_mag)
    vz = np.random.uniform(-0.01*v_mag,0.01*v_mag)
    vel = np.array([vx, vy, vz], dtype=float)
    
    mass = np.random.uniform(*big_mass_range) if i <= N_big else disk_mass
    stars.append({"mass": mass, "pos": pos, "v": vel})
    positions.append([pos.copy()])

# 外侧干扰质点
outer_r = p*3
theta = np.random.uniform(0, 2*np.pi)
x, y, z = outer_r*np.cos(theta), outer_r*np.sin(theta), 0
pos_outer = np.array([x,y,z], dtype=float)
r_vec = pos_outer - stars[0]["pos"]
r = np.linalg.norm(r_vec)
v_mag = np.sqrt(G*stars[0]["mass"]/r)
vx = -v_mag*r_vec[1]/r + np.random.uniform(-0.01*v_mag,0.01*v_mag)
vy = v_mag*r_vec[0]/r + np.random.uniform(-0.01*v_mag,0.01*v_mag)
vz = 0
vel_outer = np.array([vx, vy, vz], dtype=float)

stars.append({"mass":1e5, "pos": pos_outer, "v": vel_outer})
positions.append([pos_outer.copy()])

# ------------------ Leapfrog 积分 ------------------
start_time = time.time()
def leapfrog(stars):
    N_local = len(stars)
    accs = [np.zeros(3) for _ in range(N_local)]
    # 计算加速度
    for i in range(N_local):
        for j in range(i+1, N_local):
            diff = stars[j]["pos"] - stars[i]["pos"]
            r = np.linalg.norm(diff)
            if r == 0:
                continue
            a_i = G*stars[j]["mass"]/r**3 * diff
            accs[i] += a_i
            accs[j] -= G*stars[i]["mass"]/r**3 * diff
    # 半步速度
    for i in range(N_local):
        stars[i]["v"] += 0.5*dt*accs[i]
    # 全步位置
    for i in range(N_local):
        stars[i]["pos"] += stars[i]["v"]*dt
    # 另一半步速度
    accs = [np.zeros(3) for _ in range(N_local)]
    for i in range(N_local):
        for j in range(i+1, N_local):
            diff = stars[j]["pos"] - stars[i]["pos"]
            r = np.linalg.norm(diff)
            if r == 0:
                continue
            a_i = G*stars[j]["mass"]/r**3 * diff
            accs[i] += a_i
            accs[j] -= G*stars[i]["mass"]/r**3 * diff
    for i in range(N_local):
        stars[i]["v"] += 0.5*dt*accs[i]
    # 保存位置
    for i in range(N_local):
        positions[i].append(stars[i]["pos"].copy())

# ------------------ 先计算所有轨迹 ------------------
for step in range(tot_time):
    leapfrog(stars)
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

# ------------------ Vispy 渲染 ------------------
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white', size=(1200,900))
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=45, distance=p*3, center=stars[0]["pos"])

# 点大小与颜色
sizes = np.array([30] + [6]*(N-1) + [12])  # 中心最大，盘内稍大
colors_list = [[1,1,0,1]]  # 中心黄
for i in range(1,N):
    colors_list.append([0,1,0,1] if i<=N_big else [0,0,1,1])  # 绿大质量/蓝普通
colors_list.append([1,0,0,1])  # 外侧红
colors = np.array(colors_list)

scatter = scene.visuals.Markers(parent=view.scene)
scatter.set_data(np.array([pos[-1] for pos in positions]), face_color=colors, size=sizes)

# 拖尾
tail_scatter = scene.visuals.Markers(parent=view.scene)

# 时间显示
time_text = scene.visuals.Text('', color='black', font_size=24, parent=canvas.scene)
time_text.pos = (200, 30)

frame = 0
def update(ev):
    global frame
    if frame >= tot_time:
        return

    # 锁定中心恒星
    view.camera.center = positions[0][frame]

    # 更新拖尾
    # tail_pos, tail_col, tail_sz = [], [], []
    # for i, pos_list in enumerate(positions):
    #     traj = pos_list[max(0, frame-tail_len):frame+1]
    #     for k, pnt in enumerate(traj):
    #         tail_pos.append(pnt)
    #         c = colors[i].copy()
    #         c[3] = (k+1)/len(traj)
    #         tail_col.append(c)
    #         tail_sz.append(sizes[i]*0.006)
    # tail_scatter.set_data(np.array(tail_pos), face_color=np.array(tail_col), size=np.array(tail_sz))

    # 更新恒星位置
    scatter.set_data(np.array([pos_list[frame] for pos_list in positions]), face_color=colors, size=sizes)

    # 更新时间
    time_text.text = f"Time: {frame:.1f}"
    frame += 1

timer = app.Timer(interval=0.01, connect=update, start=True)  # ~60 FPS
app.run()