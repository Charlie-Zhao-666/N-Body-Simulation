# N 体模拟 Lab：拿哪些文件，怎样第一次跑起来

**适用版本：本地正式入口 `N body simulation Lab.py`（2026-11-08 核对）。** 如果你收到的是随附的 `NBody-Lab-source.zip`，先把它完整解压到一个可写的文件夹。所有 Python 文件都放在同一层，保留原文件名。这个包只带程序、依赖清单和说明，不带历史模拟数据。

## 一眼选方案

| 你想做什么 | 带哪些项目文件 | 运行时选择 |
|---|---|---|
| 第一次试跑，看预设轨道、跑 Leapfrog 或 RK4 | 下表前 **6 个文件**（入口 + 5 个辅助文件），以及基础 Python 软件包 | 在界面把计算设备改成 **CPU (dictionary / original)** |
| 按界面默认配置运行 Leapfrog / RK4 | 前 6 个，再加 `nbody_array_cpu.py`，安装 Numba | **CPU (array / multi-thread)**；默认配置与最小配置不同 |
| 发给别人，让对方能选择全部方法 | **整个 ZIP**：共 12 个 Python 文件。按所选功能安装软件包 | 不需要 GPU 也能使用 CPU 模式 |

## 1. 基础清单：这 6 个文件放在一起

| 文件 | 为什么要带 |
|---|---|
| `N body simulation Lab.py` | **唯一需要启动的文件。** 包含设置窗口、模拟主循环、原始 CPU 版 Leapfrog/RK4、结果显示与记录。 |
| `stable_orbits.py` | **轨道文件。** 提供内置轨道目录及初始位置、速度等参数；主程序还会扩展目录。不需要另找轨道 CSV 或从网站下载。 |
| `nbody_cpu_startup.py` | 管理启动时的后台准备工作。数组 CPU 等加速功能可选，但主程序开头就导入这个管理模块，所以文件本身必带。 |
| `nbody_kepler.py` | 生成可调偏心率的开普勒双星初态，提供解析参考轨道及误差计算。 |
| `nbody_two_body.py` | 计算双体参考位置和位置误差，也调用上一个文件。 |
| `nbody_position_analysis.py` | 提供位置分析界面与结果数组导出；它会调用上面两个参考轨道模块。 |

后面三个文件即使暂时不用位置分析，也不能从这个版本的基础包中删掉：主程序加载时就导入它们，缺文件会在打开设置窗口之前报错。

## 2. 按功能追加：另 6 个辅助文件

下表均以“已经带齐基础 6 个”为前提。**Python 文件和已安装的软件包是两件事**：例如，复制了 `nbody_array_cpu.py`，还要在运行 Lab 的同一个 Python 环境中安装 Numba。

| 要用的功能 | 另外要带的本地文件 | 额外软件或条件 |
|---|---|---|
| 数组 CPU 单线程或多线程，Leapfrog / RK4 | `nbody_array_cpu.py` | Numba，见 `requirements-array-cpu.txt` |
| DOP853，原始字典 CPU | `nbody_dop853.py` | SciPy，见 `requirements-dop853.txt` |
| DOP853，数组 CPU 单线程或多线程 | `nbody_dop853.py`、`nbody_dop853_cpu.py`、`nbody_array_cpu.py` | SciPy + Numba |
| Taylor45，单线程 float64 | `nbody_taylor45.py`、`nbody_taylor45_cpu.py`、`nbody_array_cpu.py` | Numba；这条路径不用 mpmath |
| Taylor100 / 150 / 200，单线程 float64 | `nbody_taylor45.py`、`nbody_taylor_cpu.py`、`nbody_array_cpu.py` | Numba。虽然入口模块名字含 45，但它负责全部四种阶数 |
| Taylor45 / 100 / 150 / 200，80 位十进制精度 CPU | `nbody_taylor45.py` | mpmath；高精度积分本身不依赖上述 float64 编译内核 |
| NVIDIA GPU，Leapfrog / RK4 | 无额外本地模块；GPU 实现在主程序中 | 兼容的 NVIDIA GPU、驱动及与 CUDA 环境匹配的 CuPy。GPU 上选 DOP853 还要带 `nbody_dop853.py` 并安装 SciPy；Taylor 不支持 GPU |

**为什么启动时可能提示预热失败？** Lab 会提前尝试准备多个加速内核。缺少可选模块时，相关后端可能不可用，但这不等于基础窗口无法打开。第一次验证请明确选择字典 CPU，不要保留默认的数组多线程设备。

## 3. 在 Windows 上第一次运行

1. 安装带 Tk/Tcl 的桌面版 Python。本次基础验证用的是 Python 3.14.3。Tkinter 是 Python 的图形窗口组件。打开解压后的文件夹，在资源管理器地址栏输入 `powershell` 并回车；出现的窗口用于在这个文件夹中运行下面的命令。
2. 输入以下命令。第一行创建仅供此项目使用的 Python 环境；第二行安装基础软件包；第三行使用刚才的同一环境启动 Lab。不要在 ZIP 预览窗口里双击程序。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-base.txt
.\.venv\Scripts\python.exe "N body simulation Lab.py"
```

3. 在设置窗口选择 **Known orbit**（已知轨道），保留第一条预设；积分器选 **Leapfrog** 或 **RK4**，设备选 **CPU (dictionary / original)**，再点 **Start / Rerun**。出现结果窗口和轨迹图，就跑通了基础路径。
4. 想用默认数组 CPU 或高级方法时，关闭 Lab，按需要运行下面对应的行，然后再启动。基础包清单包含 NumPy、Matplotlib、Pillow、python-dateutil；不含 GPU 软件。

```powershell
# 数组 CPU 或 Taylor 的 float64 路径
.\.venv\Scripts\python.exe -m pip install -r requirements-array-cpu.txt
# DOP853；数组版本还需上一行
.\.venv\Scripts\python.exe -m pip install -r requirements-dop853.txt
# 同时准备 Taylor 的两种 CPU 精度（这份清单包含数组 CPU 的要求和 mpmath）
.\.venv\Scripts\python.exe -m pip install -r requirements-taylor45.txt
```

若提示找不到 `stable_orbits` 或 `nbody_...`，先检查对应的同版本 `.py` 文件是否和入口同层；它们不是要用 pip 下载的软件。若提示找不到 `numpy`、`scipy` 等，则检查是否给**启动 Lab 的那个 Python**安装了相应软件包。若提示找不到 `_tkinter`，先运行 `.\.venv\Scripts\python.exe -m tkinter` 检查；正常会打开一个小测试窗口。

## 4. 不必发给首次运行者的文件

- `graph and data/`、历史 `runs.dat`、PNG/WebP 图、原始轨迹 `.npz`、旧记录 CSV：这些是输出、历史记录或旧数据迁移输入。**从预设轨道开始一次新模拟不依赖它们。** 保存新记录时要让程序有权写入入口旁的输出目录；接收者不需要复刻你的 D 盘、E 盘布局。
- **批量实验计划 JSON**：只有想用 **Test mode → Load plan** 重跑那份特定计划时才要附上。复查某次历史运行的数据与从头运行程序，是两种不同的交付需求。
- `N body simulation.py`、独立 CuPy 主程序、星系碰撞脚本、`galaxy_gpu_backend.py`、测试器、`seeds` 和 `__pycache__/` 均不在此 Lab 的启动依赖链中。

## 5. 如果接收者是从 GitHub 下载的

核对的[公开版本 d91161a](https://github.com/Charlie-Zhao-666/N-Body-Simulation/tree/d91161a278980bf95084d42f7040a74c126d2490)入口是 **`interactive_nbody_lab.py`**。这个版本的仓库根目录带有 `stable_orbits.py`、`nbody_kepler.py`、`nbody_cpu_startup.py`，以及数组 CPU 用的 `nbody_array_cpu.py`。下载该版本的完整源码，在仓库根目录用同一 Python 安装 `requirements.txt`；要用默认数组 CPU，再安装 `requirements-array-cpu.txt`，最后运行 `python interactive_nbody_lab.py`。不要把报告目录里的历史快照当成入口。

**公开版本和本地版不是只差一个文件名。** 公开版本的根目录没有本地版的 DOP853、Taylor 和单独的位置分析模块。想重现本地版，请直接用这份同版本 ZIP，不要从两套代码中各取一半。

### 本次核验范围

在只复制基础 6 个文件、没有历史数据的独立目录中，用 Python 3.14.3 创建了 Lab 的 Tk 设置窗口，并禁用 Numba、SciPy、mpmath、CuPy 的导入。随后使用预设轨道，在字典 CPU 上分别调用 Leapfrog 和 RK4；两者均完成 3 步（步长 0.001，终止时刻 0.003），能量记录为有限数。高级模式的文件清单依据主程序和辅助模块的实际导入、后端选择分支及随项目提供的依赖清单核对；这次没有逐一运行高级模式或验证新机器上的安装。

