# N 体模拟 Lab 运行包

此包核对于 2026-11-13，包含本地 2026-11-11 主程序及其配套模块。GitHub 根目录入口为 `interactive_nbody_lab.py`，ZIP 内入口为 `N body simulation Lab.py`，两者代码相同。

## 第一次运行

完整解压 ZIP 到可写文件夹。安装带 Tk 的桌面 Python。在解压后的文件夹打开 PowerShell，运行以下命令。独立环境用于将本项目的软件包放在一起，后两条命令使用这个相同环境安装依赖、启动程序。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-base.txt
.\.venv\Scripts\python.exe "N body simulation Lab.py"
```

在界面选择 CPU (dictionary / original) 和预设轨道，可运行 Leapfrog 或 RK4。默认数组多线程模式需要额外安装 `requirements-array-cpu.txt`。可选模块预热失败时，字典 CPU 仍可用。

## 单步能量误差区间

勾选 Enable adaptive time step，在控制器中选择 Step energy error band。设置单步能量变化的下限 L、上限 U，以及 Node drift limit。数值使用程序的绝对能量单位。Node drift limit 设为 0 时关闭累计限制。设为正值时，程序对接受节点相对初始能量的偏移施加限制，并在能量图显示红色虚线。

模块以 float64 总能量快照决定是否接受试算，拒步后从同一起点重试。最后一步可豁免下限。累计漂移优先时也有低于下限的接受例外。控制器不会给出轨迹位置误差上界。非常严格的阈值可能受浮点分辨率限制，运行失败时保留错误提示并调整设置。

## 文件清单与作用

| 文件 | 作用 |
|---|---|
| N body simulation Lab.py | 主入口、设置窗口、积分循环和结果图 |
| stable_orbits.py | 内置轨道及初始位置、速度、质量 |
| nbody_cpu_startup.py | 启动时准备可选 CPU 后端 |
| nbody_kepler.py、nbody_two_body.py | 双体初态、解析参考和位置误差 |
| nbody_position_analysis.py | 位置分析界面及数组导出 |
| nbody_step_energy_band.py | 单步能量变化区间与节点累计漂移控制 |
| nbody_array_cpu.py | Numba 数组 CPU 后端 |
| nbody_dop853.py、nbody_dop853_cpu.py | DOP853 及其数组 CPU 实现 |
| nbody_taylor45.py、nbody_taylor45_cpu.py、nbody_taylor_cpu.py | Taylor45/100/150/200 和 mpmath 后端 |
| nbody_taylor_mpfr.py、nbody_taylor_mpfr.dll | MPFR Python 接口及 Windows x64 库 |
| nbody_taylor_mpfr.cpp、build_taylor_mpfr.ps1 | MPFR 内核源码与 Windows 重建脚本 |
| TAYLOR_README.md | Taylor 精度选择、数值范围与构建说明 |

所有 Python 文件保持同层。只需启动主入口。基础窗口需要前六项相关模块，能量带模式额外使用 `nbody_step_energy_band.py`。历史 DAT、图片、NPZ 和你原有磁盘路径均不需要随包提供。批量复现特定实验时还需要相应计划 JSON。

## 可选依赖

在用于启动 Lab 的同一个环境执行对应命令。

```powershell
# 数组 CPU
.\.venv\Scripts\python.exe -m pip install -r requirements-array-cpu.txt
# DOP853，数组设备还需要上一个清单
.\.venv\Scripts\python.exe -m pip install -r requirements-dop853.txt
# Taylor float64 与 mpmath
.\.venv\Scripts\python.exe -m pip install -r requirements-taylor45.txt
```

MPFR 随包 DLL 对应 Windows x64，其他系统需要相应平台库，或选择 mpmath。GPU 需要兼容的 NVIDIA 驱动和 CuPy，见仓库安装说明。

网页版可在浏览器体验，无需自行安装 Python 或依赖，首次使用自动加载资源。

https://nbody-lab-charlie.zsh13718187275.chatgpt.site
