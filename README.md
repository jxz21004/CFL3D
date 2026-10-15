# CFL3D Python / MPI

本仓库在 NASA CFL3D V6 结构网格 RANS 求解器上提供 Python 接口：Python 读取 CGNS
网格、设置工况和数值参数，通过 NumPy f2py 调用原生 Fortran 求解器，返回流场、
收敛历史及按部件积分的气动力。支持串行和原生主控/计算进程 MPI 路径。
Python 求解不依赖 `.inp` 模板，也不把 CGNS 网格转换为 PLOT3D。

当前 Python 包版本为 `0.1.0`，原生接口版本为 **4**。
完整参数说明见 [Python 接口文档](docs/python-wrapper.md)，
各轮验证结果见 [MPI_PROGRESS.md](MPI_PROGRESS.md)，
原始 CFL3D 文档见 [NASA CFL3D](https://nasa.github.io/CFL3D/)。

## 1. 环境要求

以下流程假定已有可用的 Python 环境和相应依赖，不要求复刻开发容器。

| 组件 | 要求与用途 |
| --- | --- |
| 操作系统 | 当前构建脚本面向 Linux，生成 `.so`；Windows 可通过 Linux 容器或 WSL 使用 |
| Python | `>=3.10`，须有与解释器匹配的开发头文件；当前实际验证为 3.12.12 |
| NumPy | `>=2.0`，提供数组、f2py 和编译所需头文件；当前验证为 2.3.5 |
| Python 安装工具 | `pip`、`setuptools>=64`、`wheel`，用于下文的 editable 安装 |
| C / Fortran 工具链 | C 编译器 `cc`（支持 C11）、GNU Fortran、`ar`；当前 GNU 验证为 gfortran 13 |
| CGNS | 已安装的共享库 `libcgns.so` 和头文件 `cgnslib.h`；当前验证为 CGNS 4.5.0，支持 32/64 位 `cgsize_t` |
| MPI，按需 | MPI 运行时及与 Fortran 编译器匹配的编译包装器；当前验证为 Intel MPI 2021.13 |
| mpi4py，按需 | `>=4.0`，MPI 求解时需要；当前验证为 4.1.1，须与运行时 MPI 一致 |
| HDF5 读取，按需 | CGNS 库自带 HDF5 支持，或安装 `h5py>=3.8`；ADF 路径不需要 h5py |

默认构建直接使用 f2py 生成接口，**不需要 Meson/Ninja**。仅选择 `--backend meson`
时需要额外安装 `meson>=1.3` 和 `ninja`。
构建脚本也支持 Intel Fortran Classic `ifort/mpiifort`；当前版本的分组积分回归使用 GNU
扩展与 Intel MPI，未单独复测 Intel Fortran 编译的分组积分。

CGNS、MPI、编译器的共享库在编译和运行时都须可被动态链接器找到。
按自己的工具链加载环境；`mpirun`、mpi4py 和 Fortran MPI 扩展应使用同一个 MPI 实现。
MPI 内存用量随网格和进程数增加，当前每个 rank 都持有完整网格/结果副本。
容器内使用共享内存通信时应提供足够的 `/dev/shm`，小型回归可从 `--shm-size=1g` 起步；
生产网格所需内存需另行评估。

## 2. 最简安装与编译

在已准备上述依赖的 Python 环境中，从仓库根目录执行：

```bash
cd /path/to/CFL3D

# 注册 Python 包；复用已有依赖，不下载 NumPy 等运行依赖。
python -m pip install --no-deps --no-build-isolation -e .

# 串行求解器（默认优化构建）。
python tools/build_python.py --fc gfortran --jobs 8

# CGNS 读取桥：替换为安装前缀，目录内须有 include/ 和 lib/ 或 lib64/。
python tools/build_cgns.py --prefix /path/to/cgns
```

需要 MPI 时再编译一个独立扩展：

```bash
# mpif90 必须调用 GNU Fortran，并链接所选 MPI 实现。
python tools/build_python.py --mpi --fc mpif90 --jobs 8
```

若已设置 `CGNS_HOME`，可直接执行 `python tools/build_cgns.py`。
Intel Fortran Classic 用户将求解器构建命令替换为：

```bash
python tools/build_python.py --compiler-family intel --fc ifort --jobs 8
python tools/build_python.py --mpi --compiler-family intel --fc mpiifort --jobs 8
```

构建产物写入 `python/cfl3d/`：串行 `_native`、MPI `_native_mpi` 和 `_cgns_bridge.so`。
`pip install` 只注册 Python 包，**不会自动编译这些库**。
切换 Python ABI、NumPy 主版本、编译器、MPI 实现或更新原生接口后，应重新编译相关扩展并
重启 Python 进程。调试时在求解器构建命令追加 `--debug`；并发编译数可用 `--jobs` 调整。

编译后检查串行和 CGNS 加载：

```bash
python -c "from cfl3d import CFL3D, SolverOptions, _native; print('native API:', _native.py_api_version())"
python -c "from cfl3d._cgns_native import library; print('CGNS has HDF5:', library().has_hdf5)"
```

第一条应输出接口版本 `4`。第二条输出 `False` 时仍可正常读取 CGNS/ADF；
若要读取 HDF5，需使用支持 HDF5 的 CGNS 库或可选 h5py。

## 3. 输入网格与量纲

当前接口要求单个三维 CGNS base 中的结构化 zones；二维算例使用 i 方向有两个平面的
挤出网格。边界必须是 Vertex `PointRange` 矩形面，全部外表面须由边界或一对一连接覆盖，
分段不能重叠。支持 `GridConnectivity1to1_t` 块连接。
多 base 文件先使用 `CGNSMesh.from_file("mesh.cgns", base="BaseName")` 选择 base。

可直接识别 `BCFarfield`、`BCSymmetryPlane`、`BCWallInviscid`、
`BCOutflowSupersonic`、绝热 `BCWallViscous` 和零热流 `BCWallViscousHeatFlux`。
`FamilySpecified` 可通过当前 base 内唯一的 `FamilyBC_t` 解析。
其他边界或附加数据需按 [详细文档](docs/python-wrapper.md#cgns-网格和边界)
设置 `boundary_overrides`；不支持的数据不会被静默忽略。

| 输入 | 单位 / 定义 |
| --- | --- |
| 网格坐标、`cref`、`bref`、`moment_center` | m，接口不自动换算 CGNS 坐标单位 |
| `sref` | m²，所有部件统一使用这一参考面积 |
| `alpha`、`beta`、`alpha_step_limit` | 度 |
| `temperature` | K |
| `freestream_pressure` | 自由来流静压，Pa；用于有量纲载荷换算 |
| `reynolds` | 每米 Reynolds 数；若已知基于弦长的 `Re_c`，填 `Re_c / cref` |

## 4. 标准串行样例

下面数值只是工况模板，运行自己的网格前须修改工况、参考量和坐标轴约定。
可保存为 `case.py`，执行 `python case.py`：

```python
from cfl3d import CFL3D

options = {
    "input_grid": "mesh.cgns",
    "output_directory": "case-results",
    "writevolumesolution": True,           # CGNS 体网格 + 单元中心流场
    "writeTecplotSurfaceSolution": True,   # Tecplot 壁面结果，含 y+
    "writeflowsolution": False,           # CFL3D PLOT3D .x/.q
    "mach": 0.2,
    "alpha": 2.0,
    "beta": 0.0,
    "reynolds": 6e6,                 # 每米 Reynolds 数
    "temperature": 288.15,           # K
    "freestream_pressure": 101325.0, # Pa，改为实际工况静压
    "gamma": 1.4,
    "turbulence": "sst",
    "full_navier_stokes": True,
    "cfl": 1.0,
    "iterations": 1000,
    "limiter": 2,
    "flux": 1,
    "inversion": 1,
    "kappa": 1.0 / 3.0,
    "eddy_viscosity_limit": 1e5,
    "multigrid_levels": 1,           # 多重网格要求见下文
    "multigrid_cycle": 2,            # 1: V cycle；2: W cycle
    "y_up": False,                   # z 向上；y 向上时设 True
    "sref": 1.0,                    # m²，改为实际参考面积
    "cref": 1.0,                    # m
    "bref": 1.0,                    # m
    "moment_center": (0.0, 0.0, 0.0),
}

solver = CFL3D(options=options)
result = solver.solve()
print("Final alpha:", result.final_alpha)
print("Groups:", solver.getGroupNames())
print(solver.getSolution())       # 全部原生积分表面的系数和有量纲载荷
# print(solver.getSolution("Wing"))  # 将 Wing 替换为实际 CGNS 分组名
```

可直接运行的完整模板见 [examples/python/standard_case.py](examples/python/standard_case.py)。
它将网格、工况、数值参数和输出开关集中在 options 字典中，支持串行/MPI、迭代次数、多重网格、定升和部件选择：

```bash
python examples/python/standard_case.py mesh.cgns --output case-results
python examples/python/standard_case.py mesh.cgns --iterations 1000 --multigrid-levels 3 --group Wing
python examples/python/standard_case.py mesh.cgns --target-cl 0.5 --output lift-results
```

## 5. MPI 样例

将上述样例的构造、求解和输出部分替换为下面代码，保留 `CFL3D` 导入和 options 字典定义：

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
solver = CFL3D(
    options=options if comm.rank == 0 else None,
    comm=comm,
)
result = solver.solve()  # 使用 options["output_directory"]
if comm.rank == 0:
    print(solver.getSolution())
```

保存后用 `mpirun -np 3 python case.py` 运行。也可以直接运行完整模板：

```bash
mpirun -np 3 python examples/python/standard_case.py mesh.cgns --mpi --output mpi-results
```

rank 0 只做主控，其余 rank 按完整 zone 分配，**不会自动切分一个 zone**。
进程数必须满足 `2 <= 进程数 <= zone 数 + 1`：单 zone 用 2 进程，双 zone 可用 2 或 3 进程。
配置和文件读取以 communicator 的 rank 0 为准。求解后可仅在 rank 0 调用 `getSolution()`；
结果已汇总缓存，不要求其他 rank 参与查询。只有 rank 0 写 `solution.npz`，
各 worker 的原生临时输出位于 `rank-0001/` 等子目录。
单机 Intel MPI 可设置 `I_MPI_FABRICS=shm`；多节点通信需按集群实际网络配置 MPI。

## 6. 分组气动力与输出

默认可使用 CGNS Family（含 AdditionalFamily）、zone、BC 名称或完整 `zone/BC` 路径。
`getGroupNames()` 返回可用名称。同名 BC 跨 zone 合并；Family、zone、BC 同名时按此顺序优先，
完整路径可精确选定边界。新集合可以在构造时通过 `groups=` 指定，也可以求解后增加：

```python
# Wing 和 Tail 须是实际存在的 CGNS 名称，Aircraft 须为未被占用的新名字。
solver.addGroups({"Aircraft": ["Wing", "Tail"]})
loads = solver.getSolution("Aircraft")
print(loads["CL"], loads["CD"], loads["CDp"], loads["CDv"])
print(loads["D"], loads["Dp"], loads["Dv"], loads["My"])
```

集合名称不能与已有 CGNS 分组或集合重名；成员只接受 CGNS 名称，不能递归引用自定义集合。
重叠成员按表面并集计算，不会重复积分。所有组使用同一套 `sref/cref/bref/moment_center`。

| `getSolution()` 字段 | 含义 / 单位 |
| --- | --- |
| `CL, CD, CY` | 升力、阻力、输入坐标 Y 方向力系数 |
| `Cmx, Cmy, Cmz` | 输入坐标轴上的力矩系数 |
| `CDp, CDv` | 压差、黏性阻力系数，`CD = CDp + CDv` |
| `L, D, Y, Dp, Dv` | 对应力，N；`D = Dp + Dv` |
| `Mx, My, Mz` | 对应力矩，N·m，相对于 `moment_center` |
| `units` | 有量纲字段的单位字典 |

CL/CD/CDp/CDv 使用 CFL3D 原生风轴定义；CY 和力矩使用输入网格坐标轴。
`y_up=True` 时 CY 沿竖直 Y 轴，不能将它解释为横向侧力。
二维计算保持原生的部分力矩置零约定。
换算采用 `q∞ = 0.5 * gamma * freestream_pressure * mach**2`，力为 `C * q∞ * sref`；
`y_up=False` 的三个力矩参考长度为 `bref, cref, bref`，`y_up=True` 为 `bref, bref, cref`。
未设置静压时仍可求解和读取 `result.coefficients`，但 `getSolution()` 会报错。

`result.fields[zone_name]` 是 `(ni-1, nj-1, nk-1, 5)` 单元中心数组，最后一维为
`rho,u,v,w,p`，使用 CFL3D 原生无量纲化（来流 `rho=1`、声速 `a=1`、`p=1/gamma`）。
`result.history` 每行为 `residual,CL,CD,CX,CY,CMX,CMY,CMZ`；
`result.coefficients` 使用原有大写力矩键 `CMX/CMY/CMZ`，并包含总 `CDp/CDv`。
`solution.npz` 保存流场、历史、最终攻角及逐边界系数和名称；原生日志和 restart 文件也保留。

## 7. I/O 开关与求解后补写

推荐使用 `CFL3D(options={...})`，也支持 `CFL3D(options_dict)`。
原有 `CFL3D(mesh, SolverOptions(...))` 保留；输入网格只能在 mesh 参数和 `input_grid` 中选一处设置。
`solve()` 默认使用 options 的输出目录；旧的 `solve(output_directory=...)` 仍可临时覆盖目录。

| options 键 | 默认值 | 含义 |
| --- | --- | --- |
| `input_grid` | None | 输入 CGNS 路径；使用 options 入口时必填 |
| `output_directory` | `cfl3d-output` | 本次计算及自动输出目录 |
| `writevolumesolution` | True | 写 CGNS/ADF 体结果，并在 `solution.npz` 中保存流场 |
| `writeTecplotSurfaceSolution` | True | 写 Tecplot ASCII 壁面网格和解 |
| `writeflowsolution` | False | 写与原生 CFL3D IPTYPE=1 兼容的 PLOT3D `.x/.q` |
| `volume_filename` | `volume.cgns` | 自动 CGNS 文件名 |
| `surface_filename` | `surface.dat` | 自动 Tecplot 文件名 |
| `flow_prefix` | `flow` | 自动 PLOT3D 文件名前缀，对应 `flow.x`、`flow.q` |

开关须为 Python `True/False`，不是字符串。键名区分大小写，未知键会报错。
关闭输出只影响文件写出，流场、壁面数据和载荷仍缓存在对象中：

```python
options.update({
    "writevolumesolution": False,
    "writeTecplotSurfaceSolution": False,
    "writeflowsolution": False,
})
solver = CFL3D(options=options)
result = solver.solve()

# 计算结束后按需输出，无需重新求解，也不需要重新打开输入网格。
solver.writeVolumeSolution()                           # 默认输出目录 / volume.cgns
solver.writeTecplotSurfaceSolution()                   # 默认输出目录 / surface.dat
solver.writeFlowSolution()                             # 默认输出目录 / flow.x、flow.q
solver.writeTecplotSurfaceSolution("wing.dat", groupName="Wing")
```

显式传入的文件路径相对于调用时工作目录；省略路径则使用本次计算输出目录及 options 文件名。
MPI 下自动输出只由 rank 0 执行；补写方法也是本地调用，在其他 rank 上调用会返回 None。
补写基于本对象最近一次成功计算的快照，不受其他 solver 后续计算或原始网格变动影响。
三个开关全关时只保留历史、积分元数据和原生日志，不写流场数组、体/表面解或 restart 数据。
补写要求保留当前 Python 对象；没有实现从仅含历史的文件重建整个解。

表面默认选择求解器实际使用的 wall 边界（含 CGNS FamilySpecified 解析及壁面覆盖），不输出远场、
对称面和块连接。每个壁面 BC 对应一个 Tecplot `FEQUADRILATERAL` zone，变量为
`X,Y,Z,Cp,T,Cfx,Cfy,Cfz,y+`：坐标位于原始壁面顶点，其他六个量位于面单元中心，`T` 单位为 K。
Cp、Cf 直接提取自原生壁面压力及剪切应力，Cf 的三个分量对应输入网格坐标轴。
`y+ = rho_wall * u_tau * d / mu_wall`，d 为第一层单元中心的法向距离，使用原生壁面度量
`V/(2A)`；分子黏度采用内核的 Sutherland 关系。该定义不采用旧文本日志中的节点平均 y+ 统计。
无黏壁面的 Cf 和 y+ 为零；表面导出不需要设置自由来流静压。

CGNS 体文件包括原始顶点坐标、zone/边界/连接和单元中心的 `Density,VelocityX,VelocityY,VelocityZ,Pressure,Temperature`。
前五个量保持 CFL3D 原生无量纲定义，Temperature 为 K，文件内 `CFL3DUnits` 描述记录量纲。
这是后处理文件，不是 Python restart 接续文件。
PLOT3D `.x/.q` 使用原生端序、32 位 Fortran 记录标记、64 位实数和 32 位 IBLANK（启用 blanking）；
`.x` 为单元中心坐标，`.q` 为守恒量。三维是五分量，二维是 x-z（y-up 为 x-y）平面的四分量。
当前不支持网格点 IPTYPE=0 输出；缓存补写不重新进入求解内核。

## 8. 定升与数值参数

options 字典在构造时复制并校验，之后修改原字典不会改变已有 solver。
方向设置接受标量或三元 tuple/list；显式多重网格阶段也可使用字典列表。
`SolverOptions` 对象同样可用且不可变；以下展示与现有类型接口的互通：

```python
from dataclasses import replace
from cfl3d import MultigridStage, SolverOptions

base_options = SolverOptions.from_dict(options)

# 定升：alpha 是初始攻角，由原生反馈控制更新。
lift_options = replace(
    base_options, target_cl=0.5, alpha_relaxation=0.5,
    alpha_step_limit=0.1, alpha_update_interval=5,
)

# 分方向设置按网格索引 (i,j,k)，所有 zone 使用同一套参数。
mg_options = replace(
    base_options, limiter=(0, 2, 4), flux=(1, 0, 1),
    inversion=(1, 0, 1), kappa=(0.0, 1/3, 1/3),
    multigrid_levels=3, multigrid_cycle=2, multigrid_up_iterations=1,
    multigrid_stages=(
        MultigridStage(iterations=100, levels=2, first_order_iterations=10,
                       smoothing_steps=(1, 2)),
        MultigridStage(iterations=1000, levels=2, first_order_iterations=20,
                       smoothing_steps=(2, 1)),
    ),
)
```

`multigrid_levels` 为总网格层数（1..5，含最细层）；各方向单元数及边界端点须与
`2**(multigrid_levels-1)` 对齐，二维两平面 i 方向除外。
简化网格序列使用 `mesh_sequence=True` 和 `coarse_iterations`；显式 `multigrid_stages`
优先于简化阶段参数。`smoothing_steps` / `multigrid_smoothing` 按粗到细对应 MIT，
`multigrid_up_iterations` 对应 MTT。完整规则及 CFL 渐增、湍流子迭代等选项见
[详细文档](docs/python-wrapper.md)。

支持 `euler/laminar/sa/komega/sst`，`eddy_viscosity_limit` 默认值为 **1e5**。
定升仍执行指定迭代次数，不保证达到目标 CL 或自动收敛；结果是最后一次原生力积分。
原生反馈在力积分后更新攻角，因此 `result.final_alpha` 可能包含最后一次积分后的攻角更新。
输出文件存在不代表已收敛，须检查残差、载荷历史及流场。

## 9. 验证与当前范围

已有串行/MPI 扩展和 CGNS 桥时，可执行：

```bash
python -m unittest discover -s tests/python -v
CFL3D_VALIDATION_DIRECTORY=build/mpi-forces-validation bash tools/validate_mpi.sh --adf --controls --forces
```

MPI 回归脚本需要 `bash`、`timeout`、`rg`；若检测到 Valgrind，会额外检查非法访问和未初始化值，
不进行泄漏扫描。测试自己生成真实 ADF 文件，不需要用户提供网格。
当前版本已通过 27 项单元测试，3 项可选 HDF5 测试因环境无 h5py 跳过；
ADF 串行、2/3 进程、两个独立子通信器的比较项与串行基准一致。
分组与 I/O 测试覆盖 CDp/CDv、集合去重、有量纲换算、三维力矩中心、壁面 y+、恒温壁面、三个索引方向的壁面、y-up 旋转和延迟输出。
新增标准模板也已使用双块 CGNS/ADF 网格完成串行与三进程运行，覆盖定升、多重网格和
分组载荷输出；保存的流场、历史、最终攻角和逐边界积分逐项一致。
这是接口一致性与短程计算验证，尚无生产规模或长期收敛保证。

Python 接口目前支持稳态、实数、结构网格、一对一多块连接和多重网格。
尚未接入非定常、overset、非匹配 patch、周期连接、移动网格、伴随或 Python restart 接续。
`restart_interval` 只控制写出频率。原始 CFL3D 的完整能力不等同于 Python 接口已支持的范围。
内核使用进程全局状态，不支持同一 Python 进程内并发 `solve()`；MPI 并行按上述独立进程运行。
原生错误后应启动新 Python 进程；MPI 原生错误会终止当前作业。

## 许可证

Copyright 2001 United States Government as represented by the Administrator
of the National Aeronautics and Space Administration. All Rights Reserved.

The CFL3D platform is licensed under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0/.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
