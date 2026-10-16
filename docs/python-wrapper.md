# CFL3D 原生 Python wrapper

本工作区已加入实验性 MPI 接口和原生 CGNS/ADF 读取器；当前检查点与复测限制见
[MPI_PROGRESS.md](../MPI_PROGRESS.md)。下文原有串行一致性数据来自旧版交付，
本轮新增验证请以检查点记录为准。

Python 通过原生 CGNS 库（ADF）或可选 h5py（HDF5）读取网格、边界和连接信息，
把坐标与求解配置的 NumPy 数组直接传入
f2py 扩展 `_native.so`，再调用实际 CFL3D 求解器并取回流场和收敛历史。
计算不需要 `.inp` 模板、不生成 `.inp`，也不把网格写成 PLOT3D 后再读取。
CFL3D 原生的旧 CGNS 接口不参与这条路径。

## 安装和编译

当前验证环境为 Linux、GNU Fortran / Intel Fortran Classic、Intel MPI、Python 3.12。
默认直接编译 NumPy f2py 生成的 C 接口，不需要 Meson/Ninja；
`--backend meson` 保留原来的构建方式。
需要 C 编译器、`gfortran`、`ar`，以及 Python 开发头文件。

```bash
python -m pip install -e '.[build]'
python tools/build_python.py --jobs 8
python tools/build_python.py --mpi --jobs 8
# 使用已有 CGNS 安装；也可显式传 --prefix /path/to/cgns。
python tools/build_cgns.py
```

`FC` 环境变量或 `--fc /path/to/gfortran` 可以指定编译器。
MPI 默认使用 `mpif90`。`--compiler-family intel` 提供 Intel 编译参数并默认选择
`ifort`/`mpiifort`，已通过本轮构建和 MPI 回归。
两类编译器均以统一的 8 字节 REAL 编译求解器。
构建输出安装到 `cfl3d/_native.<ABI>.so`，无需修改原有 makefiles。
`--debug` 开启调试符号和新接口的数组边界检查；旧 Fortran 内核有合法的跨维展开访问，
不能整体启用 `-fcheck=bounds`。

`pip install` 安装 Python 包和依赖；共享库需要执行上述单独的构建命令。
已有依赖时可以直接设置 `PYTHONPATH=.:tests/python` 在源码目录测试。
内存网格和 CGNS/ADF 读取均不需要 h5py。`h5py` 已改为可选的 `hdf5` 依赖组。
原生读取器需 C 编译器和已有 CGNS 头文件/共享库；`CGNS_HOME` 指向安装目录。
当前容器可先 `source /home/mdolabuser/.bashrc_mdolab`，再执行 `python tools/build_cgns.py`。
该命令生成 `_cgns_bridge.so`，按实际头文件确定 cgsize_t 宽度，避免混用 32/64 位索引。
CGNS 库若依赖 Intel 运行库，运行 Python 时也需要加载相应环境。
复制整个环境到其他机器时，请为目标机器的 Python ABI 重新构建，不要直接复制 `.so`。

## 仅凭 CGNS 文件运行

```python
from cfl3d import CFL3D, SolverOptions

solver = CFL3D(
    "wing.cgns",
    SolverOptions(
        mach=0.8,
        alpha=2.0,
        beta=0.0,
        reynolds=6e6,
        temperature=288.15,
        turbulence="sst",
        cfl=5.0,
        iterations=1000,
        multigrid_levels=3,
        sref=93.0,
        cref=3.88,
        bref=30.0,
        moment_center=(0.0, 0.0, 0.0),
    ),
)
result = solver.solve(output_directory="wing-results")
print(result.coefficients)
print(result.fields.keys())
```

参考面积、长度和力矩中心默认分别为 1、1 和原点，实际气动力系数需要设置正确的参考量。
CGNS 坐标的单位由网格提供，接口不自动换算坐标。
`reynolds` 是**每个网格长度单位的 Reynolds 数**，接口转换为 CFL3D 的 REUE（百万）。
`temperature` 采用 Kelvin，传入核心时转换为 Rankine；alpha/beta 采用度。
`y_up=False` 时 alpha 在 x-z 平面；`y_up=True` 时采用原生 CFL3D 的 y-up 网格旋转。
返回的速度分量已经转换回输入网格的物理 Cartesian 坐标。
原生文本日志中的 Cartesian 力/力矩分量仍按核心内部坐标系输出；CL/CD 保持原生定义。

也可运行脚本：

```bash
python examples/python/run_cgns.py wing.cgns --model sst \
  --mach 0.8 --alpha 2 --cfl 5 --iterations 1000 --multigrid-levels 3
```

## 多重网格、网格序列与其他选项

`multigrid_levels=3` 表示生成两个粗网格层，加上细网格共三层。
默认从最细网格初始化并执行多重网格；`multigrid_cycle=1/2` 分别对应 V/W cycle。

```python
options = SolverOptions(
    turbulence="sa",
    multigrid_levels=3,
    mesh_sequence=True,
    coarse_iterations=100,
    iterations=1000,
    turbulence_subiterations=2,
    turbulence_cfl=5.0,
    full_navier_stokes=True,
    limiter=4,
    flux=1,
)
```

开启 `mesh_sequence` 时先从最粗网格开始，每个非最终阶段计算 coarse_iterations 次，
最终阶段执行 iterations 次。网格各方向单元数及边界分段端点必须与所有粗网格层对齐；
接口会检查。`I2D` 情况的两平面方向不进行粗化。

支持的湍流/黏性配置：

| turbulence | CFL3D IVISC |
| --- | --- |
| euler | 0 |
| laminar | 1 |
| sa | 5 |
| komega | 6 |
| sst | 7 |

二维两平面网格的 i 方向关闭黏性，其他方向按所选模型设置。
Euler 自动关闭 full_navier_stokes。湍流 CFL 统一设置核心的七个 CFLTURB 槽位。
`eddy_viscosity_limit`、`gamma`、`kappa` 和参考参数也可以在 SolverOptions 中配置。
SolverOptions 是不可变配置模板；solver.options 可通过字典、属性或 setOptions() 原地修改，下一次 solve() 同步传入 Fortran。

本轮新增稳态计算控制参数：

| 参数 | 默认值 | 原生含义 |
| --- | --- | --- |
| `cfl_ramp_steps` | 0 | IFLAGTS；0 关闭 CFL 渐增 |
| `cfl_ramp_factor` | 1.0 | FMAX；目标 CFL 为 `cfl * cfl_ramp_factor` |
| `first_order_iterations` | 0 | 最终网格序列阶段的 NITFO |
| `coarse_first_order_iterations` | 0 | 每个非最终网格序列阶段的 NITFO |
| `restart_interval` | None | NWREST；None 使用总迭代数 |

这几个选项已在双块 SA/SST 平板串行和 MPI 计算中确认传入核心并正常完成。
`restart_interval` 控制段内原生 checkpoint 写出频率；每段结束缓存完整状态，restartFile 支持文件接续。
一阶迭代数不得超过相应阶段迭代数；CFL 渐增步数须非负，倍数须至少为 1。

## 定升计算与最终攻角

`target_cl=None` 使用固定攻角；设为数值后启用原生 CFL3D 的 CL 反馈控制。
不需要在 Python 外层反复重新启动求解。参数对应关系为：

| Python 参数 | 默认值 | 原生变量 |
| --- | --- | --- |
| `target_cl` | None | CLTARG；None 使用原生关闭标记 |
| `alpha_relaxation` | 1.0 | RLXALPH |
| `alpha_step_limit` | 0.2 度 | DALIM；每次攻角更新的限幅 |
| `alpha_update_interval` | 1 | ICYCUPDT；每隔多少次细网格循环更新攻角 |

```python
options = SolverOptions(
    turbulence="sst", alpha=0.0, target_cl=0.5,
    alpha_relaxation=0.5, alpha_step_limit=0.1,
    alpha_update_interval=5, iterations=1000,
)
result = CFL3D("wing.cgns", options).solve(output_directory="lift-results")
print(result.coefficients["CL"], result.final_alpha)
```

`result.final_alpha` 为实际最终攻角（度），并写入 `solution.npz`。
`cfl3d.alpha` 保存迭代号、log(res)、CL、更新后的攻角和攻角增量，返回 Python 前已关闭刷新。
同目录再次求解会重置该日志，关闭定升后也不会残留旧记录。
`options.alpha` 保持初始值不变。原生控制在力计算后更新攻角，因此最后一行 CL 来自最后一次
力计算，而 final_alpha 包含随后的攻角更新；下一次迭代才会重新评价该攻角下的力。
定升仍执行指定迭代次数，不保证自动满足 CL 容差，也没有新增提前停止判据。

## 分方向格式与显式多重网格阶段

`limiter`、`flux`、`inversion` 和 `kappa` 接受标量或 `(i,j,k)` 三元组。
标量保持原有行为，三元组分别控制三个网格索引方向；目前所有 zone 使用同一套方向设置。
`limiter` 对应 IFLIM（0..4），`flux` 对应 IFDS（0/1），`inversion` 对应 IDIAG（0/1），
`kappa` 对应 RKAP。IDIAG=0 为块三对角逆，=1 为标量三对角逆。
`eddy_viscosity_limit` 默认继续使用用户选定的 `1e5`，传入 EDVISLIM。

```python
from cfl3d import MultigridStage, SolverOptions

options = SolverOptions(
    turbulence="sst", limiter=(0, 2, 4), flux=(1, 0, 1),
    inversion=(1, 0, 1), kappa=(0.0, 1/3, 1/3),
    multigrid_levels=3, multigrid_cycle=2,
    multigrid_up_iterations=1,  # MTT：多重网格上行的附加更新次数
    multigrid_stages=(
        MultigridStage(iterations=100, levels=2, first_order_iterations=10,
                       smoothing_steps=(1, 2)),
        MultigridStage(iterations=1000, levels=2, first_order_iterations=20,
                       smoothing_steps=(2, 1)),
    ),
)
```

`multigrid_levels` 决定总网格层数，`MultigridStage.levels` 决定该阶段参与循环的活跃层数
（MGLEVG）。原生网格序列每阶段细化一次，最后阶段总在最细网格；例如总共三层、两个阶段，
先计算第二细网格，再计算最细网格。阶段数不得超过总层数；第一阶段可用两层、第二阶段可用三层。
阶段 levels 可以小于可用层数，只使用该阶段较细的若干层。

`smoothing_steps` 对应每阶段 MIT，按该阶段活跃层的粗到细顺序给出。
也可以设置 `multigrid_smoothing=(1,2,1)`，按所有全局网格层的粗到细顺序提供默认 MIT；
各阶段自动取实际活跃网格层的对应项。阶段显式 smoothing_steps 优先于全局默认，未设置时为 1。
MIT 须为 1..20 的整数，MTT 非负，接口检查 MIT+MTT 不超过原生工作数组上限 20。

显式 `multigrid_stages` 优先于 `mesh_sequence/coarse_iterations/iterations` 自动生成的阶段表，
每阶段的 iterations、levels、first_order_iterations 对应 NCYC、MGLEVG、NITFO。
因此实际总迭代数为显式阶段 iterations 之和。未提供阶段表时，现有简化参数的用法保持有效。
各阶段 iterations 须为正整数，first_order_iterations 不得超过本阶段 iterations。
阶段表及 MIT 使用不可变 tuple，可通过 dataclasses.replace 创建修改后的配置。
embedded 网格尚未接入，NEMGL/ICONSF 仍没有可用的 embedded 计算接口。

本版原生接口版本为 6，新增完整状态接续及 MPI 历史长度修正后必须重建串行和 MPI 扩展并重启 Python：

```bash
python tools/build_python.py --debug --jobs 8
python tools/build_python.py --mpi --debug --jobs 8
```

串行和 MPI 示例都支持 `--target-cl`、`--limiter 0 2 4`、`--flux 1 0 1`、
`--multigrid-smoothing 1 2 1` 等常用开关；显式阶段表通过上述 Python 对象设置。

## 实验性 MPI 接口

先构建 `--mpi` 扩展，再在所有 rank 上调用构造函数和 `solve()`：

```python
from mpi4py import MPI
from cfl3d import CFL3D, SolverOptions

comm = MPI.COMM_WORLD
solver = CFL3D(
    "mesh.cgns" if comm.rank == 0 else None,
    SolverOptions(turbulence="sst", iterations=100),
    comm=comm,
)
result = solver.solve(output_directory="mpi-output")
if comm.rank == 0:
    print(result.coefficients)
```

每个 rank 保留并校验自己的 options，构造/求解前在当前 communicator 内检查一致；网格由其 rank 0 读取广播。
原生 rank 0 专用于主控，其他 rank 按完整 zone 分配；这版不自动切分单个 zone，
要求 `2 <= 进程数 <= zone数+1`。一个 zone 使用两个进程，两个 zone 最多三个进程。
Python 管理 MPI 初始化和结束；求解期间使用复制的 communicator，结束后释放副本。

返回时各 rank 均持有完整流场和历史，只有 rank 0 写 `solution.npz`。
主日志在输出根目录，计算进程的原生临时输出位于 `rank-0001` 等子目录。
当前广播和汇总会复制完整网格/结果，尚未实现按 rank 分布的 Python 存储。

Python 侧配置错误会汇总并向所有 rank 抛异常。进入原生计算后若某 rank 出错，
使用 `MPI_Abort` 终止作业以避免其他 rank 永久等待；不支持原生 MPI 错误后的恢复。
已通过双块 Euler/SA/SST 的 2/3 进程串并行比较、连续求解和两个独立子通信器测试。
GNU MPI 与 GNU 串行比较项一致；Intel Fortran MPI 相对 GNU 串行最大差异约 `3e-15`。
均匀 Euler 历史只比较力/力矩列；SA/SST 比较全部历史列。
Valgrind 双块 Euler 测试各 rank 为零错误，未检查泄漏。范围和日志见检查点记录。
ADF 文件入口已在 GNU 求解扩展下通过相同串行/MPI 验收；较大算例和长期收敛仍待验证。

## CGNS 网格和边界

`CGNSMesh.from_file(..., backend="auto")` 默认自动选择：ADF 使用原生 CGNS Mid-Level API；
HDF5 在原生库具有 HDF5 支持时使用该库，否则使用可选 h5py。
可指定 `backend="cgns"` 或 `backend="h5py"`；求解器传入文件名时使用自动选择。
当前 CGNS 4.5 库编译时关闭 HDF5，已经验证 ADF 文件读取与求解，无需转换格式。
HDF5 的原有 h5py 读取器保留，但本轮环境没有 h5py，未复测该路径。
文件读取通过 ctypes 调用编译后的 CGNS ABI 桥；求解器仍通过 f2py 调用 Fortran 核心。
没有绑定 CFL3D 旧的 CGNS API。
支持单个 3D base 内的结构化 zones；多个 base 时先使用
`CGNSMesh.from_file("mesh.cgns", base="BaseName")` 再传给 CFL3D。

边界必须是 Vertex GridLocation 的矩形 PointRange。接口检查每个面的单元覆盖，
边界分段与连接面不能重叠，也不能缺失。共享边界线不视为重叠。

| CGNS BC 类型 | CFL3D BCTYPE / data |
| --- | --- |
| BCFarfield | 1003 |
| BCSymmetryPlane | 1001 |
| BCWallInviscid | 1005 |
| BCOutflowSupersonic | 1002 |
| BCWallViscous | 2004 / (0,0)，默认绝热且无吹吸 |
| BCWallViscousHeatFlux | 2004 / (0,0)，仅支持未附加数据或显式零 HeatFlux |

其他类型或带未识别 BCDataSet 的边界需要显式覆盖；不会默默丢弃附加数据。
覆盖使用 CGNS 的 **zone 名称和 BC 名称**：

```python
solver = CFL3D(
    "mesh.cgns", SolverOptions(turbulence="sst"),
    boundary_overrides={
        ("Zone1", "inlet"): (2009, (1.1, 1.0, 0.0, 0.0)),
        ("Zone1", "outlet"): (2002, (1.0,)),
        ("Zone1", "wall"): (2004, (1.05, 0.0)),
    },
)
```

数据顺序采用 CFL3D 对应类型的原生定义：2004 是 `(Tw/Tinf, CQ)`，
其中 Twtype=0 为绝热，负值为停滞温度壁面；2009 是总压比、总温比、入流角度；
2002 是出口静压比。壁温、吹吸及入口热力学数据应由用户根据物理工况确定。
BCDataSet 不会自动进行单位推断。拼错的覆盖名称会报错。

读取 GridConnectivity1to1_t 的 donor range 和 Transform，转换为 CFL3D 的一对一块连接。
接口检查配对点坐标一致、变换与区间一致，并去重两侧重复定义。
块接口由 CGNS 连接自动设置为 BC 0，不能通过边界覆盖伪造。

## 返回结果

本版推荐把网格路径、输出目录、计算设置及输出开关统一放进 options 字典：

```python
solver = CFL3D(options={
    "input_grid": "mesh.cgns",
    "output_directory": "results",
    "mach": 0.2, "alpha": 2.0, "turbulence": "sst",
    "iterations": 1000, "cfl": 1.0,
    "freestream_pressure": 101325.0,
    "writevolumesolution": True,
    "writeTecplotSurfaceSolution": True,
    "writeflowsolution": False,
})
result = solver.solve()
```

也可使用 `CFL3D(options_dict)`。原有 `CFL3D(mesh, SolverOptions(...))` 保持可用。
`input_grid` 与 mesh 参数不能同时设置；`solve(output_directory=...)` 仍可覆盖本次输出目录。
字典在构造时复制为可修改 RuntimeOptions，修改原字典不会影响已有 solver；运行时修改 solver.options。
标量/方向参数支持三元 list，`multigrid_stages` 支持字典列表，内部统一成 tuple 和 MultigridStage。
未知键和非布尔输出开关会报错；options 和阶段字典 key 不区分大小写，同一字典的大小写重复键会报错。

自动输出文件和独立补写方法：

| options 开关 | 默认值 | 自动输出 | 补写方法 |
| --- | --- | --- | --- |
| `writevolumesolution` | True | `volume.cgns`，及 NPZ 中的体流场 | `writeVolumeSolution(filename=None)` |
| `writeTecplotSurfaceSolution` | True | `surface.dat` | `writeTecplotSurfaceSolution(filename=None, groupName=None)` |
| `writeflowsolution` | False | `flow.x`、`flow.q` | `writeFlowSolution(prefix=None)` |

文件名分别可通过 `volume_filename`、`surface_filename`、`flow_prefix` 设置。
三个开关全关时不保留体/表面解和原生 restart 文件，但仍保存历史、积分元数据和原生日志；完整接续状态缓存在 Python 内存。
场数据仍保留在本 solver 的 Python 快照中，补写方法不受开关限制，也不再进入原生 CFD 计算。
补写以最近一次成功计算为准，不依赖输入文件继续存在，也不受另一个 solver 重用 Fortran 状态影响。
尚未提供从只含历史的 NPZ 恢复整个快照的接口。

```python
# 自动输出曾关闭时，计算结束后仍可补写。
solver.writeVolumeSolution()
solver.writeTecplotSurfaceSolution()
solver.writeTecplotSurfaceSolution("wing.dat", groupName="Wing")
solver.writeFlowSolution("flow-later")
```

默认路径位于本次输出目录；显式传入的相对路径则相对于调用时工作目录。
文件通过临时文件写完再替换目标，不允许覆盖原输入 CGNS 网格。
MPI 中只由 rank 0 写文件，补写方法可仅由 root 调用；在其他 rank 调用返回 None。

CGNS 体文件默认采用 ADF，包括原始顶点、结构化 zone、边界/连接和 CellCenter FlowSolution。
Density、VelocityX/Y/Z、Pressure 保持 CFL3D 无量纲定义，Temperature 为 K；
量纲记录在 FlowSolution 的 CFL3DUnits 描述中。该文件用于后处理，不提供原生 restart 接续。
PLOT3D 按原生 `IPTYPE=1` 的单元中心二进制格式从缓存写出，含 32 位记录标记及 IBLANK、
64 位坐标/守恒量；二维为两个坐标和四个守恒量，三维为三个坐标和五个守恒量。
未实现 IPTYPE=0 的网格点解导出。SST 平板与独立原生程序的 `.q` 逐记录相同，
`.x` 坐标最大差异为 `2.22e-16`，IBLANK 相同。

表面输出使用 Tecplot ASCII FEQUADRILATERAL，每个实际 wall BC 一个 zone，
可进一步用 CGNS/custom group 过滤。无 wall 时只有文件头。
X/Y/Z 是原始壁面顶点；Cp、T、Cfx、Cfy、Cfz、y+ 为面单元中心变量（VARLOCATION 标明）。
T 使用原生壁面边界状态换算为 K；Cf 从 force.F 的剪切力除以壁面面积取得，使用输入坐标轴。
`y+ = rho_w*u_tau*d/mu_w`，使用第一层单元中心的法向距离 `d=V/(2A)`，
以及内核的 Sutherland 分子黏度。它不采用旧 calyplus.F 文本统计的节点平均值。
无黏壁面 Cf 和 y+ 为零。表面分布在最细网格最后一次力计算时缓存，时序与 getSolution 一致。
`result.wall_fields[(zone,bc)]` 形状为 `(面方向1单元数,面方向2单元数,6)`，
列依次为 Cp、T(K)、Cfx、Cfy、Cfz、y+，面方向按输入网格未固定的两个索引轴排序。

`result.fields[zone_name]` 形状是 `(ni-1,nj-1,nk-1,5)`，依次为单元中心
`rho,u,v,w,p`，**采用 CFL3D 的无量纲化**：自由来流 rho=1、声速=1、p=1/gamma。
最后一个索引依然使用网格的 i/j/k 次序，不是核心内部 j/k/i 储存顺序。

`result.history` 每行包含：

```text
residual, CL, CD, CX, CY, CMX, CMY, CMZ
```

`result.coefficients` 返回最后一次迭代的系数字典，包含 `CDp`、`CDv`。
输出目录还保存 `solution.npz`，包括 history、final_alpha、以 `zone_<name>` 命名的流场数组，
以及逐边界的 `surface_names`、`surface_coefficients` 和 `surface_coefficient_names`。
表面积分列依次为 `CL, CD, CX, CY, CZ, Cmx, Cmy, Cmz, CDp, CDv`；
`surface_names` 每行是对应的 zone、BC 名称。原生 CFL3D 日志与 restart 文件也保留。
输出文件不代表已收敛；迭代次数由配置决定，本版不实现 Python 收敛停止判据。
关闭 `writevolumesolution` 后，自动 NPZ 不包含 `zone_<name>` 体流场数组。

同一对象重复 solve() 从 Python 缓存的完整状态接续；resetFlow() 后从自由来流初始化，新对象也可用 restartFile 或 setFlowState() 接续。
核心使用 COMMON 全局状态，禁止并发 solve；计算期间输出目录通过进程 cwd 切换，
请勿在其他线程中依赖相对文件路径。
原生错误通过 C 守卫返回 Python RuntimeError，不会用 STOP 终止解释器。
错误中断可能留下核心分配的内存，因此失败后禁止重用该核心，应新建 Python 进程。
构建新网格、更改选项或正常完成后的连续调用则已通过验证。

## 按 CGNS 分组提取气动力

`CFL3D.getSolution(groupName=None)` 在成功 `solve()` 后返回一个字典。
不传名称时提取全部原生积分表面，传名称时提取对应部件：

```python
solver = CFL3D(
    "wing.cgns",
    SolverOptions(
        mach=0.8, alpha=2.0, turbulence="sst", iterations=1000,
        sref=93.0, cref=3.88, bref=30.0,
        moment_center=(0.0, 0.0, 0.0),
        freestream_pressure=101325.0,  # 自由来流静压，Pa；应改为实际工况值
    ),
    groups={"Aircraft": ["Wing", "Tail"]},  # Wing/Tail 是文件中的 CGNS 名称
)
result = solver.solve(output_directory="wing-results")
print(solver.getGroupNames())
loads = solver.getSolution("Aircraft")
print(loads["CL"], loads["CD"], loads["CDp"], loads["CDv"])
print(loads["D"], loads["Dp"], loads["Dv"], loads["My"])
solver.addGroups({"WingOnly": ["Wing"]})  # 也可在求解后增加集合
```

默认分组包括 CGNS 的 `FamilyName_t` / `AdditionalFamilyName_t`、zone 名、BC 名，
以及明确的 `zone/BC` 路径。zone 的 Family 包含其全部边界；BC 的 Family 包含该 BC。
同名 BC 跨 zone 合并。不同类型同名时优先级为 Family、zone、BC；
`zone/BC` 路径始终可用于精确选择。名称区分大小写。
ADF 和 HDF5 读取器均保留这些 Family 信息；当前环境实际验证的是 ADF。
`FamilySpecified` 可通过当前 base 内唯一的 `FamilyBC_t` 解析边界类型。

新增集合使用 `{新名字: [CGNS 名称, ...]}`，不能与已有 CGNS 分组或新增集合重名，
成员必须是已有 CGNS 名称，不支持集合递归引用。重叠成员取并集，每个边界只计一次。
接口保持 CFL3D 的原生积分范围；远场和连接面等不参与原生力积分的表面贡献为零。
这些结果来自最细网格上原生 `force.F` 的压力和黏性积分，并在内核释放数组前缓存，
无需从壁面附近流场近似重算。定升时同样对应最后一次力计算，攻角时序见前文。

返回字段与量纲：

| 字段 | 含义 / 单位 |
| --- | --- |
| `CL, CD, CY` | 升力、阻力、输入坐标 Y 方向力系数 |
| `Cmx, Cmy, Cmz` | 输入坐标轴上的力矩系数 |
| `CDp, CDv` | 压差阻力、黏性阻力系数；`CD = CDp + CDv` |
| `L, D, Y, Dp, Dv` | 对应的力，N |
| `Mx, My, Mz` | 对应的力矩，N·m |
| `units` | 上述有量纲字段的单位字典 |

`CL/CD/CDp/CDv` 沿 CFL3D 原生风轴定义；`CY`、力矩使用输入网格坐标轴。
`CY` 沿输入 Y 方向，`y_up=True` 时它不是横向侧力。
本接口的有量纲输出要求坐标、参考长度、力矩中心使用 m，`sref` 使用 m²，
`freestream_pressure` 使用 Pa。使用理想气体关系
`q∞ = 0.5 * gamma * freestream_pressure * mach**2`，力为 `C * q∞ * sref`。
`y_up=False` 的力矩参考长度依次为 `bref, cref, bref`；
`y_up=True` 为 `bref, bref, cref`。力矩均相对于 `moment_center`。
未设置静压仍可求解和读取 `result.coefficients`，但 `getSolution()` 会报错，
避免在未指定工况压力时给出错误的 N / N·m 数值。

MPI 求解结束时各 rank 都缓存完整分组积分。`getSolution()`、`getGroupNames()` 和
`addGroups()` 是本地调用，可仅在 rank 0 上执行；构造函数及 `solve()` 仍须所有 rank 参与。
一个对象缓存的结果不受另一个 CFL3D 对象后续求解影响。

串行与 MPI 示例支持 `--freestream-pressure`、`--group` 和三个参考量开关，例如：

```bash
mpirun -np 3 python examples/python/run_mpi.py wing.cgns --model sst \
  --mach 0.8 --alpha 2 --freestream-pressure 101325 \
  --sref 93 --cref 3.88 --bref 30 --group Wing
```

## 升力分布与截面 Cp

在 `solve()` 前注册，可随表面结果自动输出；也可在计算结束后注册并补写：

```python
solver = CFL3D(options=options, groups={"Aircraft": ["Wing", "Tail"]})
solver.addLiftDistribution(200, "y", groupName="Wing")
solver.addSlices("y", [-3.0, -1.5, 1.5, 3.0], sliceType="absolute", groupName="Wing")
# relative 同样输入实际坐标（m），截线绑定注册时的面单元，随后跟随网格变化。
solver.addSlices("y", [1.5], sliceType="relative", groupName="Wing")
result = solver.solve()

lift = solver.getLiftDistribution(0)  # 第一个分布；索引从 0 开始
print(lift["stations"], lift["dCL_ds"], lift["dCDp_ds"], lift["dCDv_ds"])
print("CL from strips:", (lift["dCL_ds"] * lift["width"]).sum())
print("CL from wall forces:", solver.getSolution("Wing")["CL"])
sections = solver.getSlices()        # 列表；getSlices(0) 返回单个截面字典
print(sections[0]["coordinates"], sections[0]["Cp"])
solver.writeTecplotSurfaceSolution() # 自动输出关闭时也可补写
```

`direction` 为输入网格的 `x/y/z`，`groupName=None` 选择全部 wall 边界；支持 CGNS 名称和
自定义集合，重叠表面只计一次。`nSegments` 为正整数，通常可取 150–250。
上例假定 CGNS 中存在 `Wing/Tail`；截面位置须替换为实际网格的坐标。

| 查询结果 | 含义 / 单位 |
| --- | --- |
| `edges`, `stations`, `width` | 等宽条带边界、中点、宽度，m |
| `bin_CL/CD/CDp/CDv` | 每个条带的积分系数；求和等于该 wall 分组的总系数 |
| `dCL_ds/dCD_ds/dCDp_ds/dCDv_ds` | 单位长度的积分系数，1/m，使用全局 `sref` |
| `dL_ds/dD_ds/dDp_ds/dDv_ds` | 单位长度的力，N/m；仅设置 `freestream_pressure` 时提供 |
| 截面 `points`, `segments` | 截线顶点坐标（m），及从 0 开始的两端点连接索引 |
| 截面 `coordinates`, `Cp`, `T`, `Cfx/Cfy/Cfz`, `y+` | 线段中点（m）及面中心流动量，T 为 K |

升力分布将原生逐面载荷按几何交叠面积守恒分配到条带，包含压力和黏性贡献。
这里的 `dCL_ds` 不是按局部弦长归一化的二维翼型 `cl`。
定升计算使用实际力积分时的迎角；该角度（度）记录在 `force_angles`，可能与
`result.final_alpha` 不同。

以 `surface.dat` 为表面文件名时，已注册的分析分别写入 `surface_lift.dat` 和
`surface_slices.dat`，每个请求一个 Tecplot zone；无交点的截面不写 zone。
补写 `wing.dat` 则生成 `wing_lift.dat` / `wing_slices.dat`。
`writeTecplotSurfaceSolution(groupName=...)` 只筛选主壁面文件；分析使用各自注册的分组。
分析文件随 `writeTecplotSurfaceSolution` 开关一起控制，也可由上述补写方法重新生成。
MPI 注册、查询、补写均无需集体通信，可仅在 rank 0 调用；`solve()` 仍须所有 rank 参与。

`relative` 追踪注册时的三角面交点，`absolute` 每次按固定平面重新截取；两者的位置都以 m
输入，不是展向比例。拓扑改变后须先 `clearSurfaceAnalysis()` 再重新注册。
当前支持稳态计算间的几何变化追踪，尚未接入原生非定常移动网格。
共形块交界处合并重合端点和重复线段；未实现 overset zipper 或非匹配表面的补缝。
Cp 保留面中心值，截线恰好沿共享边时使用相邻面的面积加权值；与平面完全共面的壁面
不生成截线。`clearSurfaceAnalysis()` 清除全部请求，`nLiftDist/nSlice` 返回已注册数量。

## 自动 MPI、可修改 options 与接续计算

省略 `comm` 时，普通 `python case.py` 使用串行扩展，`mpirun -np N python case.py`
自动使用 MPI 扩展。`comm=None` 显式选择串行；传入单 rank 通信器也走串行。
需要其他 rank 做优化、其他算例或其他工作时，必须显式传入本算例的子通信器；
构造、检查和求解只涉及该通信器，不会对 `COMM_WORLD` 做集体通信。
例如在四进程作业中排除 WORLD rank 0：

```python
from mpi4py import MPI
from cfl3d import CFL3D

world = MPI.COMM_WORLD
case_comm = world.Split(0 if world.rank in (1, 2, 3) else MPI.UNDEFINED, world.rank)
if case_comm != MPI.COMM_NULL:
    # 每个参与 rank 都构造并保留自己的 options 副本。
    solver = CFL3D(options=options, comm=case_comm)
    solver.solve()
    case_comm.Free()
```

字典 key 不区分大小写：`CFL/cfl/CfL`、`writeVolumeSolution/writevolumesolution` 均可。
下划线仍属于参数名称；同一字典中出现大小写重复键会报错。
各 rank 的显式 options 会分别校验并保留，在构造和每次 `solve()` 前检查一致性；
不会用 root 配置静默覆盖其他 rank 的显式设置。为兼容旧脚本，非 root 的
`options=None` 可继承 root 的初始配置。

`solver.options` 是可修改的 `RuntimeOptions`，支持字典及属性访问：

```python
solver = CFL3D(options=options)     # 自动识别串行/MPI
solver.solve()                     # 第一次从自由来流或 restartFile 初始化

# MPI 下所有参与 rank 执行相同更新；多参数更新会整体校验，失败时不部分修改。
solver.setOptions({"CFL": 3.0, "ITERATIONS": 100})
solver.options["WriteVolumeSolution"] = False
# 等价写法：solver.setOption("cfl", 3.0) 或 solver.options.cfl = 3.0
solver.checkOptions()              # comm 内集体检查；solve() 也会自动检查
solver.solve()                     # 从上一次完整状态继续，再执行本段 100 步

if solver.comm is None or solver.comm.rank == 0:
    solver.writeRestartFile("checkpoint.npz")  # root 本地调用，无需其他 rank 配合

# 新对象从文件接续；也支持原生 restart.bin（下述格式）。
continued = CFL3D(options={**options, "restartFile": "checkpoint.npz"})
continued.solve()
```

修改值在下一次 `solve()` 开始时统一传入 Fortran；当前不支持计算执行中的线程热更新。
`solver.options.check()` 与 `solver.checkOptions()` 等价。`getOption(name)` 读取当前值，`getFlowState()` 返回完整 Python 状态副本，
`setFlowState(state)` 恢复此前保存的完整状态；其中 `fields` 是读取用的 primitive 流场，
恢复以 `native` 完整状态为准。`resetFlow()` 清除内存状态和 restartFile 设置，下次从自由来流开始。
`SolverOptions` 仍可作为不可变配置模板传入，运行时修改使用 `solver.options`。
`input_grid` 在构造时加载，切换输入网格文件应创建新对象。

接续状态包含主流场、湍流变量、涡黏性及 ghost/壁面状态，缓存在各 rank 的 Python 内存中，
不依赖旧输出目录或另一个 solver 的 Fortran 全局状态。
当前每段通过内部临时原生 restart 文件与 Fortran 交换状态，结束即清理；尚未实现无文件的状态数组传递。
体/表面输出开关全关仍可继续计算及手动保存检查点。
`writeRestartFile()` 默认写本次输出目录下的 `restart.npz`，包含完整原生载荷、校验和、
兼容性元数据和 primitive 流场；它与后处理 `solution.npz` 用途不同。

`restartFile` 默认 None；支持上述检查点和原生 real64、int32 记录标记、原生端序且含 ghost 数据的
`restart.bin`。读入前检查记录完整性、块尺寸及湍流模型；Python 检查点还检查模型、gamma、
y-up 和网格拓扑。原生文件不包含完整 Python 元数据，坐标约定和物理参考量须由使用者匹配。
当前接续使用最后一级网格，不重复已完成的粗网格序列；`iterations` / 显式阶段表的最终阶段
迭代数表示本段追加步数，返回 history 也仅包含本段。
每段重新使用所设的 CFL 渐增、一阶迭代和定升反馈计数；启用这些控制时，分段点会影响计算路径。
原生 checkpoint 与当前数值精度、记录格式和模型相关，跨平台或更换编译配置需另行验证。

## 当前明确范围

本版已验证稳态、实数、串行求解；包含多块一对一连接和多重网格。
MPI/f2py communicator 为实验性实现，验证状态见上文。
尚未实现非定常、overset、非匹配 patch、周期连接、
移动网格、气动弹性、伴随及 OpenMDAO 接口。
不支持真正的二维 base、PointList、非零坐标 rind、非 Vertex 边界，
或多个时刻的 GridCoordinates。读取到不支持的数据会报错。
二维计算需使用三维 base 中 i 方向有两个平面的结构化网格。
原生读取器目前要求坐标节点名为 `GridCoordinates`，且最多一个 `ZoneGridConnectivity` 节点。
BCDataSet 的数值数组会读出并传给现有边界检查；不支持的类型或非零热流不会被忽略。

## 本地验证和 CI

新增升力分布、截面 Cp、字典配置与延迟 I/O 回归：

```bash
PYTHONPATH=.:tests/python python -m unittest test_surface_analysis test_solution_io -v
PYTHONPATH=.:tests/python mpirun -np 3 python tests/python/validate_solution_io_mpi.py --directory build/mpi-io-validation
PYTHONPATH=.:tests/python mpirun -np 4 python tests/python/validate_solution_io_mpi.py --split --directory build/mpi-io-validation
python tools/build_python.py --standalone --debug --jobs 8
PYTHONPATH=.:tests/python python tests/python/validate_plot3d.py --directory build/mpi-io-validation/plot3d-parity
```

覆盖输出开关全关后的三种补写、不同 solver 的快照隔离、CGNS 流场读回、壁面温度、
Cf 再积分、y+ 定义、三维 i/j/k 壁面及 y-up、PLOT3D 原生格式一致性。

分组积分回归（真实 CGNS/ADF、2/3 进程及两个独立子通信器）：

```bash
PYTHONPATH=.:tests/python python -m unittest test_surface_forces test_cgns_native -v
CFL3D_VALIDATION_DIRECTORY=build/mpi-forces-validation bash tools/validate_mpi.sh --adf --controls --forces
```

检查分组和全部积分的一致性、重叠集合去重、`CDp+CDv=CD`、SI 换算、缓存独立性，
并在三维 SST 网格上检查力矩中心平移与 y-up 旋转。
三维 SST 验证还修复了旧 `twoeqn.F` 中普通 SST 模式读取 DES 专用 `fdsav` 数组的问题；
Fortran 的逻辑 AND 不保证短路求值，现用嵌套条件确保仅 DES 模式访问该数组。

新增计算控制的回归（包括真实 ADF 文件读入）：

```bash
CFL3D_VALIDATION_DIRECTORY=build/mpi-controls-validation bash tools/validate_mpi.sh --adf --controls
```

该测试增加攻角限幅/间隔、自定义阶段与 MIT/MTT、分方向格式，以及 200 次迭代的 SST 定升平板。
Valgrind 检查的是自定义 MIT 和块矩阵求逆算例，不进行泄漏扫描。

当前 ADF 文件回归：

```bash
source /home/mdolabuser/.bashrc_mdolab
python tools/build_cgns.py
PYTHONPATH=.:tests/python python -m unittest test_cgns_native -v
CFL3D_VALIDATION_DIRECTORY=build/cgns-validation bash tools/validate_mpi.sh --adf
```

脚本用官方 CGNS API 写出双块 Euler/SA/SST 的 ADF 文件，再分别读取文件做串行及 MPI 求解，
与内存网格串行基准比较。ADF 读取回归还覆盖负号 Transform、多 base 选择、
BC 数据集，以及 PointList、CellCenter、周期连接的拒绝行为。
HDF5 专项单元测试在未安装可选 h5py 时显示为跳过。


```bash
python tools/build_python.py --debug
python tools/build_cgns.py
python -m unittest discover -s tests/python -v
python tools/build_python.py --debug --standalone
python tests/python/validate_parity.py --directory parity-output
```

测试覆盖 CGNS 坐标/BC/连接读取、重复计算、单块和多块均匀来流、3D beta 和 y-up、
SA/SST/层流真实平板网格、三级网格序列、输出数组及错误返回。
真实网格来自本仓库 NASA 的 Flatplateyplus 归档；CGNS 测试文件使用同一坐标生成。
原生调用测试会检查计算目录内没有 `.inp` 或 CGNS 输入副本。

对比验证另行编译同一套 CFL3D 源码的原生可执行程序，以归档的输入文件为参考，
显式统一壁温、重构系数等设置；然后逐元素比较 restart 中的单元流场与迭代历史。
参考测试的 `.inp` 只用于原生可执行程序，Python/f2py 计算不会读取它。
SA 和 SST 各三级网格序列、15 次迭代的 debug 验证中，流场和历史最大差值均为 0。
优化构建与原生 debug 构建的对比差异不超过 2.4e-15。
测试用的平板和多块 CGNS 文件还通过了官方 CGNS 仓库的 cgnscheck 检查，
并验证了由官方 CGNS C API 写出的 CGNS/HDF5 文件可正常读取和求解。
这是接口一致性验证，不是平板网格充分收敛或湍流模型精度验证。

GitHub Actions 的 `python-native.yml` 会构建扩展、运行测试、编译原生程序并执行对比。
原生 makefiles 仍保留原有行为；额外修复了 sizer.F 一个缺少右括号的编译错误。
