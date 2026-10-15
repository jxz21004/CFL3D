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
构建输出安装到 `python/cfl3d/_native.<ABI>.so`，无需修改原有 makefiles。
`--debug` 开启调试符号和新接口的数组边界检查；旧 Fortran 内核有合法的跨维展开访问，
不能整体启用 `-fcheck=bounds`。

`pip install` 安装 Python 包和依赖；共享库需要执行上述单独的构建命令。
已有依赖时可以直接设置 `PYTHONPATH=python:tests/python` 在源码目录测试。
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
参数对象不可变；修改工况时可使用 `dataclasses.replace` 创建新的 options，再创建 solver。

本轮新增稳态计算控制参数：

| 参数 | 默认值 | 原生含义 |
| --- | --- | --- |
| `cfl_ramp_steps` | 0 | IFLAGTS；0 关闭 CFL 渐增 |
| `cfl_ramp_factor` | 1.0 | FMAX；目标 CFL 为 `cfl * cfl_ramp_factor` |
| `first_order_iterations` | 0 | 最终网格序列阶段的 NITFO |
| `coarse_first_order_iterations` | 0 | 每个非最终网格序列阶段的 NITFO |
| `restart_interval` | None | NWREST；None 使用总迭代数 |

这几个选项已在双块 SA/SST 平板串行和 MPI 计算中确认传入核心并正常完成。
`restart_interval` 只控制原生 restart 文件写出频率，不提供 restart 读入接续接口。
一阶迭代数不得超过相应阶段迭代数；CFL 渐增步数须非负，倍数须至少为 1。

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

配置、网格和边界覆盖以 communicator 的 rank 0 为准，并广播到其他 rank。
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

`result.fields[zone_name]` 形状是 `(ni-1,nj-1,nk-1,5)`，依次为单元中心
`rho,u,v,w,p`，**采用 CFL3D 的无量纲化**：自由来流 rho=1、声速=1、p=1/gamma。
最后一个索引依然使用网格的 i/j/k 次序，不是核心内部 j/k/i 储存顺序。

`result.history` 每行包含：

```text
residual, CL, CD, CX, CY, CMX, CMY, CMZ
```

`result.coefficients` 返回最后一次迭代的系数字典。输出目录还保存 `solution.npz`，
包括 history 和以 `zone_<name>` 命名的流场数组，以及原生 CFL3D 日志与 restart 文件。
输出文件不代表已收敛；迭代次数由配置决定，本版不实现 Python 收敛停止判据。

同一进程可以连续求解多个工况，每次重新传入配置和网格、从自由来流初始化。
核心使用 COMMON 全局状态，禁止并发 solve；计算期间输出目录通过进程 cwd 切换，
请勿在其他线程中依赖相对文件路径。
原生错误通过 C 守卫返回 Python RuntimeError，不会用 STOP 终止解释器。
错误中断可能留下核心分配的内存，因此失败后禁止重用该核心，应新建 Python 进程。
构建新网格、更改选项或正常完成后的连续调用则已通过验证。

## 当前明确范围

本版已验证稳态、实数、串行求解；包含多块一对一连接和多重网格。
MPI/f2py communicator 为实验性实现，验证状态见上文。
尚未实现非定常、overset、非匹配 patch、周期连接、
移动网格、气动弹性、Python restart/continuation、伴随及 OpenMDAO 接口。
不支持真正的二维 base、PointList、非零坐标 rind、非 Vertex 边界，
或多个时刻的 GridCoordinates。读取到不支持的数据会报错。
二维计算需使用三维 base 中 i 方向有两个平面的结构化网格。
原生读取器目前要求坐标节点名为 `GridCoordinates`，且最多一个 `ZoneGridConnectivity` 节点。
BCDataSet 的数值数组会读出并传给现有边界检查；不支持的类型或非零热流不会被忽略。

## 本地验证和 CI

当前 ADF 文件回归：

```bash
source /home/mdolabuser/.bashrc_mdolab
python tools/build_cgns.py
PYTHONPATH=python:tests/python python -m unittest test_cgns_native -v
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
