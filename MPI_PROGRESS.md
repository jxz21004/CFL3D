# CFL3D Python/MPI 第十一轮检查点（动态参数与分类初始化策略）— 2026-10-16

最新动态参数及自动初始化策略见文末“第十一轮”；前面保留此前各轮记录。

本轮在 `CFL3D/` 修改；旧的 `CFL3D-f2py-CGNS-source/` 保留作为参考。
MPI 已通过下述内存网格验收；本轮又补齐 CGNS/ADF 文件入口的串行和 MPI 验收。
较大生产算例和更多边界类型仍需单独验收。

## 已实现的接口

- 串行 `_native` 与并行 `_native_mpi` 独立构建；支持 GNU Fortran、Intel Fortran Classic 和 Intel MPI。
  默认直接编译 f2py 生成的 C 接口，不需要 Meson/Ninja。
- `CFL3D(..., comm=mpi4py_comm)` 调用原生主控/计算进程路径。
  Python 管理 MPI 生命周期；求解使用 communicator 副本，不自行 Init/Finalize。
- communicator 的 rank 0 读取并广播配置和网格；计算进程按完整 zone 分配。
  仅拥有者提取流场，再向所有 rank 汇总完整结果。
- rank 0 写 `solution.npz`；计算进程的原生输出使用独立子目录。
- Python 配置错误集体传播；原生 MPI 错误使用 MPI_Abort，暂不支持恢复。
- Python 新增 `cfl_ramp_steps`、`cfl_ramp_factor`、`first_order_iterations`、
  `coarse_first_order_iterations`、`restart_interval`，并检查取值。
  SA/SST 验证日志确认 IFLAGTS=3、FMAX=2、NITFO=1/1/2、NWREST=2。
  restart_interval 当前只控制写出，不支持读入接续。

## 修复的两个原生 MPI 问题

1. `mit` 应为五个多重网格层 × 五个网格序列阶段，旧声明却为 `mit(5,maxbl)`。
   MPI 固定打包/解包 25 个整数，总块数小于五时越界读写。
   已统一相关分配、形参和 COMMON 为 `mit(5,5)`，初始化并修正内存计数。
2. `mgbl.F` 计算通信缓冲区长度时使用 `nkey`，却缺少 `common /key/ nkey`。
   隐式局部变量未初始化，导致缓冲区长度检查和清零循环使用不确定值。
   已补上 COMMON 声明，与 main/readkey/trnsfr_vals 一致。

Valgrind 先后定位到上述越界和未初始化值。修复后的双块、两级多重网格测试中，
三个 MPI 进程均报告 `ERROR SUMMARY: 0 errors from 0 contexts`。
本项检查使用 `--leak-check=no`，覆盖非法访问和未初始化值，不构成泄漏验收。

## 验收范围与结果

环境由 `/home/mdolabuser/.bashrc_mdolab` 加载：Python 3.12.12、NumPy 2.3.5、
已有 mpi4py、GNU Fortran 13、Intel Fortran Classic、Intel MPI 2021.13。
未下载或安装依赖。测试使用内存 CGNSMesh。

算例包括双块均匀 Euler 流和双块 SA/SST 平板；SA/SST 使用三级多重网格与网格序列，
每个粗阶段三步、细阶段五步，共十一行历史，并启用新增计算参数。
每个作业内连续求解 Euler、SA、SST，逐 rank 比较流场与历史，求解后继续执行 MPI barrier。
均匀 Euler 的近零残差归一化受舍入影响，因此只比较完整流场和七列力/力矩历史；
SA/SST 比较全部八列历史（包括残差）。

| 检查 | 结果 |
| --- | --- |
| GNU 串行 debug 构建及既有内存网格回归 | 通过 |
| GNU + Intel MPI debug 构建 | 通过 |
| Intel Fortran Classic + Intel MPI debug 构建 | 通过 |
| 2/3 进程 Euler、SA、SST 与 GNU 串行逐元素比较 | 通过 |
| 四个 WORLD rank 分成两个独立的二进程 communicator | 两组均通过，不依赖 WORLD 主控 |
| 同作业连续求解、求解后 MPI barrier | 通过 |
| root 无效参数向所有 rank 传播，随后继续求解 | 通过 |
| 原生负体积触发 MPI_Abort | Intel 返回状态 1，GNU 返回状态 255，均无超时 |
| 三进程 Valgrind，双块 Euler 多重网格 | 每个进程 0 个错误；未检查泄漏 |

GNU MPI 与 GNU 串行的比较项最大差异为 0。
Intel MPI 扩展与 GNU 串行的 SA 流场最大绝对差异为 `2.031e-15`，
历史最大绝对差异为 `2.809e-15`；Euler/SST 比较项差异为 0。
这些是短程回归结果，不代表已完成长时间收敛或大规模并行验证。

## 可复现命令

```bash
cd /CFL3d_temp/CFL3D
source /home/mdolabuser/.bashrc_mdolab
export PYTHONPATH="$PWD:$PWD/tests/python"
python tools/build_python.py --debug --jobs 8
python tools/build_python.py --mpi --debug --jobs 8
CFL3D_VALIDATION_DIRECTORY=build/mpi-validation-gnu bash tools/validate_mpi.sh

# 切换到 Intel Fortran，保留独立验收结果目录。
python tools/build_python.py --mpi --compiler-family intel --debug --jobs 8
CFL3D_VALIDATION_DIRECTORY=build/mpi-validation-intel bash tools/validate_mpi.sh
```

脚本准备串行基准，依次运行二/三进程、子通信器、Valgrind（若已安装）和原生错误退出检查。
默认采用 `I_MPI_FABRICS=shm`，各作业带超时保护。
当前安装的串行和 MPI 扩展均为 GNU debug 构建；Intel debug 构建也已完成上述验收。
切换编译器会替换 `_native_mpi.so`，应在 MPI 作业结束后构建，并启动新的 Python 进程测试。

结果与日志：

- GNU：`build/mpi-validation-gnu/`；逐项报告在 `mpi-2/report.json`、`mpi-3/report.json` 等目录。
- Intel：`build/mpi-validation-intel/`；结构相同，`logs/` 保存内存检查与错误退出日志。
- 测试实现：`tests/python/validate_mpi.py`、`tools/validate_mpi.sh`。

## 环境问题与剩余边界

新容器的实际 `/dev/shm` 已扩大到 1 GiB，普通 MPI 测试正常运行。
先前 Valgrind 默认退出泄漏扫描扫描大型 MPI 共享映射，超时后有测试进程遗留并占满共享内存。
已终止本次遗留进程，并关闭泄漏扫描；后续完整验收正常退出。
沙箱本身禁止 MPI 启动器创建监听 socket，多进程测试需在沙箱外运行。

当前每个 communicator 至少两个进程，rank 0 为主控，计算进程数不得超过 zone 数。
尚不支持自动 zone 切分、空闲计算进程、分布式 Python 流场存储、非定常、
restart 读入接续或原生 MPI 错误后的恢复。

现有 CGNS 4.5 库关闭了 HDF5，但支持 ADF。本轮已使用该库完成文件入口适配，无需 h5py。
下一阶段可逐项扩展分方向格式/黏性选项、逐阶段网格序列和 restart 接续。

仓库已有 CRLF/执行权限差异保留；本版提交只纳入 Python/MPI/CGNS 适配相关文件及文档。


## 第三轮：真实 CGNS/ADF 文件入口

新增 `tools/build_cgns.py` 和 `source/python/cgns_bridge.c`，使用已有 CGNS_HOME 编译 ABI 桥，
由头文件确定索引宽度和枚举值。Python 的 `_cgns_native.py` 通过 ctypes 调用 CGNS API，
读取坐标、BC/数值 BCDataSet 和一对一连接，生成现有 CGNSMesh。
核心计算依然通过 f2py 扩展，不修改原生求解器的网格入口。

`CGNSMesh.from_file(path, base=..., backend="auto")` 可直接读 ADF；
显式 `backend="cgns"` 强制原生 API，`backend="h5py"` 保留旧的 HDF5 路径。
h5py 改为可选依赖。没有下载、安装或重编译 CGNS 库。

### 本轮实测

- 全部单元测试共 15 项：12 项通过，3 项 HDF5 专项因未安装可选 h5py 明确跳过。
- 原生读取测试 8 项通过：单块/双块/真实平板坐标和边界逐项还原、
  带负号的一对一 Transform、多 base 选择、热流数据集，以及非法文件、
  PointList、CellCenter 和周期连接的拒绝检查。
- 用官方 CGNS API 写出双块 Euler、SA、SST 的真实 ADF 文件，随后通过 CFL3D 文件名入口读入。
  文件串行、2/3 进程和两组独立子通信器均与内存串行基准一致，比较项最大绝对差异为 0。
  均匀 Euler 比较完整流场和系数历史；SA/SST 包含全部残差历史。
- root 缺失文件的读取错误传播到所有 rank，随后仍能完成正常求解。
- 三进程 Valgrind 的 ADF 双块 Euler 测试各进程为 0 个错误，未检查泄漏。
- 官方 cgnscheck 检查平板文件完成，无错误；提示 8 个可选元数据警告（family/dataclass 未设置）。

复现：

```bash
source /home/mdolabuser/.bashrc_mdolab
python tools/build_cgns.py
PYTHONPATH=.:tests/python python -m unittest test_cgns_native -v
CFL3D_VALIDATION_DIRECTORY=build/cgns-validation bash tools/validate_mpi.sh --adf
```

产物在 `build/cgns-validation/`：`meshes/{box,sa,sst}.cgns` 为真实 ADF 文件；
`serial-adf/report.json`、`mpi-2/report.json`、`mpi-3/report.json` 及各组目录保存比较结果，
`logs/` 保存 MPI 和 Valgrind 日志。当前求解扩展为 GNU debug 构建。
本轮 ADF 路径未单独切换 Intel Fortran 核心复测；此前 Intel 的内存网格验收仍有效。

原生读取器保留结构化、3D base（可为挤出的二维网格）、Vertex PointRange、
单个 GridCoordinates 和一对一连接的范围。
尚不解析真正二维 base、PointList、非零 rind、一般/overset/周期/平均接口或多时刻连接。
FamilySpecified 仍需显式边界覆盖，未增加 family 到边界类型的解析。
原 HDF5 读取器保留，本环境无 h5py 且 CGNS 库不支持 HDF5，本轮未复测 HDF5。

维护备注：MPI 生命周期、错误同步、独立输出目录、CGNS 索引宽度及数组顺序、
mit 固定通信布局和 nkey 初始化原因已在对应代码处解释。编译产物和验证输出不纳入提交。


## 第四轮：定升、分方向格式和多重网格计算控制

本轮补齐上一轮列出的参数缺口，eddy_viscosity_limit 默认继续保留为用户指定的 1e5。

- 定升：target_cl、alpha_relaxation、alpha_step_limit、alpha_update_interval 映射到
  原生 CLTARG/RLXALPH/DALIM/ICYCUPDT，直接使用原生攻角反馈和 MPI 广播。
- 分方向：limiter、flux、inversion、kappa 接受标量或 i/j/k 三元组。
- 多重网格：新增 multigrid_up_iterations（MTT）、全局 multigrid_smoothing（MIT）和
  不可变 MultigridStage 阶段表，每阶段独立设置 NCYC/MGLEVG/NITFO/MIT。
  阶段细化顺序遵循原生规则；不支持任意跳级、embedded 网格或逐 zone 的不同格式设置。
- 输出：SolverResult.final_alpha 与 NPZ 记录实际最终攻角；修复 unit 27 的日志刷新和重用。
- 原生 ABI 升到版本 2；Python 会拒绝旧扩展，并明确提示重建和重启解释器。

验证使用 GNU debug 串行/MPI 扩展与现有 Intel MPI，没有安装新依赖。

- 保留原有 Euler/SA/SST 真实 ADF 回归。
- 无承力壁面的均匀流测试：初始 alpha=1 度、target_cl=0.1、每两步更新、限幅 0.02 度，
  六步后实际攻角为 1.06 度；随后关闭定升并复用目录，确认状态和日志重置。
- SST 平板定升：target_cl=0.01，200 步后 CL=0.0104563329803824、alpha=0.0689091483339882 度。
  这证明反馈链路有效；不构成充分收敛或普适 CL 容差保证。
- 2/3 进程及两个独立的二进程子通信器：所有比较项与串行基准差异为 0，最终攻角一致。
- 自定义阶段采用两阶段、不同 MIT、MTT=1 和不同方向的 IFLIM/IFDS/IDIAG/RKAP；
  三进程 Valgrind 各 rank 零错误，未检查泄漏。原生负体积退出仍无超时。
- 单元测试共 21 项：18 项通过，3 项可选 HDF5 测试跳过。
- 单元回归还覆盖参数非法值、标量/三元组等价、V/W 循环和全局 MIT 与显式活跃层设置等价。
- 串行 CLI 验证新开关后返回 final_alpha=1.06 度。

复现：

```bash
source /home/mdolabuser/.bashrc_mdolab
export PYTHONPATH="$PWD:$PWD/tests/python"
python tools/build_python.py --debug --jobs 8
python tools/build_python.py --mpi --debug --jobs 8
python -m unittest discover -s tests/python -v
CFL3D_VALIDATION_DIRECTORY=build/mpi-controls-validation bash tools/validate_mpi.sh --adf --controls
```

新增报告位于 build/mpi-controls-validation/ 下的 serial-adf/report.json、mpi-2/report.json、
mpi-3/report.json、mpi-2-group-*/report.json；logs/ 保存 MPI、Valgrind 和错误退出记录。
本轮尚未切换 Intel Fortran 编译器重复新参数验收；现有 MPI 库仍为 Intel MPI。

## 第五轮：CGNS 分组、力矩与压差/黏性阻力

新增 `CFL3D.getSolution(groupName=None)`、`getGroupNames()` 和 `addGroups({...})`。
默认按 CGNS Family、zone、BC 或 zone/BC 路径选择表面；自定义集合成员为 CGNS 名称列表，
集合名字不得与已有分组重名。重叠表面只计一次。
ADF 与 HDF5 读取器保留 primary/additional Family；当前实际复测 ADF。
单一 FamilyBC 的 FamilySpecified 可自动解析，其他情况仍需显式覆盖。

原生 `force.F` 在每个边界分段前后记录积分增量，保留最细网格结果；
MPI 仅由块拥有者写入，再向全部 rank 汇总。继续通过 f2py 返回 Python，原生 ABI 升至 3。
结果缓存在各 CFL3D 对象中，查询不参与 MPI 通信，可只在 rank 0 执行。
返回 CL/CD/CY/Cmx/Cmy/Cmz/CDp/CDv，以及 L/D/Y/Dp/Dv（N）和 Mx/My/Mz（N·m）。
CDp 为压差阻力，CDv 为黏性阻力；两者来自原生积分，和等于 CD。
NPZ 额外保存逐边界系数和名称，原有八列 history 格式保留。

有量纲输出要求 `freestream_pressure` 为 Pa、坐标及参考长度为 m、参考面积为 m²；
按 `q∞=0.5*gamma*p∞*Mach²` 换算。不设置压力可正常求解，但 getSolution 会明确报错。
具体坐标定义、参考长度和示例见 docs/python-wrapper.md。

三维 SST 测试定位并修复旧 twoeqn.F 的 DES 数组越界：逻辑 AND 不保证短路，
普通 SST 也可能读取没有完整分配的 fdsav。现用嵌套条件隔离 DES 数组访问，
保留 DES 模式原有公式。

本轮环境为 GNU debug 串行/MPI 扩展及现有 Intel MPI，无新依赖下载。

- 全部单元测试 26 项：23 项通过，3 项可选 HDF5 测试跳过。
- 分组测试覆盖真实 ADF Family、额外 Family、集合去重、分部件相加、CDp/CDv、SI 缩放、
  多个求解对象缓存独立性，以及三维 SST 力矩中心平移和 y-up 旋转。
- 保留六个 Euler/SA/SST、显式阶段、定升算例；真实 ADF 串行、2/3 进程及两个独立
  二进程子通信器均通过。流场、历史和逐边界系数的比较项最大绝对差异为 0。
- 仅 root 增加集合和读取载荷后，各 rank 继续通过 barrier，确认查询无需 collective。
- 三进程 Valgrind 自定义 MIT / 块矩阵求逆 / ADF 分组算例，各 rank 零错误，未检查泄漏。
- 三维 SST 力矩中心与 y-up 测试完成 Valgrind 检查，零错误，未检查泄漏。
- 原生负体积仍通过 MPI_Abort 退出，返回 255，无超时。
- 串行示例实测输出 CDp/CDv 及 Dp/Dv；两个示例均新增压力、参考量、分组命令行开关。

复现：

```bash
source /home/mdolabuser/.bashrc_mdolab
export PYTHONPATH="$PWD:$PWD/tests/python"
python tools/build_python.py --debug --jobs 8
python tools/build_python.py --mpi --debug --jobs 8
python -m unittest discover -s tests/python -v
CFL3D_VALIDATION_DIRECTORY=build/mpi-forces-validation bash tools/validate_mpi.sh --adf --controls --forces
```

结果在 `build/mpi-forces-validation/`，目录结构与此前回归相同。
这些短程回归验证接口和积分一致性，不代表生产算例充分收敛。
本轮未切换 Intel Fortran 编译器复测，也未复测 HDF5 文件路径。


## 第六轮：options 字典与独立输出

新增 `CFL3D(options={...})` 和 `CFL3D(options_dict)`，兼容原有 SolverOptions / mesh 参数。
input_grid、output_directory 与三个开关统一由 options 管理：writevolumesolution 默认 True，
writeTecplotSurfaceSolution 默认 True，writeflowsolution 默认 False。字典拷贝并校验，
支持 list 方向参数和多重网格阶段字典列表，未知键明确报错。

- CGNS/ADF 体结果包含原始顶点、结构化块、边界、连接和 CellCenter primitive fields，
  同时写 Temperature（K）和 CFL3DUnits 量纲说明。
- Tecplot ASCII 壁面文件包含 X/Y/Z、Cp、T(K)、Cfx/Cfy/Cfz、y+。
  顶点为原网格点，流动量为面单元中心。原生 force.F 在最细网格上缓存壁面状态和剪切力；
  y+ 使用第一层单元中心法向距离 V/(2A)，不是旧日志的节点平均统计。
- PLOT3D .x/.q 采用原生 IPTYPE=1、64 位实数、32 位记录标记和 IBLANK。
  缓存序列化与原生 plot3d.F 的排列一致，支持二维和三维及 y-up。
- writeVolumeSolution / writeTecplotSurfaceSolution / writeFlowSolution 可在求解后独立调用。
  快照与原网格、返回数组以及另一个 solver 的原生状态隔离，补写不重新计算。
- 三个开关全关时保留历史、积分元数据与原生日志，不写体/表面解及 restart 数据。
  MPI 所有 rank 仍缓存完整结果，仅 root 写文件；补写无需 collective。
- 原生接口升到版本 4，已重建 GNU debug 串行/MPI 扩展和独立原生程序。

验证：全部 30 项单元测试中 27 项通过，3 项可选 HDF5 测试跳过。
覆盖字典配置、开关独立性、补写、缓存隔离、CGNS 流场读回、恒温壁面、Cf 再积分和 y+，
以及三维 i/j/k 壁面与 y-up 转换。
真实 ADF 双块的 2/3 进程和两个独立子通信器中，壁面六列数据、流场、历史和积分均与串行一致；
root 补写的 Tecplot / PLOT3D 文件也与串行逐字节一致。
独立原生 CFL3D 的 SST 平板 IPTYPE=1 对比：.q 差异为 0；.x 坐标最大差异 2.22e-16，
IBLANK 和所有记录大小一致。测试实现为 tests/python/validate_plot3d.py。

结果目录为 build/mpi-io-validation/，测试命令和新接口使用说明见 README.md 与 docs/python-wrapper.md。
本轮没有下载依赖，现有 Intel MPI 与 CGNS/ADF 库继续复用。

三进程 Valgrind 的新 I/O / 壁面缓存 / 补写测试各 rank 零错误，未检查泄漏。
既有六算例的真实 ADF / 定升 / 自定义阶段 / 分组积分 MPI 回归也再次通过，
记录位于 build/mpi-io-regression/；原生负体积 MPI_Abort 返回 255，没有超时。


## 第七轮：升力分布与截面 Cp

CFL3D 对象新增 addLiftDistribution(nSegments, direction, groupName=None)、
addSlices(direction, positions, sliceType="relative", groupName=None)，
getLiftDistribution(index=0)、getSlices(index=None)、clearSurfaceAnalysis()，
以及 nLiftDist / nSlice 计数。可在 solve 前或后注册，支持 CGNS 与自定义去重集合。

- 原生 force.F 保存每个 wall 面单元的压力/黏性积分载荷，以及实际力积分时的 alpha/beta。
  结果通过独立 f2py getter 返回；MPI 按面所有者汇总，并从有效计算进程广播力积分角度。
  API 升到 5，当前环境已重建 GNU debug 串行和 MPI 扩展。
- 升力分布按三角裁剪面积守恒分配原生逐面载荷，提供条带 CL/CD/CDp/CDv，
  对应 dC/ds [1/m]；设置自由来流静压时另提供 dL/ds、dD/ds、dDp/ds、dDv/ds [N/m]。
  定升最后一次反馈更新后的 final_alpha 不用于重新投影已积分载荷。
- positions 对 relative / absolute 均为输入网格轴上的实际坐标 [m]。
  relative 保存注册时的三角面交点重心坐标，后续网格变化时追踪材料位置；
  absolute 每次重新与固定平面求交。当前用于稳态计算之间的几何变化，不涉及非定常移动网格求解。
- 截面返回连接线段及面中心 Cp/T/Cf/y+，合并共形块的重合端点和重复线段。
  与平面共面的面不产生线段；无交点时返回空数组；沿共享边时面积加权取相邻面值。
  尚未实现非匹配表面或 overset zipper。
- writeTecplotSurfaceSolution() 在主壁面文件旁写 *_lift.dat / *_slices.dat。
  自动开关关闭时仍可注册、查询与补写，MPI 可仅 root 注册和输出，无额外 collective。

验证：37 项单元测试中 34 项通过，3 项可选 HDF5 测试因没有 h5py 跳过。
新增 7 项测试覆盖梯形面解析分布、翘曲面原生载荷守恒、共形跨块截线、共享边去重、
拓扑检查、定升积分角度、原生面中心 Cp、SI 换算、relative/absolute 网格平移追踪、
自动/延迟输出、三维 i/j/k 壁面、y-up 与非零侧滑角。
真实 CGNS/ADF 的 2/3 进程及两个独立二进程通信器中，逐面载荷、流场、历史、
分布、截面均与串行逐元素一致，主表面及两个分析 Tecplot 文件逐字节一致。
三进程 Valgrind 使用 3 个条带以降低几何后处理开销，各 rank 零错误；未检查泄漏。
完整 37 条带的 Valgrind 曾超过 90 秒时限，未将该超时作业计为通过。

```bash
python -m unittest discover -s tests/python -v
PYTHONPATH=.:tests/python mpirun -np 2 python tests/python/validate_solution_io_mpi.py --directory build/mpi-slices-validation
PYTHONPATH=.:tests/python mpirun -np 3 python tests/python/validate_solution_io_mpi.py --directory build/mpi-slices-validation
PYTHONPATH=.:tests/python mpirun -np 4 python tests/python/validate_solution_io_mpi.py --split --directory build/mpi-slices-validation
# Valgrind 需使用真实 Python 可执行文件，不要传 pyenv shell shim。
# python_executable 应替换为当前解释器的 sys.executable。
OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:tests/python mpirun -np 3 valgrind --leak-check=no --error-exitcode=97 python_executable tests/python/validate_solution_io_mpi.py --segments 3 --directory build/mpi-slices-small-memcheck
```

本轮日志在 build/mpi-slices-validation/logs/，测试源为 tests/python/test_surface_analysis.py
和 tests/python/validate_solution_io_mpi.py。使用方法与单位见 README.md 和 docs/python-wrapper.md。
本轮未下载依赖；没有新增 commit。


## 目录布局调整 — 2026-10-16

将 Python 包从 python/cfl3d/ 迁移到仓库根目录 cfl3d/（flat layout），包内 Python 源码逐字节保持不变。
pyproject.toml 显式限定只发现 cfl3d 与其子包，避免将 source/build/docs 等目录识别为 Python 包。
串行/MPI 扩展及 CGNS 桥的构建目标同步移到 cfl3d/；验证脚本与文档的 PYTHONPATH 改为仓库根目录。
source/python/、tests/python/ 和 examples/python/ 的用途与位置保持不变。

验证使用 /tmp 下的临时环境，复用已有 NumPy、mpi4py、setuptools 和 pip，没有下载依赖。
editable 安装成功，在仓库外、不设置 PYTHONPATH 时仍能导入新目录的 cfl3d，顶层打包名称仅为 cfl3d。
GNU debug 串行/MPI 扩展及 CGNS 桥已通过实际构建并写入新位置。
加载现有 Intel/CGNS 动态库环境后，37 项测试中 34 项通过，3 项可选 HDF5 测试跳过；
真实 CGNS/ADF 三进程的流场、壁面载荷、升力分布、截面及延迟输出与串行一致。
标准样例在仓库外可正常加载；验证日志位于 build/mpi-flat-layout-validation/logs/。
已有 editable 安装需在迁移后重新安装；手动指定的 PYTHONPATH 应改为仓库根目录。

随后已按要求在当前 Bash Python 环境（/home/mdolabuser/.pyenv/versions/3.12.12/bin/python）
重新执行 editable 安装，没有下载或更新依赖。在仓库外且不将仓库加入 PYTHONPATH 时，
确认导入来自根目录 cfl3d/，串行/MPI API 均为 5，CGNS 桥可加载；
原生载荷回积分、截面、延迟输出和网格追踪的集成测试通过。


## 第八轮：自动 MPI、rank-local options 与完整状态接续

- 省略 comm 时，检测 MPI 启动环境或已初始化的 mpi4py，多个 WORLD rank 自动使用 MPI 扩展；
  普通 Python 使用串行，显式 comm=None 强制串行。单 rank 通信器使用串行核心。
  显式子通信器的所有集体操作仅发生在该 comm 内；成员检查使用本地 MPI group 映射，
  不对 WORLD 做集体通信。仍采用一个 host、其余进程按完整 zone 分配。
- 每个 rank 的显式 options 独立拷贝并校验，不用 root 的值覆盖。为兼容旧脚本，
  非 root options=None 可继承 root 初始配置。构造及 solve 前检查当前 comm 内参数一致性，
  网格/边界生成的原生数组和缓存状态也分别检查。
- solver.options 为可修改 RuntimeOptions，key 大小写无关，支持属性和字典访问，
  setOption/getOption/setOptions；多参数修改整体校验，失败时不部分写入。
  solver.options.check() 与 solver.checkOptions() 在当前 comm 内集体检查。
  新参数于下一次 solve 入口整体传入 Fortran，支持追加迭代和修改 CFL。
  input_grid 在构造时加载，后续切换文件须建立新对象。
- 每个对象缓存完整原生 restart 载荷及 primitive fields，不依赖旧输出目录或活跃 COMMON。
  重复 solve 默认接续；getFlowState/setFlowState 读取和恢复完整快照，resetFlow 重新冷启动。
  当前每段仍通过内部 .cfl3d-state.bin 文件与 Fortran 交换，成功后清理；尚未实现无文件的内存传递。
- writeRestartFile 默认写 restart.npz（全状态、元数据、校验和、primitive fields）；
  options.restartFile 可读该文件或原生 real64/int32/原生端序/含 ghost 的 restart.bin。
  Python 检查点检查拓扑、模型、gamma、y-up；原生文件检查块尺寸、模型、记录格式、
  历史/流场/ghost 记录长度及主流场有限性、密度/压力正值。
  visualization solution.npz 不用于接续。模型/坐标约定等不兼容时需 resetFlow。
- 接续时只使用最终网格序列阶段，返回本段历史。每段重新应用其 CFL 渐增、
  一阶迭代和定升反馈计数，所以启用这些控制时分段点可能影响计算路径。
- 修正 rrest.F 在 Python 负 IREST 重启中，用旧历史长度发送新 MPI 数组的潜在越界；
  Python 分段历史独立保存，旧 ntt 在 MPI 发送前置零，完整湍流/ghost 状态仍通过原生读取。
  原生 API 升到 6，已重建 GNU debug 串行及 MPI 扩展。

验证：41 项单元测试中 38 项通过，3 项可选 HDF5 测试跳过。新增 4 项覆盖参数大小写、
独立副本、原子修改，五种模型的 5+2 步接续与连续 7 步比较，CFL 修改、完整文件与原生文件
恢复、快照安装、冷启动重置、模型不兼容/截断/误用后处理文件，以及多重网格序列不重复。
三进程自动 MPI、四进程中仅 WORLD rank 1/2/3 参与、四进程两个独立子通信器均通过；
覆盖参数不一致/状态不一致的集体报错、较短迭代段、文件跨串行/MPI 接续。
流场、历史和逐面载荷与串行基准逐元素一致。既有三进程升力分布/截面/延迟输出回归也通过。
三进程自动 MPI 续算 Valgrind 各 rank 零错误，未做泄漏扫描。

```bash
python -m unittest discover -s tests/python -v
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_runtime_mpi.py --auto --directory build/mpi-runtime-validation
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_runtime_mpi.py --exclude --directory build/mpi-runtime-validation
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_runtime_mpi.py --split --directory build/mpi-runtime-validation
```

测试源为 tests/python/test_runtime.py、tests/python/validate_runtime_mpi.py；
日志在 build/mpi-runtime-validation/logs/，现有 I/O 回归结果在 build/mpi-runtime-io-validation/。
新增行为和状态交换限制已写入 README.md / docs/python-wrapper.md；标准模板默认支持自动 MPI。
本轮未下载依赖。

标准 standard_case.py 未提前导入 mpi4py、未传 --mpi 的三进程启动也已通过，确认自动识别 MPI 启动环境的分支可用。


## 第九轮：算例名、逐次编号和 Tecplot 分析自动输出

- CFL3D 增加 name 关键字和属性，options.name 也可设置，默认 out；MPI 下纳入当前 comm 的一致性检查。
  每次完成原生求解分配一个从 000 开始的编号，result.name/solve_index、solver.nSolve
  和 result.output_paths 可用于读取身份与输出目标路径，NPZ 中也保存 name 和 solve_index。
- 自动文件使用同一前缀，例如 out_000_surf.dat、out_000_slices.dat、out_000_lift.dat、
  out_000_volume.cgns、out_000_flow.x/.q、out_000_solution.npz、out_000_restart.bin。
  writeRestartFile 默认补写 out_000_restart.npz。表面、截面、分布均使用 Tecplot ASCII，分析固定 .dat。
- volume_filename、surface_filename、flow_prefix 默认 None，指定时作为 name_编号 后的自定义标签；
  显式补写路径仍由调用者控制。默认补写使用最近结果的原名称和编号，不受随后修改 name 影响。
- 只要已经 addSlices/addLiftDistribution，每次 solve 后自动写分析，即使主壁面输出开关关闭。
  writeTecplotSurfaceSolution 仅控制主壁面自动输出；writeSurfaceAnalysis 可独立补写。
  clearSurfaceAnalysis 清除注册，下一次不再生成对应新编号的分析文件。
- 非空原生日志也保存编号副本，固定日志名仍提供最近一次内容。配置检查失败不占号；
  原生计算完成但输出失败时该号保留，可补写，后续继续递增。新对象从 000 开始，
  不同对象应使用不同 name 或输出目录，避免同名覆盖。
- 根据用户决定继续采用原生 restart 方式：缓存最细层完整状态，续算重建粗层工作数据，
  保留 V/W 循环，不重复首次粗到细网格序列。没有实现保存全部粗层工作数组。
  resetFlow 一行清除续算状态及 restartFile，下一次重新冷启动/执行首次网格序列；
  保留分析注册、已输出文件和递增编号。

验证：43 项测试中 40 项通过，3 项可选 HDF5 测试跳过。新增 2 项覆盖自动分析不受主壁面
开关影响、相同目录连续求解编号、旧结果字节不变、原生接续与连续迭代一致、
resetFlow 冷启动且编号不重置、name 修改后的快照隔离、失败不占号、标签和独立补写。
多重网格序列测试另外验证冷启动 7 步、续算 2 步、清空缓存后重做粗层准备共 6 步。
三进程及两个独立二进程子通信器内，连续两次求解的结果/编号和 Tecplot 文件与串行一致，
第二次分析文件自动生成，第一批文件保持逐字节不变。标准模板 --name wing 实测生成 wing_000_solution.npz。

```bash
python -m unittest discover -s tests/python -v
PYTHONPATH=tests/python mpirun -np 3 python tests/python/validate_solution_io_mpi.py --directory build/mpi-numbered-validation
PYTHONPATH=tests/python mpirun -np 4 python tests/python/validate_solution_io_mpi.py --split --directory build/mpi-numbered-validation
```

日志在 build/mpi-numbered-validation/logs/，规则和示例见 README.md / docs/python-wrapper.md。
本轮只修改 Python、测试及文档，未修改 Fortran 或下载依赖；原生 API 仍为 6。

## 第十轮：原生日志直接命名与同编号追加（2026-10-16）

- 原生 API 7，GNU debug 串行及 Intel MPI 扩展均已重建。计算从开始即写
  `<name>_<index>.out/.res/.turres/.alpha`，预处理日志为 `<name>_<index>_precfl3d.out`。
  取消固定名日志的事后复制，保留原有后缀；MPI 各 rank 的工作目录仍独立。
- 新增大小写无关的 options.numberSolutions（默认 True）。False 时同一对象的缓存续算
  沿用上一输出编号：日志及收敛历史追加，显示迭代号连续；流场/分析/restart 更新为最新状态。
  result.history 保留当前段，cumulative_history 为本编号累计历史，iteration_offset 表示起点。
  NPZ 保存累计 history 和本段 chunk_history。nSolve 独立统计完成的调用次数。
- 新流场、切换输出身份、新建对象都避免认领旧文件；分配新编号时跳过已存在文件。
  resetFlow 后即使关闭编号递增，也不会把冷启动接入旧日志。restartFile 不自动附加磁盘旧日志。
- 修复 CGNS 互为反向端点的连接声明被重复计数的问题；先校验映射和坐标，再按几何范围去重。
- 全套 Python 测试 46 项，43 项通过、3 项可选 HDF5 测试跳过。三进程 MPI 和两个独立
  双进程通信器的流场、分析文件、编号和追加历史与串行一致。
- 用户 L2 半翼（9 块、24,192 单元）完成两个 50 步求解，共用 cruise_000，日志/历史延续到 100。
  此前 1000 步单层结果未改动；原始网格未改动。review 文件保存在 ../case_test/。

验证日志：build/native-output-full-tests.log、build/native-output-mpi3.log、
build/native-output-mpi-split.log。源码与说明已更新，未下载依赖。

## 第十一轮：动态参数与分类初始化策略（2026-10-16）

- SolverOptions 移除 frozen=True，支持校验后的属性赋值和多参数 update。
  RuntimeOptions 直接保存运行参数字典，校验候选值后整体替换，不在更新时重建 SolverOptions。
  历史配置通过独立副本隔离；修改已返回快照不会改变运行参数或旧结果。
- 新增 getOptionCategories()，物理模型/尺度和网格改变默认重新初始化；数值控制、积分参考量、
  输出和可调工况保留最细层流场。判断在 solve 前进行，以最后成功配置为基准，改了又改回不重置。
  input_grid 支持重新加载及 MPI root 广播，内存几何/边界/连接改变也会触发初始化。
- 原生 API 8，显式修改 alpha 时覆盖 restart 中旧的定升攻角；未修改时保留原生定升接续。
  GNU debug 串行与 MPI 扩展均已重建。beta 和参考量更新也通过续算回归验证。
- result/NPZ 新增 start_mode、changed_options、reset_reasons。自动重置打印原因，使用新编号，
  不把冷启动接入旧日志。原有 restartFile 未变时不会重新载入来抵消物理参数重置。
- 完整回归 52 项，49 项通过、3 项可选 HDF5 跳过；最终 8 项针对性检查通过，包含新增的
  旧 restartFile 不抵消重置的测试。三进程自动 MPI 和两个独立双进程算例的参数更新、
  显式定升攻角、物理重置及网格重载与串行一致。

日志：build/native-output-dynamic-full.log、build/native-output-dynamic-final-targeted.log、
build/native-output-dynamic-mpi3.log、build/native-output-dynamic-mpi-split.log。
本轮未下载依赖。

## 第十二轮：gamma-Re_theta 四方程转捩接口（2026-10-19）

- Python 新增 turbulence="gamma-theta"，使用原生 IVISC=40 的 Langtry-Menter
  gamma-Re_theta v1.1；未改写模型方程，不与原生 IVISC=30 三方程模型混用。
- 新增 freestream_turbulence_intensity（百分数）、freestream_eddy_viscosity
  （mu_t/mu_inf）和 keep_ambient_turbulence。这三个设置用于 komega/SST/gamma-theta，
  默认沿用原生初始化；修改时按物理参数策略重新初始化流场。
- 原生接口升级为 API 9，将环境湍流设置通过 f2py 传入 readkey.F。串行和 Intel MPI
  扩展均已重编译；接续计算与文件 restart 保留全部四个输运变量和边界状态。
- 修复 foureqn.F 的 RANS 分支可能读取不存在的 DES 数组 fdsav 的问题：Fortran 的
  .and. 不保证短路，改为先判断 ides、再访问数组。原问题在调试版原生对照中触发崩溃。
- 修复后的调试版原生程序与 Python 在三层多重网格/网格序列的平均流场、收敛历史和
  四个模型变量完全一致，最大差值为 0。完整回归 59 项，56 项通过、3 项 HDF5 跳过。
- 三进程自动 MPI 的缓存/文件续算、显式工况修改、物理重置和网格重载均与串行一致；
  四方程状态也逐值相同。新增三维计算、模型切换和环境参数校验测试。
- 当前验证证明接口、数值执行及状态接续一致，尚未完成转捩位置的实验验证。
  体场可视化仍只导出已有平均流变量，暂未增加 gamma/Re_theta 独立可视化字段。

最终日志：build/native-output-gamma-theta-final-tests.log、
build/native-output-gamma-theta-parity-fixed.log、
build/native-output-gamma-theta-mpi-fixed.log。
可重复原生对照脚本：tests/python/validate_transition_parity.py。
本轮未下载依赖。

## 第十三轮：基础两方程变体（2026-10-19）

- 新增 turbulence="bsl"、"sst2003"、"wilcox2006"，分别映射原生 I_BSL、
  ISST2003、I_WILCOX06；原有 "sst"、"komega" 默认模型不变。
- 用互斥模型名表达变体，不开放互相冲突的原生开关组合。每次配置显式重置全部变体标志，
  切回旧模型不遗留前次设置；Wilcox 2006 保留原生涡伸展修正默认值。
- 原生 API 升至 10，串行和 Intel MPI 扩展均已重编译。环境湍流参数同时支持三个新变体。
- 缓存和 NPZ 重启保持完整模型名。切换变体按物理模型改变冷启动；即使 IVISC 相同，
  不同变体的 NPZ 重启也会拒绝。原生裸二进制头只有 IVISC，不能单凭它辨别变体。
- 新增模型标志隔离、版本切换、文件接续、两输运变量逐值对照测试，并扩展缓存续算测试。
- 调试版原生程序对照覆盖 gamma-theta、BSL、SST-2003、Wilcox 2006；三层多重网格/
  网格序列下的平均流场、历史及输运变量最大差值均为 0。
- 四个模型各自通过三进程自动 MPI 回归，覆盖可变参数、显式攻角、物理重置、网格重载、
  缓存和文件续算，与串行一致。完整回归 62 项，59 项通过、3 项可选 HDF5 跳过。

日志：build/native-output-variants-full-tests.log、build/native-output-variants-parity.log、
build/native-output-variants-mpi-{bsl,sst2003,wilcox2006,gamma-theta}.log。
本轮未下载依赖，其他模型暂未扩展。
