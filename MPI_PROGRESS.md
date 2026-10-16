# CFL3D Python/MPI 第七轮检查点（升力分布与截面 Cp）— 2026-10-15

最新升力分布及截面 Cp 接口和验证见文末“第七轮”；前面保留此前各轮记录。

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
