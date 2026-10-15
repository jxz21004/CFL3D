# CFL3D Python/MPI 第四轮检查点（计算控制）— 2026-10-15

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
export PYTHONPATH="$PWD/python:$PWD/tests/python"
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
PYTHONPATH=python:tests/python python -m unittest test_cgns_native -v
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
export PYTHONPATH="$PWD/python:$PWD/tests/python"
python tools/build_python.py --debug --jobs 8
python tools/build_python.py --mpi --debug --jobs 8
python -m unittest discover -s tests/python -v
CFL3D_VALIDATION_DIRECTORY=build/mpi-controls-validation bash tools/validate_mpi.sh --adf --controls
```

新增报告位于 build/mpi-controls-validation/ 下的 serial-adf/report.json、mpi-2/report.json、
mpi-3/report.json、mpi-2-group-*/report.json；logs/ 保存 MPI、Valgrind 和错误退出记录。
本轮尚未切换 Intel Fortran 编译器重复新参数验收；现有 MPI 库仍为 Intel MPI。
