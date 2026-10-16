"""CGNS case template: edit the physical/reference values for your own model.

Serial: python examples/python/standard_case.py mesh.cgns
MPI:    mpirun -np 3 python examples/python/standard_case.py mesh.cgns
"""
import argparse
import json

from cfl3d import CFL3D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", help="Structured CGNS mesh (ADF or supported HDF5)")
    parser.add_argument("--mpi", action="store_true", help="Use a host/worker MPI communicator")
    parser.add_argument("--output", default="standard-results")
    parser.add_argument("--name", default="out", help="Case name for numbered result files")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--multigrid-levels", type=int, default=1)
    parser.add_argument("--target-cl", type=float, help="Enable native lift matching")
    parser.add_argument("--group", help="CGNS/custom group; omitted means all integrated surfaces")
    parser.add_argument("--write-volume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-surface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-flow", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    comm = None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    rank = 0 if comm is None else comm.Get_rank()

    # 工况与参考量只是示例，必须改为实际模型的值；网格坐标采用 m。
    options = {
        "input_grid": args.mesh,
        "output_directory": args.output,
        "name": args.name,
        "writevolumesolution": args.write_volume,
        "writeTecplotSurfaceSolution": args.write_surface,
        "writeflowsolution": args.write_flow,
        "mach": 0.2,
        "alpha": 2.0,                       # 度；定升时为初始攻角
        "beta": 0.0,                        # 度
        "reynolds": 6e6,                    # 每米 Reynolds 数，并非默认按弦长定义
        "temperature": 288.15,              # K
        "freestream_pressure": 101325.0,    # Pa；用于将系数换算为 N / N*m
        "gamma": 1.4,
        "turbulence": "sst",
        "full_navier_stokes": True,
        "cfl": 1.0,
        "iterations": args.iterations,
        "limiter": 2,                       # 也可设为 (i, j, k) 三元组
        "flux": 1,
        "inversion": 1,
        "kappa": 1.0 / 3.0,
        "eddy_viscosity_limit": 1e5,
        "multigrid_levels": args.multigrid_levels,
        "multigrid_cycle": 2,               # 1: V cycle；2: W cycle
        "multigrid_smoothing": (1,) * args.multigrid_levels,
        "target_cl": args.target_cl,        # None: 固定攻角；数值: 定升反馈
        "alpha_relaxation": 0.5,
        "alpha_step_limit": 0.1,
        "alpha_update_interval": 5,
        "y_up": False,                      # False: z 向上；True: y 向上
        "sref": 1.0,                        # m^2
        "cref": 1.0,                        # m
        "bref": 1.0,                        # m
        "moment_center": (0.0, 0.0, 0.0),   # m，输入网格坐标系
    }

    # 可改为 {"Aircraft": ["Wing", "Tail"]}，成员须为网格中已有的 CGNS 名称。
    groups = {}
    # MPI 下各 rank 提供自己的 options，构造/求解时在当前 comm 内检查一致。
    solver = CFL3D(
        options=options,
        groups=groups,
        **({"comm": comm} if args.mpi else {}),
    )
    rank = 0 if solver.comm is None else solver.comm.Get_rank()
    result = solver.solve()
    if rank == 0:
        print("Available groups:", solver.getGroupNames())
        print("Final alpha (degrees):", result.final_alpha)
        # 结果已缓存在各 rank；仅 root 查询无需其他 rank 配合。
        loads = solver.getSolution(args.group)
        print(json.dumps(loads, indent=2, ensure_ascii=False))
        print("Output directory:", result.output_directory)
        # 三个开关全关时，也可在这里或之后单独补写：
        # solver.writeVolumeSolution()
        # solver.writeTecplotSurfaceSolution(groupName=args.group)
        # solver.writeFlowSolution()


if __name__ == "__main__":
    main()
