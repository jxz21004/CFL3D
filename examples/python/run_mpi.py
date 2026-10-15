"""Run collectively: mpirun -np 3 python examples/python/run_mpi.py mesh.cgns.

Rank 0 is the native host; the remaining ranks compute whole CGNS zones.
Use no more workers than zones. Input files are read only by rank 0.
"""
import argparse
from mpi4py import MPI
from cfl3d import CFL3D, SolverOptions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mesh")
    parser.add_argument("--output", default="cfl3d-mpi-output")
    parser.add_argument("--model", default="sa", choices=("euler", "laminar", "sa", "sst", "komega"))
    parser.add_argument("--mach", type=float, default=.5)
    parser.add_argument("--alpha", type=float, default=0.)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cfl", type=float, default=2.)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    options = SolverOptions(mach=args.mach, alpha=args.alpha, turbulence=args.model,
                            iterations=args.iterations, cfl=args.cfl)
    solver = CFL3D(args.mesh if comm.rank == 0 else None, options, comm=comm)
    result = solver.solve(output_directory=args.output)
    if comm.rank == 0:
        print(result.coefficients)


if __name__ == "__main__":
    main()
