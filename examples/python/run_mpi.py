"""Run collectively: mpirun -np 3 python examples/python/run_mpi.py mesh.cgns.

Rank 0 is the native host; the remaining ranks compute whole CGNS zones.
Use no more workers than zones. Input files are read only by rank 0.
"""
import argparse
from mpi4py import MPI
from cfl3d import CFL3D
from numerical_options import add_numerical_options, numerical_options


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mesh")
    parser.add_argument("--output", default="cfl3d-mpi-output")
    parser.add_argument("--model", default="sa", choices=("euler", "laminar", "sa", "sst", "komega", "gamma-theta", "bsl", "sst2003", "wilcox2006"))
    parser.add_argument("--mach", type=float, default=.5)
    parser.add_argument("--alpha", type=float, default=0.)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cfl", type=float, default=2.)
    add_numerical_options(parser)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    options = dict(input_grid=args.mesh, output_directory=args.output,
                            mach=args.mach, alpha=args.alpha, turbulence=args.model,
                            iterations=args.iterations, cfl=args.cfl, **numerical_options(parser, args))
    solver = CFL3D(options=options, comm=comm)
    try:
        result = solver.solve()
        if comm.rank == 0:
            print(result.coefficients)
            print(f"Final alpha: {result.final_alpha:.10g} degrees")
            if args.freestream_pressure is not None:
                # Cached loads are local: querying only rank 0 cannot strand workers.
                print(solver.getSolution(args.group))
    finally:
        solver.close()  # Collective within comm, including the native host.


if __name__ == "__main__":
    main()
