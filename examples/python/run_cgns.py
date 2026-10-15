"""Usage: python examples/python/run_cgns.py mesh.cgns --model sst --iterations 500"""
import argparse
from cfl3d import CFL3D, SolverOptions


def main():
    p = argparse.ArgumentParser(description="Run CFL3D directly from a structured CGNS mesh (ADF or HDF5)")
    p.add_argument("mesh")
    p.add_argument("--model", choices=("euler", "laminar", "sa", "sst", "komega"), default="sa")
    p.add_argument("--mach", type=float, default=.5)
    p.add_argument("--alpha", type=float, default=0.)
    p.add_argument("--beta", type=float, default=0.)
    p.add_argument("--reynolds", type=float, default=1e6)
    p.add_argument("--temperature", type=float, default=288.15)
    p.add_argument("--cfl", type=float, default=2.)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--multigrid-levels", type=int, default=1)
    p.add_argument("--output-directory", default="cfl3d-output")
    a = p.parse_args()
    result = CFL3D(a.mesh, SolverOptions(mach=a.mach, alpha=a.alpha, beta=a.beta,
                   reynolds=a.reynolds, temperature=a.temperature, turbulence=a.model,
                   cfl=a.cfl, iterations=a.iterations, multigrid_levels=a.multigrid_levels)).solve(
                   output_directory=a.output_directory)
    print(result.coefficients)
    print(f"Saved fields and convergence history to {result.output_directory / 'solution.npz'}")


if __name__ == "__main__":
    main()
