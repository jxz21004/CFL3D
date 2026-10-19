"""Shared numerical and output flags for the serial and MPI example scripts."""
import argparse


def add_numerical_options(parser):
    parser.add_argument("--write-volume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-surface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-flow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freestream-pressure", type=float, help="Static pressure in Pa; enables SI loads")
    parser.add_argument("--group", help="CGNS force group; default: all integrated surfaces")
    parser.add_argument("--sref", type=float, default=1., help="Reference area in m^2")
    parser.add_argument("--cref", type=float, default=1., help="Reference chord in m")
    parser.add_argument("--bref", type=float, default=1., help="Reference span in m")
    parser.add_argument("--target-cl", type=float)
    parser.add_argument("--alpha-relaxation", type=float, default=1.)
    parser.add_argument("--alpha-step-limit", type=float, default=.2, help="Degrees per CL-control update")
    parser.add_argument("--alpha-update-interval", type=int, default=1)
    parser.add_argument("--limiter", type=int, nargs="+", default=[4], choices=range(5),
                        help="One value for all directions, or three values in i j k order")
    parser.add_argument("--flux", type=int, nargs="+", default=[1], choices=(0, 1))
    parser.add_argument("--inversion", type=int, nargs="+", default=[1], choices=(0, 1))
    parser.add_argument("--kappa", type=float, nargs="+", default=[1/3])
    parser.add_argument("--multigrid-levels", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--multigrid-cycle", type=int, default=2, choices=(1, 2))
    parser.add_argument("--mesh-sequence", action="store_true")
    parser.add_argument("--coarse-iterations", type=int, default=50)
    parser.add_argument("--first-order-iterations", type=int, default=0)
    parser.add_argument("--coarse-first-order-iterations", type=int, default=0)
    parser.add_argument("--multigrid-up-iterations", type=int, default=0)
    parser.add_argument("--multigrid-smoothing", type=int, nargs="+")
    parser.add_argument("--eddy-viscosity-limit", type=float, default=1e5)


def numerical_options(parser, args):
    if args.group is not None and args.freestream_pressure is None:
        parser.error("--group requires --freestream-pressure in Pa for dimensional loads")
    result = {name: getattr(args, name) for name in (
        "freestream_pressure", "sref", "cref", "bref",
        "target_cl", "alpha_relaxation", "alpha_step_limit", "alpha_update_interval",
        "multigrid_levels", "multigrid_cycle", "mesh_sequence", "coarse_iterations",
        "first_order_iterations", "coarse_first_order_iterations", "multigrid_up_iterations",
        "eddy_viscosity_limit")}
    result.update(writevolumesolution=args.write_volume,
                  writeTecplotSurfaceSolution=args.write_surface,
                  writeflowsolution=args.write_flow)
    for name in ("limiter", "flux", "inversion", "kappa"):
        values = getattr(args, name)
        if len(values) not in (1, 3):
            parser.error(f"--{name} requires one value or three values in i j k order")
        result[name] = values[0] if len(values) == 1 else tuple(values)
    result["multigrid_smoothing"] = tuple(args.multigrid_smoothing) if args.multigrid_smoothing else None
    return result
