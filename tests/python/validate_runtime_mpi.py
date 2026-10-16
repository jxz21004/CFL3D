"""mpirun -np 3 ... [--auto], or -np 4 ... --exclude/--split."""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
from cfl3d import CFL3D
from cgns_fixtures import write_adf
from test_surface_forces import grouped_plate


def compare(result, reference):
    for name in result.fields:
        np.testing.assert_array_equal(result.fields[name], reference.fields[name])
    np.testing.assert_array_equal(result.history, reference.history)
    for name in result.wall_forces:
        np.testing.assert_array_equal(result.wall_forces[name], reference.wall_forces[name])


def run(comm, root, automatic):
    color = comm.Get_rank()  # Only for rank-varying spelling below, never numerical settings.
    root.mkdir(parents=True, exist_ok=True)
    options = dict(
        input_grid=root / "mesh.cgns",
        output_directory=root / "mpi-first",
        turbulence="sst",
        mach=0.2,
        alpha=2.0,
        iterations=5,
        cfl=1.5 if root.name == "group-1" else 1.0,
        multigrid_levels=2,
        writevolumesolution=False,
        writeTecplotSurfaceSolution=False,
    )
    reference = error = None
    if comm.rank == 0:
        try:
            write_adf(grouped_plate(), options["input_grid"])
            serial = CFL3D(options={**options, "output_directory": root / "serial-first"}, comm=None)
            reference = serial.solve()
            serial.writeRestartFile(root / "checkpoint.npz")
        except Exception as exc:
            error = repr(exc)
    error, reference = comm.bcast((error, reference), root=0)
    if error:
        raise RuntimeError(error)
    local = {key.upper() if color % 2 else key: value for key, value in options.items()}
    solver = CFL3D(options=local) if automatic else CFL3D(options=local, comm=comm)
    assert solver.comm == comm
    result = solver.solve()
    solver._checked(lambda: compare(result, reference))
    # Mutating only one rank must be rejected collectively, rather than silently overwritten.
    if comm.rank == comm.size - 1:
        solver.options["CFL"] = 9.0
    try:
        solver.options.check()
    except RuntimeError as exc:
        assert "cfl" in str(exc)
    else:
        raise AssertionError("Expected rank-local options mismatch")
    solver.setOptions({"CFL": 2.0, "ITERATIONS": 2})
    solver.checkOptions()
    reference = None
    if comm.rank == 0:
        serial.setOptions({"CFL": 2.0, "ITERATIONS": 2})
        reference = serial.solve(output_directory=root / "serial-next")
    reference = comm.bcast(reference, root=0)
    result = solver.solve(output_directory=root / "mpi-next")
    solver._checked(lambda: compare(result, reference))
    # A new object resumes from the file created by the serial solver, using a different worker count.
    other = CFL3D(options={**options, "cfl": 2.0, "iterations": 2, "restartFile": root / "checkpoint.npz"}, comm=comm)
    result = other.solve(output_directory=root / "file-next")
    solver._checked(lambda: compare(result, reference))
    # Cached states are also checked before native collectives.
    if comm.rank == comm.size - 1:
        other._flow_state = None
    try:
        other.solve(output_directory=root / "mismatched-state")
    except RuntimeError as exc:
        assert "states differ" in str(exc)
    else:
        raise AssertionError("Expected cached flow state mismatch")
    comm.Barrier()
    if comm.rank == 0:
        print(f"RUNTIME_PASS world_rank={MPI.COMM_WORLD.rank} comm_size={comm.size} auto={automatic}", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--exclude", action="store_true")
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    if args.exclude:
        assert world.size == 4
        comm = world.Split(MPI.UNDEFINED if world.rank == 0 else 0, world.rank)
        if world.rank != 0:
            run(comm, args.directory.resolve() / "exclude", False)
            comm.Free()
        else:
            assert sum(i * i for i in range(1000)) == 332833500  # This rank creates no CFL3D object.
    elif args.split:
        assert world.size == 4
        comm = world.Split(world.rank // 2, world.rank)
        run(comm, args.directory.resolve() / f"group-{world.rank // 2}", False)
        comm.Free()
    else:
        assert world.size in (2, 3)
        run(world, args.directory.resolve() / f"auto-{world.size}", args.auto)
    world.Barrier()


if __name__ == "__main__":
    main()
