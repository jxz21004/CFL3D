"""Native warpers: distributed interfaces, surface targets and resident reuse."""

from pathlib import Path
import argparse
import tempfile
import numpy as np
from mpi4py import MPI
from cfl3d import CFL3D
from cfl3d.mesh import CFL3DMeshAdapter
from test_cfl3d_mesh_adapter import mesh3d, target_surface


def run(comm, root):
    config = dict(
        resident=True,
        iterations=2,
        multigrid_levels=3,
        turbulence="sst",
        mach=0.2,
        cfl=1.0,
        writevolumesolution=False,
        writeTecplotSurfaceSolution=False,
        progress=False,
    )
    for method in ("exponential", "fmem"):
        reference = None
        error = None
        if comm.rank == 0:
            try:
                grids = []
                with CFL3D(mesh3d(), config, comm=None) as solver:
                    solver.solve(output_directory=root / (method + "-serial"))
                    adapter = CFL3DMeshAdapter(solver.mesh, solver.getSurfaceMesh(), method=method, control_stride=2)
                    for _ in range(2):
                        candidate = adapter.warp(target_surface(solver.getSurfaceMesh(), 0.001))
                        grids.append(adapter.layout.unpack(candidate))
                        solver.setVolumeCoordinates(candidate)
                        result = solver.solve(output_directory=root / (method + "-serial"))
                        assert not result.fail
                    reference = (result.history, result.fields, result.wall_forces, grids)
            except Exception as exc:
                error = repr(exc)
        error, reference = comm.bcast((error, reference), root=0)
        if error:
            raise RuntimeError(error)
        with CFL3D(mesh3d(), config, comm=comm) as solver:
            solver.solve(output_directory=root / (method + "-mpi"))
            surface = solver.getSurfaceMesh()
            adapter = CFL3DMeshAdapter(solver.mesh, surface, comm=comm, method=method, control_stride=2)
            for step in range(2):
                before = solver.getRuntimeInfo().copy()
                warped = adapter.warp(target_surface(solver.getSurfaceMesh(), 0.001))
                assert before == solver.getRuntimeInfo()
                for name, coordinates in adapter.layout.unpack(warped).items():
                    np.testing.assert_allclose(coordinates, reference[3][step][name], rtol=0, atol=5e-15)
                solver.setVolumeCoordinates(warped)
                result = solver.solve(output_directory=root / (method + "-mpi"))
                assert not result.fail
            assert solver.getRuntimeInfo()["initializations"] == 1
            assert solver.getRuntimeInfo()["geometry_updates"] == 2
            np.testing.assert_allclose(result.history, reference[0], rtol=2e-12, atol=5e-14)
            for z, values in result.fields.items():
                np.testing.assert_allclose(values, reference[1][z], rtol=2e-12, atol=5e-14)
            for key, values in result.wall_forces.items():
                np.testing.assert_allclose(values, reference[2][key], rtol=2e-12, atol=5e-14)
            # A local validation error is reported on all ranks before native calls.
            invalid = target_surface(solver.getSurfaceMesh())
            if len(invalid["coordinates"]) and comm.rank == 1:
                invalid["coordinates"][0, 0] = np.nan
            try:
                adapter.warp(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Expected collective invalid-input error")
        if comm.rank == 0:
            print("NATIVE_WARPER_PASS", method, flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--split", action="store_true")
args = parser.parse_args()
world = MPI.COMM_WORLD
comm = world.Split(world.rank // 2, world.rank) if args.split else world
scratch = tempfile.TemporaryDirectory() if comm.rank == 0 else None
root = Path(comm.bcast(scratch.name if scratch else None, root=0))
try:
    run(comm, root)
    comm.Barrier()
    if scratch:
        scratch.cleanup()
except BaseException:
    import traceback

    traceback.print_exc()
    world.Abort(1)
