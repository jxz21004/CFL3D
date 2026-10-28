"""Native warpers: distributed interfaces, surface targets and resident reuse."""

from pathlib import Path
import argparse
import tempfile
import numpy as np
from mpi4py import MPI
from cfl3d import CFL3D
from cfl3d.mesh import CFL3DMeshAdapter
from test_cfl3d_mesh_adapter import mesh3d, target_surface


class PartitionComm:
    """Reject accidental global payload collection in the adapter itself."""

    def __init__(self, comm):
        self._comm = comm
        self.max_batch = 0

    def __getattr__(self, name):
        return getattr(self._comm, name)

    def gather(self, *args, **kwargs):
        raise AssertionError("Global gather is forbidden in the mesh adapter")

    def scatter(self, *args, **kwargs):
        raise AssertionError("Global scatter is forbidden in the mesh adapter")

    def allgather(self, value):
        assert not isinstance(value, (dict, np.ndarray)), "No replicated grid payload"
        return self._comm.allgather(value)

    def bcast(self, value, root=0):
        result = self._comm.bcast(value, root=root)
        if isinstance(result, tuple):
            assert len(result) == 3 and len(result[0]) <= 256
            self.max_batch = max(self.max_batch, len(result[0]))
        return result


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
            partition = PartitionComm(comm)
            adapter = CFL3DMeshAdapter(solver.mesh, surface, comm=partition, method=method, control_stride=2)
            assert set(adapter.graphs) == set(adapter.layout.owned)
            assert len(adapter.nodes) == sum(len(adapter.graphs[g]) for g in adapter.layout.owned)
            if not adapter.layout.owned:
                assert len(adapter.nodes) == len(adapter.unique) == 0
            effective = adapter.getControlIndices()
            explicit = CFL3DMeshAdapter(
                solver.mesh, surface, comm=partition, method=method, control_mode="indices", control_indices=effective
            )
            np.testing.assert_array_equal(explicit.warp(surface), adapter.layout.coordinates(solver.mesh))
            for mode in ("minimal", "auto_stride"):
                automatic = CFL3DMeshAdapter(solver.mesh, surface, comm=partition, method=method, control_mode=mode)
                np.testing.assert_array_equal(automatic.warp(surface), adapter.layout.coordinates(solver.mesh))
            # Incorrect index input on one rank must fail collectively before
            # shared-node communication rather than stranding other ranks.
            bad_indices = dict(effective)
            if comm.rank == 1:
                bad_indices[next(iter(bad_indices))] = ([0, -1], [0], [0])
            try:
                CFL3DMeshAdapter(
                    solver.mesh,
                    surface,
                    comm=partition,
                    method=method,
                    control_mode="indices",
                    control_indices=bad_indices,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("Expected collective control-index validation error")
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
            if method == "fmem":
                from cfl3d import _native

                original = _native.py_warp_matrix

                def fail_assembly(*args, **kwargs):
                    raise ValueError("injected owner-local assembly failure")

                if comm.rank == 1:
                    _native.py_warp_matrix = fail_assembly
                try:
                    try:
                        adapter.warp(target_surface(solver.getSurfaceMesh(), 0.001))
                    except ValueError as exc:
                        assert "injected owner-local" in str(exc)
                    else:
                        raise AssertionError("Assembly failure did not reach every rank")
                finally:
                    _native.py_warp_matrix = original
                # Retrying the adapter does not require reset/CFD reinitialization.
                adapter.warp(target_surface(solver.getSurfaceMesh(), 0.001))
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
