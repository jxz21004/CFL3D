"""Native warpers: distributed interfaces, surface targets and resident reuse."""

from pathlib import Path
import argparse
import tempfile
import numpy as np
from mpi4py import MPI
from optional_dependencies import petsc_skip_reason
from cfl3d import CFL3D
from cfl3d.mesh import CFL3DMeshAdapter
from test_cfl3d_mesh_adapter import mesh3d, target_surface


class PartitionComm:
    """Reject accidental global payload collection in the adapter itself."""

    def __init__(self, comm, native=False):
        self._comm = comm
        self.native = native
        self.max_batch = 0

    def __getattr__(self, name):
        return getattr(self._comm, name)

    def gather(self, *args, **kwargs):
        assert self.native, "Global gather is forbidden in the distributed backend"
        for data in args[0].values():
            assert len(data) == 5 and data[4].shape == (len(data[0]), 21)
        return self._comm.gather(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        assert self.native, "Global scatter is forbidden in the distributed backend"
        return self._comm.scatter(*args, **kwargs)

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
    cases = [("exponential", "native-gauss-seidel"), ("fmem", "native-gauss-seidel"), ("fmem", "petsc-gmres")]
    if args.mixed:
        cases = [("fmem", "native-gauss-seidel" if world.rank // 2 == 0 else "petsc-gmres")]
    if args.petsc_skip_reason is not None:
        # Still run native validation when the optional mixed-backend scenario
        # cannot run. Never silently label native-only runs as mixed coverage.
        cases = [("exponential", "native-gauss-seidel"), ("fmem", "native-gauss-seidel")]
    for method, backend in cases:
        backend_options = dict(linear_solver=backend)
        if backend == "petsc-gmres":
            backend_options["linear_solver_options"] = {"restart": 12}
        reference = None
        error = None
        if comm.rank == 0:
            try:
                grids = []
                with CFL3D(mesh3d(), config, comm=None) as solver:
                    solver.solve(output_directory=root / (method + "-serial"))
                    adapter = CFL3DMeshAdapter(
                        solver.mesh, solver.getSurfaceMesh(), method=method, control_stride=2, **backend_options
                    )
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
            partition = comm if backend == "petsc-gmres" else PartitionComm(comm, native=True)
            adapter = CFL3DMeshAdapter(
                solver.mesh, surface, comm=partition, method=method, control_stride=2, **backend_options
            )
            assert set(adapter.graphs) == set(adapter.layout.owned)
            assert len(adapter.nodes) == sum(len(adapter.graphs[g]) for g in adapter.layout.owned)
            if not adapter.layout.owned:
                assert len(adapter.nodes) == len(adapter.unique) == 0
            if backend == "petsc-gmres":
                try:
                    CFL3DMeshAdapter(
                        solver.mesh,
                        surface,
                        comm=comm,
                        method=method,
                        linear_solver=backend,
                        linear_solver_options={"restart": 12 + int(comm.rank == 1)},
                    )
                except ValueError as exc:
                    assert "differ between ranks" in str(exc)
                else:
                    raise AssertionError("Inconsistent solver options were not rejected collectively")
            effective = adapter.getControlIndices()
            explicit = CFL3DMeshAdapter(
                solver.mesh,
                surface,
                comm=partition,
                method=method,
                control_mode="indices",
                control_indices=effective,
                **backend_options,
            )
            np.testing.assert_array_equal(explicit.warp(surface), adapter.layout.coordinates(solver.mesh))
            for mode in ("minimal", "auto_stride"):
                automatic = CFL3DMeshAdapter(
                    solver.mesh, surface, comm=partition, method=method, control_mode=mode, **backend_options
                )
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
                    **backend_options,
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
            if method in ("exponential", "fmem"):
                from cfl3d import _native

                function = "py_warp_matrix" if backend == "petsc-gmres" else "py_warp_controls"
                original = getattr(_native, function)

                def fail_assembly(*args, **kwargs):
                    raise ValueError("injected owner-local assembly failure")

                failure_rank = 1 if backend == "petsc-gmres" else 0
                if comm.rank == failure_rank:
                    setattr(_native, function, fail_assembly)
                try:
                    try:
                        adapter.warp(target_surface(solver.getSurfaceMesh(), 0.001))
                    except ValueError as exc:
                        assert "injected owner-local" in str(exc)
                    else:
                        raise AssertionError("Assembly failure did not reach every rank")
                finally:
                    setattr(_native, function, original)
                # Retrying the adapter does not require reset/CFD reinitialization.
                adapter.warp(target_surface(solver.getSurfaceMesh(), 0.001))
        if comm.rank == 0:
            print("NATIVE_WARPER_PASS", method, backend, flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--split", action="store_true")
parser.add_argument("--mixed", action="store_true")
parser.add_argument("--skip-petsc", action="store_true", help="Skip optional PETSc cases; native cases still run")
args = parser.parse_args()
world = MPI.COMM_WORLD
args.petsc_skip_reason = petsc_skip_reason(world, skip=args.skip_petsc)
if args.petsc_skip_reason is not None and world.rank == 0:
    print("PETSC_TESTS_SKIPPED:", args.petsc_skip_reason, flush=True)
    if args.mixed:
        print("MIXED_BACKEND_TEST_SKIPPED: running native cases only", flush=True)
comm = world.Split(world.rank // 2, world.rank) if args.split or args.mixed else world
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
