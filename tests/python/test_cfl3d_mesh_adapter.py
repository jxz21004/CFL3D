"""Both native deformation paths, interface continuity and CFD isolation."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cfl3d import CFL3D
from cfl3d.mesh import CFL3DMeshAdapter, VolumeMeshLayout
from cfl3d.surface_mesh import SurfaceTopology
from fixtures import box_mesh


def mesh3d(multiblock=True):
    mesh = box_mesh(multiblock)
    for z in mesh.zones:
        z.coordinates = np.linspace(z.coordinates[0], z.coordinates[-1], 9)
        for bc in z.boundaries:
            bc.point_range[0][bc.point_range[0] == 2] = 9
            if bc.name == "face20":
                bc.bc_type = "BCWallViscous"
        for link in z.connections:
            link.point_range[0][link.point_range[0] == 2] = 9
            link.donor_range[0][link.donor_range[0] == 2] = 9
    return mesh


def target_surface(surface, scale=0.01):
    target = deepcopy(surface)
    xyz = target["coordinates"]
    xyz[:, 2] += scale * np.sin(np.pi * xyz[:, 0] / 2) * np.sin(-np.pi * xyz[:, 1])
    return target


class NativeMeshAdapterTests(unittest.TestCase):
    def test_methods_preserve_surface_interfaces_and_do_not_mutate_input(self):
        for method in ("exponential", "fmem"):
            with self.subTest(method=method):
                mesh = mesh3d()
                with CFL3D(mesh, comm=None) as solver:
                    surface = solver.getSurfaceMesh()
                    adapter = CFL3DMeshAdapter(mesh, surface, method=method, control_stride=2)
                    before = VolumeMeshLayout(mesh).coordinates(mesh)
                    np.testing.assert_array_equal(adapter.warp(surface), before)
                    target = target_surface(surface)
                    result = adapter.warp(target)
                    np.testing.assert_array_equal(VolumeMeshLayout(mesh).coordinates(mesh), before)
                    zones = adapter.layout.unpack(result)
                    topology = SurfaceTopology(mesh)
                    for node, xyz in zip(target["node_ids"], target["coordinates"]):
                        g, p = topology.decode(node)
                        np.testing.assert_allclose(zones[mesh.zones[g].name][p], xyz, rtol=0, atol=1e-15)
                    np.testing.assert_allclose(zones["zone0"][:, -1], zones["zone1"][:, 0], rtol=0, atol=1e-12)
                    self.assertGreater(
                        np.max(abs(zones["zone0"][1:-1, 1:-1, 1:-1] - mesh.zones[0].coordinates[1:-1, 1:-1, 1:-1])), 0.0
                    )
                    np.testing.assert_array_equal(result, adapter.warp(target))

    def test_warp_does_not_change_resident_flow_or_counters(self):
        config = dict(
            resident=True,
            iterations=2,
            multigrid_levels=3,
            turbulence="sst",
            cfl=1.0,
            mach=0.2,
            writevolumesolution=False,
            writeTecplotSurfaceSolution=False,
            progress=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with CFL3D(mesh3d(), config, comm=None) as solver:
                solver.solve(output_directory=root / "reference")
                expected = solver.solve(output_directory=root / "reference")
            for method in ("exponential", "fmem"):
                with CFL3D(mesh3d(), config, comm=None) as solver:
                    solver.solve(output_directory=root / method)
                    before = solver.getRuntimeInfo().copy()
                    surface = solver.getSurfaceMesh()
                    adapter = CFL3DMeshAdapter(solver.mesh, surface, method=method, control_stride=2)
                    warped = adapter.warp(target_surface(surface, 0.001))
                    self.assertEqual(solver.getRuntimeInfo(), before)
                    # Discard the candidate: uninterrupted flow must be exact.
                    actual = solver.solve(output_directory=root / method)
                    np.testing.assert_array_equal(actual.history, expected.history)
                    for name in actual.fields:
                        np.testing.assert_array_equal(actual.fields[name], expected.fields[name])
                    solver.setVolumeCoordinates(warped)
                    result = solver.solve(output_directory=root / method)
                    self.assertFalse(result.fail)
                    self.assertEqual(solver.getRuntimeInfo()["initializations"], 1)
                    # Multiple absolute-coordinate exchanges with accepted mesh as base.
                    surface = solver.getSurfaceMesh()
                    solver.setVolumeCoordinates(adapter.warp(target_surface(surface, 0.001)))
                    self.assertFalse(solver.solve(output_directory=root / method).fail)
                    self.assertEqual(solver.getRuntimeInfo()["geometry_updates"], 2)

    def test_reject_invalid_input(self):
        mesh = mesh3d()
        with CFL3D(mesh, comm=None) as solver:
            surface = solver.getSurfaceMesh()
            with self.assertRaises(ValueError):
                CFL3DMeshAdapter(mesh, surface, method="other")
            adapter = CFL3DMeshAdapter(mesh, surface)
            bad = deepcopy(surface)
            bad["coordinates"][0, 0] = np.nan
            with self.assertRaises(ValueError):
                adapter.warp(bad)
            bad = deepcopy(surface)
            bad["node_ids"][0] = -1
            with self.assertRaises(ValueError):
                adapter.warp(bad)
