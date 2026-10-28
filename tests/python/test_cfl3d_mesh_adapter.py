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

    def test_single_zone_control_solution_against_original_kernel(self):
        from cfl3d import _native

        # The parallel solver changes iteration order/tolerance, not the native
        # macro-element stiffness. Compare a one-block problem to the old kernel.
        for method in ("exponential", "fmem"):
            mesh = mesh3d(False)
            with CFL3D(mesh, comm=None) as solver:
                surface = solver.getSurfaceMesh()
                adapter = CFL3DMeshAdapter(mesh, surface, method=method, control_stride=2, smoothing_iterations=2)
                targets = adapter._targets(target_surface(surface, 0.001))
                actual = adapter._controls(targets)[0]
                graph = adapter.graphs[0].copy()
                points = np.array([mesh.zones[g].coordinates[p] for g, p in adapter.nodes])
                prescribed = np.zeros_like(points)
                for i, node in enumerate(adapter.canonical):
                    if int(node) in targets:
                        graph[i, 7] = 0
                        prescribed[i] = targets[int(node)] - points[i]
                graph[graph[:, 7] == 0, 9] = 0
                expected, status = _native.py_warp_controls(
                    points, graph, prescribed, int(method == "fmem"), adapter.parameters, adapter.sweeps, 1
                )
                self.assertEqual(status, 0)
                # Original GS stops at 1e-6 displacement-update norm. GMRES uses
                # a tighter preconditioned residual; agreement is not bitwise.
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6 if method == "fmem" else 1e-15)

    def test_control_storage_is_owned_only(self):
        mesh = mesh3d()
        with CFL3D(mesh, comm=None) as solver:
            adapter = CFL3DMeshAdapter(mesh, solver.getSurfaceMesh(), method="fmem")
            self.assertEqual(set(adapter.graphs), set(adapter.layout.owned))
            for g, graph in adapter.graphs.items():
                self.assertEqual(len(graph), len(adapter.zone_nodes[g]))

    def test_control_selection_modes_and_interface_propagation(self):
        mesh = mesh3d()
        with CFL3D(mesh, comm=None) as solver:
            surface = solver.getSurfaceMesh()
            minimal = CFL3DMeshAdapter(mesh, surface, control_mode="minimal")
            for axes in minimal.getControlIndices().values():
                self.assertEqual(axes, ((0, 8), (0, 8), (0, 8)))
            strides = {"zone0": (2, 4, 4), "zone1": (4, 2, 4)}
            adapter = CFL3DMeshAdapter(mesh, surface, control_stride=strides)
            actual = adapter.getControlIndices()
            self.assertEqual(actual["zone0"][0], (0, 2, 4, 6, 8))
            # The matching j-face propagates i indices to the other block.
            self.assertEqual(actual["zone1"][0], actual["zone0"][0])
            self.assertEqual(actual["zone0"][1], (0, 4, 8))
            self.assertEqual(actual["zone1"][1], (0, 2, 4, 6, 8))
            requested = {"zone0": ([0, 3, 8], [0, 8], [0, 8]), "zone1": ([0, 8], [0, 8], [0, 5, 8])}
            explicit = CFL3DMeshAdapter(mesh, surface, control_mode="indices", control_indices=requested)
            resolved = explicit.getControlIndices()
            self.assertEqual(resolved["zone0"][0], resolved["zone1"][0])
            self.assertEqual(resolved["zone0"][2], (0, 5, 8))
            self.assertEqual(requested["zone0"][2], [0, 8])
            # A physical segment endpoint at index 3 forces automatic uniform
            # spacing gcd(3,8)=1; minimal mode inserts only that endpoint.
            segmented = mesh3d(False)
            next(b for b in segmented.zones[0].boundaries if b.name == "face20").point_range[0, 1] = 4
            with CFL3D(segmented, comm=None) as segmented_solver:
                skin = segmented_solver.getSurfaceMesh()
                auto = CFL3DMeshAdapter(segmented, skin, control_mode="auto_stride")
                sparse_auto = CFL3DMeshAdapter(segmented, skin, control_mode="minimal")
                self.assertEqual(auto.getControlIndices()["zone0"][0], tuple(range(9)))
                self.assertEqual(sparse_auto.getControlIndices()["zone0"][0], (0, 3, 8))
            resolved.clear()
            self.assertEqual(len(explicit.getControlIndices()), 2)
            # No hidden five-indices-per-axis floor for FMEM.
            sparse = CFL3DMeshAdapter(mesh, surface, method="fmem", control_stride=8)
            self.assertEqual(sparse.getControlIndices(), minimal.getControlIndices())
            # Equivalent effective lattices give identical candidates.
            explicit = CFL3DMeshAdapter(mesh, surface, control_mode="indices", control_indices=actual)
            target = target_surface(surface, 0.001)
            np.testing.assert_array_equal(adapter.warp(target), explicit.warp(target))

    def test_control_selection_rejects_ambiguous_or_invalid_options(self):
        mesh = mesh3d(False)
        with CFL3D(mesh, comm=None) as solver:
            surface = solver.getSurfaceMesh()
            invalid = [
                dict(control_mode="unknown"),
                dict(control_mode="minimal", control_stride=2),
                dict(control_indices={"zone0": ([0, 8],) * 3}),
                dict(control_mode="indices"),
                dict(control_stride={}),
                dict(control_stride=(2, 0, 2)),
                dict(control_stride=True),
                dict(control_mode="indices", control_indices={"zone0": ([0, 0, 8],) * 3}),
                dict(control_mode="indices", control_indices={"zone0": ([0, 9],) * 3}),
                dict(control_mode="indices", control_indices={"zone0": ([0.0, 8.0],) * 3}),
            ]
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    CFL3DMeshAdapter(mesh, surface, **kwargs)

    def test_minimal_all_prescribed_controls_need_no_fmem_solve(self):
        mesh = mesh3d(False)
        with CFL3D(mesh, comm=None) as solver:
            surface = solver.getSurfaceMesh()
            target = target_surface(surface, 0.001)
            target["coordinates"][:, 2] += 0.001
            adapter = CFL3DMeshAdapter(mesh, surface, method="fmem", control_mode="minimal")
            candidate = adapter.layout.unpack(adapter.warp(target))["zone0"]
            topology = SurfaceTopology(mesh)
            for node, xyz in zip(target["node_ids"], target["coordinates"]):
                _, point = topology.decode(node)
                np.testing.assert_allclose(candidate[point], xyz, rtol=0, atol=1e-15)
