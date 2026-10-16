"""Analytical conservation, material cuts and native wall-load regression."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D, Boundary, CGNSMesh, Zone
from cfl3d.surface_analysis import SurfaceMesh, lift_distribution, slice_references, evaluate_slice
from test_surface_forces import grouped_plate
from fixtures import flatplate_mesh


def trapezoid():
    xyz = np.zeros((2, 2, 2, 3))
    for i, j, k in np.ndindex(2, 2, 2):
        xyz[i, j, k] = [i * (2 + 2 * j), j, k]
    bc = Boundary("wall", "BCWallInviscid", np.array([[1, 2], [1, 2], [1, 1]]))
    mesh = CGNSMesh([Zone("wing", xyz, boundaries=[bc])])
    values = np.array([[[-1.0, 300.0, 0.1, 0.0, 0.0, 0.0]]])
    return mesh, {("wing", "wall"): values}


class GeometryTests(unittest.TestCase):
    def test_trapezoid_exact_strip_density_and_native_conservation(self):
        mesh, walls = trapezoid()
        surface = SurfaceMesh(mesh, walls)
        members = set(walls)
        d = lift_distribution(surface, members, 200, 1, [0.0, 0.0], scale=10.0)
        np.testing.assert_allclose(d["dCL_ds"], 2 + 2 * d["stations"], rtol=2e-13)
        np.testing.assert_allclose(d["dCDv_ds"], 0.1 * d["dCL_ds"], rtol=2e-13)
        self.assertAlmostEqual(d["bin_CL"].sum(), 3.0)
        self.assertAlmostEqual(d["bin_CDv"].sum(), 0.3)
        np.testing.assert_allclose(d["dL_ds"], 10 * d["dCL_ds"])
        # Allocation must preserve solver loads even for a warped face whose projected area differs.
        mesh.zones[0].coordinates[1, 1, 0, 2] = 0.7
        forces = {next(iter(walls)): np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])}
        d = lift_distribution(SurfaceMesh(mesh, walls, forces), members, 73, 1, [0.2, 0.1])
        np.testing.assert_allclose(d["pressure_coefficients"].sum(axis=0), [1, 2, 3], rtol=1e-14)
        np.testing.assert_allclose(d["viscous_coefficients"].sum(axis=0), [4, 5, 6], rtol=1e-14)

    def test_material_cut_and_empty_intersections(self):
        mesh, walls = trapezoid()
        surface = SurfaceMesh(mesh, walls)
        members = set(walls)
        refs = slice_references(surface, members, 1, 0.5)
        cut = evaluate_slice(surface, members, refs, 1, 0.5, False)
        self.assertEqual(len(cut["segments"]), 2)
        self.assertEqual(len(cut["points"]), 3)
        np.testing.assert_allclose(cut["points"][:, 1], 0.5)
        np.testing.assert_allclose(cut["Cp"], -1)
        moved = deepcopy(mesh)
        moved.zones[0].coordinates[..., 1] += 0.2
        tracked = evaluate_slice(SurfaceMesh(moved, walls), members, refs, 1, 0.5, False)
        np.testing.assert_allclose(tracked["points"][:, 1], 0.7)
        # Coplanar cuts have no line; nonintersecting cuts also return empty data.
        for axis, pos in ((1, 5.0), (2, 0.0)):
            refs = slice_references(surface, members, axis, pos)
            empty = evaluate_slice(surface, members, refs, axis, pos, True)
            self.assertEqual(empty["points"].shape, (0, 3))
            self.assertEqual(empty["segments"].shape, (0, 2))

    def test_shared_block_edge_and_topology_check(self):
        solver = CFL3D(grouped_plate())
        surface = solver._analysis_surface()
        members = set(surface.patches)
        junction = solver.mesh.zones[1].coordinates[0, 0, 0, 0]
        refs = slice_references(surface, members, 0, junction)
        cut = evaluate_slice(surface, members, refs, 0, junction, True)
        # Both wall patches include the junction, but the line must be exported once.
        self.assertEqual(cut["points"].shape, (2, 3))
        self.assertEqual(cut["segments"].shape, (1, 2))
        self.assertAlmostEqual(np.linalg.norm(np.diff(cut["points"], axis=0)), 1.0)
        patch = surface.patches[refs[0].key]
        patch.topology = ("changed",)
        with self.assertRaisesRegex(ValueError, "topology"):
            evaluate_slice(surface, members, refs, 0, junction, False)

    def test_validation_before_solve(self):
        solver = CFL3D(grouped_plate())
        for n, d, g in (
            (0, "y", None),
            (True, "y", None),
            (2.2, "y", None),
            (2, "bad", None),
            (2, "y", "typo"),
            (2, "y", "face00"),
            (2, "z", "Wing"),
        ):
            with self.subTest(n=n, d=d, g=g), self.assertRaises(ValueError):
                solver.addLiftDistribution(n, d, g)
        for positions in ([], [float("nan")], [[0.5]], True, "0.5"):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                solver.addSlices("y", positions)
        with self.assertRaises(ValueError):
            solver.addSlices("y", 0.5, "invalid")
        solver.addLiftDistribution(17, "Y", "Wing")
        solver.addSlices("y", [-0.25, -0.5], groupName="Wing")
        self.assertEqual((solver.nLiftDist, solver.nSlice), (1, 2))
        with self.assertRaisesRegex(RuntimeError, "solve"):
            solver.getLiftDistribution()
        solver.clearSurfaceAnalysis()
        self.assertEqual((solver.nLiftDist, solver.nSlice), (0, 0))


class NativeAnalysisTests(unittest.TestCase):
    def test_load_integrals_cuts_outputs_and_mesh_tracking(self):
        options = dict(
            turbulence="sst",
            mach=0.2,
            alpha=2.0,
            iterations=6,
            cfl=1.0,
            target_cl=0.01,
            alpha_step_limit=0.1,
            freestream_pressure=101325.0,
            writevolumesolution=False,
            writeTecplotSurfaceSolution=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver = CFL3D(grouped_plate(), options, groups={"Overlap": ["Wing", "Forward"]})
            solver.addLiftDistribution(31, "y", "Overlap")
            solver.addSlices("y", [-0.25, -0.5], groupName="Wing")
            solver.addSlices("y", -0.5, "absolute", "Wing")
            result = solver.solve(output_directory=root / "first")
            d = solver.getLiftDistribution()
            total = solver.getSolution("Wing")
            for key in ("CL", "CD", "CDp", "CDv"):
                self.assertAlmostEqual(d["bin_" + key].sum(), total[key], places=13)
                self.assertAlmostEqual(np.sum(d["d" + key + "_ds"] * d["width"]), total[key], places=13)
            self.assertAlmostEqual(np.sum(d["dL_ds"] * d["width"]), total["L"], places=10)
            self.assertGreater(abs(d["force_angles"][0] - result.final_alpha), 1e-4)
            cuts = solver.getSlices()
            for cut in cuts:
                self.assertGreater(len(cut["segments"]), 5)
                np.testing.assert_allclose(cut["points"][:, 1], cut["position"], atol=1e-15)
                # One continuous line crosses the two CGNS zones, with no duplicated junction.
                degree = np.bincount(cut["segments"].ravel())
                self.assertEqual(np.count_nonzero(degree == 1), 2)
                self.assertTrue(np.all((degree == 1) | (degree == 2)))
                # Segment Cp matches raw native face-cell Cp without nodal smoothing.
                for midpoint, cp in zip(cut["coordinates"], cut["Cp"], strict=True):
                    for zone in solver._last_mesh.zones:
                        if zone.coordinates[..., 0].min() < midpoint[0] < zone.coordinates[..., 0].max():
                            wall = next(b for b in zone.boundaries if b.name == "plate")
                            coords = zone.coordinates[0, :, 0, 0]
                            j = np.searchsorted(coords, midpoint[0]) - int(wall.point_range[1, 0])
                            self.assertAlmostEqual(cp, result.wall_fields[(zone.name, "plate")][0, j, 0], places=13)
            path = solver.writeTecplotSurfaceSolution()
            self.assertTrue(path.exists())
            self.assertIn("dCDp/ds", path.with_name("out_000_lift.dat").read_text())
            self.assertIn("FELINESEG", path.with_name("out_000_slices.dat").read_text())
            # Snapshot must survive modification of the publicly returned result arrays.
            for value in result.wall_forces.values():
                value[:] = 0
            np.testing.assert_array_equal(solver.getLiftDistribution()["bin_CL"], d["bin_CL"])
            for zone in solver.mesh.zones:
                zone.coordinates[..., 1] += 0.2
            solver.solve(output_directory=root / "moved")
            np.testing.assert_allclose(solver.getSlices(1)["points"][:, 1], -0.3, atol=1e-15)
            np.testing.assert_allclose(solver.getSlices(2)["points"][:, 1], -0.5, atol=1e-15)
            # Also permit requests registered after solve, including empty sections.
            solver.addSlices("y", 100.0, "absolute", "Wing")
            self.assertEqual(len(solver.getSlices(3)["Cp"]), 0)
            solver.addLiftDistribution(7, "x", "Forward")
            self.assertAlmostEqual(
                solver.getLiftDistribution(1)["bin_CL"].sum(), solver.getSolution("Forward")["CL"], places=13
            )
            solver.writeTecplotSurfaceSolution(root / "custom.dat")
            self.assertTrue((root / "custom_slices.dat").exists())

    def test_three_wall_axes_and_y_up_with_sideslip(self):
        with tempfile.TemporaryDirectory() as directory:
            for order, y_up in (((0, 1, 2), False), ((1, 2, 0), False), ((2, 0, 1), False), ((0, 1, 2), True)):
                with self.subTest(order=order, y_up=y_up):
                    mesh = flatplate_mesh()
                    zone = mesh.zones[0]
                    zone.coordinates = np.linspace(zone.coordinates[0], zone.coordinates[-1], 5)
                    for bc in zone.boundaries:
                        bc.point_range[0][bc.point_range[0] == 2] = 5
                        if bc.point_range[0, 0] == bc.point_range[0, 1]:
                            bc.bc_type = "BCSymmetryPlane"
                        bc.point_range = bc.point_range[list(order)].copy()
                    zone.coordinates = zone.coordinates.transpose((*order, 3)).copy()
                    if y_up:
                        xyz = zone.coordinates.copy()
                        zone.coordinates[..., 1] = xyz[..., 2]
                        zone.coordinates[..., 2] = -xyz[..., 1]
                    solver = CFL3D(
                        mesh,
                        dict(
                            turbulence="sst",
                            mach=0.2,
                            alpha=2.0,
                            beta=3.0,
                            iterations=2,
                            cfl=1.0,
                            freestream_pressure=101325.0,
                            sref=2.0,
                            y_up=y_up,
                            writevolumesolution=False,
                            writeTecplotSurfaceSolution=False,
                        ),
                    )
                    solver.addLiftDistribution(19, "z" if y_up else "y")
                    solver.solve(output_directory=Path(directory) / ("".join(map(str, order)) + str(y_up)))
                    d = solver.getLiftDistribution()
                    total = solver.getSolution()
                    for name in ("CL", "CD", "CDp", "CDv"):
                        self.assertAlmostEqual(d["bin_" + name].sum(), total[name], places=13)

    def test_automatic_outputs_and_no_pressure(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = CFL3D(grouped_plate(), dict(turbulence="sst", iterations=2, cfl=1.0, writevolumesolution=False))
            solver.addLiftDistribution(10, "y", "Wing")
            solver.addSlices("y", -0.5, "absolute", "Wing")
            solver.solve(output_directory=directory)
            self.assertNotIn("dL_ds", solver.getLiftDistribution())
            self.assertTrue((Path(directory) / "out_000_lift.dat").exists())
            self.assertTrue((Path(directory) / "out_000_slices.dat").exists())


if __name__ == "__main__":
    unittest.main()
