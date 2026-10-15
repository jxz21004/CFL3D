from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D, CGNSMesh, SolverOptions
from cgns_fixtures import write_adf
from fixtures import box_mesh, flatplate_mesh
from validate_mpi import split_plate


def grouped_plate():
    mesh = split_plate()
    for zone, name in zip(mesh.zones, ("Forward", "Aft")):
        wall = next(bc for bc in zone.boundaries if bc.name == "plate")
        wall.family = name
        wall.additional_families = ("Wing",)
    return mesh


class GroupValidationTests(unittest.TestCase):
    def test_group_names_members_and_before_solve(self):
        mesh = grouped_plate()
        for groups in ({"Wing": ["Forward"]}, {"plate0": ["Forward"]}, {"plate": ["Wing"]},
                       {"Bad": ["typo"]}, {"Bad": []}, {"Bad": "Wing"},
                       {"A": ["Forward"], "B": ["A"]}):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                CFL3D(mesh, groups=groups)
        solver = CFL3D(mesh, groups={"Combined": ["Forward", "Aft"]})
        self.assertIn("Wing", solver.getGroupNames())
        self.assertIn("plate0/plate", solver.getGroupNames())
        with self.assertRaisesRegex(RuntimeError, "solve"):
            solver.getSolution()
        with self.assertRaises(ValueError):
            SolverOptions(freestream_pressure=-1.).validate()


@unittest.skipUnless(importlib.util.find_spec("cfl3d._native"), "Build the native extension first")
class SurfaceForceTests(unittest.TestCase):
    def test_native_group_sums_drag_split_scaling_and_cache(self):
        options = SolverOptions(turbulence="sst", mach=.2, alpha=2., iterations=8,
            cfl=1., multigrid_levels=2, sref=2., cref=.7, bref=1.3,
            freestream_pressure=90000.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_adf(grouped_plate(), root / "plate.cgns")
            solver = CFL3D(path, options, groups={"Combined": ["Forward", "Aft"]})
            result = solver.solve(output_directory=root / "first")
            total = solver.getSolution()
            combined = solver.getSolution("Combined")
            family = solver.getSolution("Wing")
            solver.addGroups({"Overlap": ["Wing", "Forward", "Wing"]})
            self.assertEqual(total, combined)
            self.assertEqual(total, family)
            self.assertEqual(total, solver.getSolution("Overlap"))
            a, b = solver.getSolution("Forward"), solver.getSolution("Aft")
            for key in total:
                if key != "units":
                    self.assertAlmostEqual(total[key], a[key]+b[key], places=11)
            for key in ("CL", "CD", "CY", "Cmx", "Cmy", "Cmz"):
                native_key = key.upper() if key.startswith("Cm") else key
                self.assertAlmostEqual(total[key], float(result.coefficients[native_key]), places=12)
            self.assertAlmostEqual(total["CD"], total["CDp"]+total["CDv"], places=13)
            self.assertGreater(abs(total["CDp"]), 1e-8)
            self.assertGreater(total["CDv"], 0)
            scale = .5 * options.gamma * options.freestream_pressure * options.mach**2 * options.sref
            self.assertAlmostEqual(total["D"], total["CD"]*scale, places=12)
            self.assertAlmostEqual(total["Dp"]+total["Dv"], total["D"], places=11)
            self.assertAlmostEqual(total["My"], total["Cmy"]*scale*options.cref, places=12)
            self.assertEqual(total["units"]["D"], "N")
            self.assertEqual(total["units"]["My"], "N*m")
            with np.load(root / "first/solution.npz") as saved:
                self.assertEqual(saved["surface_coefficients"].shape[1], 10)
                self.assertEqual(saved["surface_coefficient_names"][-2:].tolist(), ["CDp", "CDv"])
            # Another solver replaces Fortran COMMON state; the first object's cache stays valid.
            other = CFL3D(path, replace(options, freestream_pressure=180000.))
            other.solve(output_directory=root / "second")
            self.assertEqual(solver.getSolution(), total)
            doubled = other.getSolution()
            for key in ("CL", "CD", "CDp", "CDv", "Cmy"):
                self.assertEqual(doubled[key], total[key])
            for key in ("L", "D", "Dp", "Dv", "My"):
                self.assertAlmostEqual(doubled[key], 2*total[key], places=11)

    def test_three_dimensional_moment_center_and_y_up(self):
        mesh = flatplate_mesh()
        zone = mesh.zones[0]
        zone.coordinates = np.linspace(zone.coordinates[0], zone.coordinates[-1], 5)
        for bc in zone.boundaries:
            bc.point_range[0][bc.point_range[0] == 2] = 5
            if bc.point_range[0, 0] == bc.point_range[0, 1]:
                bc.bc_type = "BCSymmetryPlane"
        options = SolverOptions(turbulence="sst", mach=.2, alpha=2., iterations=5,
            cfl=1., sref=2., cref=.7, bref=1.3, freestream_pressure=90000.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver = CFL3D(mesh, options)
            r = solver.solve(output_directory=root / "origin")
            a = solver.getSolution()
            offset = np.array([.2, .1, .3])
            shifted = CFL3D(mesh, replace(options, moment_center=tuple(offset)))
            shifted.solve(output_directory=root / "shifted")
            b = shifted.getSolution()
            raw = np.sum(list(r.surface_coefficients.values()), axis=0)
            scale = .5 * options.gamma * options.freestream_pressure * options.mach**2 * options.sref
            expected = -np.cross(offset, raw[2:5]*scale)
            np.testing.assert_allclose([b[k]-a[k] for k in ("Mx", "My", "Mz")], expected,
                                       rtol=2e-11, atol=1e-11)
            xyz = zone.coordinates.copy()
            zone.coordinates[..., 1] = xyz[..., 2]
            zone.coordinates[..., 2] = -xyz[..., 1]
            y_up = CFL3D(mesh, replace(options, y_up=True))
            y_up.solve(output_directory=root / "y-up")
            c = y_up.getSolution()
            np.testing.assert_allclose([c[k] for k in ("Mx", "My", "Mz")],
                                       [a["Mx"], a["Mz"], -a["My"]], rtol=2e-11, atol=1e-11)
            self.assertAlmostEqual(c["CL"], a["CL"], places=12)
            self.assertAlmostEqual(c["CD"], a["CD"], places=12)

    def test_missing_pressure_and_unknown_group(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = CFL3D(box_mesh(), SolverOptions(turbulence="euler", iterations=1))
            solver.solve(output_directory=directory)
            with self.assertRaisesRegex(ValueError, "freestream_pressure"):
                solver.getSolution()
            with self.assertRaisesRegex(ValueError, "Unknown surface group"):
                solver.getSolution("missing")


if __name__ == "__main__":
    unittest.main()
