from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
from cfl3d import CFL3D, CGNSMesh, SolverOptions
from cgns_fixtures import write_adf
from fixtures import box_mesh, flatplate_mesh
from validate_mpi import split_plate


def grouped_plate(*, split=33):
    mesh = split_plate(split)
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
        self.assertIn("Combined", solver.getGroupNames())
        self.assertNotIn("plate0/plate", solver.getGroupNames())
        self.assertIn("plate0/plate", solver.getGroupNames(detailed=True))
        with self.assertRaisesRegex(RuntimeError, "solve"):
            solver.getSolution()
        with self.assertRaises(ValueError):
            SolverOptions(freestream_pressure=-1.).validate()

    def test_numbered_boundary_patches_have_one_component_name(self):
        mesh = grouped_plate()
        for zone, number in zip(mesh.zones, (2854, 2855)):
            wall = next(bc for bc in zone.boundaries if bc.name == "plate")
            wall.name = f"FAR_{number}"
            wall.family = None
            wall.additional_families = ()
        solver = CFL3D(mesh)
        self.assertIn("FAR", solver.getGroupNames())
        self.assertNotIn("FAR_2854", solver.getGroupNames())
        self.assertIn("FAR_2854", solver.getGroupNames(detailed=True))
        self.assertEqual(solver._cgns_surface_groups()["FAR"],
                         {("plate0", "FAR_2854"), ("plate1", "FAR_2855")})
        with self.assertRaises(ValueError):
            solver.addGroups({"FAR": ["FAR_2854"]})
        with self.assertRaises(ValueError):
            solver.getGroupNames(detailed="yes")


@unittest.skipUnless(importlib.util.find_spec("cfl3d._native"), "Build the native extension first")
class SurfaceForceTests(unittest.TestCase):
    def test_numbered_patch_load_is_sum_of_exact_patches(self):
        mesh = grouped_plate()
        for zone, number in zip(mesh.zones, (2854, 2855)):
            wall = next(bc for bc in zone.boundaries if bc.name == "plate")
            wall.name = f"FAR_{number}"
            wall.family = None
            wall.additional_families = ()
        options = SolverOptions(turbulence="sst", mach=.2, alpha=2., iterations=3,
                                freestream_pressure=101325.)
        with tempfile.TemporaryDirectory() as directory:
            solver = CFL3D(mesh, options, comm=None)
            solver.solve(output_directory=directory)
            total = solver.getSolution("FAR")
            patches = (solver.getSolution("FAR_2854"), solver.getSolution("FAR_2855"))
            for key, value in total.items():
                if key == "units":
                    continue
                suffix = key.removeprefix("FAR_")
                self.assertAlmostEqual(value, sum(part[f"FAR_{number}_{suffix}"]
                                                  for number, part in zip((2854, 2855), patches)),
                                       places=11)
            self.assertGreater(total["FAR_wettedArea"], 0.)

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
            solver.addGroups({"Overlap": ["Wing", "Forward", "Wing"]})
            for group in ("Combined", "Wing", "Overlap"):
                grouped = solver.getSolution(group)
                self.assertEqual(set(grouped), {group+'_'+key for key in total if key != 'units'} | {'units'})
                for key in total:
                    if key != 'units':
                        self.assertEqual(total[key], grouped[group+'_'+key])
                self.assertEqual(grouped['units'], {group+'_'+key: unit for key, unit in total['units'].items()})
            a, b = solver.getSolution("Forward"), solver.getSolution("Aft")
            for key in total:
                if key != "units":
                    self.assertAlmostEqual(total[key], a['Forward_'+key]+b['Aft_'+key], places=11)
            for key in ("CL", "CD", "CX", "CY", "Cmx", "Cmy", "Cmz"):
                native_key = key.upper() if key.startswith("Cm") else key
                self.assertAlmostEqual(total[key], float(result.coefficients[native_key]), places=12)
            self.assertAlmostEqual(total["CD"], total["CDp"]+total["CDv"], places=13)
            self.assertGreater(abs(total["CDp"]), 1e-8)
            self.assertGreater(total["CDv"], 0)
            scale = .5 * options.gamma * options.freestream_pressure * options.mach**2 * options.sref
            self.assertAlmostEqual(total["D"], total["CD"]*scale, places=12)
            self.assertAlmostEqual(total["Dp"]+total["Dv"], total["D"], places=11)
            self.assertAlmostEqual(total["My"], total["Cmy"]*scale*options.cref, places=12)
            for axis in 'xyz':
                self.assertAlmostEqual(total['F'+axis], total['C'+axis.upper()]*scale, places=12)
            self.assertNotIn('Y', total)
            mesh = solver._last_mesh
            # This fixture is a planar plate: span times wall length is its exact area.
            area = 0.
            for zone in mesh.zones:
                wall = next(bc for bc in zone.boundaries if bc.name == 'plate')
                lo, hi = np.sort(wall.point_range, axis=1).T - 1
                a0, b0 = zone.coordinates[tuple(lo)], zone.coordinates[tuple(hi)]
                area += abs((b0[0]-a0[0])*(b0[1]-a0[1]))
            self.assertAlmostEqual(total['wettedArea'], area, places=13)
            self.assertEqual(total['units']['wettedArea'], 'm^2')
            self.assertEqual(total["units"]["D"], "N")
            self.assertEqual(total["units"]["My"], "N*m")
            with np.load(root / "first/out_000_solution.npz") as saved:
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
            # Live edits cannot reinterpret pressure or geometry saved with an existing solution.
            solver.options.freestream_pressure = 180000.
            for zone in solver.mesh.zones:
                zone.coordinates *= 2.
            self.assertEqual(solver.getSolution(), total)

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
            rotated = y_up.solve(output_directory=root / "y-up")
            c = y_up.getSolution()
            np.testing.assert_allclose([c[k] for k in ("Mx", "My", "Mz")],
                                       [a["Mx"], a["Mz"], -a["My"]], rtol=2e-11, atol=1e-11)
            self.assertAlmostEqual(c["CL"], a["CL"], places=12)
            self.assertAlmostEqual(c["CD"], a["CD"], places=12)
            for key, wall in r.wall_fields.items():
                expected = wall.copy()
                expected[...,3], expected[...,4] = wall[...,4], -wall[...,3]
                np.testing.assert_allclose(rotated.wall_fields[key], expected, rtol=2e-12, atol=1e-12)
            np.testing.assert_allclose([c[k] for k in ('Fx', 'Fy', 'Fz')],
                                       [a['Fx'], a['Fz'], -a['Fy']], rtol=2e-11, atol=1e-11)
            self.assertAlmostEqual(c['wettedArea'], a['wettedArea'], places=13)

    def test_missing_pressure_and_unknown_group(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = CFL3D(box_mesh(), SolverOptions(turbulence="euler", iterations=1))
            solver.solve(output_directory=directory)
            with self.assertWarnsRegex(RuntimeWarning, "No freestream pressure.*absolute forces"):
                loads = solver.getSolution()
            self.assertEqual(set(loads), {'CL', 'CD', 'CX', 'CY', 'CZ', 'Cmx', 'Cmy', 'Cmz',
                                         'CDp', 'CDv', 'wettedArea', 'units'})
            self.assertEqual(loads['wettedArea'], 0.)
            self.assertEqual(loads['units'], {'wettedArea': 'm^2'})
            with self.assertWarns(RuntimeWarning):
                grouped = solver.getSolution('zone0')
            self.assertEqual(set(grouped), {'zone0_'+key for key in loads if key != 'units'} | {'units'})
            self.assertEqual(grouped['units'], {'zone0_wettedArea': 'm^2'})
            # Dimensional information can be supplied after solving, without touching cached flow.
            solver.options.freestream_pressure = 90000.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                dimensional = solver.getSolution()
            self.assertFalse(caught)
            scale = .5 * solver._last_options.gamma * 90000. * solver._last_options.mach**2 * solver._last_options.sref
            for axis in 'xyz':
                self.assertEqual(dimensional['F'+axis], loads['C'+axis.upper()]*scale)
            with self.assertRaisesRegex(ValueError, "Unknown surface group"):
                solver.getSolution("missing")


if __name__ == "__main__":
    unittest.main()
