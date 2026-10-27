"""Resident geometry-only refresh, without resetting transported flow state."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cfl3d import _native
from cfl3d.mesh import VolumeLayout
from test_surface_forces import grouped_plate
from test_resident_native import native_session, readout


class GeometryNativeTests(unittest.TestCase):
    def test_refresh_preserves_flow_and_workspace(self):
        mesh = grouped_plate()
        xyz = VolumeLayout(mesh).coordinates(mesh)
        options = dict(turbulence='sst', mach=.2, cfl=1., iterations=4,
                       multigrid_levels=3, writevolumesolution=False,
                       writeTecplotSurfaceSolution=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with native_session(mesh, options, root/'reference', 2, open_ended=True):
                self.assertEqual(_native.py_solve()[0], 0)
                self.assertEqual(_native.py_advance_resident(2, 0.)[0], 0)
                reference, _ = readout(mesh, 4)
            with native_session(mesh, options, root/'updated', 2, open_ended=True):
                self.assertEqual(_native.py_solve()[0], 0)
                before, _ = readout(mesh, 2)
                info = _native.py_resident_info()
                self.assertEqual(_native.py_stage_geometry(np.asfortranarray(xyz), 0), 0)
                staged, _ = readout(mesh, 2)
                for name in before:
                    np.testing.assert_array_equal(staged[name], before[name])
                self.assertEqual(_native.py_advance_resident(2, 0.)[0], 0)
                refreshed, _ = readout(mesh, 4)
                for name in reference:
                    np.testing.assert_allclose(refreshed[name], reference[name], atol=1e-13, rtol=1e-13)
                after = _native.py_resident_info()
                self.assertEqual(after[3], info[3])  # no repeated initialization
                self.assertEqual(after[5], info[5])  # same main workspace
                self.assertEqual(_native.py_geometry_info(), 1)
                moved = xyz.copy()
                moved[:, 2] *= 1.01
                self.assertEqual(_native.py_stage_geometry(np.asfortranarray(moved), 0), 0)
                self.assertEqual(_native.py_advance_resident(1, 0.)[0], 0)
                result, _ = readout(mesh, 5)
                self.assertTrue(all(np.isfinite(q).all() for q in result.values()))
                self.assertEqual(_native.py_geometry_info(), 2)
                self.assertEqual(_native.py_advance_resident(1, 0.)[0], 0)
                self.assertEqual(_native.py_geometry_info(), 2)

    def test_public_staging_and_interface_validation(self):
        from cfl3d import CFL3D
        with tempfile.TemporaryDirectory() as directory:
            with CFL3D(grouped_plate(), dict(resident=True, iterations=2,
                       multigrid_levels=3, turbulence='sa', cfl=1.,
                       writevolumesolution=False, writeTecplotSurfaceSolution=False,
                       progress=False), comm=None) as solver:
                solver.solve(output_directory=directory, final=False)
                original = solver.getVolumeCoordinates()
                bad = original.copy()
                bad[0, 0] = np.nan
                with self.assertRaises(ValueError):
                    solver.setVolumeCoordinates(bad)
                layout = VolumeLayout(solver.mesh)
                broken = original.copy()
                count = int(np.prod(solver.mesh.zones[0].dimensions))
                broken[:count, 0] += .01
                with self.assertRaisesRegex(ValueError, 'interface'):
                    solver.setVolumeCoordinates(broken)
                moved = original.copy()
                moved[:, 2] *= 1.01
                solver.setVolumeCoordinates(moved)
                np.testing.assert_array_equal(solver.getVolumeCoordinates(), original)
                result = solver.solve(output_directory=directory)
                np.testing.assert_array_equal(solver.getVolumeCoordinates(), moved)
                self.assertEqual(solver.getRuntimeInfo()['geometry_updates'], 1)
                self.assertEqual(solver.getRuntimeInfo()['initializations'], 1)
                self.assertEqual(result.start_mode, 'restart')
                solver.setVolumeCoordinates(moved)
                solver.solve(output_directory=directory)
                self.assertEqual(solver.getRuntimeInfo()['geometry_updates'], 1)

    def test_negative_volume_returns_failure_and_can_reset(self):
        from cfl3d import CFL3D, SolveFailure
        with tempfile.TemporaryDirectory() as directory:
            with CFL3D(grouped_plate(), dict(resident=True, iterations=2,
                       multigrid_levels=3, turbulence='sst', cfl=1.,
                       writevolumesolution=False, writeTecplotSurfaceSolution=False,
                       progress=False), comm=None) as solver:
                solver.solve(output_directory=directory)
                xyz = solver.getVolumeCoordinates()
                bad = xyz.copy()
                bad[:, 2] *= -1
                solver.setVolumeCoordinates(bad)
                before = solver.getRuntimeInfo()['cycles']
                result = solver.solve(output_directory=directory)
                self.assertIsInstance(result, SolveFailure)
                self.assertTrue(result.fail)
                self.assertEqual(solver.getRuntimeInfo()['cycles'], before)
                self.assertTrue(solver.solve(output_directory=directory).fail)
                np.testing.assert_array_equal(solver.getVolumeCoordinates(), xyz)
                with self.assertRaises(RuntimeError):
                    solver.getSolution()
                solver.resetFlow()
                # A rejected resident grid must not poison a later legacy solve.
                solver.options.resident = False
                self.assertFalse(solver.solve(output_directory=directory).fail)


if __name__ == '__main__':
    unittest.main()
