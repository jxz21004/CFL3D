import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import numpy as np
from cfl3d import CGNSMesh, CFL3D, SolverOptions
from fixtures import box_mesh, flatplate_mesh, write_cgns


class MeshTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("h5py"), "Optional h5py is required for this HDF5 test")
    def test_cgns_roundtrip(self):
        for mesh in (box_mesh(), box_mesh(True), flatplate_mesh()):
            with tempfile.TemporaryDirectory() as directory:
                restored = CGNSMesh.from_file(write_cgns(mesh, Path(directory) / "mesh.cgns"))
                self.assertEqual([z.name for z in restored.zones], [z.name for z in mesh.zones])
                for a, b in zip(mesh.zones, restored.zones):
                    np.testing.assert_array_equal(a.coordinates, b.coordinates)
                    self.assertEqual(len(a.boundaries), len(b.boundaries))
                    self.assertEqual(len(a.connections), len(b.connections))
                arrays, xyz = CFL3D(restored, SolverOptions(turbulence="sst", multigrid_levels=3))._arrays()
                self.assertEqual(arrays["blocks"][0, 0], 2)
                self.assertEqual(len(xyz), sum(np.prod(z.dimensions) for z in mesh.zones))

    def test_invalid_options_and_coverage(self):
        for options in (SolverOptions(iterations=0), SolverOptions(cfl=-1), SolverOptions(turbulence="bad"),
                        SolverOptions(mach=float("nan")), SolverOptions(multigrid_levels=6)):
            with self.assertRaises(ValueError):
                CFL3D(box_mesh(), options)
        mesh = box_mesh(); mesh.zones[0].boundaries.pop()
        with self.assertRaises(ValueError): CFL3D(mesh)._arrays()
        mesh = box_mesh(); mesh.zones[0].boundaries.append(mesh.zones[0].boundaries[0])
        with self.assertRaises(ValueError): CFL3D(mesh)._arrays()

    def test_unknown_overrides_and_connectivity(self):
        with self.assertRaises(ValueError):
            CFL3D(box_mesh(), boundary_overrides={("typo", "wall"): (2004, (0, 0))})._arrays()
        mesh = box_mesh(True); mesh.zones[1].coordinates += .01
        with self.assertRaises(ValueError): CFL3D(mesh)._arrays()

    def test_generic_bcwall_matches_native_converter_and_respects_dataset(self):
        mesh = flatplate_mesh()
        wall = next(b for b in mesh.zones[0].boundaries if b.name == "plate")
        wall.bc_type = "BCWall"
        automatic, _ = CFL3D(mesh)._arrays()
        explicit, _ = CFL3D(mesh, boundary_overrides={
            ("flatplate", "plate"): (2004, (0., 0.))})._arrays()
        np.testing.assert_array_equal(automatic["bcs"], explicit["bcs"])
        np.testing.assert_array_equal(automatic["bcdata"], explicit["bcdata"])
        self.assertIn(2004, automatic["bcs"][2])
        wall.data = {"HeatFlux": np.array([1.])}
        with self.assertRaisesRegex(NotImplementedError, "BCDataSet needs an explicit override"):
            CFL3D(mesh)._arrays()

    def test_reciprocal_interface_with_reversed_endpoints(self):
        mesh = box_mesh(True)
        expected, _ = CFL3D(mesh, comm=None)._arrays()
        reciprocal = mesh.zones[1].connections[0]
        reciprocal.point_range = reciprocal.point_range[:, ::-1].copy()
        reciprocal.donor_range = reciprocal.donor_range[:, ::-1].copy()
        actual, _ = CFL3D(mesh, comm=None)._arrays()
        np.testing.assert_array_equal(actual['connections'], expected['connections'])
        np.testing.assert_array_equal(actual['counts'], expected['counts'])
        # Even a redundant reciprocal declaration must have a valid mapping.
        reciprocal.transform[2] = -3
        with self.assertRaisesRegex(ValueError, 'Transform and donor range disagree'):
            CFL3D(mesh, comm=None)._arrays()


@unittest.skipUnless(importlib.util.find_spec("cfl3d._native"), "Build the native extension first")
class NativeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("h5py"), "Optional h5py is required for this HDF5 test")
    def test_repeated_cgns_solve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for two_blocks, levels, alpha in ((False, 1, 0.), (False, 3, 2.), (True, 2, 0.)):
                path = write_cgns(box_mesh(two_blocks), root / "mesh.cgns")
                options = SolverOptions(mach=.3, alpha=alpha, turbulence="euler", iterations=5,
                                        multigrid_levels=levels)
                result = CFL3D(path, options).solve(output_directory=root / f"run{two_blocks}{levels}")
                self.assertEqual(result.history.shape, (5, 8))
                self.assertTrue((np.abs(result.history[:, 0]) < 1e-12).all())
                for field in result.fields.values():
                    np.testing.assert_allclose(field[..., 0], 1., atol=1e-12)
                    np.testing.assert_allclose(field[..., 1], .3 * np.cos(np.deg2rad(alpha)), atol=1e-12)
                    np.testing.assert_allclose(field[..., 4], 1 / 1.4, atol=1e-12)
                self.assertFalse(list(result.output_directory.glob("*.inp")))
                self.assertFalse((result.output_directory / "mesh.cgns").exists())
                self.assertTrue((result.output_directory / "out_000_volume.cgns").is_file())
                self.assertTrue((result.output_directory / "out_000_solution.npz").is_file())

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "Optional h5py is required for this HDF5 test")
    def test_viscous_models_and_mesh_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_cgns(flatplate_mesh(), root / "flatplate.cgns")
            for model in ("sa", "sst", "laminar", "komega"):
                options = SolverOptions(mach=.2, reynolds=6e6, temperature=460/1.8, turbulence=model,
                                        iterations=5, cfl=5, multigrid_levels=3,
                                        mesh_sequence=True, coarse_iterations=2)
                result = CFL3D(path, options).solve(output_directory=root / model)
                self.assertEqual(result.history.shape, (9, 8))
                self.assertTrue(np.isfinite(result.fields["flatplate"]).all())
                self.assertGreater(result.coefficients["CD"], 0.)
                self.assertGreater(np.ptp(result.fields["flatplate"][..., 1]), .01)

    def test_y_up_and_3d(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = box_mesh()
            xyz = mesh.zones[0].coordinates.copy()
            mesh.zones[0].coordinates[..., 1] = xyz[..., 2]
            mesh.zones[0].coordinates[..., 2] = -xyz[..., 1]
            result = CFL3D(mesh, SolverOptions(mach=.3, alpha=3., y_up=True,
                            turbulence="euler", iterations=3)).solve(output_directory=root / "y-up")
            np.testing.assert_allclose(result.fields["zone0"][..., 2], .3*np.sin(np.deg2rad(3)), atol=1e-12)
            np.testing.assert_allclose(result.fields["zone0"][..., 3], 0., atol=1e-12)
            from cfl3d import Boundary, CGNSMesh, Zone
            xyz = np.empty((5, 9, 9, 3))
            for i, j, k in np.ndindex(xyz.shape[:3]): xyz[i, j, k] = [j/8, -i/4, k/8]
            z = Zone("volume", xyz)
            for a in range(3):
                for side in (0, 1):
                    r = np.array([[1, n] for n in z.dimensions]); r[a] = [1 if not side else z.dimensions[a]]*2
                    z.boundaries.append(Boundary(f"face{a}{side}", "BCFarfield", r))
            result = CFL3D(CGNSMesh([z]), SolverOptions(mach=.3, beta=2., turbulence="euler",
                            iterations=3, multigrid_levels=2)).solve(output_directory=root / "3d")
            np.testing.assert_allclose(result.fields["volume"][..., 2], -.3*np.sin(np.deg2rad(2)), atol=1e-12)
            self.assertLess(np.max(result.history[:, 0]), 1e-12)

    def test_native_failure_returns_to_python(self):
        code = '''
from fixtures import box_mesh
from cfl3d import CFL3D, SolverOptions
import tempfile
mesh=box_mesh()
mesh.zones[0].coordinates[...,1] *= -1
with tempfile.TemporaryDirectory() as directory:
    try:
        CFL3D(mesh,SolverOptions(turbulence="euler",iterations=1)).solve(output_directory=directory)
    except RuntimeError as exc:
        assert "status -1" in str(exc)
        print("ERROR_RETURNED")
    else:
        raise AssertionError("Expected negative-volume failure")
'''
        import os
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        self.assertIn("ERROR_RETURNED", result.stdout)


if __name__ == "__main__": unittest.main()
