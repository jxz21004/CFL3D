import ctypes as C
from pathlib import Path
import tempfile
import unittest

import numpy as np
import cfl3d
from cfl3d import CFL3D, CGNSMesh, SolverOptions
from cgns_fixtures import write_adf, writer
from fixtures import box_mesh, flatplate_mesh


@unittest.skipUnless(Path(cfl3d.__file__).with_name("_cgns_bridge.so").is_file(),
                     "Build the CGNS bridge first: python tools/build_cgns.py")
class NativeCGNSTests(unittest.TestCase):
    def assert_mesh_equal(self, expected, actual):
        self.assertEqual([z.name for z in expected.zones], [z.name for z in actual.zones])
        for a, b in zip(expected.zones, actual.zones):
            np.testing.assert_array_equal(a.coordinates, b.coordinates)
            self.assertEqual(len(a.boundaries), len(b.boundaries))
            self.assertEqual(len(a.connections), len(b.connections))
            for x, y in zip(a.boundaries, b.boundaries):
                self.assertEqual((x.name, x.bc_type), (y.name, y.bc_type))
                np.testing.assert_array_equal(x.point_range, y.point_range)
                self.assertEqual(x.data.keys(), y.data.keys())
                for key in x.data:
                    np.testing.assert_array_equal(x.data[key], y.data[key])
            for x, y in zip(a.connections, b.connections):
                self.assertEqual(x.donor, y.donor)
                for key in ("point_range", "donor_range", "transform"):
                    np.testing.assert_array_equal(getattr(x, key), getattr(y, key))

    def test_adf_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for mesh in (box_mesh(), box_mesh(True), flatplate_mesh()):
                path = write_adf(mesh, Path(directory) / "mesh.cgns")
                restored = CGNSMesh.from_file(path)
                self.assert_mesh_equal(mesh, restored)
                arrays, xyz = CFL3D(restored, SolverOptions(turbulence="sst", multigrid_levels=3))._arrays()
                self.assertEqual(len(xyz), sum(np.prod(z.dimensions) for z in mesh.zones))
                self.assertEqual(arrays["blocks"][0, 0], 2)

    def test_signed_connection_transform(self):
        mesh = box_mesh(True)
        mesh.zones[1].coordinates = mesh.zones[1].coordinates[::-1, :, ::-1].copy()
        for bc in mesh.zones[1].boundaries:
            for axis in (0, 2):
                bc.point_range[axis] = mesh.zones[1].dimensions[axis]+1-bc.point_range[axis]
        for i, zone in enumerate(mesh.zones):
            cn = zone.connections[0]
            p = cn.donor_range if i == 0 else cn.point_range
            for axis in (0, 2):
                p[axis] = mesh.zones[1].dimensions[axis]+1-p[axis]
            cn.transform = np.array([-1, 2, -3])
        with tempfile.TemporaryDirectory() as directory:
            path = write_adf(mesh, Path(directory) / "signed.cgns")
            restored = CGNSMesh.from_file(path, backend="cgns")
            self.assert_mesh_equal(mesh, restored)
            CFL3D(restored, SolverOptions(turbulence="euler"))._arrays()

    def test_bc_dataset_and_heatflux_guard(self):
        mesh = flatplate_mesh()
        wall = next(b for b in mesh.zones[0].boundaries if b.name == "plate")
        wall.bc_type = "BCWallViscousHeatFlux"
        wall.data = {"HeatFlux": np.array([0.])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heatflux.cgns"
            restored = CGNSMesh.from_file(write_adf(mesh, path))
            self.assert_mesh_equal(mesh, restored)
            CFL3D(restored)._arrays()
            wall.data["HeatFlux"][:] = 1.
            restored = CGNSMesh.from_file(write_adf(mesh, path))
            with self.assertRaises(NotImplementedError):
                CFL3D(restored)._arrays()

    def test_base_selection(self):
        api = writer()
        with tempfile.TemporaryDirectory() as directory:
            path = write_adf(box_mesh(), Path(directory) / "bases.cgns", base="MeshBase")
            with api.open(path, api.constant("CG_MODE_MODIFY")) as fn:
                b = C.c_int()
                api.call("cg_base_write", fn, b"Other", 3, 3, C.byref(b))
            with self.assertRaisesRegex(ValueError, "Select exactly one"):
                CGNSMesh.from_file(path)
            self.assert_mesh_equal(box_mesh(), CGNSMesh.from_file(path, base="MeshBase"))

    def test_cellcenter_boundary_rejected(self):
        api = writer()
        with tempfile.TemporaryDirectory() as directory:
            path = write_adf(box_mesh(), Path(directory) / "location.cgns")
            with api.open(path, api.constant("CG_MODE_MODIFY")) as fn:
                api.call("cg_boco_gridlocation_write", fn, 1, 1, 1, api.constant("CellCenter"))
            with self.assertRaisesRegex(NotImplementedError, "GridLocation must be Vertex"):
                CGNSMesh.from_file(path)

    def test_periodic_connection_rejected(self):
        api = writer()
        with tempfile.TemporaryDirectory() as directory:
            path = write_adf(box_mesh(True), Path(directory) / "periodic.cgns")
            zero = (C.c_float * 3)()
            with api.open(path, api.constant("CG_MODE_MODIFY")) as fn:
                api.call("cg_1to1_periodic_write", fn, 1, 1, 1, zero, zero, zero)
            with self.assertRaisesRegex(NotImplementedError, "periodic/averaged"):
                CGNSMesh.from_file(path)

    def test_pointlist_boundary_rejected(self):
        api = writer()
        mesh = box_mesh()
        bc = mesh.zones[0].boundaries[0]
        points = (api.index * 6)(*bc.point_range.flatten(order="F"))
        with tempfile.TemporaryDirectory() as directory:
            path = write_adf(mesh, Path(directory) / "pointlist.cgns")
            with api.open(path, api.constant("CG_MODE_MODIFY")) as fn:
                index = C.c_int()
                bctype = next(i for i in range(api.constant("NofValidBCTypes"))
                              if api.lib.cg_BCTypeName(i).decode() == bc.bc_type)
                api.call("cg_boco_write", fn, 1, 1, bc.name.encode(), bctype,
                         api.constant("PointList"), 2, points, C.byref(index))
            with self.assertRaisesRegex(NotImplementedError, "PointList is not supported"):
                CGNSMesh.from_file(path)

    def test_bad_file_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.cgns"
            with self.assertRaises(FileNotFoundError):
                CGNSMesh.from_file(path)
            path.write_text("not a CGNS file")
            with self.assertRaisesRegex(ValueError, "cg_open"):
                CGNSMesh.from_file(path)


if __name__ == "__main__":
    unittest.main()
