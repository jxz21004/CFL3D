"""Mesh exchange ordering and staging without a native geometry mutation."""
import unittest
import numpy as np
from cfl3d.cgns import CGNSMesh, Zone
from cfl3d.mesh import VolumeLayout
from cfl3d.mesh.idwarp import local_surface


class MeshExchangeTests(unittest.TestCase):
    def test_unequal_zone_roundtrip_and_partial_ownership(self):
        a = np.arange(2*3*4*3.).reshape(2, 3, 4, 3)
        b = np.arange(3*2*2*3.).reshape(3, 2, 2, 3)+1000
        mesh = CGNSMesh([Zone('a', a), Zone('b', b)])
        layout = VolumeLayout(mesh)
        xyz = layout.coordinates(mesh)
        expected = [v[i, j, k] for v in (a, b) for k in range(v.shape[2])
                    for j in range(v.shape[1]) for i in range(v.shape[0])]
        np.testing.assert_array_equal(xyz, expected)
        unpacked = layout.unpack(xyz)
        np.testing.assert_array_equal(unpacked['a'], a)
        np.testing.assert_array_equal(unpacked['b'], b)
        unpacked['a'][:] = -1
        self.assertTrue(np.all(a >= 0))
        mesh.zones[0].coordinate_dimensions = a.shape[:3]
        mesh.zones[0].coordinates = None
        with self.assertRaises(ValueError):
            layout.coordinates(mesh)
        partial = VolumeLayout(mesh)
        np.testing.assert_array_equal(partial.component_indices(), np.arange(a.size, a.size+b.size))
        np.testing.assert_array_equal(partial.unpack(partial.coordinates(mesh))['b'], b)
        with self.assertRaises(ValueError):
            partial.unpack(np.full((partial.size, 3), np.nan))

    def test_surface_global_ids_become_local_connectivity(self):
        surface = dict(node_ids=np.array([90, 10, 70, 30]),
                       coordinates=np.array([[9,0,0], [1,0,0], [7,0,0], [3,0,0]]),
                       connectivity=np.array([[90, 10, 70, 30]]))
        xyz, conn, sizes = local_surface(surface)
        np.testing.assert_array_equal(xyz[conn], surface['coordinates'])
        np.testing.assert_array_equal(sizes, [4])


if __name__ == '__main__':
    unittest.main()
