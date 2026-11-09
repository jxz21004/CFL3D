import unittest
import numpy as np
from cfl3d import CGNSMesh
from cfl3d.cgns import Zone, Connection
from cfl3d.mesh import repair_interface_orientations


def fixture():
    i,j,k=np.indices((3,4,5))
    a=Zone('a',np.stack((i,j,k),axis=-1).astype(float))
    b=Zone('b',np.stack((i+2,3-j,k),axis=-1).astype(float))
    p=np.array([[3,3],[1,4],[1,5]])
    q=np.array([[1,1],[1,4],[1,5]])
    a.connections=[Connection('b',p.copy(),q.copy(),np.array([1,2,3]))]
    b.connections=[Connection('a',q.copy(),p.copy(),np.array([1,2,3]))]
    return CGNSMesh([a,b])


class TestRepairConnectivity(unittest.TestCase):
    def test_axis_swap_and_reversed_source_range(self):
        mesh=fixture()
        b=mesh.zones[1]
        b.coordinates=b.coordinates.transpose(1,0,2,3).copy()
        a=mesh.zones[0].connections[0]
        a.point_range[1]=[4,1]
        a.donor_range=np.array([[1,4],[1,1],[1,5]])
        a.transform=np.array([2,-1,3])
        # The endpoint direction, not just Transform, must be repaired.
        a.donor_range[0]=[4,1]
        b.connections=[]
        self.assertEqual(len(repair_interface_orientations(mesh)),1)
        np.testing.assert_array_equal(a.transform,[2,-1,3])
        np.testing.assert_array_equal(a.donor_range[0],[1,4])
        self.assertEqual(repair_interface_orientations(mesh),[])

    def test_reciprocal_reversal_and_idempotence(self):
        mesh=fixture()
        coords=[z.coordinates.copy() for z in mesh.zones]
        report=repair_interface_orientations(mesh)
        self.assertEqual(len(report),2)
        for z,x in zip(mesh.zones,coords):
            np.testing.assert_array_equal(z.coordinates,x)
            np.testing.assert_array_equal(z.connections[0].transform,[1,-2,3])
            np.testing.assert_array_equal(z.connections[0].donor_range[1],[4,1])
        self.assertEqual(repair_interface_orientations(mesh),[])

    def test_failure_is_atomic(self):
        mesh=fixture()
        mesh.zones[1].connections[0].donor_range[0]=[1,1]
        with self.assertRaisesRegex(ValueError,'found 0'):
            repair_interface_orientations(mesh)
        np.testing.assert_array_equal(mesh.zones[0].connections[0].transform,[1,2,3])

    def test_ambiguous_geometry_rejected(self):
        mesh=fixture()
        for z in mesh.zones:z.coordinates[:]=0
        mesh.zones[0].connections[0].transform=np.array([1,1,3])
        with self.assertRaisesRegex(ValueError,'found 4'):
            repair_interface_orientations(mesh)

    def test_requires_coordinates(self):
        mesh=fixture(); mesh.zones[1].coordinates=None
        with self.assertRaisesRegex(ValueError,'complete mesh'):
            repair_interface_orientations(mesh)

if __name__=='__main__':unittest.main()
