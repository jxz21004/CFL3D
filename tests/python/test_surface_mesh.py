"""Coupling geometry identities, independent copies and solve-time partitions."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cfl3d import CFL3D, CGNSMesh, Connection
from cfl3d.surface_mesh import SurfaceTopology
from cgns_fixtures import write_adf
from fixtures import box_mesh
from test_surface_forces import grouped_plate


def wall_boxes():
    mesh = box_mesh(True)
    for zone, family in zip(mesh.zones, ('Front', 'Aft')):
        wall = next(b for b in zone.boundaries if b.name == 'face20')
        wall.bc_type = 'BCWallViscous'
        wall.family = family
        wall.additional_families = ('Wing',)
    return mesh


def tiled_wall():
    """Four zones share one corner; only one side declares each connection."""
    zones = []
    for row in range(2):
        for column in range(2):
            zone = box_mesh().zones[0]
            zone.name = f'z{row}{column}'
            zone.coordinates += [column, -row, 0.]
            next(b for b in zone.boundaries if b.name == 'face20').bc_type = 'BCWallViscous'
            zones.append(zone)
    for index, zone in enumerate(zones):
        row, column = divmod(index, 2)
        for axis, neighbor in ((0, index+2 if row == 0 else None),
                               (1, index+1 if column == 0 else None)):
            if neighbor is None:
                continue
            donor = zones[neighbor]
            points = np.array([[1, n] for n in zone.dimensions])
            points[axis] = zone.dimensions[axis]
            other = points.copy()
            other[axis] = 1
            zone.connections.append(Connection(donor.name, points, other, np.array([1, 2, 3])))
            zone.boundaries = [b for b in zone.boundaries if b.name != f'face{axis}1']
            donor.boundaries = [b for b in donor.boundaries if b.name != f'face{axis}0']
    return CGNSMesh(zones)


class SurfaceMeshTests(unittest.TestCase):
    def test_before_solve_groups_topology_and_area(self):
        solver = CFL3D(wall_boxes(), comm=None, groups={'Both': ['Wing', 'Front', 'Wing']})
        data = solver.getSurfaceMesh()
        self.assertEqual(data['coordinates'].shape, (34, 3))  # 36 patch vertices minus the shared edge.
        self.assertEqual(data['connectivity'].shape, (16, 4))
        self.assertEqual(len(np.unique(data['node_ids'])), 34)
        np.testing.assert_array_equal(data['node_owners'], 0)
        self.assertEqual(len(np.unique(data['face_ids'])), 16)
        points = dict(zip(data['node_ids'], data['coordinates']))
        quads = np.array([[points[node] for node in face] for face in data['connectivity']])
        vectors = .5*np.cross(quads[:, 2]-quads[:, 0], quads[:, 3]-quads[:, 1])
        np.testing.assert_allclose(vectors[:, :2], 0.)
        self.assertTrue((vectors[:, 2] > 0).all())
        self.assertAlmostEqual(np.linalg.norm(vectors, axis=1).sum(), 2.)
        for name in ('Wing', 'Both'):
            group = solver.getSurfaceMesh(name)
            for key in ('coordinates', 'node_ids', 'connectivity', 'face_ids'):
                np.testing.assert_array_equal(group[key], data[key])
        front, aft = solver.getSurfaceMesh('Front'), solver.getSurfaceMesh('Aft')
        self.assertEqual(len(set(front['node_ids']) & set(aft['node_ids'])), 2)
        self.assertEqual(len(front['connectivity']), 8)
        self.assertEqual(len(aft['connectivity']), 8)
        self.assertEqual(data['units']['coordinates'], 'm')
        self.assertIsNone(solver._last_result)  # Geometry access never initializes Fortran.

    def test_coordinates_are_not_used_to_merge_nodes(self):
        mesh = wall_boxes()
        for zone in mesh.zones:
            zone.connections.clear()
        mesh.zones[1].coordinates = mesh.zones[0].coordinates.copy()
        data = CFL3D(mesh, comm=None).getSurfaceMesh()
        self.assertEqual(len(data['node_ids']), 36)
        self.assertEqual(len(np.unique(data['coordinates'], axis=0)), 18)

    def test_transitive_corner_and_one_sided_connections(self):
        mesh = tiled_wall()
        data = CFL3D(mesh, comm=None).getSurfaceMesh()
        self.assertEqual(data['coordinates'].shape, (51, 3))
        self.assertEqual(data['connectivity'].shape, (32, 4))
        topology = SurfaceTopology(mesh)
        corner = topology.canonical(0, (1, 8, 0))
        for zone, point in ((1, (1, 0, 0)), (2, (0, 8, 0)), (3, (0, 0, 0))):
            self.assertEqual(topology.canonical(zone, point), corner)

    def test_signed_permuted_reciprocal_connections(self):
        mesh = wall_boxes()
        expected = CFL3D(deepcopy(mesh), comm=None).getSurfaceMesh()
        zone = mesh.zones[1]
        zone.coordinates = zone.coordinates[::-1, :, ::-1].copy()
        for boundary in zone.boundaries:
            for axis in (0, 2):
                boundary.point_range[axis] = zone.dimensions[axis]+1-boundary.point_range[axis]
        for index, z in enumerate(mesh.zones):
            connection = z.connections[0]
            points = connection.donor_range if index == 0 else connection.point_range
            for axis in (0, 2):
                points[axis] = zone.dimensions[axis]+1-points[axis]
            connection.transform = np.array([-1, 2, -3])
        # Permute the second zone's axes; update both directions independently.
        permutation = [2, 1, 0]
        zone.coordinates = zone.coordinates.transpose(2, 1, 0, 3).copy()
        for boundary in zone.boundaries:
            boundary.point_range = boundary.point_range[permutation]
        left, right = mesh.zones[0].connections[0], zone.connections[0]
        left.donor_range = left.donor_range[permutation]
        left.transform = np.array([(1 if t > 0 else -1)*(permutation.index(abs(int(t))-1)+1)
                                   for t in left.transform])
        right.point_range = right.point_range[permutation]
        right.transform = right.transform[permutation]
        actual = CFL3D(mesh, comm=None).getSurfaceMesh()
        self.assertEqual(len(actual['node_ids']), len(expected['node_ids']))
        self.assertEqual(set(map(tuple, actual['coordinates'])), set(map(tuple, expected['coordinates'])))
        shared = set(expected['node_ids']) & set(actual['node_ids'])
        a = dict(zip(actual['node_ids'], actual['coordinates']))
        b = dict(zip(expected['node_ids'], expected['coordinates']))
        # Canonical vertices in zone 0 are invariant under donor reindexing.
        for node in shared:
            if node < np.prod(mesh.zones[0].dimensions):
                np.testing.assert_array_equal(a[node], b[node])

    def test_reversed_patch_endpoints_and_independent_arrays(self):
        solver = CFL3D(wall_boxes(), comm=None)
        first = solver.getSurfaceMesh()
        for zone in solver.mesh.zones:
            for boundary in zone.boundaries:
                boundary.point_range = boundary.point_range[:, ::-1].copy()
        actual = solver.getSurfaceMesh()
        for key in ('coordinates', 'node_ids', 'connectivity', 'face_ids'):
            np.testing.assert_array_equal(actual[key], first[key])
        first['coordinates'][:] = 999.
        first['node_ids'][:] = -1
        first['connectivity'][:] = -1
        self.assertTrue((solver.getSurfaceMesh()['node_ids'] >= 0).all())
        self.assertLess(solver.getSurfaceMesh()['coordinates'].max(), 999.)

    def test_invalid_group_mapping_and_geometry(self):
        solver = CFL3D(wall_boxes(), comm=None)
        for group in ('missing', 'face21', ['Wing'], 1):
            with self.subTest(group=group), self.assertRaises(ValueError):
                solver.getSurfaceMesh(group)
        bad = wall_boxes()
        bad.zones[1].coordinates += .01
        with self.assertRaisesRegex(ValueError, 'coordinates do not coincide'):
            CFL3D(bad, comm=None).getSurfaceMesh('Aft')
        bad = wall_boxes()
        bad.zones[0].connections[0].transform[0] = 2
        with self.assertRaisesRegex(ValueError, 'signed permutation'):
            CFL3D(bad, comm=None).getSurfaceMesh()
        bad = wall_boxes()
        bad.zones[0].boundaries[0].point_range[1] = [0, 9]
        with self.assertRaisesRegex(ValueError, 'PointRange'):
            CFL3D(bad, comm=None).getSurfaceMesh()
        with self.assertRaisesRegex(ValueError, 'no wall'):
            CFL3D(box_mesh(), comm=None).getSurfaceMesh()

    def test_adf_and_solution_geometry_snapshot(self):
        config = dict(turbulence='sst', mach=.2, cfl=1., iterations=2,
                      writevolumesolution=False, writeTecplotSurfaceSolution=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_adf(grouped_plate(split=25), root/'mesh.cgns')
            solver = CFL3D(path, config, comm=None)
            before = solver.getSurfaceMesh('Wing')
            solver.solve(output_directory=root/'run')
            for zone in solver.mesh.zones:
                zone.coordinates *= 2.
            after = solver.getSurfaceMesh('Wing')
            for key in ('coordinates', 'node_ids', 'connectivity', 'face_ids'):
                np.testing.assert_array_equal(before[key], after[key])
            after['coordinates'][:] = 999.
            np.testing.assert_array_equal(solver.getSurfaceMesh('Wing')['coordinates'], before['coordinates'])
            solver.close()
            np.testing.assert_array_equal(solver.getSurfaceMesh('Wing')['coordinates'], before['coordinates'])


if __name__ == '__main__':
    unittest.main()
