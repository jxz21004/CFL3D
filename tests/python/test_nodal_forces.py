"""Nodal force conservation, native moment consistency and dimensional scaling."""
import tempfile
import unittest

import numpy as np

from cfl3d import CFL3D
from cfl3d.solution_io import wall_vertices
from cfl3d.surface_mesh import distribute_surface_forces
from fixtures import box_mesh
from test_surface_forces import grouped_plate
from test_surface_mesh import wall_boxes


def coupling_mesh(y_up=True):
    """Unequal 3D zones without importing/initializing MPI in serial tests."""
    mesh = grouped_plate(split=25)
    for zone in mesh.zones:
        zone.coordinates = np.linspace(zone.coordinates[0], zone.coordinates[-1], 5)
        for boundary in zone.boundaries:
            boundary.point_range[0][boundary.point_range[0] == 2] = 5
            if boundary.point_range[0, 0] == boundary.point_range[0, 1]:
                boundary.bc_type = 'BCSymmetryPlane'
        for connection in zone.connections:
            for points in (connection.point_range, connection.donor_range):
                points[0][points[0] == 2] = 5
        if y_up:
            xyz = zone.coordinates.copy()
            zone.coordinates[..., 1] = xyz[..., 2]
            zone.coordinates[..., 2] = -xyz[..., 1]
    return mesh


def cell_resultants(solver, group, component='total'):
    """Independent native-cell reference, using its four-corner moment arm."""
    members = solver._group_members(group)
    options = solver._last_options
    pressure = options.freestream_pressure
    if pressure is None:
        pressure = solver.options.freestream_pressure
    scale = .5*options.gamma*pressure*options.mach**2*options.sref
    force, moment = np.zeros(3), np.zeros(3)
    for zone in solver._last_mesh.zones:
        for boundary in zone.boundaries:
            key = zone.name, boundary.name
            if key not in members or key not in solver._last_wall_forces:
                continue
            xyz = wall_vertices(zone, boundary)
            center = .25*(xyz[:-1, :-1]+xyz[1:, :-1]+xyz[1:, 1:]+xyz[:-1, 1:])
            raw = solver._last_wall_forces[key]
            values = raw[..., :3] if component == 'pressure' else raw[..., 3:]
            if component == 'total':
                values = raw[..., :3]+raw[..., 3:]
            values = values.reshape(-1, 3)*scale
            force += values.sum(axis=0)
            moment += np.cross(center.reshape(-1, 3)-options.moment_center, values).sum(axis=0)
    return force, moment


class NodalForceTests(unittest.TestCase):
    def test_native_all_six_wall_faces_and_euler_zero_viscosity(self):
        mesh = box_mesh()
        zone = mesh.zones[0]
        zone.coordinates = np.linspace(zone.coordinates[0], zone.coordinates[-1], 5)
        for boundary in zone.boundaries:
            boundary.point_range[0][boundary.point_range[0] == 2] = 5
            boundary.bc_type = 'BCWallInviscid'
        config = dict(turbulence='euler', mach=.2, alpha=12., beta=5.,
                      freestream_pressure=90000., moment_center=(.2, -.3, .4),
                      sref=2., cref=.7, bref=1.3, cfl=.5, iterations=2,
                      writevolumesolution=False, writeTecplotSurfaceSolution=False)
        with tempfile.TemporaryDirectory() as directory:
            solver = CFL3D(mesh, config, comm=None)
            solver.solve(output_directory=directory)
            surface = solver.getSurfaceMesh()
            forces = solver.getSurfaceForces()
            np.testing.assert_array_equal(forces, solver.getSurfaceForces(component='pressure'))
            np.testing.assert_array_equal(solver.getSurfaceForces(component='viscous'), 0.)
            loads = solver.getSolution()
            np.testing.assert_allclose(forces.sum(axis=0), [loads['F'+a] for a in 'xyz'], rtol=2e-12, atol=1e-11)
            np.testing.assert_allclose(np.cross(surface['coordinates']-config['moment_center'], forces).sum(axis=0),
                                       [loads['M'+a] for a in 'xyz'], rtol=2e-12, atol=1e-11)

    def test_two_quads_add_shared_corner_forces_without_averaging(self):
        # Two one-cell quads, each with pressure Fz=100 N and viscous Fx=8 N.
        # No native solve is needed for this analytic conservation fixture.
        mesh = wall_boxes()
        for zone in mesh.zones:
            zone.coordinates = zone.coordinates[:, [0, -1]][:, :, [0, -1]].copy()
            for boundary in zone.boundaries:
                boundary.point_range[boundary.point_range == 9] = 2
            for connection in zone.connections:
                for points in (connection.point_range, connection.donor_range):
                    points[points == 9] = 2
        solver = CFL3D(mesh, comm=None)
        surface = solver.getSurfaceMesh()
        raw = {(z.name, 'face20'): np.array([[[0., 0., 100., 8., 0., 0.]]]) for z in mesh.zones}
        forces = distribute_surface_forces(mesh, surface, raw, 1., None, solver._checked)
        shared = surface['coordinates'][:, 0] == 1.
        self.assertEqual(len(forces), 6)
        self.assertEqual(shared.sum(), 2)
        np.testing.assert_array_equal(forces[shared, 2], 50.)
        np.testing.assert_array_equal(forces[~shared, 2], 25.)
        np.testing.assert_array_equal(forces[shared, 3], 4.)
        np.testing.assert_array_equal(forces[~shared, 3], 2.)
        np.testing.assert_array_equal(forces[:, :3].sum(axis=0), [0., 0., 200.])
        np.testing.assert_array_equal(forces[:, 3:].sum(axis=0), [16., 0., 0.])

    def test_warped_quad_preserves_force_moment_and_virtual_work(self):
        mesh = wall_boxes()
        mesh.zones = mesh.zones[:1]
        mesh.zones[0].connections.clear()
        # A warped surface on an unequal rectangular parameter grid.
        xyz = mesh.zones[0].coordinates
        xyz[..., 2] += .3*xyz[..., 0]*xyz[..., 1]
        solver = CFL3D(mesh, comm=None)
        surface = solver.getSurfaceMesh()
        count = len(surface['face_ids'])
        raw = np.arange(1., count*6+1).reshape(1, count, 6)
        values = distribute_surface_forces(mesh, surface, {('zone0', 'face20'): raw}, 2., None, solver._checked)
        rows = np.searchsorted(surface['node_ids'], surface['connectivity'])
        points = surface['coordinates'][rows]
        center = points.mean(axis=1)
        offset = [.2, -.4, .6]
        cell_force = (raw[..., :3]+raw[..., 3:]).reshape(-1, 3)*2.
        nodal_force = values[:, :3]+values[:, 3:]
        np.testing.assert_allclose(nodal_force.sum(axis=0), cell_force.sum(axis=0), rtol=1e-14)
        np.testing.assert_allclose(np.cross(surface['coordinates']-offset, nodal_force).sum(axis=0),
                                   np.cross(center-offset, cell_force).sum(axis=0), rtol=1e-14, atol=1e-12)
        displacement = np.random.default_rng(27).normal(size=nodal_force.shape)
        cell_displacement = displacement[rows].mean(axis=1)
        np.testing.assert_allclose(np.sum(nodal_force*displacement), np.sum(cell_force*cell_displacement),
                                   rtol=1e-14, atol=1e-12)

    def test_native_components_groups_and_resultants(self):
        config = dict(turbulence='sst', mach=.2, alpha=2., beta=3., y_up=True,
                      freestream_pressure=90000., sref=2., cref=.7, bref=1.3,
                      moment_center=(.2, -.1, .3), cfl=1., iterations=5,
                      multigrid_levels=3, writevolumesolution=False, writeTecplotSurfaceSolution=False)
        with tempfile.TemporaryDirectory() as directory:
            solver = CFL3D(coupling_mesh(), config, comm=None,
                           groups={'Overlap': ['Wing', 'Forward', 'Wing', 'Aft']})
            result = solver.solve(output_directory=directory)
            surface = solver.getSurfaceMesh('Wing')
            total = solver.getSurfaceForces('Wing')
            pressure = solver.getSurfaceForces('Wing', component='pressure')
            viscous = solver.getSurfaceForces('Wing', component='viscous')
            self.assertEqual(total.shape, surface['coordinates'].shape)
            np.testing.assert_array_equal(total, pressure+viscous)
            self.assertGreater(np.linalg.norm(pressure), 0.)
            self.assertGreater(np.linalg.norm(viscous), 0.)
            for group in (None, 'Wing', 'Overlap', 'Forward', 'Aft'):
                geometry = solver.getSurfaceMesh(group)
                for kind in ('total', 'pressure', 'viscous'):
                    forces = solver.getSurfaceForces(group, component=kind)
                    force, moment = cell_resultants(solver, group, kind)
                    np.testing.assert_allclose(forces.sum(axis=0), force, rtol=2e-12, atol=1e-11)
                    np.testing.assert_allclose(np.cross(geometry['coordinates']-config['moment_center'], forces).sum(axis=0),
                                               moment, rtol=2e-12, atol=1e-11)
                loads = solver.getSolution(group)
                prefix = '' if group is None else group+'_'
                force = solver.getSurfaceForces(group)
                np.testing.assert_allclose(force.sum(axis=0), [loads[prefix+'F'+a] for a in 'xyz'],
                                           rtol=2e-12, atol=1e-11)
                np.testing.assert_allclose(np.cross(geometry['coordinates']-config['moment_center'], force).sum(axis=0),
                                           [loads[prefix+'M'+a] for a in 'xyz'], rtol=2e-12, atol=1e-11)
            # Selecting a group must not contaminate its shared edge with the
            # unselected neighbour's load. Assemble both subgroups by global ID.
            combined = {int(node): np.zeros(3) for node in surface['node_ids']}
            for group in ('Forward', 'Aft'):
                ids = solver.getSurfaceMesh(group)['node_ids']
                for node, force in zip(ids, solver.getSurfaceForces(group)):
                    combined[int(node)] += force
            np.testing.assert_allclose([combined[int(n)] for n in surface['node_ids']], total, rtol=1e-14, atol=1e-12)
            # Saved native readouts remain authoritative if the user edits the
            # public SolverResult, returned arrays, or current mesh/options.
            saved = total.copy()
            total[:] = 0.
            for values in result.wall_forces.values():
                values[:] = 0.
            for zone in solver.mesh.zones:
                zone.coordinates *= 2.
            solver.setOptions({'freestream_pressure': 180000., 'mach': .3, 'sref': 4.})
            np.testing.assert_array_equal(solver.getSurfaceForces('Wing'), saved)
            solver.close()
            np.testing.assert_array_equal(solver.getSurfaceForces('Wing'), saved)

    def test_pressure_required_then_supplied_after_solve(self):
        config = dict(turbulence='sst', mach=.2, iterations=2, cfl=1.,
                      writevolumesolution=False, writeTecplotSurfaceSolution=False)
        with tempfile.TemporaryDirectory() as directory:
            solver = CFL3D(grouped_plate(), config, comm=None)
            with self.assertRaisesRegex(RuntimeError, 'solve'):
                solver.getSurfaceForces()
            solver.solve(output_directory=directory)
            with self.assertRaisesRegex(ValueError, 'freestream pressure'):
                solver.getSurfaceForces()
            solver.options.freestream_pressure = 90000.
            first = solver.getSurfaceForces('Wing')
            solver.options.freestream_pressure = 180000.
            np.testing.assert_array_equal(solver.getSurfaceForces('Wing'), 2.*first)
            for component in ('traction', 'bad', 1, True):
                with self.subTest(component=component), self.assertRaises(ValueError):
                    solver.getSurfaceForces(component=component)
            for group in ('missing', 'face21'):
                with self.subTest(group=group), self.assertRaises(ValueError):
                    solver.getSurfaceForces(group)


if __name__ == '__main__':
    unittest.main()
