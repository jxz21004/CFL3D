"""Native gamma-Re_theta selection, ambient data, and all four restart fields."""
from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D, SolverOptions
from cfl3d.restart import records
from fixtures import flatplate_mesh
from test_surface_forces import grouped_plate


def transition_fields(state, mesh):
    """Read wrest.F's paired omega/k and gamma/Re_theta records for each block."""
    if state.zone_indices == ():
        return {}
    blocks = records(state.native)
    fields = {}
    start = 0
    indices = range(len(mesh.zones)) if state.zone_indices is None else state.zone_indices
    for index in indices:
        zone = mesh.zones[index]
        count = int(np.prod(np.array(zone.dimensions)-1))
        for i in range(max(1, start), len(blocks)-1):
            if (len(blocks[i]) == 4 and struct.unpack('=i', blocks[i])[0] == 4
                    and len(blocks[i-1]) == 2*count*8 and len(blocks[i+1]) == 2*count*8):
                fields[zone.name] = np.column_stack((
                    np.frombuffer(blocks[i-1], dtype=np.float64).reshape(count, 2, order='F'),
                    np.frombuffer(blocks[i+1], dtype=np.float64).reshape(count, 2, order='F')))
                start = i+2
                break
        else:
            raise AssertionError(f'Four-equation turbulence records missing for {zone.name}')
    return fields


class TransitionTests(unittest.TestCase):
    def test_parameters_and_case_insensitive_options(self):
        options = dict(TURBULENCE='gamma-theta', FREESTREAM_TURBULENCE_INTENSITY=.2,
                       FREESTREAM_EDDY_VISCOSITY=1., KEEP_AMBIENT_TURBULENCE=True)
        solver = CFL3D(grouped_plate(), options, comm=None)
        arrays, _ = solver._arrays()
        np.testing.assert_array_equal(arrays['blocks'][5:], 40)
        np.testing.assert_array_equal(arrays['keywords'][9:], [.2, 1., 1., 0., 0., 0.])
        for updates in ({'freestream_turbulence_intensity': 0},
                        {'freestream_eddy_viscosity': -1},
                        {'freestream_turbulence_intensity': float('nan')},
                        {'keep_ambient_turbulence': 1}, {'turbulence': 'sa'}):
            before = dict(solver.options)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                solver.setOptions(updates)
            self.assertEqual(dict(solver.options), before)

    def test_four_equation_memory_and_file_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = SolverOptions(turbulence='gamma-theta', mach=.2, iterations=5, cfl=1.,
                multigrid_levels=2, freestream_turbulence_intensity=.2,
                freestream_eddy_viscosity=1., keep_ambient_turbulence=True,
                writevolumesolution=False, writeTecplotSurfaceSolution=False)
            solver = CFL3D(grouped_plate(), options, comm=None)
            solver.solve(output_directory=root/'first')
            first_fields = transition_fields(solver.getFlowState(), solver.mesh)
            for values in first_fields.values():
                self.assertEqual(values.shape[1], 4)
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue((values[:, 3] > 0).all())
            checkpoint = solver.writeRestartFile(root/'checkpoint.npz')
            solver.options.iterations = 2
            warm = solver.solve(output_directory=root/'warm')
            self.assertEqual(warm.start_mode, 'restart')
            saved = solver.getFlowState()
            expected = CFL3D(grouped_plate(), replace(options, iterations=7), comm=None)
            full = expected.solve(output_directory=root/'full')
            resumed = CFL3D(grouped_plate(), replace(options, iterations=2, restartFile=checkpoint), comm=None)
            from_file = resumed.solve(output_directory=root/'file')
            for zone in warm.fields:
                np.testing.assert_array_equal(warm.fields[zone], full.fields[zone])
                np.testing.assert_array_equal(warm.fields[zone], from_file.fields[zone])
            for reference in (expected, resumed):
                actual_fields = transition_fields(saved, solver.mesh)
                reference_fields = transition_fields(reference.getFlowState(), reference.mesh)
                for zone in actual_fields:
                    np.testing.assert_array_equal(actual_fields[zone], reference_fields[zone])
            solver.options.freestream_turbulence_intensity = .4
            cold = solver.solve(output_directory=root/'new-ambient')
            self.assertEqual(cold.start_mode, 'cold')
            self.assertEqual(cold.reset_reasons, ('freestream_turbulence_intensity',))

    def test_three_dimensional_model_switch_starts_fresh(self):
        mesh = flatplate_mesh()
        zone = mesh.zones[0]
        zone.coordinates = np.linspace(zone.coordinates[0], zone.coordinates[-1], 5)
        for boundary in zone.boundaries:
            boundary.point_range[0][boundary.point_range[0] == 2] = 5
            if boundary.point_range[0, 0] == boundary.point_range[0, 1]:
                boundary.bc_type = 'BCSymmetryPlane'
        options = SolverOptions(turbulence='gamma-theta', mach=.2, iterations=1, cfl=.5,
            freestream_turbulence_intensity=.2, freestream_eddy_viscosity=1.,
            writevolumesolution=False, writeTecplotSurfaceSolution=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver = CFL3D(mesh, options, comm=None)
            result = solver.solve(output_directory=root/'transition')
            self.assertTrue(np.isfinite(result.fields[zone.name]).all())
            solver.getFlowState().validate(mesh, solver.options)
            solver.options.turbulence = 'sst'
            changed = solver.solve(output_directory=root/'switched')
            self.assertEqual((changed.start_mode, changed.reset_reasons), ('cold', ('turbulence',)))
            fresh = CFL3D(mesh, replace(options, turbulence='sst'), comm=None).solve(
                output_directory=root/'fresh')
            np.testing.assert_array_equal(changed.fields[zone.name], fresh.fields[zone.name])


if __name__ == '__main__':
    unittest.main()
