"""Dynamic settings, warm control changes, and automatic cold-start policy."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D, SolverOptions
from cgns_fixtures import write_adf
from fixtures import box_mesh
from test_surface_forces import grouped_plate


class DynamicOptionsTests(unittest.TestCase):
    def test_typed_options_are_mutable_and_updates_are_atomic(self):
        config = SolverOptions(iterations=4, first_order_iterations=2)
        config.CFL = 3.0
        config.update({'ITERATIONS': 1, 'FIRST_ORDER_ITERATIONS': 0})
        self.assertEqual((config.cfl, config.iterations, config.first_order_iterations), (3., 1, 0))
        with self.assertRaises(ValueError):
            config.cfl = 0
        self.assertEqual(config.cfl, 3.)
        with self.assertRaises(ValueError):
            config.update({'cfl': 4., 'first_order_iterations': 2})
        self.assertEqual((config.cfl, config.first_order_iterations), (3., 0))
        categories = CFL3D.getOptionCategories()
        self.assertIn('mach', categories['physics'])
        self.assertIn('cfl', categories['numerical'])
        self.assertEqual(set().union(*map(set, categories.values())), set(SolverOptions.__dataclass_fields__))

    def test_physics_resets_but_reverted_edits_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = dict(turbulence='sst', mach=.2, iterations=2, cfl=1.,
                           numberSolutions=False, writevolumesolution=False,
                           writeTecplotSurfaceSolution=False, output_directory=root)
            solver = CFL3D(grouped_plate(), options, comm=None)
            first = solver.solve()
            solver.options.mach = .3
            solver.options.mach = .2
            warm = solver.solve()
            self.assertEqual((warm.start_mode, warm.reset_reasons, warm.solve_index), ('restart', (), 0))
            full = CFL3D(grouped_plate(), {**options, 'iterations': 4}, comm=None).solve(output_directory=root/'full')
            for zone in warm.fields:
                np.testing.assert_array_equal(warm.fields[zone], full.fields[zone])
            saved_log = first.output_paths['log'].read_bytes()
            for changes in ({'mach': .3}, {'gamma': 1.3}, {'turbulence': 'sa'},
                            {'temperature': 270.}, {'reynolds': 2e6}, {'full_navier_stokes': False}):
                solver.setOptions(changes)
                result = solver.solve()
                self.assertEqual(result.start_mode, 'cold')
                self.assertEqual(set(result.reset_reasons), set(changes))
                self.assertEqual(result.iteration_offset, 0)
                fresh = CFL3D(grouped_plate(), dict(solver.options), comm=None).solve(
                    output_directory=root/f'fresh-{next(iter(changes))}')
                for zone in result.fields:
                    np.testing.assert_array_equal(result.fields[zone], fresh.fields[zone])
            self.assertEqual(first.output_paths['log'].read_bytes(), saved_log)

    def test_old_restart_file_does_not_cancel_a_physical_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = dict(turbulence='sst', iterations=2, cfl=1., numberSolutions=False,
                           writevolumesolution=False, writeTecplotSurfaceSolution=False)
            original = CFL3D(grouped_plate(), options, comm=None)
            original.solve(output_directory=root/'original')
            checkpoint = original.writeRestartFile(root/'checkpoint.npz')
            solver = CFL3D(grouped_plate(), {**options, 'restartFile': checkpoint}, comm=None)
            solver.solve(output_directory=root/'continued')
            solver.options.gamma = 1.3
            cold = solver.solve(output_directory=root/'continued')
            self.assertEqual((cold.start_mode, cold.reset_reasons), ('cold', ('gamma',)))
            fresh = CFL3D(grouped_plate(), {**options, 'gamma': 1.3}, comm=None).solve(
                output_directory=root/'fresh')
            for zone in cold.fields:
                np.testing.assert_array_equal(cold.fields[zone], fresh.fields[zone])
            warm = solver.solve(output_directory=root/'continued')
            self.assertEqual(warm.start_mode, 'restart')

    def test_reference_and_multigrid_changes_retain_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = dict(turbulence='sst', mach=.2, alpha=2., iterations=2, cfl=1.,
                           freestream_pressure=101325., numberSolutions=False,
                           writevolumesolution=False, writeTecplotSurfaceSolution=False)
            solver = CFL3D(grouped_plate(), options, comm=None)
            solver.solve(output_directory=root/'stream')
            old_loads = solver.getSolution()
            solver.setOptions({'sref': 2., 'cref': 3., 'moment_center': [0.2, 0., 0.],
                               'iterations': 1, 'cfl': 2., 'multigrid_levels': 2})
            self.assertEqual(solver.getSolution(), old_loads)  # Old results keep their own reference values.
            warm = solver.solve(output_directory=root/'stream')
            self.assertEqual((warm.start_mode, warm.reset_reasons, warm.iteration_offset), ('restart', (), 2))
            reference = CFL3D(grouped_plate(), options, comm=None)
            reference.solve(output_directory=root/'reference')
            reference.setOptions({'iterations': 1, 'cfl': 2., 'multigrid_levels': 2})
            expected = reference.solve(output_directory=root/'reference')
            for zone in warm.fields:
                np.testing.assert_array_equal(warm.fields[zone], expected.fields[zone])
            loads, base = solver.getSolution(), reference.getSolution()
            self.assertAlmostEqual(loads['CL'], base['CL']/2, places=13)
            self.assertAlmostEqual(loads['L'], base['L'], places=9)

    def test_explicit_alpha_overrides_trim_restart_angle(self):
        with tempfile.TemporaryDirectory() as directory:
            options = dict(turbulence='euler', iterations=2, alpha=1., cfl=1., target_cl=.1,
                           alpha_step_limit=.02, alpha_update_interval=1,
                           writevolumesolution=False, writeTecplotSurfaceSolution=False,
                           output_directory=directory, numberSolutions=False)
            solver = CFL3D(box_mesh(True), options, comm=None)
            first = solver.solve()
            self.assertAlmostEqual(first.final_alpha, 1.04)
            solver.setOptions({'alpha': 4., 'beta': 3., 'iterations': 1})
            changed = solver.solve()
            self.assertEqual((changed.start_mode, changed.reset_reasons), ('restart', ()))
            self.assertAlmostEqual(changed.final_alpha, 4.02)
            continued = solver.solve()
            self.assertAlmostEqual(continued.final_alpha, 4.04)  # No explicit angle edit: preserve trim state.

    def test_grid_reloads_and_memory_geometry_changes_initialize_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = write_adf(box_mesh(), root/'a.cgns')
            b = write_adf(box_mesh(True), root/'b.cgns')
            solver = CFL3D(options=dict(input_grid=a, turbulence='euler', iterations=2,
                           numberSolutions=False, output_directory=root/'stream',
                           writevolumesolution=False, writeTecplotSurfaceSolution=False), comm=None)
            solver.solve()
            solver.options.input_grid = b
            new = solver.solve()
            self.assertEqual((new.start_mode, new.solve_index, len(new.fields)), ('cold', 1, 2))
            self.assertIn('input_grid', new.reset_reasons)
            for zone in solver.mesh.zones:
                zone.coordinates[..., 0] *= 1.01
            moved = solver.solve()
            self.assertEqual((moved.start_mode, moved.reset_reasons, moved.solve_index), ('cold', ('mesh',), 2))
            # An invalid new file must not consume an index or destroy usable cached flow.
            solver.options.input_grid = root/'missing.cgns'
            with self.assertRaises(FileNotFoundError):
                solver.solve()
            solver.options.input_grid = b
            warm = solver.solve()
            self.assertEqual((warm.start_mode, warm.reset_reasons, warm.solve_index), ('restart', (), 2))


if __name__ == '__main__':
    unittest.main()
