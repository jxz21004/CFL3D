"""Final calculation output is independent of native iteration chunks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from cfl3d import CFL3D
from test_surface_forces import grouped_plate


def make_solver(directory, resident=False, **options):
    config = dict(turbulence='sst', mach=.2, iterations=2, cfl=1.,
                  resident=resident, output_directory=directory, progress=False,
                  writevolumesolution=False, writeTecplotSurfaceSolution=True)
    config.update(options)
    return CFL3D(grouped_plate(), config, comm=None, name='cruise')


class FinalizationTests(unittest.TestCase):
    def test_chunks_and_final_solve(self):
        for resident in (False, True):
            with self.subTest(resident=resident), tempfile.TemporaryDirectory() as directory:
                with make_solver(directory, resident) as solver:
                    first = solver(final=False)
                    second = solver.solve(final=False)
                    self.assertFalse(first.output_paths['surface'].exists())
                    self.assertFalse(first.output_paths['solution'].exists())
                    self.assertEqual((solver.nSolve, solver.nSolutions), (2, 0))
                    final = solver.solve()
                    self.assertEqual([r.solve_index for r in (first, second, final)], [0, 0, 0])
                    self.assertEqual([r.chunk_index for r in (first, second, final)], [0, 1, 2])
                    self.assertEqual(final.iteration_offset, 4)
                    self.assertEqual(len(final.cumulative_history), 6)
                    self.assertTrue(final.finalized)
                    self.assertFalse(first.finalized)
                    self.assertTrue(final.output_paths['surface'].exists())
                    self.assertEqual((solver.nSolve, solver.nSolutions), (3, 1))
                    rows = np.loadtxt(final.output_paths['residual'])
                    np.testing.assert_array_equal(rows[:, 0], np.arange(1, 7))
                    # Finalization retains the warm state and starts a new output number.
                    next_result = solver.solve()
                    self.assertEqual((next_result.solve_index, next_result.chunk_index), (1, 0))
                    self.assertEqual(next_result.start_mode, 'restart')
                    if resident:
                        self.assertEqual(solver.getRuntimeInfo()['initializations'], 1)
                with make_solver(Path(directory)/'reference', resident) as reference:
                    reference.options.iterations = 8
                    continuous = reference.solve()
                    for key in continuous.fields:
                        np.testing.assert_array_equal(next_result.fields[key], continuous.fields[key])

    def test_explicit_finalization_and_intermediate_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            with make_solver(directory, True, writeIntermediateSolutions=True) as solver:
                solver.addSlices('y', -.5, 'absolute', 'Wing')
                first = solver.solve(final=False)
                snapshot = first.chunk_output_paths['surface']
                self.assertEqual(snapshot.name, 'cruise_000_chunk_000_surf.plt')
                self.assertTrue(snapshot.exists())
                self.assertTrue(first.chunk_output_paths['slices'].exists())
                self.assertFalse(first.output_paths['surface'].exists())
                saved = snapshot.read_bytes()
                result = solver.solve(final=False)
                self.assertEqual(result.chunk_output_paths['surface'].name, 'cruise_000_chunk_001_surf.plt')
                runtime = solver.getRuntimeInfo().copy()
                self.assertIs(solver.finalizeSolution(), result)
                self.assertEqual(solver.getRuntimeInfo(), runtime)
                self.assertEqual(solver.nSolve, 2)
                self.assertEqual(solver.nSolutions, 1)
                self.assertTrue(result.output_paths['surface'].exists())
                self.assertEqual(snapshot.read_bytes(), saved)
                with patch.object(solver, '_write_solution_outputs', side_effect=AssertionError('duplicate output')):
                    self.assertIs(solver.finalizeSolution(), result)
                self.assertEqual(solver.nSolutions, 1)

    def test_failed_finalization_retries_without_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            with make_solver(directory, True) as solver:
                solver.solve(final=False)
                runtime = solver.getRuntimeInfo().copy()
                with patch.object(solver, 'writeTecplotSurfaceSolution', side_effect=OSError('injected')):
                    with self.assertRaisesRegex(OSError, 'injected'):
                        solver.finalizeSolution()
                self.assertFalse(solver._last_result.finalized)
                self.assertEqual(solver.nSolutions, 0)
                solver.finalizeSolution()
                self.assertEqual(solver.getRuntimeInfo(), runtime)
                self.assertEqual(solver.nSolutions, 1)

    def test_volume_and_restart_exports_follow_final_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            with make_solver(directory, writevolumesolution=True,
                             writeIntermediateSolutions=True) as solver:
                result = solver.solve(final=False)
                for kind in ('volume', 'restart', 'solution'):
                    self.assertTrue(result.chunk_output_paths[kind].exists())
                    self.assertFalse(result.output_paths[kind].exists())
                with np.load(result.chunk_output_paths['solution']) as data:
                    self.assertFalse(bool(data['finalized']))
                    self.assertEqual(int(data['chunk_index']), 0)
                state = solver.getFlowState().native
                solver.finalizeSolution()
                for kind in ('volume', 'restart', 'solution'):
                    self.assertTrue(result.output_paths[kind].exists())
                with np.load(result.output_paths['solution']) as data:
                    self.assertTrue(bool(data['finalized']))
                self.assertEqual(solver.getFlowState().native, state)
                self.assertEqual((solver.nSolve, solver.nSolutions), (1, 1))

    def test_validation_reset_and_same_number(self):
        with tempfile.TemporaryDirectory() as directory:
            with make_solver(directory, True, numberSolutions=False, writeIntermediateSolutions=True) as solver:
                with self.assertRaises(RuntimeError):
                    solver.finalizeSolution()
                with self.assertRaises(TypeError):
                    solver.solve(final='False')
                result = solver.solve(final=False)
                solver.name = 'changed'
                with self.assertRaisesRegex(ValueError, 'Finalize or resetFlow'):
                    solver.solve()
                solver.name = 'cruise'
                self.assertIs(solver._last_result, result)
                solver.finalizeSolution()
                result = solver.solve(final=False)
                self.assertEqual((result.solve_index, result.chunk_index), (0, 1))
                self.assertTrue(result.chunk_output_paths['surface'].exists())
                solver.resetFlow()
                result = solver.solve()
                self.assertEqual((result.solve_index, result.chunk_index), (1, 0))
                self.assertEqual(result.start_mode, 'cold')


if __name__ == '__main__':
    unittest.main()
