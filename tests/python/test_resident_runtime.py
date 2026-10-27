"""Public resident lifetime, control updates, readouts and numbered output."""
from pathlib import Path
from copy import deepcopy
from contextlib import redirect_stdout
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from cfl3d import CFL3D, MultigridStage, RuntimeOptions
from test_surface_forces import grouped_plate


def options(**updates):
    return dict(dict(turbulence='sst', mach=.2, cfl=1., iterations=2, freestream_pressure=101325.,
        resident=True, multigrid_levels=3, writevolumesolution=False,
        writeTecplotSurfaceSolution=False), **updates)


class ResidentRuntimeTests(unittest.TestCase):
    def test_progress_checkpoints_show_startup_and_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            config = options(iterations=1, coarse_grid_stages=(
                MultigridStage(1, 1), MultigridStage(1, 2)))
            with redirect_stdout(output):
                with CFL3D(grouped_plate(), config, comm=None) as solver:
                    solver.solve(output_directory=Path(directory) / 'run')
                    solver.solve(output_directory=Path(directory) / 'run')
            log = output.getvalue()
            self.assertIn('Mesh ready:', log)
            self.assertIn('Native input ready', log)
            self.assertIn('1 level(s) x 1 cycles, 2 level(s) x 1 cycles, 3 level(s) x 1 cycles', log)
            self.assertIn('Native solve running: 3 level(s) x 1 cycles', log)
            self.assertIn('Solve 000 complete', log)
            self.assertIn('Solve 001 complete', log)
            with redirect_stdout(output := io.StringIO()):
                with CFL3D(grouped_plate(), {**config, 'progress': False}, comm=None) as quiet:
                    quiet.solve(output_directory=Path(directory) / 'quiet')
            self.assertNotIn('CFL3D[', output.getvalue())

    def test_startup_spans_short_public_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = options(iterations=1,first_order_iterations=3,cfl_ramp_steps=3,cfl_ramp_factor=2.)
            reference = CFL3D(grouped_plate(),{**config,'resident':False,'iterations':7},comm=None).solve(
                output_directory=root/'reference')
            with CFL3D(grouped_plate(),config,comm=None) as solver:
                chunks = [solver.solve(output_directory=root/'resident'),solver.solve(output_directory=root/'resident')]
                solver.setOption('iterations',5)
                chunks.append(solver.solve(output_directory=root/'resident'))
                self.assertEqual([len(c.history) for c in chunks],[1,1,5])
                np.testing.assert_array_equal(np.concatenate([c.history for c in chunks]),reference.history)
                for name in reference.fields:
                    np.testing.assert_array_equal(chunks[-1].fields[name],reference.fields[name])
                self.assertEqual(solver.getRuntimeInfo()['initializations'],1)

    def test_output_failure_preserves_advanced_context_for_deferred_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with CFL3D(grouped_plate(),options(writevolumesolution=True),comm=None) as solver:
                with patch.object(Path,'mkdir',side_effect=OSError('injected directory failure')):
                    with self.assertRaisesRegex(OSError,'directory failure'):
                        solver.solve(output_directory=root/'resident')
                self.assertIsNone(solver._resident_context)
                self.assertFalse(solver.getRuntimeInfo()['active'])
                with patch.object(solver,'writeVolumeSolution',side_effect=OSError('injected output failure')):
                    with self.assertRaisesRegex(OSError,'injected'):
                        solver.solve(output_directory=root/'resident')
                self.assertEqual((solver.nSolve,solver.getRuntimeInfo()['cycles']),(1,2))
                solver.finalizeSolution()
                self.assertEqual(solver.nSolutions, 1)
                self.assertEqual(solver.getRuntimeInfo()['cycles'],2)
                result = solver.solve(output_directory=root/'resident')
                self.assertEqual((result.solve_index,solver.getRuntimeInfo()['cycles']),(1,4))
                self.assertEqual(solver.getRuntimeInfo()['initializations'],1)
            reference = CFL3D(grouped_plate(),options(resident=False,iterations=4),comm=None).solve(
                output_directory=root/'reference')
            for name in result.fields:
                np.testing.assert_array_equal(result.fields[name],reference.fields[name])

    def test_physics_hierarchy_and_mesh_changes_reinitialize_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            with CFL3D(grouped_plate(split=25),options(),comm=None) as solver:
                solver.solve(output_directory=root/'resident')
                for label,edits in (('mach',{'mach':.3}),('multigrid_levels',{'multigrid_levels':2}),('mesh',{})):
                    previous = solver._resident_context
                    solver.setOptions(edits)
                    if label == 'mesh':
                        solver.mesh = grouped_plate(split=33)
                    result = solver.solve(output_directory=root/'resident')
                    self.assertEqual((result.start_mode,result.reset_reasons),('cold',(label,)))
                    self.assertIsNot(solver._resident_context,previous)
                    self.assertFalse(previous.started)
                    self.assertEqual(solver.getRuntimeInfo()['cycles'],2)
                    cases.append((label,result,deepcopy(solver.mesh),dict(solver.options)))
            for label,result,mesh,config in cases:
                reference = CFL3D(mesh,{**config,'resident':False},comm=None).solve(output_directory=root/label)
                np.testing.assert_array_equal(result.history,reference.history)
                for name in result.fields:
                    np.testing.assert_array_equal(result.fields[name],reference.fields[name])

    def test_file_mesh_is_not_read_again_between_chunks(self):
        from cfl3d import CGNSMesh
        from cgns_fixtures import write_adf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_adf(grouped_plate(),root/'mesh.cgns')
            with CFL3D(options=options(input_grid=path),comm=None) as solver:
                solver.solve(output_directory=root/'resident')
                with patch.object(CGNSMesh,'from_file',side_effect=AssertionError('grid reread')):
                    solver.setOption('iterations',3)
                    result = solver.solve(output_directory=root/'resident')
                self.assertEqual(len(result.history),3)
                self.assertEqual(solver.getRuntimeInfo()['initializations'],1)

    def test_public_chunks_retain_workspace_and_match_continuous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ('sa','sst','gamma-theta'):
                for cycle in (1,2):
                    with self.subTest(model=model,cycle=cycle):
                        config = options(turbulence=model,multigrid_cycle=cycle,
                            first_order_iterations=2,cfl_ramp_steps=3,cfl_ramp_factor=2.)
                        mesh = grouped_plate(split=25)
                        reference = CFL3D(mesh,{**config,'resident':False,'iterations':29},comm=None).solve(
                            output_directory=root/f'reference-{model}-{cycle}')
                        with CFL3D(mesh,config,comm=None) as solver:
                            with patch.object(solver,'_native_arrays',wraps=solver._native_arrays) as prepare, \
                                 patch.object(solver,'_stage_flow_state',side_effect=AssertionError('restart staging')):
                                target = root/f'resident-{model}-{cycle}'
                                chunks = [solver.solve(output_directory=target)]
                                for steps in (3,1,7,16):
                                    solver.setOption('iterations',steps)
                                    chunks.append(solver(output_directory=target))
                                self.assertEqual(prepare.call_count,1)
                            self.assertEqual(solver.getRuntimeInfo()['initializations'],1)
                            self.assertEqual(solver.getRuntimeInfo()['cycles'],29)
                            self.assertEqual(solver.getRuntimeInfo()['input_coordinate_values'],0)
                            self.assertEqual([c.start_mode for c in chunks],['cold']+['restart']*4)
                            self.assertEqual([c.solve_index for c in chunks],list(range(5)))
                            np.testing.assert_array_equal(np.concatenate([c.history for c in chunks]),reference.history)
                            for name in reference.fields:
                                np.testing.assert_array_equal(chunks[-1].fields[name],reference.fields[name])
                            for key in reference.wall_fields:
                                np.testing.assert_array_equal(chunks[-1].wall_fields[key],reference.wall_fields[key])
                            self.assertEqual(solver.getSolution('Wing')['Wing_CL'],solver.getSolution()['CL'])
                            self.assertFalse(list(target.glob('*_solution.npz')))
                            self.assertFalse(list(target.glob('*restart*')))
                            self.assertFalse((target/'.cfl3d-state.bin').exists())
                            for number,chunk in enumerate(chunks):
                                rows = np.loadtxt(chunk.output_paths['residual'],ndmin=2)
                                np.testing.assert_array_equal(rows[:,0],np.arange(1,len(chunk.history)+1))
                        self.assertFalse(solver.getRuntimeInfo()['active'])
                        solver.close()  # Idempotent, readouts survive cleanup.
                        self.assertEqual(solver.getSolution()['CL'],chunks[-1].coefficients['CL'])

    def test_cfl_updates_match_native_schedule_without_setup(self):
        from test_resident_native import native_session,readout
        from cfl3d import _native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = grouped_plate(split=25)
            config = options()
            with native_session(mesh,{**config,'resident':False},root/'native',2,open_ended=True):
                self.assertEqual(_native.py_solve(),(0,2))
                for steps,cfl in ((3,.5),(4,0.),(7,2.)):
                    status,rows = _native.py_advance_resident(steps,cfl)
                    self.assertEqual(status,0)
                fields,history = readout(mesh,rows)
            with CFL3D(mesh,config,comm=None) as solver:
                chunks = [solver.solve(output_directory=root/'public')]
                for steps,cfl in ((3,.5),(4,.5),(7,2.)):
                    solver.setOptions({'CFL':cfl,'ITERATIONS':steps})
                    chunks.append(solver.solve(output_directory=root/'public'))
                self.assertEqual(solver.getRuntimeInfo()['initializations'],1)
                np.testing.assert_array_equal(np.concatenate([c.history for c in chunks]),history)
                for name in fields:
                    np.testing.assert_array_equal(chunks[-1].fields[name],fields[name])

    def test_numbering_append_analysis_and_deferred_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = options(name='case',freestream_pressure=101325.,numberSolutions=False,
                             writeTecplotSurfaceSolution=True,writeflowsolution=True)
            with CFL3D(grouped_plate(),config,comm=None) as solver:
                solver.addLiftDistribution(7,'y','Wing')
                solver.addSlices('y',-.5,groupName='Wing')
                first = solver.solve(output_directory=root)
                original = first.output_paths['log'].read_bytes()
                second = solver.solve(output_directory=root)
                self.assertEqual((second.solve_index,second.iteration_offset),(0,2))
                self.assertTrue(second.output_paths['log'].read_bytes().startswith(original))
                np.testing.assert_array_equal(np.loadtxt(second.output_paths['residual'])[:,0],np.arange(1,5))
                for key in ('surface','slices','lift'):
                    self.assertTrue(second.output_paths[key].exists())
                self.assertFalse(second.output_paths['restart'].exists())
                self.assertEqual(len(second.cumulative_history),4)
                saved = second.output_paths['log'].read_bytes()
                solver.setOptions({'numberSolutions':True,'name':'renamed','writevolumesolution':True})
                third = solver.solve(output_directory=root/'moved')
                self.assertEqual(third.solve_index,1)
                self.assertEqual(second.output_paths['log'].read_bytes(),saved)
                self.assertTrue(third.output_paths['log'].exists())
                self.assertTrue(third.output_paths['volume'].exists())
                self.assertTrue(third.output_paths['solution'].exists())
                self.assertEqual(solver.getRuntimeInfo()['initializations'],1)
                solver.close()
                self.assertTrue(solver.writeVolumeSolution(root/'deferred.cgns').exists())

    def test_reset_context_owner_and_incompatible_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = grouped_plate()
            ordinary = CFL3D(mesh,{**options(),'resident':False},comm=None)
            with CFL3D(mesh,options(),comm=None) as solver:
                first = solver.solve(output_directory=root/'resident')
                info = solver.getRuntimeInfo()
                with self.assertRaisesRegex(RuntimeError,'Another CFL3D'):
                    ordinary.solve(output_directory=root/'blocked')
                self.assertFalse((root/'blocked').exists())
                self.assertEqual(solver.getRuntimeInfo(),info)
                with self.assertRaisesRegex(NotImplementedError,'snapshots'):
                    solver.getFlowState()
                with self.assertRaisesRegex(NotImplementedError,'checkpoint'):
                    solver.writeRestartFile(root/'unavailable.npz')
                solver.options.limiter = 0
                cold = solver.solve(output_directory=root/'resident')
                self.assertEqual(cold.start_mode,'cold')
                self.assertEqual(cold.reset_reasons,('limiter',))
                self.assertEqual(solver.getRuntimeInfo()['cycles'],2)
                solver.resetFlow()
                self.assertFalse(solver.getRuntimeInfo()['active'])
                again = solver.solve(output_directory=root/'resident')
                self.assertEqual(again.start_mode,'cold')
                self.assertGreater(again.solve_index,cold.solve_index)
                solver.setOption('resident',False)
                legacy = solver.solve(output_directory=root/'resident')
                self.assertEqual(legacy.start_mode,'cold')
                self.assertIn('resident',legacy.reset_reasons)
                self.assertIsNotNone(solver.getFlowState())
            ordinary_result = ordinary.solve(output_directory=root/'ordinary')
            ordinary.setOption('resident',True)
            with ordinary:
                switched = ordinary.solve(output_directory=root/'ordinary')
                self.assertEqual((switched.start_mode,switched.reset_reasons),('cold',('resident',)))
                for name in switched.fields:
                    np.testing.assert_array_equal(switched.fields[name],ordinary_result.fields[name])

    def test_unsupported_modes_are_explicit_and_atomic(self):
        config = RuntimeOptions(options())
        original = dict(config)
        for edits in ({'restartFile':'some.npz'},):
            with self.subTest(edits=edits),self.assertRaises(NotImplementedError):
                config.update(edits)
            self.assertEqual(dict(config),original)


if __name__ == '__main__':
    unittest.main()
