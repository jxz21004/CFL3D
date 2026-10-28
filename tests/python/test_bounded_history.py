"""Public continuation keeps chunk histories without changing native numerics."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cfl3d import CFL3D, MultigridStage
from test_surface_forces import grouped_plate


class BoundedHistoryTests(unittest.TestCase):
    def test_trim_startup_and_repeated_chunks_match_continuous(self):
        for cycle in (1, 2):
            with self.subTest(cycle=cycle), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = dict(turbulence='sst', mach=.2, alpha=2., target_cl=.01,
                    alpha_step_limit=.1, cfl=1., iterations=24, multigrid_levels=3,
                    multigrid_cycle=cycle, first_order_iterations=3,
                    cfl_ramp_steps=4, cfl_ramp_factor=2.,
                    coarse_grid_stages=(MultigridStage(2,1),MultigridStage(2,2)),
                    writevolumesolution=False, writeTecplotSurfaceSolution=False,
                    progress=False, numberSolutions=False)
                with CFL3D(grouped_plate(),config,comm=None) as reference_solver:
                    reference = reference_solver.solve(output_directory=root/'reference')
                with CFL3D(grouped_plate(),{**config,'resident':True,'iterations':1},comm=None) as solver:
                    chunks = []
                    capacities = []
                    for steps in (1,1,4,2,4,4,4,4):
                        solver.setOption('iterations',steps)
                        result = solver.solve(output_directory=root/'resident',final=False)
                        chunks.append(result.history.copy())  # Only the test opts into accumulation.
                        capacities.append(solver.getRuntimeInfo()['history_capacity'])
                        self.assertIs(result.cumulative_history,result.history)
                        self.assertEqual(solver._output_session_iterations,sum(map(len,chunks)))
                    np.testing.assert_array_equal(np.concatenate(chunks),reference.history)
                    for name in reference.fields:
                        np.testing.assert_array_equal(result.fields[name],reference.fields[name])
                    self.assertEqual(result.final_alpha,reference.final_alpha)
                    self.assertEqual(capacities[-1],capacities[2])
                    self.assertLess(capacities[-1],solver.getRuntimeInfo()['cycles'])
                    rows = np.loadtxt(result.output_paths['residual'])
                    np.testing.assert_array_equal(rows[:,0],np.arange(1,29))
                    solver.finalizeSolution()
