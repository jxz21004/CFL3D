"""Unambiguous fine budgets and native coarse-startup/resident continuation."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D, MultigridStage, RuntimeOptions, SolverOptions
from test_surface_forces import grouped_plate


class StageOptionsTests(unittest.TestCase):
    def test_automatic_coarse_grid_budgets(self):
        options = RuntimeOptions({"MULTIGRID_LEVELS": 3, "MESH_SEQUENCE": True,
            "COARSE_GRID_ITERATIONS": 4,
            "COARSE_GRID_FIRST_ORDER_ITERATIONS": 2, "ITERATIONS": 7})
        self.assertEqual([(stage.iterations, stage.first_order_iterations)
                          for stage in options.stage_schedule()], [(4, 2), (4, 2), (7, 0)])
        with self.assertRaisesRegex(ValueError, "coarse_grid_first_order_iterations"):
            options.update({"coarse_grid_first_order_iterations": 5})

    def test_objects_and_case_insensitive_dictionaries_agree(self):
        typed = SolverOptions(iterations=5, multigrid_levels=3, resident=True,
            coarse_grid_stages=(MultigridStage(3, 1, 1), MultigridStage(4, 2, 2)))
        live = RuntimeOptions({"ITERATIONS":5,"MULTIGRID_LEVELS":3,"RESIDENT":True,
            "COARSE_GRID_STAGES":[{"ITERATIONS":3,"LEVELS":1,"FIRST_ORDER_ITERATIONS":1},
                                  {"iterations":4,"levels":2,"first_order_iterations":2}]})
        self.assertEqual(typed.stage_schedule(),live.stage_schedule())
        saved = live.snapshot()
        live["iterations"] = 9
        self.assertEqual([s.iterations for s in live.stage_schedule()],[3,4,9])
        self.assertEqual([s.iterations for s in saved.stage_schedule()],[3,4,5])
        self.assertEqual(replace(saved,iterations=7).stage_schedule()[-1].iterations,7)

    def test_removed_full_schedule_is_rejected(self):
        stages=(MultigridStage(3,1),MultigridStage(7,2,2))
        with self.assertRaises(TypeError):
            SolverOptions(multigrid_levels=2,multigrid_stages=stages)
        with self.assertRaisesRegex(ValueError,"Unknown CFL3D option"):
            RuntimeOptions({"multigrid_levels":2,"multigrid_stages":stages})

    def test_invalid_stage_combinations_are_atomic(self):
        live=RuntimeOptions({"multigrid_levels":3,"iterations":5,
                             "coarse_grid_stages":[{"iterations":2,"levels":2}]})
        before=dict(live)
        for edits in ({"mesh_sequence":True},
                      {"multigrid_stages":[{"iterations":5,"levels":3}]},
                      {"coarse_grid_stages":[{"iterations":2,"levels":3}]},
                      {"coarse_grid_stages":[{"iterations":0,"levels":1}]},
                      {"coarse_grid_stages":[{"iterations":2,"levels":1,"smoothing_steps":[1,2]}]}):
            with self.subTest(edits=edits),self.assertRaises(ValueError):
                live.update(edits)
            self.assertEqual(dict(live),before)


class ResidentCoarseStartupTests(unittest.TestCase):
    def test_fine_budget_edit_does_not_restart_resident_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            mesh=grouped_plate()
            config=dict(turbulence="sst",mach=.2,cfl=1.,multigrid_levels=3,
                writevolumesolution=False,writeTecplotSurfaceSolution=False)
            coarse=(MultigridStage(3,1),MultigridStage(4,2))
            reference=CFL3D(mesh,{**config,"iterations":5,"coarse_grid_stages":coarse},comm=None).solve(
                output_directory=root/"reference")
            with CFL3D(mesh,{**config,"resident":True,"iterations":2,
                            "coarse_grid_stages":coarse},comm=None) as solver:
                first=solver.solve(output_directory=root/"resident")
                solver.options.iterations=3
                final=solver.solve(output_directory=root/"resident")
                self.assertEqual(final.start_mode,"restart")
                self.assertEqual((solver.getRuntimeInfo()["cycles"],len(final.history)),(12,3))
                np.testing.assert_array_equal(np.concatenate((first.history,final.history)),reference.history)
                for name in final.fields:
                    np.testing.assert_array_equal(final.fields[name],reference.fields[name])

    def test_bounded_native_schedule_completes_after_fine_stage(self):
        from cfl3d import _native
        from test_resident_native import native_session, readout
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            mesh=grouped_plate()
            config=dict(turbulence="sst",mach=.2,cfl=1.,iterations=5,multigrid_levels=3,
                coarse_grid_stages=(MultigridStage(3,1),MultigridStage(4,2)),
                writevolumesolution=False,writeTecplotSurfaceSolution=False)
            reference=CFL3D(mesh,config,comm=None).solve(output_directory=root/"reference")
            lines=reference.output_paths["log"].read_text().splitlines()
            mseq=next(i for i,line in enumerate(lines) if "mseq    mgflag" in line)
            ncyc=next(i for i,line in enumerate(lines) if "ncyc    mglevg" in line)
            mit=next(i for i,line in enumerate(lines) if "mit1" in line and "mit2" in line)
            self.assertEqual(lines[mseq+1].split()[0],"3")
            self.assertEqual([line.split()[:2] for line in lines[ncyc+1:ncyc+4]],
                             [["3","1"],["4","2"],["5","3"]])
            self.assertEqual([line.split() for line in lines[mit+1:mit+4]],
                             [["1"],["1","1"],["1","1","1"]])
            with native_session(mesh,config,root/"bounded",9):
                self.assertEqual(_native.py_solve(),(0,9))
                before=_native.py_resident_info()
                self.assertEqual(_native.py_advance_resident(4,0.)[0],1)
                self.assertEqual(_native.py_resident_info(),before)
                self.assertEqual(_native.py_advance_resident(3,0.),(0,12))
                self.assertEqual(_native.py_resident_info()[:4],(0,0,12,1))
                fields,history=readout(mesh,12)
                np.testing.assert_array_equal(history,reference.history)
                for name in fields:
                    np.testing.assert_array_equal(fields[name],reference.fields[name])

    def test_explicit_startup_and_chunks_match_continuous_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            mesh=grouped_plate(split=25)
            for model in ("sa","sst","gamma-theta"):
                for cycle in (1,2):
                    with self.subTest(model=model,cycle=cycle):
                        config=dict(turbulence=model,mach=.2,cfl=1.,alpha=2.,
                            multigrid_levels=3,multigrid_cycle=cycle,
                            coarse_grid_stages=(MultigridStage(3,1,1),MultigridStage(4,2,2)),
                            first_order_iterations=4,multigrid_smoothing=(1,2,1),
                            cfl_ramp_steps=3,cfl_ramp_factor=2.,
                            writevolumesolution=False,writeTecplotSurfaceSolution=False)
                        label=f"{model}-{cycle}"
                        reference=CFL3D(mesh,{**config,"iterations":13},comm=None).solve(
                            output_directory=root/(label+"-continuous"))
                        with CFL3D(mesh,{**config,"iterations":2,"resident":True},comm=None) as solver:
                            chunks=[]
                            for steps in (2,3,1,7):
                                solver.options.iterations=steps
                                chunks.append(solver.solve(output_directory=root/label))
                                info=solver.getRuntimeInfo()
                                self.assertEqual(info["initializations"],1)
                                self.assertEqual(info["startup_cycles"],7)
                                self.assertEqual(info["fine_cycles"],sum((2,3,1,7)[:len(chunks)]))
                                np.testing.assert_array_equal(
                                    np.concatenate([c.history for c in chunks]),
                                    reference.history[:info["cycles"]])
                            self.assertEqual(len(chunks[0].history),9)
                            self.assertEqual(chunks[1].start_mode,"restart")
                            self.assertEqual(solver.getRuntimeInfo()["cycles"],20)
                            for name in reference.fields:
                                np.testing.assert_array_equal(chunks[-1].fields[name],reference.fields[name])
                            for name in reference.wall_fields:
                                np.testing.assert_array_equal(chunks[-1].wall_fields[name],reference.wall_fields[name])
                            self.assertFalse(list(root.rglob(".cfl3d-state.bin")))
                            solver.resetFlow()
                            solver.options.iterations=2
                            restarted=solver.solve(output_directory=root/(label+"-reset"))
                            self.assertEqual((restarted.start_mode,len(restarted.history)),("cold",9))
                            for name in restarted.fields:
                                np.testing.assert_array_equal(restarted.fields[name],chunks[0].fields[name])

    def test_automatic_startup_and_nonresident_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=dict(turbulence="sst",mach=.2,cfl=1.,multigrid_levels=3,
                mesh_sequence=True,coarse_grid_iterations=3,
                writevolumesolution=False,writeTecplotSurfaceSolution=False)
            mesh=grouped_plate()
            reference=CFL3D(mesh,{**config,"iterations":5},comm=None).solve(output_directory=root/"full")
            with CFL3D(mesh,{**config,"iterations":2,"resident":True},comm=None) as solver:
                first=solver.solve(output_directory=root/"resident")
                solver.options.iterations=3
                final=solver.solve(output_directory=root/"resident")
                np.testing.assert_array_equal(np.concatenate((first.history,final.history)),reference.history)
                for name in final.fields:
                    np.testing.assert_array_equal(final.fields[name],reference.fields[name])
            ordinary=CFL3D(mesh,{**config,"iterations":2},comm=None)
            self.assertEqual(len(ordinary.solve(output_directory=root/"ordinary").history),8)
            ordinary.options.iterations=3
            self.assertEqual(len(ordinary.solve(output_directory=root/"ordinary").history),3)


if __name__ == "__main__":
    unittest.main()
