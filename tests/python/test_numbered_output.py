"""Per-solve Tecplot output and cache-reset behavior without file overwrites."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D
from test_surface_forces import grouped_plate


class NumberedOutputTests(unittest.TestCase):
    def test_lift_matching_history_continues_iteration_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = CFL3D(grouped_plate(), dict(turbulence="sst", mach=.2,
                            iterations=2, cfl=1., target_cl=.01, numberSolutions=False,
                            output_directory=directory, writevolumesolution=False,
                            writeTecplotSurfaceSolution=False), comm=None)
            first = solver.solve()
            previous = first.output_paths["alpha_history"].read_bytes()
            second = solver.solve()
            self.assertTrue(second.output_paths["alpha_history"].read_bytes().startswith(previous))
            rows = [line.split() for line in second.output_paths["alpha_history"].read_text().splitlines()
                    if line.split() and line.split()[0].isdigit()]
            self.assertEqual([int(row[0]) for row in rows], [1, 2, 3, 4])

    def test_same_number_appends_logs_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = dict(turbulence="sst", mach=.2, alpha=2., iterations=3, cfl=1.,
                           numberSolutions=False, output_directory=root,
                           writevolumesolution=True, writeTecplotSurfaceSolution=True)
            solver = CFL3D(grouped_plate(), options, name="cruise", comm=None)
            solver.addSlices("y", -.5, "absolute", "Wing")
            first = solver.solve()
            saved = {name: first.output_paths[name].read_bytes() for name in
                     ("log", "residual", "turbulence_history", "preprocess_log")}
            surface = first.output_paths["surface"].read_bytes()
            solver.setOptions({"NUMBERSOLUTIONS": False, "iterations": 2})
            second = solver.solve()
            self.assertEqual((second.solve_index, solver.nSolve, second.iteration_offset), (0, 2, 3))
            self.assertEqual(second.history.shape, (2, 8))
            np.testing.assert_array_equal(second.cumulative_history, np.concatenate([first.history, second.history]))
            for name, content in saved.items():
                self.assertTrue(second.output_paths[name].read_bytes().startswith(content), name)
                self.assertGreater(second.output_paths[name].stat().st_size, len(content), name)
            for name in ("residual", "turbulence_history"):
                rows = np.loadtxt(second.output_paths[name])
                np.testing.assert_array_equal(rows[:, 0], np.arange(1, 6))
            self.assertNotEqual(second.output_paths["surface"].read_bytes(), surface)
            self.assertFalse((root / "cfl3d.out").exists())
            self.assertFalse((root / "cfl3d.res").exists())
            self.assertFalse((root / "precfl3d.out").exists())
            self.assertFalse((root / "cruise_000_cfl3d.out").exists())
            with np.load(second.output_paths["solution"]) as data:
                np.testing.assert_array_equal(data["history"], second.cumulative_history)
                np.testing.assert_array_equal(data["chunk_history"], second.history)
                self.assertEqual(int(data["iteration_offset"]), 3)
            full = CFL3D(grouped_plate(), {**options, "iterations": 5}, comm=None).solve(output_directory=root / "full")
            for zone in second.fields:
                np.testing.assert_array_equal(second.fields[zone], full.fields[zone])
            saved = second.output_paths["residual"].read_bytes()
            solver.setOption("numberSolutions", True)
            third = solver.solve()
            self.assertEqual((third.solve_index, solver.nSolve, third.iteration_offset), (1, 3, 0))
            self.assertEqual(first.output_paths["residual"].read_bytes(), saved)
            solver.setOption("numberSolutions", False)
            fourth = solver.solve()
            self.assertEqual((fourth.solve_index, fourth.iteration_offset), (1, 2))
            # Clearing state starts a fresh stream, even if numbering is disabled.
            solver.resetFlow()
            fifth = solver.solve()
            self.assertEqual((fifth.solve_index, fifth.iteration_offset, solver.nSolve), (2, 0, 5))
            # Another object does not claim an existing stream by filename alone.
            another = CFL3D(grouped_plate(), options, name="cruise", comm=None).solve()
            self.assertEqual(another.solve_index, 3)

    def test_analyses_auto_output_numbering_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = dict(
                turbulence="sst",
                mach=0.2,
                alpha=2.0,
                iterations=2,
                cfl=1.0,
                writevolumesolution=False,
                writeTecplotSurfaceSolution=False,
                writeflowsolution=False,
                output_directory=root,
            )
            solver = CFL3D(grouped_plate(), options, name="out", comm=None)
            solver.addLiftDistribution(9, "y", "Wing")
            solver.addSlices("y", -0.5, "absolute", "Wing")
            first = solver.solve()
            self.assertEqual((first.name, first.solve_index, solver.nSolve), ("out", 0, 1))
            self.assertFalse((root / "out_000_surf.dat").exists())
            saved = {}
            for name in ("out_000_slices.dat", "out_000_lift.dat", "out_000_solution.npz"):
                saved[name] = (root / name).read_bytes()
            self.assertIn("ZONETYPE=FELINESEG", (root / "out_000_slices.dat").read_text())
            self.assertIn("DATAPACKING=POINT", (root / "out_000_lift.dat").read_text())
            with np.load(root / "out_000_solution.npz") as data:
                self.assertEqual(int(data["solve_index"]), 0)
                self.assertEqual(str(data["name"]), "out")
            solver.setOptions(
                {
                    "ITERATIONS": 1,
                    "WriteTecplotSurfaceSolution": True,
                    "WriteVolumeSolution": True,
                    "writeFlowSolution": True,
                }
            )
            second = solver.solve()
            self.assertEqual((second.solve_index, solver.nSolve), (1, 2))
            for name in (
                "out_001_surf.dat",
                "out_001_slices.dat",
                "out_001_lift.dat",
                "out_001_volume.cgns",
                "out_001_solution.npz",
                "out_001_restart.bin",
                "out_001_flow.x",
                "out_001_flow.q",
            ):
                self.assertTrue((root / name).is_file(), name)
            full = CFL3D(grouped_plate(), {**options, "iterations": 3}, comm=None).solve(output_directory=root / "full")
            for zone in second.fields:
                np.testing.assert_array_equal(second.fields[zone], full.fields[zone])
            # Clearing computational state restores cold initialization but retains the file counter.
            solver.resetFlow().setOption("iterations", 2)
            third = solver.solve()
            self.assertEqual((third.solve_index, solver.nSolve), (2, 3))
            for zone in third.fields:
                np.testing.assert_array_equal(third.fields[zone], first.fields[zone])
            for name, content in saved.items():
                self.assertEqual((root / name).read_bytes(), content)
            # Changing the live name cannot rename the cached result's deferred outputs.
            solver.name = "nextcase"
            self.assertEqual(solver.writeTecplotSurfaceSolution().name, "out_002_surf.dat")
            self.assertEqual(solver.writeRestartFile().name, "out_002_restart.npz")
            fourth = solver.solve()
            self.assertEqual((fourth.name, fourth.solve_index), ("nextcase", 3))
            self.assertTrue((root / "nextcase_003_surf.dat").exists())
            # Configuration errors do not consume an output index.
            solver.setOption("input_grid", "another.cgns")
            with self.assertRaises(FileNotFoundError):
                solver.solve()
            self.assertEqual(solver.nSolve, 4)

    def test_case_names_and_custom_labels(self):
        for name in ("", "../wing", "a/b", "a\\b", ".", "..", "\x00bad"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                CFL3D(grouped_plate(), name=name, comm=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver = CFL3D(
                grouped_plate(),
                dict(
                    NAME="wing",
                    turbulence="sst",
                    iterations=1,
                    cfl=1.0,
                    output_directory=root,
                    volume_filename="grid.cgns",
                    surface_filename="plots/wall.dat",
                    flow_prefix="native",
                    writeflowsolution=True,
                ),
                comm=None,
            )
            solver.addLiftDistribution(3, "y", "Wing")
            solver.addSlices("y", -0.5, "absolute", "Wing")
            result = solver.solve()
            for name in (
                "wing_000_grid.cgns",
                "plots/wing_000_wall.dat",
                "plots/wing_000_slices.dat",
                "plots/wing_000_lift.dat",
                "wing_000_native.x",
                "wing_000_native.q",
            ):
                self.assertTrue((root / name).exists(), name)
            self.assertEqual(result.output_paths["surface"], root / "plots/wing_000_wall.dat")
            # Manual prefixes stay caller-controlled; analysis format always uses .dat.
            files = solver.writeSurfaceAnalysis(root / "manual")
            self.assertEqual(files["slices"], root / "manual_slices.dat")
            self.assertEqual(files["lift"], root / "manual_lift.dat")


if __name__ == "__main__":
    unittest.main()
