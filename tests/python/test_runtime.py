"""Mutable rank-local settings and complete, Python-owned warm starts."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D, SolverOptions
from fixtures import flatplate_mesh, box_mesh
from test_surface_forces import grouped_plate


class RuntimeOptionsTests(unittest.TestCase):
    def test_case_insensitive_atomic_updates_and_snapshots(self):
        source = {
            "CFL": 1.0,
            "ITERATIONS": 6,
            "writeVolumeSolution": False,
            "writeTECPLOTsurfaceSOLUTION": False,
            "TURBULENCE": "sst",
        }
        solver = CFL3D(grouped_plate(), source, comm=None)
        source["CFL"] = 999
        self.assertEqual(solver.options["cFl"], 1.0)
        solver.options["CFL"] = 2.0
        solver.options.ITERATIONS = 3
        self.assertEqual(solver.getOption("cfl"), 2.0)
        self.assertEqual(solver.options.iterations, 3)
        solver.setOptions({"CFL": 3.0, "ITERATIONS": 4})
        before = dict(solver.options)
        with self.assertRaises(ValueError):
            solver.setOptions({"CFL": 0.0, "ITERATIONS": 2})
        self.assertEqual(dict(solver.options), before)
        for opts in ({"CFL": 1, "cfl": 2}, {"unknown": 3}):
            with self.assertRaises(ValueError):
                CFL3D(box_mesh(), opts, comm=None)
        self.assertTrue(solver.checkOptions())
        self.assertTrue(solver.options.check())
        stages = SolverOptions.from_dict(
            {"MULTIGRID_LEVELS": 2, "MULTIGRID_STAGES": [{"ITERATIONS": 2, "LEVELS": 2, "SMOOTHING_STEPS": [1, 2]}]}
        )
        stages.validate()
        self.assertEqual(stages.multigrid_stages[0].iterations, 2)


class ContinuationTests(unittest.TestCase):
    def test_split_iterations_equal_uninterrupted_for_all_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ("euler", "laminar", "sa", "sst", "komega"):
                with self.subTest(model=model):
                    mesh = grouped_plate()
                    if model == "euler":
                        for zone in mesh.zones:
                            for boundary in zone.boundaries:
                                if boundary.bc_type == "BCWallViscous":
                                    boundary.bc_type = "BCWallInviscid"
                    o = dict(
                        turbulence=model,
                        alpha=2.0,
                        mach=0.2,
                        iterations=5,
                        cfl=1.0,
                        multigrid_levels=2,
                        writevolumesolution=False,
                        writeTecplotSurfaceSolution=False,
                    )
                    solver = CFL3D(deepcopy(mesh), o, comm=None)
                    first = solver.solve(output_directory=root / model / "first")
                    saved = solver.getFlowState()
                    self.assertEqual(saved.fields.keys(), first.fields.keys())
                    # Returned snapshots and another solver's COMMON state cannot corrupt this object's state.
                    for value in saved.fields.values():
                        value[:] = 0
                    CFL3D(
                        box_mesh(),
                        dict(
                            turbulence="euler",
                            iterations=1,
                            writevolumesolution=False,
                            writeTecplotSurfaceSolution=False,
                        ),
                        comm=None,
                    ).solve(output_directory=root / model / "other")
                    solver.setOption("ITERATIONS", 2)  # Smaller than the previous chunk tests restart history bounds.
                    continued = solver.solve(output_directory=root / model / "continued")
                    full = CFL3D(mesh, {**o, "iterations": 7}, comm=None).solve(output_directory=root / model / "full")
                    for name in continued.fields:
                        np.testing.assert_allclose(continued.fields[name], full.fields[name], rtol=2e-12, atol=2e-13)
                    np.testing.assert_allclose(continued.history[:, 1:], full.history[-2:, 1:], rtol=2e-11, atol=2e-13)
                    self.assertEqual(continued.history.shape, (2, 8))
                    self.assertFalse((root / model / "continued/out_001_restart.bin").exists())
                    self.assertFalse((root / model / "continued/.cfl3d-state.bin").exists())

    def test_files_changed_controls_reset_and_input_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = flatplate_mesh()
            opts = dict(
                turbulence="sst",
                iterations=5,
                cfl=1.0,
                alpha=2.0,
                mach=0.2,
                writevolumesolution=False,
                writeTecplotSurfaceSolution=False,
            )
            solver = CFL3D(deepcopy(mesh), opts, comm=None)
            first = solver.solve(output_directory=root / "first")
            saved = solver.getFlowState()
            path = solver.writeRestartFile(root / "state.npz")
            native = root / "raw.bin"
            native.write_bytes(saved.native)
            # Delete the old output directory's checkpoint: memory/file state owns all required state.
            solver.setOptions({"cFl": 2.0, "Iterations": 2})
            updated = solver.solve(output_directory=root / "updated")
            self.assertEqual(first.history.shape, (5, 8))
            self.assertEqual(solver._last_options.cfl, 2.0)
            self.assertIn("2.000", (root / "updated/cfl3d.out").read_text())
            for file in (path, native):
                resumed = CFL3D(deepcopy(mesh), {**opts, "cfl": 2.0, "iterations": 2, "RESTARTFILE": file}, comm=None)
                result = resumed.solve(output_directory=root / file.stem)
                np.testing.assert_array_equal(result.fields["flatplate"], updated.fields["flatplate"])
            installed = CFL3D(deepcopy(mesh), {**opts, "cfl": 2.0, "iterations": 2}, comm=None)
            installed.setFlowState(saved)
            result = installed.solve(output_directory=root / "installed")
            np.testing.assert_array_equal(result.fields["flatplate"], updated.fields["flatplate"])
            solver.resetFlow().setOptions({"cfl": 1.0, "iterations": 5})
            fresh = solver.solve(output_directory=root / "fresh")
            np.testing.assert_array_equal(fresh.fields["flatplate"], first.fields["flatplate"])
            with self.assertRaisesRegex(ValueError, "incompatible"):
                CFL3D(deepcopy(mesh), {**opts, "turbulence": "sa", "restartFile": path}, comm=None).solve(
                    output_directory=root / "invalid"
                )
            with self.assertRaises(ValueError):
                CFL3D(deepcopy(mesh), {**opts, "restartFile": root / "first/out_000_solution.npz"}, comm=None).solve(
                    output_directory=root / "visualization"
                )
            bad = root / "bad.bin"
            bad.write_bytes(native.read_bytes()[:-5])
            with self.assertRaises(ValueError):
                CFL3D(deepcopy(mesh), {**opts, "restartFile": bad}, comm=None).solve(output_directory=root / "bad")

    def test_continuation_skips_completed_mesh_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver = CFL3D(
                flatplate_mesh(),
                dict(
                    turbulence="sst",
                    iterations=3,
                    cfl=1.0,
                    multigrid_levels=3,
                    mesh_sequence=True,
                    coarse_iterations=2,
                    writevolumesolution=False,
                    writeTecplotSurfaceSolution=False,
                ),
                comm=None,
            )
            first = solver.solve(output_directory=root / "first")
            self.assertEqual(len(first.history), 7)
            solver.setOption("iterations", 2)
            second = solver.solve(output_directory=root / "second")
            self.assertEqual(len(second.history), 2)
            self.assertTrue(np.isfinite(second.fields["flatplate"]).all())
            solver.resetFlow()
            restarted = solver.solve(output_directory=root / "fresh-sequence")
            self.assertEqual(len(restarted.history), 6)  # Two coarse stages + the configured two fine steps.
            self.assertEqual(restarted.solve_index, 2)


if __name__ == "__main__":
    unittest.main()
