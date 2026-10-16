from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D, MultigridStage, SolverOptions
from fixtures import box_mesh, flatplate_mesh


class ControlValidationTests(unittest.TestCase):
    def test_invalid_controls_rejected_before_native(self):
        invalid = (
            dict(target_cl=float("nan")), dict(target_cl=True), dict(target_cl=99999.),
            dict(alpha_relaxation=0), dict(alpha_step_limit=-.1), dict(alpha_update_interval=0),
            dict(limiter=(0, 1)), dict(limiter=(0, 1, 5)), dict(limiter=True),
            dict(flux=(0, 2, 1)), dict(inversion=(1, -1, 1)), dict(kappa=(0., float("inf"), 0.)),
            dict(multigrid_levels=2, multigrid_smoothing=(1,)),
            dict(multigrid_levels=2, multigrid_smoothing=(0, 1)),
            dict(multigrid_smoothing=(20,), multigrid_up_iterations=1),
            dict(multigrid_stages=()), dict(multigrid_stages=[MultigridStage(3, 1)]),
            dict(multigrid_stages=("bad",)),
            dict(multigrid_levels=2, multigrid_stages=(MultigridStage(2, 2), MultigridStage(3, 2))),
            dict(multigrid_stages=(MultigridStage(2, 1, first_order_iterations=3),)),
            dict(multigrid_stages=(MultigridStage(2, 1, smoothing_steps=(1, 2)),)),
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CFL3D(box_mesh(), SolverOptions(**kwargs))
        self.assertEqual(SolverOptions().eddy_viscosity_limit, 1e5)


@unittest.skipUnless(importlib.util.find_spec("cfl3d._native"), "Build the native extension first")
class NativeControlTests(unittest.TestCase):
    def test_cl_updates_respect_step_limit_interval_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 无承力壁面的均匀流 CL=0，可精确检验原生控制器的限幅与更新间隔。
            options = SolverOptions(turbulence="euler", alpha=1., iterations=6,
                                    target_cl=.1, alpha_relaxation=3.,
                                    alpha_step_limit=.02, alpha_update_interval=2)
            result = CFL3D(box_mesh(True), options).solve(output_directory=root / "lift")
            self.assertAlmostEqual(result.final_alpha, 1.06, places=12)
            updates = np.loadtxt(root / "lift/cfl3d.alpha", skiprows=1, ndmin=2)
            np.testing.assert_array_equal(updates[:, 0], [2, 4, 6])
            np.testing.assert_allclose(updates[:, 4], .02, atol=1e-12)
            with np.load(root / "lift/out_000_solution.npz") as saved:
                self.assertAlmostEqual(float(saved["final_alpha"]), result.final_alpha)
            fixed = CFL3D(box_mesh(), replace(options, target_cl=None, alpha=2.)).solve(
                output_directory=root / "lift")
            self.assertAlmostEqual(fixed.final_alpha, 2., places=12)
            self.assertEqual((root / "lift/cfl3d.alpha").read_text(), "")

    def test_scalar_and_directional_forms_give_same_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = SolverOptions(turbulence="sst", iterations=5, limiter=2, flux=1,
                                    multigrid_levels=2, multigrid_smoothing=(1, 2))
            a = CFL3D(flatplate_mesh(), options).solve(output_directory=root / "scalar")
            b = CFL3D(flatplate_mesh(), replace(options, limiter=(2, 2, 2), flux=(1, 1, 1),
                       inversion=(1, 1, 1), kappa=(1/3, 1/3, 1/3))).solve(output_directory=root / "tuple")
            np.testing.assert_array_equal(a.history, b.history)
            np.testing.assert_array_equal(a.fields["flatplate"], b.fields["flatplate"])

    def test_custom_stages_and_directional_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = SolverOptions(turbulence="sst", cfl=.5, multigrid_levels=3, multigrid_cycle=1,
                limiter=(0, 2, 4), flux=(1, 0, 1), inversion=(1, 0, 1), kappa=(0., 1/3, 1/3),
                multigrid_up_iterations=1,
                multigrid_stages=(MultigridStage(3, 2, 1, (1, 2)),
                                  MultigridStage(5, 2, 2, (2, 1))))
            result = CFL3D(flatplate_mesh(), options).solve(output_directory=root / "custom")
            self.assertEqual(result.history.shape, (8, 8))
            self.assertTrue(np.isfinite(result.fields["flatplate"]).all())
            self.assertGreater(result.coefficients["CD"], 0)

    def test_lift_matching_moves_plate_toward_target(self):
        with tempfile.TemporaryDirectory() as directory:
            options = SolverOptions(mach=.2, reynolds=6e6, temperature=460/1.8,
                turbulence="sst", iterations=200, cfl=3., multigrid_levels=2,
                limiter=2, target_cl=.01, alpha_relaxation=.5,
                alpha_step_limit=.1, alpha_update_interval=5)
            result = CFL3D(flatplate_mesh(), options).solve(output_directory=directory)
            self.assertLess(abs(result.coefficients["CL"]-options.target_cl), 1e-3)
            self.assertGreater(result.final_alpha, 0.)

    def test_global_smoothing_matches_explicit_active_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = SolverOptions(turbulence="sst", cfl=.5, multigrid_levels=3,
                multigrid_smoothing=(1, 2, 3),
                multigrid_stages=(MultigridStage(3, 1), MultigridStage(4, 2)))
            a = CFL3D(flatplate_mesh(), options).solve(output_directory=root / "global")
            explicit = replace(options, multigrid_smoothing=None,
                multigrid_stages=(MultigridStage(3, 1, smoothing_steps=(2,)),
                                  MultigridStage(4, 2, smoothing_steps=(2, 3))))
            b = CFL3D(flatplate_mesh(), explicit).solve(output_directory=root / "explicit")
            np.testing.assert_array_equal(a.history, b.history)
            np.testing.assert_array_equal(a.fields["flatplate"], b.fields["flatplate"])


if __name__ == "__main__":
    unittest.main()
