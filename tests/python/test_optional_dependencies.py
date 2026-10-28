"""Skip policy must not hide installed PETSc failures or diverge across ranks."""

import unittest
from unittest.mock import patch
from optional_dependencies import petsc_skip_reason


class OptionalDependencyTests(unittest.TestCase):
    def test_explicit_skip_does_not_probe_or_import(self):
        with (
            patch.dict("os.environ", {"CFL3D_SKIP_PETSC_TESTS": "1"}),
            patch("optional_dependencies.importlib.util.find_spec", side_effect=AssertionError("Unexpected probe")),
        ):
            self.assertIn("disabled", petsc_skip_reason())
        with patch("optional_dependencies.importlib.util.find_spec", side_effect=AssertionError("Unexpected probe")):
            self.assertIn("disabled", petsc_skip_reason(skip=True))

    def test_absent_skips_but_discoverable_installation_runs(self):
        with patch.dict("os.environ", {"CFL3D_SKIP_PETSC_TESTS": "0"}):
            with patch("optional_dependencies.importlib.util.find_spec", return_value=None):
                self.assertIn("not installed", petsc_skip_reason())
            with patch("optional_dependencies.importlib.util.find_spec", return_value=object()):
                self.assertIsNone(petsc_skip_reason())

    def test_one_rank_missing_dependency_skips_collectively(self):
        class Comm:
            def allgather(self, reason):
                return [reason, "Optional petsc4py not installed", None]

        with (
            patch.dict("os.environ", {"CFL3D_SKIP_PETSC_TESTS": "0"}),
            patch("optional_dependencies.importlib.util.find_spec", return_value=object()),
        ):
            self.assertIn("not installed", petsc_skip_reason(Comm()))
