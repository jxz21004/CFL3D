"""Optional test dependencies: detect presence without importing MPI libraries."""

import importlib.util
import os


def petsc_skip_reason(comm=None, *, skip=False):
    """Agree on skipping before any rank imports PETSc.

    A discoverable but broken installation is not treated as absent: the real
    import/test must fail normally. This policy affects tests, never the solver.
    """
    if skip or os.environ.get("CFL3D_SKIP_PETSC_TESTS") == "1":
        reason = "PETSc tests explicitly disabled"
    elif importlib.util.find_spec("petsc4py") is None:
        reason = "Optional petsc4py not installed"
    else:
        reason = None
    reasons = [reason] if comm is None else comm.allgather(reason)
    return next((value for value in reasons if value is not None), None)
