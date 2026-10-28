"""Optional PETSc GMRES backend; imported only by an explicit user selection."""

import sys
import numpy as np
from ._distributed_warp import scalar


def load_petsc():
    try:
        import petsc4py

        if "petsc4py.PETSc" not in sys.modules:
            from mpi4py import MPI

            # Avoid implicit initialization on COMM_WORLD: another disjoint
            # case may deliberately use only native GS and never load PETSc.
            # Preserve application-owned initialization when already imported.
            petsc4py.init(args=[], comm=MPI.COMM_SELF)
        from petsc4py import PETSc
    except ImportError as exc:
        raise ImportError("linear_solver='petsc-gmres' requires a working petsc4py installation") from exc
    if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise ValueError("The mesh adapter requires a real double-precision PETSc build")
    return PETSc


def solve_petsc(PETSc, shared, matvec, rhs, diagonal, restart):
    """PETSc owns Krylov iteration/storage; only the local operator is supplied.

    PETSc vectors contain canonical owned nodes only. Interface copies are filled
    on each matrix-vector product. No matrix, vector or surface is gathered.
    """
    if scalar(shared.comm, int(np.any(~np.isfinite(diagonal)) or np.any(diagonal == 0)), "max"):
        raise ValueError("Degenerate FMEM control matrix")
    comm = PETSc.COMM_SELF if shared.comm is None else shared.comm
    owned = shared.owned
    local_size = 3 * int(np.count_nonzero(owned))

    class Operator:
        def mult(self, matrix, x, y):
            values = np.zeros_like(rhs)
            values[owned] = x.getArray(readonly=True).reshape(-1, 3)
            values = shared.sum(values)
            y.getArray()[:] = matvec(values)[owned].ravel()

        def getDiagonal(self, matrix, result):
            result.getArray()[:] = diagonal[owned].ravel()

    matrix = b = x = ksp = None
    try:
        matrix = PETSc.Mat().createPython(((local_size, None), (local_size, None)), context=Operator(), comm=comm)
        matrix.setUp()
        x, b = matrix.createVecs()
        b.getArray()[:] = rhs[owned].ravel()
        x.set(0)
        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOperators(matrix)
        ksp.setType("gmres")
        ksp.setGMRESRestart(restart)
        ksp.getPC().setType("jacobi")
        ksp.setPCSide(PETSc.PC.Side.LEFT)
        ksp.setNormType(PETSc.KSP.NormType.PRECONDITIONED)
        ksp.setTolerances(rtol=1e-11, atol=1e-13, max_it=3000)
        # No setFromOptions(): unrelated application PETSc options must not
        # silently replace the backend/algorithm explicitly selected here.
        ksp.solve(b, x)
        reason = int(ksp.getConvergedReason())
        if scalar(shared.comm, int(reason <= 0), "max"):
            raise ValueError(f"PETSc GMRES did not converge (reason {reason})")
        result = np.zeros_like(rhs)
        result[owned] = x.getArray(readonly=True).reshape(-1, 3)
        return shared.sum(result)
    finally:
        for obj in (ksp, x, b, matrix):
            if obj is not None:
                obj.destroy()
