/* Serial error exits return through f2py. MPI errors abort the job before here.
 * longjmp does not unwind Fortran allocations: after an error the Python state
 * rejects reuse, and the caller must start a fresh process for another solve.
 */
#include <setjmp.h>
static jmp_buf solver_exit;
static int active = 0;
extern void cfl3d_python_main_(void);
void py_guarded_solve_(int *status) {
    *status = setjmp(solver_exit);
    if (*status == 0) {
        active = 1;
        cfl3d_python_main_();
    }
    active = 0;
}
void py_abort_jump_(const int *status) {
    if (active) longjmp(solver_exit, *status == 0 ? 1 : *status);
}
