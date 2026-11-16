"""Large Python payload transport; never replaces the public/native communicator."""


def large_objects(comm):
    """Borrow an intracommunicator using pkl5 out-of-band array transport.

    Requires mpi4py >= 4.0 (declared by the MPI extra). Import lazily for
    serial use; borrow the handle without Dup/Free or size-dependent switching.
    """
    from mpi4py.util import pkl5
    return pkl5.Intracomm(comm)
