"""Transport selection needs no coordinate allocation or MPI runtime."""
import builtins
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cfl3d._mpi_objects import mesh_communicator


class Comm:
    def __init__(self, remote_large=False, remote_error=None):
        self.remote_large = remote_large
        self.remote_error = remote_error

    def allgather(self, value):
        return [value, self.remote_large if isinstance(value, bool) else self.remote_error]


class MeshCommunicatorTests(unittest.TestCase):
    def setUp(self):
        self.mesh = SimpleNamespace(zones=[SimpleNamespace(dimensions=(3, 3, 3))])

    def test_small_and_serial_do_not_import_mpi(self):
        comm = Comm()
        with patch('builtins.__import__', side_effect=AssertionError('unexpected import')):
            self.assertIs(mesh_communicator(comm, self.mesh), comm)
            self.assertIsNone(mesh_communicator(None, self.mesh))

    def test_remote_large_rank_requires_support_on_every_rank(self):
        # This rank has a tiny mesh, but must honor the other rank's decision.
        with patch('builtins.__import__', side_effect=ImportError('unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'pkl5 collective support'):
                mesh_communicator(Comm(remote_large=True), self.mesh)

    def test_large_dimensions_do_not_overflow_or_allocate(self):
        self.mesh.zones[0].dimensions = (2**31, 2, 2)
        with patch('builtins.__import__', side_effect=ImportError('unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'Large mesh'):
                mesh_communicator(Comm(), self.mesh)

    def test_remote_import_failure_is_collective(self):
        class Wrapped:
            def __init__(self, comm):
                self.comm = comm

        for name in ('gather', 'scatter', 'alltoall', 'allgather', 'bcast'):
            setattr(Wrapped, name, lambda *args: None)
        original = builtins.__import__

        def importing(name, *args, **kwargs):
            if name == 'mpi4py.util':
                return SimpleNamespace(pkl5=SimpleNamespace(Intracomm=Wrapped))
            return original(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=importing):
            with self.assertRaisesRegex(RuntimeError, 'remote unavailable'):
                mesh_communicator(Comm(True, 'remote unavailable'), self.mesh)


if __name__ == '__main__':
    unittest.main()
