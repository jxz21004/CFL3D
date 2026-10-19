"""Bounded native indexing/NPZ extraction and compatible array serialization."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from cfl3d import CFL3D
from cfl3d._streaming import CHUNK_BYTES, array_chunks, file_chunks
from cfl3d.restart import index_native_file, open_native_restart, write_restart, zone_payloads
from test_surface_forces import grouped_plate


class BoundedReader(io.BytesIO):
    def read(self, size=-1):
        if not 0 <= size <= CHUNK_BYTES:
            raise AssertionError(f'Unbounded native read: {size}')
        return super().read(size)


class StreamingRestartTests(unittest.TestCase):
    def test_array_chunk_layout_and_bounds(self):
        base = np.arange(3*CHUNK_BYTES//8, dtype=np.float64).reshape(3, -1)
        for array in (base, base.T, base[:, ::2], base.T[::-1]):
            for order in ('C','F'):
                with self.subTest(shape=array.shape, strides=array.strides, order=order):
                    result = []
                    for chunk in array_chunks(array, fortran_order=order == 'F'):
                        self.assertLessEqual(len(chunk), CHUNK_BYTES)
                        result.append(bytes(chunk))  # Consume before nditer's next buffered iteration.
                    self.assertEqual(b''.join(result), array.tobytes(order=order))

    def test_native_index_and_npz_extraction_all_models(self):
        original_read = zipfile.ZipExtFile.read
        def guarded_read(member, size=-1):
            if member.name == 'native.npy' and not 0 <= size <= CHUNK_BYTES:
                raise AssertionError(f'Unbounded NPZ native extraction: {size}')
            return original_read(member, size)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ('euler','laminar','sa','sst','komega','bsl','sst2003','wilcox2006','gamma-theta'):
                with self.subTest(model=model):
                    mesh = grouped_plate()
                    if model == 'euler':
                        for zone in mesh.zones:
                            for bc in zone.boundaries:
                                if bc.bc_type == 'BCWallViscous':
                                    bc.bc_type = 'BCWallInviscid'
                    solver = CFL3D(mesh, dict(turbulence=model, mach=.2, iterations=2,
                                  writevolumesolution=False, writeTecplotSurfaceSolution=False), comm=None)
                    solver.solve(output_directory=root/model)
                    state = solver.getFlowState()
                    stream = BoundedReader(state.native)
                    ranges = index_native_file(stream, mesh, solver.options)
                    parts = zone_payloads(state, mesh)
                    for index,(start,length) in enumerate(ranges):
                        self.assertEqual(b''.join(file_chunks(stream,start,length)), bytes(parts[index]))
                    raw, saved = root/(model+'.bin'), root/(model+'.npz')
                    raw.write_bytes(state.native)
                    write_restart(saved, state)
                    with patch.object(zipfile.ZipExtFile, 'read', guarded_read):
                        for path in (raw,saved):
                            native,digest = open_native_restart(path,mesh,solver.options)
                            try:
                                self.assertEqual(digest, hashlib.sha256(state.native).hexdigest() if path == saved else None)
                                self.assertEqual(index_native_file(native,mesh,solver.options), ranges)
                            finally:
                                native.close()
                    with self.assertRaisesRegex(ValueError, 'record'):
                        index_native_file(BoundedReader(state.native[:-5]),mesh,solver.options)
                    bad = root/'bad.npz'
                    with np.load(saved,allow_pickle=False) as archive:
                        np.savez_compressed(bad,metadata=archive['metadata'],native=archive['native'],sha256='0'*64)
                    with self.assertRaisesRegex(ValueError, 'checksum'):
                        open_native_restart(bad,mesh,solver.options)


if __name__ == '__main__':
    unittest.main()
