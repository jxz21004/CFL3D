"""Two-equation native variants, model isolation, and restart identity."""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D
from cfl3d.restart import records
from test_surface_forces import grouped_plate


def two_equation_fields(state, mesh):
    blocks = records(state.native)
    fields = {}
    start = 0
    for zone in mesh.zones:
        count = int(np.prod(np.array(zone.dimensions)-1))
        dims = (zone.dimensions[1], zone.dimensions[2], zone.dimensions[0])
        header = next(i for i in range(start, len(blocks))
                      if len(blocks[i]) == 208 and struct.unpack_from('=3i', blocks[i], 168) == dims)
        model = header + (4 if not fields else 3)
        # wrest.F stores viscosity/ghost data, followed by the omega,k cell record.
        field = next(i for i in range(model+1, len(blocks)) if len(blocks[i]) == 2*count*8)
        fields[zone.name] = np.frombuffer(blocks[field], dtype=np.float64).reshape(count, 2, order='F')
        start = field+1
    return fields


class ModelVariantTests(unittest.TestCase):
    def test_selection_uses_distinct_flags_and_preserves_old_defaults(self):
        solver = CFL3D(grouped_plate(), dict(turbulence='sst'), comm=None)
        for name, code, flags in (('BSL', 7, (1, 0, 0)), ('SST2003', 7, (0, 1, 0)),
                                 ('Wilcox2006', 6, (0, 0, 1)), ('sst', 7, (0, 0, 0)),
                                 ('komega', 6, (0, 0, 0)), ('gamma-theta', 40, (0, 0, 0))):
            with self.subTest(model=name):
                solver.setOptions({'TURBULENCE': name})
                arrays, _ = solver._arrays()
                np.testing.assert_array_equal(arrays['blocks'][5:], code)
                np.testing.assert_array_equal(arrays['keywords'][-3:], flags)

    def test_variant_file_restarts_preserve_both_turbulence_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('bsl', 'sst2003', 'wilcox2006'):
                with self.subTest(model=name):
                    options = dict(turbulence=name, mach=.2, iterations=3, cfl=1., multigrid_levels=2,
                        freestream_turbulence_intensity=.2, freestream_eddy_viscosity=1.,
                        keep_ambient_turbulence=True, writevolumesolution=False,
                        writeTecplotSurfaceSolution=False)
                    solver = CFL3D(grouped_plate(), options, comm=None)
                    solver.solve(output_directory=root/name/'first')
                    path = solver.writeRestartFile(root/name/'checkpoint.npz')
                    resumed = CFL3D(grouped_plate(), {**options, 'iterations': 2, 'restartFile': path}, comm=None)
                    continued = resumed.solve(output_directory=root/name/'resumed')
                    full = CFL3D(grouped_plate(), {**options, 'iterations': 5}, comm=None)
                    expected = full.solve(output_directory=root/name/'full')
                    self.assertEqual(continued.start_mode, 'restart')
                    for zone in continued.fields:
                        np.testing.assert_array_equal(continued.fields[zone], expected.fields[zone])
                    actual_turb = two_equation_fields(resumed.getFlowState(), resumed.mesh)
                    expected_turb = two_equation_fields(full.getFlowState(), full.mesh)
                    for zone in actual_turb:
                        np.testing.assert_array_equal(actual_turb[zone], expected_turb[zone])
                    # Matching native IVISC numbers must not hide differing variants in NPZ checkpoints.
                    base = 'komega' if name == 'wilcox2006' else 'sst'
                    incompatible = CFL3D(grouped_plate(), {**options, 'turbulence': base, 'restartFile': path}, comm=None)
                    with self.assertRaisesRegex(ValueError, 'turbulence'):
                        incompatible.solve(output_directory=root/name/'wrong-variant')

    def test_switching_back_clears_native_variant_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = dict(turbulence='sst', mach=.2, iterations=2, cfl=1.,
                           writevolumesolution=False, writeTecplotSurfaceSolution=False)
            solver = CFL3D(grouped_plate(), options, comm=None)
            solver.solve(output_directory=root/'initial')
            for index, name in enumerate(('bsl', 'sst2003', 'wilcox2006', 'komega', 'sst')):
                with self.subTest(model=name):
                    solver.setOption('turbulence', name)
                    changed = solver.solve(output_directory=root/f'changed-{index}')
                    self.assertEqual((changed.start_mode, changed.reset_reasons), ('cold', ('turbulence',)))
                    fresh = CFL3D(grouped_plate(), {**options, 'turbulence': name}, comm=None).solve(
                        output_directory=root/f'fresh-{index}')
                    for zone in changed.fields:
                        np.testing.assert_array_equal(changed.fields[zone], fresh.fields[zone])


if __name__ == '__main__':
    unittest.main()
