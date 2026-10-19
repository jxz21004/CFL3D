"""Experimental resident driver: retain native workspace at cycle boundaries."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D, _native
from test_surface_forces import grouped_plate


@contextmanager
def native_session(mesh, options, directory, chunk):
    solver = CFL3D(mesh, options, comm=None)
    arrays, xyz = solver._arrays()
    assert _native.py_configure(**arrays) == 0
    _native.py_set_coordinates(xyz)
    _native.py_set_output_names('resident', 0, 0)
    _native.py_set_output_flags(0)
    assert _native.py_start_resident(chunk) == 0
    previous = Path.cwd()
    Path(directory).mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    try:
        yield arrays
    finally:
        try:
            assert _native.py_close_resident() == 0
            _native.py_release_buffers()
        finally:
            os.chdir(previous)


def readout(mesh, nrows):
    owners, ncells, _ = _native.py_get_partition(len(mesh.zones))
    packed = _native.py_get_solution(ncells)
    fields = {}
    offset = 0
    for zone in mesh.zones:
        count = int(np.prod(np.array(zone.dimensions)-1))
        fields[zone.name] = packed[offset:offset+count].reshape(
            (*tuple(n-1 for n in zone.dimensions), 5), order='F').copy()
        offset += count
    return fields, _native.py_get_history(nrows)


class ResidentNativeTests(unittest.TestCase):
    def test_explicit_final_checkpoint_preserves_full_transition_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = grouped_plate(split=25)
            options = dict(turbulence='gamma-theta', mach=.2, iterations=9, cfl=1.,
                multigrid_levels=3, multigrid_cycle=2, first_order_iterations=2,
                cfl_ramp_steps=3, cfl_ramp_factor=2.,
                writevolumesolution=False, writeTecplotSurfaceSolution=False)
            reference = CFL3D(mesh, options, comm=None)
            reference.solve(output_directory=root/'reference')
            with native_session(mesh, options, root/'resident', 2):
                self.assertEqual(_native.py_solve(), (0, 2))
                self.assertFalse(Path('.cfl3d-state.bin').exists())
                self.assertEqual(_native.py_advance_resident(3, 0.), (0, 5))
                self.assertFalse(Path('.cfl3d-state.bin').exists())
                # Explicit export once, at the end, verifies the authoritative
                # native turbulence/viscosity/ghost records, not just mean flow.
                _native.py_set_output_flags(1)
                self.assertEqual(_native.py_advance_resident(4, 0.), (0, 9))
                self.assertEqual(Path('.cfl3d-state.bin').read_bytes(), reference.getFlowState().native)

    def test_cfl_changes_are_independent_of_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh = grouped_plate(split=25)
            options = dict(turbulence='sst', mach=.2, iterations=10, cfl=1.,
                multigrid_levels=2, writevolumesolution=False, writeTecplotSurfaceSolution=False)
            outputs = []
            for number, schedule in enumerate((((3,.5),(5,2.)), ((1,.5),(2,0.),(2,2.),(3,0.)))):
                with native_session(mesh, options, Path(directory)/str(number), 2):
                    self.assertEqual(_native.py_solve(), (0, 2))
                    self.assertEqual(_native.py_advance_resident(1, float('inf'))[0], 1)
                    for steps, cfl in schedule:
                        status, rows = _native.py_advance_resident(steps, cfl)
                        self.assertEqual(status, 0)
                    outputs.append(readout(mesh, rows))
                    self.assertEqual(_native.py_resident_info()[3], 1)
            np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
            for name in outputs[0][0]:
                np.testing.assert_array_equal(outputs[0][0][name], outputs[1][0][name])

    def test_continuous_and_chunked_cycles_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ('sa', 'sst', 'gamma-theta'):
                for cycle in (1, 2):
                    with self.subTest(model=model, cycle=cycle):
                        mesh = grouped_plate(split=25)
                        options = dict(turbulence=model, mach=.2, alpha=2., cfl=1.,
                            iterations=9, multigrid_levels=3, multigrid_cycle=cycle,
                            first_order_iterations=2, cfl_ramp_steps=3, cfl_ramp_factor=2.,
                            writevolumesolution=False, writeTecplotSurfaceSolution=False)
                        reference = CFL3D(mesh, options, comm=None).solve(
                            output_directory=root/f'reference-{model}-{cycle}')
                        target = root/f'chunks-{model}-{cycle}'
                        with native_session(mesh, options, target, 2) as arrays:
                            status, rows = _native.py_solve()
                            self.assertEqual((status, rows), (0, 2))
                            info = _native.py_resident_info()
                            self.assertEqual(info[:4], (1, 1, 2, 1))
                            self.assertGreater(info[5], 0)
                            self.assertEqual(info[6], 0)
                            self.assertEqual(_native.py_solve(), (2, 2))
                            # Reconfiguration must not overwrite an active COMMON/workspace.
                            self.assertNotEqual(_native.py_configure(**arrays), 0)
                            for steps, total in ((3, 5), (1, 6), (3, 9)):
                                status, rows = _native.py_advance_resident(steps, 0.)
                                self.assertEqual((status, rows), (0, total))
                                info = _native.py_resident_info()
                                self.assertEqual(info[2:4], (total, 1))
                                fields, history = readout(mesh, rows)
                                np.testing.assert_array_equal(history, reference.history[:total])
                                if total == 9:
                                    for name in fields:
                                        np.testing.assert_array_equal(fields[name], reference.fields[name])
                                    self.assertEqual(info[:2], (0, 0))
                            self.assertFalse(Path('.cfl3d-state.bin').exists())

    def test_early_close_releases_workspace_without_advancing(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh = grouped_plate()
            options = dict(turbulence='sst', mach=.2, iterations=8, cfl=1.,
                           writevolumesolution=False, writeTecplotSurfaceSolution=False)
            with native_session(mesh, options, Path(directory)/'session', 2):
                self.assertEqual(_native.py_solve(), (0, 2))
                self.assertEqual(_native.py_advance_resident(0, 0.)[0], 1)
                self.assertEqual(_native.py_advance_resident(100, 0.)[0], 1)
                self.assertEqual(_native.py_close_resident(), 0)
                self.assertEqual(_native.py_resident_info()[:4], (0, 0, 2, 1))
                self.assertEqual(_native.py_close_resident(), 0)
            # Another ordinary solver may initialize after explicit cleanup.
            result = CFL3D(mesh, options, comm=None).solve(output_directory=Path(directory)/'ordinary')
            self.assertEqual(len(result.history), 8)


if __name__ == '__main__':
    unittest.main()
