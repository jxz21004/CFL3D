"""Native zone shards retain every model/ghost record and reject lost ownership."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D
from cfl3d.restart import merge_states, partition_state, write_restart
from test_surface_forces import grouped_plate


class DistributedRestartTests(unittest.TestCase):
    def test_split_validate_merge_all_native_models(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in ('euler', 'laminar', 'sa', 'sst', 'komega', 'gamma-theta',
                          'bsl', 'sst2003', 'wilcox2006'):
                with self.subTest(model=model):
                    mesh = grouped_plate()
                    if model == 'euler':
                        for zone in mesh.zones:
                            for bc in zone.boundaries:
                                if bc.bc_type == 'BCWallViscous':
                                    bc.bc_type = 'BCWallInviscid'
                    solver = CFL3D(mesh, dict(turbulence=model, mach=.2, iterations=2,
                                  writevolumesolution=False, writeTecplotSurfaceSolution=False), comm=None)
                    solver.solve(output_directory=Path(directory)/model)
                    complete = solver.getFlowState()
                    pieces = partition_state(complete, mesh, (2, 1), 3)
                    self.assertEqual([part.zone_indices for part in pieces], [(), (1,), (0,)])
                    self.assertEqual(pieces[0].native, b'')
                    self.assertEqual(sum(len(part.native) for part in pieces), len(complete.native))
                    for part in pieces:
                        part.validate(mesh, solver.options)
                        self.assertEqual(part.global_digest, complete.digest)
                    merged = merge_states(pieces, mesh)
                    self.assertEqual(merged.native, complete.native)
                    for name in complete.fields:
                        np.testing.assert_array_equal(merged.fields[name], complete.fields[name])
                    with self.assertRaisesRegex(ValueError, 'collectively'):
                        write_restart(Path(directory)/'local.npz', pieces[1])
                    for owners in ((1,), (1, 3), (-1, 1)):
                        with self.assertRaisesRegex(ValueError, 'partition map'):
                            partition_state(complete, mesh, owners, 3)
                    mixed = deepcopy(pieces)
                    mixed[1].global_digest = 'different-solve'
                    with self.assertRaisesRegex(ValueError, 'states differ'):
                        merge_states(mixed, mesh)
                    with self.assertRaisesRegex(ValueError, 'Incomplete'):
                        merge_states(pieces[:2], mesh)
                    with self.assertRaisesRegex(ValueError, 'Duplicate'):
                        merge_states([*pieces, pieces[1]], mesh)
                    bad = deepcopy(pieces[1])
                    bad.zone_indices = (0,)
                    with self.assertRaises(ValueError):
                        bad.validate(mesh, solver.options)


if __name__ == '__main__':
    unittest.main()
