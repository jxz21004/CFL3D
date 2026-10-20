"""Stable distributed coupling nodes across native repartition and split comms."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from mpi4py import MPI

from cfl3d import CFL3D
from validate_resident_runtime_mpi import mesh3d


def compare(solver, geometry, reference):
    comm = solver.comm
    def local():
        nodes = geometry['node_ids']
        assert len(nodes) == len(np.unique(nodes))
        assert np.all(nodes[1:] > nodes[:-1])
        np.testing.assert_array_equal(geometry['node_owners'], comm.rank)
        if comm.rank == 0:
            assert geometry['coordinates'].shape == (0, 3)
            assert geometry['connectivity'].shape == (0, 4)
    solver._checked(local)
    # Only this validation explicitly gathers data; getSurfaceMesh never does.
    parts = comm.gather(geometry, root=0)
    def global_check():
        if comm.rank != 0:
            return
        nodes = np.concatenate([part['node_ids'] for part in parts])
        xyz = np.concatenate([part['coordinates'] for part in parts])
        assert len(np.unique(nodes)) == len(nodes) == len(reference['node_ids'])
        order = np.argsort(nodes)
        np.testing.assert_array_equal(nodes[order], reference['node_ids'])
        np.testing.assert_array_equal(xyz[order], reference['coordinates'])
        ids = np.concatenate([part['face_ids'] for part in parts])
        faces = np.concatenate([part['connectivity'] for part in parts])
        keys = sum((list(part['face_keys']) for part in parts), [])
        assert len(np.unique(ids)) == len(ids)
        order = np.argsort(ids)
        expected = np.argsort(reference['face_ids'])
        np.testing.assert_array_equal(ids[order], reference['face_ids'][expected])
        np.testing.assert_array_equal(faces[order], reference['connectivity'][expected])
        assert [keys[i] for i in order] == [reference['face_keys'][i] for i in expected]
        assert set(faces.flat) <= set(nodes)
    solver._checked(global_check)


def run(comm, directory):
    config = dict(turbulence='sst', mach=.2, alpha=2., beta=3., y_up=True,
                  cfl=1., iterations=2, multigrid_levels=3, resident=True,
                  writevolumesolution=False, writeTecplotSurfaceSolution=False)
    reference = (CFL3D(mesh3d(), comm=None).getSurfaceMesh('Wing') if comm.rank == 0 else None)
    # Root-only references are for checks, not a solver-side replicated cache.
    with CFL3D(mesh3d() if comm.rank == 0 else None, config, comm=comm,
               groups={'Combined': ['Wing', 'Forward', 'Aft']}) as solver:
        initial = solver.getSurfaceMesh('Wing')
        initial_ids = initial['node_ids'].copy()
        initial_xyz = initial['coordinates'].copy()
        compare(solver, initial, reference)
        for name in ('Combined', 'Forward', 'Aft'):
            serial = (CFL3D(mesh3d(), comm=None, groups={'Combined': ['Wing', 'Forward', 'Aft']})
                      .getSurfaceMesh(name) if comm.rank == 0 else None)
            compare(solver, solver.getSurfaceMesh(name), serial)
        for name in ('missing', 'face21'):
            try:
                solver.getSurfaceMesh(name)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Expected collective invalid/empty group error')
        try:
            solver.getSurfaceMesh('Forward' if comm.rank == comm.size-1 else 'Wing')
        except RuntimeError as exc:
            assert 'differs within comm' in str(exc)
        else:
            raise AssertionError('Expected mismatched group error')
        solver.addGroups({'Custom': ['Forward' if comm.rank == comm.size-1 else 'Aft']})
        try:
            solver.getSurfaceMesh('Custom')
        except RuntimeError as exc:
            assert 'membership differs' in str(exc)
        else:
            raise AssertionError('Expected mismatched membership error')
        boundary = solver.mesh.zones[0].boundaries[0]
        old_name = boundary.name
        if comm.rank == comm.size-1:
            boundary.name = 'changed-local-boundary'
        try:
            solver.getSurfaceMesh('Wing')
        except RuntimeError as exc:
            assert 'topology/coordinate ownership differs' in str(exc)
        else:
            raise AssertionError('Expected mismatched topology error')
        finally:
            boundary.name = old_name
        key = solver.mesh.zones[0].name, old_name
        if comm.rank == comm.size-1:
            solver.boundary_overrides[key] = (1005, ())
        try:
            solver.getSurfaceMesh('Wing')
        except RuntimeError as exc:
            assert 'boundary overrides differ' in str(exc)
        else:
            raise AssertionError('Expected mismatched boundary overrides error')
        finally:
            solver.boundary_overrides.pop(key, None)
        before_owners = solver._mesh_owners
        result = solver.solve(output_directory=directory)
        if comm.size == 3:
            assert before_owners == (1, 2)
            assert tuple(result.zone_owners.values()) == (2, 1)
        for step in range(2):
            with patch.object(solver, '_gather_mesh', side_effect=AssertionError('global mesh gather')), \
                 patch.object(solver, '_surface_collective', side_effect=AssertionError('global wall gather')):
                geometry = solver.getSurfaceMesh('Wing')
            compare(solver, geometry, reference)
            def unchanged():
                np.testing.assert_array_equal(geometry['node_ids'], initial_ids)
                np.testing.assert_array_equal(geometry['coordinates'], initial_xyz)
                assert solver.getRuntimeInfo()['initializations'] == 1
            solver._checked(unchanged)
            if step == 0:
                solver.solve(output_directory=directory)
        report = comm.gather(dict(rank=comm.rank, nodes=len(geometry['node_ids']),
                                  faces=len(geometry['face_ids']), bytes=geometry['coordinates'].nbytes), root=0)
        solver._checked(lambda: (directory/'surface-mesh-report.json').write_text(
            json.dumps(dict(ranks=report, zone_owners=result.zone_owners,
                            node_order_stable=True, serial_geometry_exact=True), indent=2)+'\n')
            if comm.rank == 0 else None)
    compare(solver, solver.getSurfaceMesh('Wing'), reference)
    if comm.rank == 0:
        print('Surface geometry: exact serial match, stable ownership across native repartition, no global gather', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--exclude', action='store_true')
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    if args.split or args.exclude:
        assert world.size == 4
        color = (MPI.UNDEFINED if world.rank == 0 else 0) if args.exclude else world.rank//2
        comm = world.Split(color, world.rank)
    else:
        assert world.size == 3
        color, comm = 0, world
    try:
        if comm != MPI.COMM_NULL:
            run(comm, args.directory.resolve()/f'group-{color}')
            if comm != world:
                comm.Free()
        world.Barrier()
    except BaseException:
        import traceback
        traceback.print_exc()
        world.Abort(1)


if __name__ == '__main__':
    main()
