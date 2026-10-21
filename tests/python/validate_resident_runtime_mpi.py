"""Public resident solve: native ownership, communicator lifetime and output."""
import argparse
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
from mpi4py import MPI
from cfl3d import CFL3D, MultigridStage
from test_surface_forces import grouped_plate


def digest(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def mesh3d():
    mesh = grouped_plate(split=25)
    for zone in mesh.zones:
        zone.coordinates = np.linspace(zone.coordinates[0],zone.coordinates[-1],5)
        for boundary in zone.boundaries:
            boundary.point_range[0][boundary.point_range[0] == 2] = 5
            if boundary.point_range[0,0] == boundary.point_range[0,1]:
                boundary.bc_type = 'BCSymmetryPlane'
        for connection in zone.connections:
            for points in (connection.point_range,connection.donor_range):
                points[0][points[0] == 2] = 5
        xyz = zone.coordinates.copy()
        zone.coordinates[...,1] = xyz[...,2]
        zone.coordinates[...,2] = -xyz[...,1]
    return mesh


def run(comm,root,model,coarse_startup=False):
    mesh = mesh3d()
    config = dict(turbulence=model,mach=.2,alpha=2.,beta=3.,y_up=True,
        freestream_pressure=101325.,cfl=1.,iterations=2,multigrid_levels=3,
        first_order_iterations=2,cfl_ramp_steps=3,cfl_ramp_factor=2.,
        writevolumesolution=False,writeTecplotSurfaceSolution=False)
    startup = 7 if coarse_startup else 0
    if coarse_startup:
        config['coarse_grid_stages'] = (MultigridStage(3,1,1),MultigridStage(4,2,1))
    reference = error = None
    if comm.rank == 0:
        try:
            result = CFL3D(mesh,{**config,'iterations':29},comm=None).solve(output_directory=root/'serial')
            reference = dict(history=result.history,fields={k:digest(v) for k,v in result.fields.items()},
                wall={k:digest(v) for k,v in result.wall_fields.items()},surface=result.surface_coefficients,
                alpha=result.final_alpha)
        except BaseException as exc:
            error = repr(exc)
    error,reference = comm.bcast((error,reference),root=0)
    if error:
        raise RuntimeError(error)
    output = root/'mpi'
    with CFL3D(mesh,{**config,'resident':True},comm=comm) as solver:
        with patch.object(solver,'_native_arrays',wraps=solver._native_arrays) as prepare, \
             patch.object(solver,'_stage_flow_state',side_effect=AssertionError('restart staging')):
            chunks = [solver.solve(output_directory=output)]
            borrowed = solver._resident_context.comm.py2f()
            # Rank-local inconsistent edits must fail collectively before advancing.
            if comm.rank == comm.size-1:
                solver.options.cfl = 1.1
            try:
                solver.solve(output_directory=output)
            except RuntimeError as exc:
                assert 'options differ' in str(exc)
            else:
                raise AssertionError('Expected mismatched options')
            solver.options.cfl = 1.
            for steps,total in ((3,5),(1,6),(7,13),(16,29)):
                solver.setOptions({'iterations':steps,'numberSolutions':total != 6})
                if total == 29:
                    solver.addLiftDistribution(7,'z','Wing')
                    solver.addSlices('z',.5,groupName='Wing')
                    solver.setOptions({'writevolumesolution':True,
                                       'writeTecplotSurfaceSolution':True,'writeflowsolution':True})
                result = solver(output_directory=output)
                chunks.append(result)
                def check():
                    info = solver.getRuntimeInfo()
                    assert (info['cycles'],info['initializations'],info['input_coordinate_values']) == (total+startup,1,0)
                    assert (info['startup_cycles'],info['fine_cycles']) == (startup,total)
                    assert solver._resident_context.comm.py2f() == borrowed
                    assert prepare.call_count == 1
                    if comm.size == 3:
                        assert tuple(result.zone_owners.values()) == (2,1)
                    np.testing.assert_array_equal(np.concatenate([c.history for c in chunks]),reference['history'][:total+startup])
                    if total == 6 and comm.rank == 0:
                        np.testing.assert_array_equal(np.loadtxt(result.output_paths['residual'])[:,0],np.arange(1,5))
                    if total < 29 and comm.rank == 0:
                        assert not list(output.glob('*_solution.npz'))
                    if total == 29:
                        for name,values in result.fields.items():
                            assert digest(values) == reference['fields'][name]
                        for key,values in result.wall_fields.items():
                            assert digest(values) == reference['wall'][key]
                        for key,values in result.surface_coefficients.items():
                            np.testing.assert_array_equal(values,reference['surface'][key])
                        assert result.final_alpha == reference['alpha']
                        if comm.rank == 0:
                            for key in ('volume','surface','slices','lift','solution'):
                                assert result.output_paths[key].exists(),key
                            assert not result.output_paths['restart'].exists()
                    assert not (output/'.cfl3d-state.bin').exists()
                solver._checked(check)
                comm.Barrier()
        other = CFL3D(mesh,config,comm=comm)
        try:
            other.solve(output_directory=root/'blocked')
        except RuntimeError as exc:
            assert 'Another CFL3D' in str(exc)
        else:
            raise AssertionError('Expected owner guard')
        assert solver.getRuntimeInfo()['cycles'] == 29+startup
    assert not solver.getRuntimeInfo()['active']
    solver.getSolution('Wing')  # Result queries survive closing the workspace.
    other.solve(output_directory=root/'after-close')
    assert not MPI.Is_finalized()
    comm.Barrier()
    if comm.rank == 0:
        print('PUBLIC_RESIDENT_PASS',model,root,'setup=1, communicator retained, native owners unchanged, outputs checked',flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--model',default='sst')
    parser.add_argument('--split',action='store_true')
    parser.add_argument('--exclude',action='store_true')
    parser.add_argument('--coarse-startup',action='store_true')
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    if args.split or args.exclude:
        assert world.size == 4
        color = (MPI.UNDEFINED if world.rank == 0 else 0) if args.exclude else world.rank//2
        comm = world.Split(color,world.rank)
    else:
        assert world.size in (2,3)
        color,comm = 0,world
    try:
        if comm != MPI.COMM_NULL:
            run(comm,args.directory.resolve()/f'group-{color}',args.model,args.coarse_startup)
            if comm != world:
                comm.Free()
        world.Barrier()
    except BaseException:
        import traceback
        traceback.print_exc()
        world.Abort(1)


if __name__ == '__main__':
    main()
