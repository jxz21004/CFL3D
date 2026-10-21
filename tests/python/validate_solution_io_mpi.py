"""Run with mpirun -np 2/3, or -np 4 with --split; no user mesh is needed."""
import argparse
from pathlib import Path

import numpy as np
from cfl3d import CFL3D, CGNSMesh
from cgns_fixtures import write_adf
from test_surface_forces import grouped_plate


def main():
    from mpi4py import MPI
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--segments', type=int, default=37, help='Strip count; use 3 for Valgrind')
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    if args.split and world.size != 4:
        parser.error('--split needs four processes')
    color = world.rank//2 if args.split else 0
    comm = world.Split(color,world.rank) if args.split else world
    root = args.directory.resolve()/f'{world.size}-processes-group-{color}'
    options = dict(input_grid=root/'input.cgns', turbulence='sst', mach=.2, alpha=2.,
                   iterations=6,cfl=1.,target_cl=.01,alpha_step_limit=.1,freestream_pressure=101325.,multigrid_levels=2,
                   limiter=[0,2,4], writevolumesolution=False,
                   writeTecplotSurfaceSolution=False, writeflowsolution=False)
    reference = error = None
    if comm.rank == 0:
        try:
            root.mkdir(parents=True,exist_ok=True)
            write_adf(grouped_plate(),options['input_grid'])
            serial = CFL3D(options={**options,'output_directory':root/'serial'}, comm=None)
            serial.addLiftDistribution(args.segments,'y','Wing')
            serial.addSlices('y',[-.25,-.5],groupName='Wing')
            serial.addSlices('y',-.5,'absolute','Wing')
            result = serial.solve()
            serial.writeVolumeSolution()
            serial.writeTecplotSurfaceSolution()
            serial.writeFlowSolution()
            reference = (result.fields,result.history,result.wall_fields,result.surface_coefficients,
                         result.wall_forces,serial.getLiftDistribution(),serial.getSlices())
        except Exception as exc:
            error = repr(exc)
    error, reference = comm.bcast((error,reference),root=0)
    if error:
        raise RuntimeError(error)
    solver = CFL3D(options={**options,'output_directory':root/'mpi'} if comm.rank==0 else None, comm=comm)
    # Surface registration, detailed queries and writers collect owned patches.
    solver.addLiftDistribution(args.segments,'y','Wing')
    solver.addSlices('y',[-.25,-.5],groupName='Wing')
    solver.addSlices('y',-.5,'absolute','Wing')
    result = solver.solve()

    actual_lift, actual_slices = solver.getLiftDistribution(), solver.getSlices()
    def check():
        fields,history,walls,coefficients,forces,lift,slices = reference
        np.testing.assert_array_equal(result.history,history)
        assert set(result.fields) == {name for name, owner in result.zone_owners.items() if owner == comm.rank}
        local_reference = {name: fields[name] for name in result.fields}
        owned_walls = {key: value for key,value in walls.items() if key[0] in result.fields}
        owned_forces = {key: value for key,value in forces.items() if key[0] in result.fields}
        for actual,expected in ((result.fields,local_reference),(result.wall_fields,owned_walls),
                                (result.surface_coefficients,coefficients),(result.wall_forces,owned_forces)):
            assert actual.keys()==expected.keys()
            for key in actual:
                np.testing.assert_array_equal(actual[key],expected[key])
        if comm.rank == 0:
            for actual,expected in [(actual_lift,lift),*zip(actual_slices,slices)]:
                assert actual.keys()==expected.keys()
                for key in actual:
                    if isinstance(actual[key],np.ndarray):
                        np.testing.assert_array_equal(actual[key],expected[key])
                    else:
                        assert actual[key]==expected[key],key
            total = solver.getSolution('Wing')
            for name in ('CL','CD','CDp','CDv'):
                np.testing.assert_allclose(actual_lift['bin_'+name].sum(),total['Wing_'+name],
                                           rtol=2e-13,atol=2e-14)
            output = result.output_directory
            assert not list(output.glob('rank-*')), 'successful workers left temporary directories'
            for suffix in ('.out15', '.out20', '.blomax', '.prout'):
                assert not (output/f'out_000{suffix}').exists(), suffix
            assert (output/'out_000.alpha').stat().st_size > 0  # This case uses target_cl.
            for name in ('out_000_volume.cgns','out_000_surf.plt','out_000_flow.x','out_000_flow.q','out_000_restart.bin'):
                assert not (output/name).exists(),name
    solver._checked(check)
    solver.writeTecplotSurfaceSolution()
    solver.writeVolumeSolution()
    solver.writeFlowSolution()
    def check_files():
        if comm.rank == 0:
            output = result.output_directory
            for name in ('out_000_surf.plt','out_000_lift.dat','out_000_slices.dat','out_000_flow.x','out_000_flow.q'):
                assert (output/name).read_bytes()==(root/'serial'/name).read_bytes(),name
            loaded = CGNSMesh.from_file(output/'out_000_volume.cgns')
            for a,b in zip(loaded.zones,serial.mesh.zones):
                np.testing.assert_array_equal(a.coordinates,b.coordinates)
    solver._checked(check_files)
    # A second call appends the requested steps and auto-writes registered analyses,
    # even while the main surface-output flag remains disabled.
    first_files = None
    second_reference = None
    if comm.rank == 0:
        first_files = {name:(result.output_directory/name).read_bytes() for name in
                       ('out_000_surf.plt','out_000_slices.dat','out_000_lift.dat','out_000_solution.npz')}
        serial.setOption('iterations',2)
        second_reference = serial.solve()
    second_reference = comm.bcast(second_reference,root=0)
    solver.setOption('iterations',2)
    second = solver.solve()
    def check_second():
        assert second.solve_index==1 and solver.nSolve==2
        np.testing.assert_array_equal(second.history,second_reference.history)
        for name in second.fields:
            np.testing.assert_array_equal(second.fields[name],second_reference.fields[name])
        if comm.rank==0:
            for name,content in first_files.items():
                assert (second.output_directory/name).read_bytes()==content,name
            for name in ('out_001_slices.dat','out_001_lift.dat','out_001_solution.npz'):
                assert (second.output_directory/name).read_bytes()==(root/'serial'/name).read_bytes(),name
            assert not (second.output_directory/'out_001_surf.plt').exists()
    solver._checked(check_second)
    # Keep this output number for a shorter continuation chunk on all ranks.
    append_reference = None
    before_append = None
    if comm.rank == 0:
        before_append = {key:second.output_paths[key].read_bytes() for key in
                         ('log','residual','turbulence_history')}
        serial.setOptions({'numberSolutions':False, 'iterations':1})
        append_reference = serial.solve()
    append_reference = comm.bcast(append_reference,root=0)
    solver.setOptions({'NUMBERSOLUTIONS':False, 'iterations':1})
    appended = solver.solve()
    def check_appended():
        assert appended.solve_index == 1 and solver.nSolve == 3
        assert appended.iteration_offset == 2
        np.testing.assert_array_equal(appended.history,append_reference.history)
        np.testing.assert_array_equal(appended.cumulative_history,append_reference.cumulative_history)
        for name in appended.fields:
            np.testing.assert_array_equal(appended.fields[name],append_reference.fields[name])
        if comm.rank == 0:
            for key,content in before_append.items():
                assert appended.output_paths[key].read_bytes().startswith(content),key
            for key in ('residual','turbulence_history'):
                np.testing.assert_array_equal(np.loadtxt(appended.output_paths[key])[:,0],np.arange(1,4))
            for name in ('out_001_slices.dat','out_001_lift.dat','out_001_solution.npz'):
                assert (appended.output_directory/name).read_bytes()==(root/'serial'/name).read_bytes(),name
            assert not (appended.output_directory/'cfl3d.out').exists()
    solver._checked(check_appended)
    if comm.rank == comm.size-1:
        solver.clearSurfaceAnalysis()
    try:
        solver.solve()
    except RuntimeError as exc:
        assert 'Surface registrations differ' in str(exc)
    else:
        raise AssertionError('Expected collective rejection of inconsistent surface requests')
    comm.Barrier()
    if comm.rank == 0:
        print(f'IO_PASS world={world.size} group={color} local={comm.size}: fields, wall loads, analyses, numbered outputs and same-number appended histories match serial',flush=True)
    if args.split:
        comm.Free()
    world.Barrier()


if __name__ == '__main__':
    main()
