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
    # Requests are deliberately root-only: registering/querying/output needs no collective.
    if comm.rank == 0:
        solver.addLiftDistribution(args.segments,'y','Wing')
        solver.addSlices('y',[-.25,-.5],groupName='Wing')
        solver.addSlices('y',-.5,'absolute','Wing')
    result = solver.solve()

    def check():
        fields,history,walls,coefficients,forces,lift,slices = reference
        np.testing.assert_array_equal(result.history,history)
        for actual,expected in ((result.fields,fields),(result.wall_fields,walls),
                                (result.surface_coefficients,coefficients),(result.wall_forces,forces)):
            assert actual.keys()==expected.keys()
            for key in actual:
                np.testing.assert_array_equal(actual[key],expected[key])
        if comm.rank == 0:
            for actual,expected in [(solver.getLiftDistribution(),lift),*zip(solver.getSlices(),slices)]:
                assert actual.keys()==expected.keys()
                for key in actual:
                    if isinstance(actual[key],np.ndarray):
                        np.testing.assert_array_equal(actual[key],expected[key])
                    else:
                        assert actual[key]==expected[key],key
            total = solver.getSolution('Wing')
            for name in ('CL','CD','CDp','CDv'):
                np.testing.assert_allclose(solver.getLiftDistribution()['bin_'+name].sum(),total[name],
                                           rtol=2e-13,atol=2e-14)
            output = result.output_directory
            for name in ('out_000_volume.cgns','out_000_surf.dat','out_000_flow.x','out_000_flow.q','out_000_restart.bin'):
                assert not (output/name).exists(),name
            # Root-only deferred writes must not contain any MPI collective.
            solver.writeVolumeSolution()
            solver.writeTecplotSurfaceSolution()
            solver.writeFlowSolution()
            for name in ('out_000_surf.dat','out_000_lift.dat','out_000_slices.dat','out_000_flow.x','out_000_flow.q'):
                assert (output/name).read_bytes()==(root/'serial'/name).read_bytes(),name
            loaded = CGNSMesh.from_file(output/'out_000_volume.cgns')
            for a,b in zip(loaded.zones,solver.mesh.zones):
                np.testing.assert_array_equal(a.coordinates,b.coordinates)
        else:
            assert solver.writeVolumeSolution() is None
            assert solver.writeTecplotSurfaceSolution() is None
            assert solver.writeFlowSolution() is None
    solver._checked(check)
    # A second call appends the requested steps and auto-writes registered analyses,
    # even while the main surface-output flag remains disabled.
    first_files = None
    second_reference = None
    if comm.rank == 0:
        first_files = {name:(result.output_directory/name).read_bytes() for name in
                       ('out_000_surf.dat','out_000_slices.dat','out_000_lift.dat','out_000_solution.npz')}
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
            assert not (second.output_directory/'out_001_surf.dat').exists()
    solver._checked(check_second)
    comm.Barrier()
    if comm.rank == 0:
        print(f'IO_PASS world={world.size} group={color} local={comm.size}: fields, wall loads, lift strips, tracked/absolute Cp cuts and numbered repeated outputs match serial',flush=True)
    if args.split:
        comm.Free()
    world.Barrier()


if __name__ == '__main__':
    main()
