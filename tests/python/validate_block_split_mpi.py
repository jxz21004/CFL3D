"""Run with 3 or 8 ranks: insufficient blocks, file and explicit mesh paths."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import sys
from mpi4py import MPI
from cfl3d import CFL3D, CGNSMesh
from fixtures import box_mesh
from types import SimpleNamespace
from cfl3d.solution_io import write_volume
from cfl3d.mesh.volume_layout import VolumeMeshLayout

comm=MPI.COMM_WORLD
legacy = '--legacy' in sys.argv
root=Path('build/native-output-auto-split-'+str(comm.size)+('-legacy' if legacy else ''))
mesh=box_mesh()
levels = 2
wall_name = "face20"
if '--3d' in sys.argv:
    from test_block_split import cube
    z = cube('zone0',17)
    i,j,k = np.indices(z.dimensions)
    z.coordinates = np.stack((j/16.,-i/16.,k/16.),axis=-1)
    mesh = CGNSMesh([z])
    levels = 3
    wall_name = "face5"
next(b for b in mesh.zones[0].boundaries if b.name==wall_name).bc_type="BCWallInviscid"
if comm.size==8:
    base=mesh.zones[0]
    mesh=CGNSMesh([])
    for i in range(6):
        z=deepcopy(base); z.name=f'zone{i}'; z.coordinates[:,:,:,0]+=i*2
        mesh.zones.append(z)
error = None
if comm.rank==0:
    try:
        root.mkdir(parents=True,exist_ok=True)
        fields={z.name:np.ones(tuple(n-1 for n in z.dimensions)+(5,)) for z in mesh.zones}
        write_volume(root/'input.cgns',mesh,fields,SimpleNamespace(temperature=300.,gamma=1.4,mach=.2,freestream_pressure=101325.))
    except Exception as exc:
        error=str(exc)
error=comm.bcast(error,root=0)
if error: raise RuntimeError(error)
opts=dict(mach=.2,alpha=0.,turbulence='euler',iterations=2,multigrid_levels=levels,
          mesh_sequence=True,coarse_grid_iterations=1,resident=not legacy,
          writevolumesolution=False,writeTecplotSurfaceSolution=False)
original=np.concatenate([z.coordinates.reshape(-1,3,order='F') for z in mesh.zones]).ravel()
for mode in ('file','complete'):
    inp=str(root/'input.cgns') if mode=='file' else mesh
    with CFL3D(inp,opts,comm=comm) as solver:
        assert len(solver.mesh.zones)==comm.size-1
        layout=VolumeMeshLayout(solver.mesh)
        np.testing.assert_array_equal(layout.coordinates(solver.mesh).ravel(), original[layout.component_indices()])
        assert f'zone0/{wall_name}' in solver._cgns_surface_groups()
        solver.getSurfaceMesh(f'zone0/{wall_name}')  # collective topology and ownership check
        result=solver.solve(output_directory=root/mode,final=legacy)
        if not legacy:
            assert solver.getRuntimeInfo()['initializations']==1
        if not legacy:
            xyz=solver.getVolumeCoordinates()
            solver.setVolumeCoordinates(xyz)
        result=solver.solve(output_directory=root/mode,final=legacy)
        if not legacy:
            assert solver.getRuntimeInfo()['initializations']==1
        total=solver.getSolution()
        component=solver.getSolution(f'zone0/{wall_name}')
        np.testing.assert_allclose(total['wettedArea'],len(mesh.zones),rtol=0,atol=1e-12)
        np.testing.assert_allclose(component[f'zone0/{wall_name}_wettedArea'],1.,rtol=0,atol=1e-12)
# Every rank must receive the failed plan; no automatic level downgrade.
try:
    with CFL3D(mesh,dict(opts,multigrid_levels=levels+1),comm=comm):
        raise AssertionError('Expected an impossible three-level cut to fail')
except RuntimeError as exc:
    assert 'Cannot create' in str(exc)
if comm.rank==0: print('AUTO_SPLIT_MPI_PASS',comm.size,flush=True)
