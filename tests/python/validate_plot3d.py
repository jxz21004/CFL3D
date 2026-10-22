"""Compare deferred .x/.q against actual standalone CFL3D IPTYPE=1 output."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
from cfl3d import CFL3D, SolverOptions
from fixtures import ROOT, flatplate_mesh
from validate_parity import baseline_deck, put_record, record


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    root=args.directory.resolve()
    root.mkdir(parents=True,exist_ok=True)
    mesh=flatplate_mesh()
    zone=mesh.zones[0]
    with (root/'grid.bin').open('wb') as stream:
        put_record(stream,np.array([1],np.int32).tobytes())
        put_record(stream,np.array(zone.dimensions,np.int32).tobytes())
        put_record(stream,zone.coordinates.astype('<f8').tobytes(order='F'))
    lines=baseline_deck('sst',2).splitlines()
    lines[2]='native.x'
    lines[3]='native.q'
    for i,line in enumerate(lines):
        if line.strip().startswith('NGRID'):
            lines[i+1]='-1 1 0 9999 0 1 1 1'
        if line.strip().startswith('GRID IPTYPE'):
            lines.insert(i+1,'1 1 1 2 1 1 65 1 1 97 1')
            break
    (root/'reference.inp').write_text('\n'.join(lines)+'\n')
    with (root/'reference.inp').open('rb') as inp,(root/'stdout.log').open('wb') as log:
        subprocess.run([str(ROOT/'build/python-standalone/cfl3d_seq')],cwd=root,stdin=inp,
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
    assert 'execution terminated normally' in (root/'cfl3d.error').read_text()
    solver=CFL3D(mesh,SolverOptions(mach=.2,reynolds=6e6,temperature=460/1.8,
        turbulence='sst',iterations=2,cfl=5,multigrid_levels=3,mesh_sequence=True,
        coarse_grid_iterations=2,kappa=.33333,writevolumesolution=False,
        writeTecplotSurfaceSolution=False,writeflowsolution=False),
        boundary_overrides={('flatplate','plate'):(2004,(-1.,0.))})
    result=solver.solve(output_directory=root/'python')
    solver.writeFlowSolution()
    report={}
    for suffix in ('.x','.q'):
        maxdiff=0.
        with (root/('native'+suffix)).open('rb') as native,(root/'python'/('flow'+suffix)).open('rb') as cached:
            for index in range(3 if suffix=='.x' else 4):
                left,right=record(native),record(cached)
                assert len(left)==len(right),(suffix,index)
                if suffix=='.x' and index==2:
                    blank_bytes=4*int(np.prod(result.fields['flatplate'].shape[:3]))
                    assert left[-blank_bytes:]==right[-blank_bytes:]
                    left,right=left[:-blank_bytes],right[:-blank_bytes]
                dtype='<i4' if index<2 else '<f8'
                a,b=np.frombuffer(left,dtype=dtype),np.frombuffer(right,dtype=dtype)
                np.testing.assert_allclose(a,b,rtol=2e-12,atol=1e-13)
                maxdiff=max(maxdiff,float(np.max(np.abs(a-b))))
            assert native.read()==cached.read()==b''
        report[suffix]=maxdiff
    print('Native CFL3D IPTYPE=1 maximum differences:',report)


if __name__=='__main__':
    main()
