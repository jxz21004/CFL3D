"""Compare native-array initialization against standalone CFL3D on a NASA mesh.

The standalone control decks are only reference fixtures in this test. The
Python native solve receives a CGNS file and never reads/writes an input deck.
"""
from pathlib import Path
import argparse
import json
import struct
import subprocess
import tarfile
import numpy as np
from cfl3d import CFL3D, SolverOptions
from fixtures import ROOT, flatplate_mesh, write_cgns


def record(stream):
    size = struct.unpack('<i', stream.read(4))[0]
    data = stream.read(size)
    if struct.unpack('<i', stream.read(4))[0] != size:
        raise ValueError('Invalid Fortran record')
    return data


def put_record(stream, data):
    tag = struct.pack('<i', len(data)); stream.write(tag+data+tag)


def baseline_deck(model, cycles):
    name = 'sa' if model == 'sa' else 'sst'
    with tarfile.open(ROOT / 'docs/Cfl3dv6/2DTestcases/Flatplateyplus/Flatplateyplus.tar.gz') as tar:
        lines = tar.extractfile(f'Flatplateyplus/grid_y+4.6{name}.inp').read().decode().splitlines()
    lines[1] = 'grid.bin'
    # All numerical headings are kept as in the distributed NASA deck.
    for i, line in enumerate(lines):
        if line.strip() == '<':
            lines[i:i] = [f'cflturb{j} 5.0' for j in range(1, 8)]
            break
    for i, line in enumerate(lines):
        if line.strip().startswith('NGRID'):
            lines[i+1] = '-1 0 0 9999 0 1 1 1'
        elif line.strip().startswith('ILAMLO'):
            lines[i+1] = '0 0 0 0 0 0'
        elif line.strip().startswith('NCYC'):
            lines[i+1:i+4] = [f'{cycles} 1 0 0', f'{cycles} 2 0 0', f'{cycles} 3 0 0']
        elif line.strip().startswith('GRID IPTYPE'):
            # NPLOT3D and NPRINT are zero; remove their data rows, not headings.
            j=i+1
            while j<len(lines) and lines[j].strip() and lines[j].strip()[0].isdigit():
                del lines[j]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, default=ROOT/'build/python-standalone/cfl3d_seq')
    parser.add_argument('--directory', type=Path, required=True)
    args=parser.parse_args(); root=args.directory.resolve(); root.mkdir(parents=True,exist_ok=True)
    mesh=flatplate_mesh(); path=write_cgns(mesh,root/'flatplate.cgns'); z=mesh.zones[0]
    report={}
    for model in ('sa','sst'):
        baseline=root/f'{model}-standalone'; baseline.mkdir(exist_ok=True)
        with (baseline/'grid.bin').open('wb') as f:
            put_record(f,np.array([1],np.int32).tobytes())
            put_record(f,np.array(z.dimensions,np.int32).tobytes())
            put_record(f,z.coordinates.astype('<f8').tobytes(order='F'))
        (baseline/'reference.inp').write_text(baseline_deck(model,5))
        with (baseline/'reference.inp').open('rb') as inp, (baseline/'stdout.log').open('wb') as out:
            subprocess.run([str(args.executable.resolve())],cwd=baseline,stdin=inp,stdout=out,stderr=subprocess.STDOUT,check=True)
        errors=(baseline/'cfl3d.error').read_text()
        if 'execution terminated normally' not in errors: raise RuntimeError(errors)
        result=CFL3D(path,SolverOptions(mach=.2,reynolds=6e6,temperature=460/1.8,turbulence=model,
                                      iterations=5,cfl=5,multigrid_levels=3,mesh_sequence=True,
                                      coarse_iterations=5,kappa=.33333,limiter=0),  # Match the original unlimited NASA deck.
                       boundary_overrides={("flatplate","plate"):(2004,(-1.,0.))}).solve(output_directory=root/f'{model}-native')
        with (baseline/'restart.bin').open('rb') as f:
            record(f)  # title and run metadata
            raw_history=record(f)
            raw_q=record(f)
        reference=np.frombuffer(raw_q,dtype='<f8').reshape((64,96,1,5),order='F').transpose(2,0,1,3)
        np.testing.assert_allclose(result.fields['flatplate'],reference,rtol=1e-12,atol=1e-14)
        hist=np.frombuffer(raw_history,dtype='<f8').reshape(16,15).T
        selection=hist[:,[0,1,2,5,6,8,9,10]]
        np.testing.assert_allclose(result.history,selection,rtol=1e-12,atol=1e-14)
        report[model]={'max_field_difference':float(np.max(np.abs(result.fields['flatplate']-reference))),
                       'max_history_difference':float(np.max(np.abs(result.history-selection))),
                       'iterations':len(result.history),'coefficients':result.coefficients}
    (root/'parity-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
