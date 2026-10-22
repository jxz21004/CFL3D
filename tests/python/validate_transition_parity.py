"""Compare transition/two-equation variants and their native transported states.

Build the reference with --standalone --debug to also exercise the RANS/DES
conditional without relying on optimizer short-circuit behavior.
"""
import argparse
from pathlib import Path
import subprocess

import numpy as np
from cfl3d import CFL3D, SolverOptions
from cfl3d.restart import FlowState
from cgns_fixtures import write_adf
from fixtures import ROOT, flatplate_mesh
from test_transition import transition_fields
from validate_parity import baseline_deck, put_record, record


def run_case(root, executable, model='gamma-theta'):
    baseline = root/'standalone'
    baseline.mkdir(parents=True, exist_ok=True)
    mesh = flatplate_mesh()
    zone = mesh.zones[0]
    with (baseline/'grid.bin').open('wb') as stream:
        put_record(stream, np.array([1], np.int32).tobytes())
        put_record(stream, np.array(zone.dimensions, np.int32).tobytes())
        put_record(stream, zone.coordinates.astype('<f8').tobytes(order='F'))
    code = {'gamma-theta': 40, 'bsl': 7, 'sst2003': 7, 'wilcox2006': 6}[model]
    keywords = {'gamma-theta': [], 'bsl': ['i_bsl 1'],
                'sst2003': ['isst2003 1'], 'wilcox2006': ['i_wilcox06 1']}[model]
    lines = baseline_deck('sst', 3).splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('NCG'):
            lines[i+1] = f'2 0 0 1 0 {code} {code}'
        elif line.strip().startswith('IDIAG'):
            lines[i+1] = '1 1 1 4 4 4'
    for i, line in enumerate(lines):
        if line.strip() == '<':
            lines[i:i] = ['turbintensity_inf_percent .2', 'eddy_visc_inf 1.', 'keepambient 1', *keywords]
            break
    (baseline/'reference.inp').write_text('\n'.join(lines)+'\n')
    with (baseline/'reference.inp').open('rb') as inp, (baseline/'stdout.log').open('wb') as out:
        subprocess.run([str(executable.resolve())], cwd=baseline, stdin=inp,
                       stdout=out, stderr=subprocess.STDOUT, check=True)
    assert 'execution terminated normally' in (baseline/'cfl3d.error').read_text()
    path = write_adf(mesh, root/'mesh.cgns')
    options = SolverOptions(turbulence=model, mach=.2, reynolds=6e6, temperature=460/1.8,
        iterations=3, cfl=5, multigrid_levels=3, mesh_sequence=True, coarse_grid_iterations=3,
        kappa=.33333, limiter=4, freestream_turbulence_intensity=.2,
        freestream_eddy_viscosity=1., keep_ambient_turbulence=True,
        writevolumesolution=False, writeTecplotSurfaceSolution=False)
    solver = CFL3D(path, options, comm=None,
                   boundary_overrides={('flatplate', 'plate'): (2004, (-1., 0.))})
    result = solver.solve(output_directory=root/'python')
    with (baseline/'restart.bin').open('rb') as stream:
        record(stream)
        raw_history, raw_q = record(stream), record(stream)
    expected = np.frombuffer(raw_q, dtype='<f8').reshape((64, 96, 1, 5), order='F').transpose(2, 0, 1, 3)
    history = np.frombuffer(raw_history, dtype='<f8').reshape(16, 9).T[:, [0, 1, 2, 5, 6, 8, 9, 10]]
    np.testing.assert_allclose(result.fields['flatplate'], expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.history, history, rtol=1e-12, atol=1e-14)
    if model == 'gamma-theta':
        read_turbulence = transition_fields
    else:
        from test_model_variants import two_equation_fields
        read_turbulence = two_equation_fields
    reference = read_turbulence(FlowState((baseline/'restart.bin').read_bytes(), {}, {}), mesh)
    actual = read_turbulence(solver.getFlowState(), mesh)
    np.testing.assert_allclose(actual['flatplate'], reference['flatplate'], rtol=1e-12, atol=1e-14)
    print('MODEL_PARITY_PASS', model,
          'max_mean_flow_difference=', np.max(np.abs(result.fields['flatplate']-expected)),
          'max_turbulence_difference=', np.max(np.abs(actual['flatplate']-reference['flatplate'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=ROOT/'build/python-standalone/cfl3d_seq')
    parser.add_argument('--models', nargs='+', default=['gamma-theta'],
                        choices=('gamma-theta', 'bsl', 'sst2003', 'wilcox2006'))
    args = parser.parse_args()
    for model in args.models:
        run_case(args.directory.resolve()/model, args.executable, model)


if __name__ == '__main__':
    main()
