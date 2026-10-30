# CFL3D / MELD / TACS static aeroelastic example

`run.py` is the single-condition IDWarp example. Edit `aero_options` in
`main()` for your case; select 1/2/3 grid levels with `--levels`. Each coarse
startup stage runs 500 cycles. Final component and total loads go to `out.json`.

`run_comparison.py` retains the two-scenario native CFL3D/IDWarp comparison. `structure.py` defines the TACS model; `compare.py` compares completed
native/IDWarp runs. `l2-tangential-controls.json` supplies control indices for the
external L2 mesh.

Dependencies, input paths, commands, options, multigrid startup and output
interpretation are documented together in the
[Python wrapper reference](../../../docs/python-wrapper.md#running-the-aeroelastic-example).
For the builder API and forward-only derivative contract, see
[MPhys integration](../../../docs/python-wrapper.md#mphys-static-aeroelastic-analysis-experimental-forward-only).
