# CFL3D / MELD / TACS static aeroelastic example

`run.py` runs two forward aeroelastic scenarios with native CFL3D deformation
or IDWarp. `structure.py` defines the TACS model; `compare.py` compares completed
native/IDWarp runs. `l2-tangential-controls.json` supplies control indices for the
external L2 mesh.

Dependencies, input paths, commands, options, multigrid startup and output
interpretation are documented together in the
[Python wrapper reference](../../../docs/python-wrapper.md#running-the-aeroelastic-example).
For the builder API and forward-only derivative contract, see
[MPhys integration](../../../docs/python-wrapper.md#mphys-static-aeroelastic-analysis-experimental-forward-only).
