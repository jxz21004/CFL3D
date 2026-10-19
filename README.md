# CFL3D Python / MPI

A Python interface to the NASA CFL3D V6 structured-grid CFD solver, using NumPy
f2py to call the Fortran core in the same process. The package supports serial
execution and the native MPI host/worker path.

Python package version: **0.1.0**. Native interface version: **10**.

## Why modernize the CFL3D interface?

The aim is to make CFL3D fit the way modern scientific computing software is used:
configure a case in Python, run it, inspect the results, change parameters, and
continue the calculation from an application or research script.

This project retains the Fortran solver and provides a programmable interface for:

- **Case configuration:** Python dictionaries and typed settings describe the
  physical model, numerical controls, references, and output choices.
- **Repeated calculations:** Python owns the cached flow state. Parameter changes
  can continue a calculation or trigger initialization according to their category.
- **Parallel workflows:** an explicit MPI communicator defines which processes
  belong to a case, including independent cases within one MPI job.
- **Data access:** flow arrays, convergence histories, component loads, Cp sections,
  and lift distributions are available to Python for subsequent analysis.
- **Controlled output:** computation and file writing can be separated, with
  numbered snapshots or appended histories for continued calculations.

These capabilities provide a basis for parameter studies and external algorithms.
The Python solve path reads structured CGNS grids directly and does not require an
`.inp` template or an intermediate PLOT3D grid conversion. The exposed capabilities
and limitations are documented in the interface reference.

## Requirements

Use an existing Python and compiler environment with these dependencies. The
installation below does not require reproducing the development container.

| Component | Requirement |
| --- | --- |
| Platform | Linux; Windows users can use a Linux container or WSL |
| Python | >=3.10, with matching development headers |
| NumPy | >=2.0, including f2py and its C headers |
| Packaging tools | pip, setuptools >=64 for editable installation, and wheel |
| C / Fortran tools | C11 compiler, GNU Fortran, and `ar` |
| CGNS | Shared `libcgns.so` and `cgnslib.h`; the bridge supports 32/64-bit `cgsize_t` |
| MPI, when needed | MPI runtime and a Fortran wrapper compatible with the selected compiler |
| mpi4py, when needed | >=4.0, linked to the MPI implementation used at runtime |
| CGNS/HDF5, when needed | HDF5 support in libcgns, or optional h5py >=3.8; ADF does not need h5py |

The current development checks use Python 3.12.12, NumPy 2.3.5, GNU Fortran 13,
CGNS 4.5.0, Intel MPI 2021.13, and mpi4py 4.1.1. These are tested versions, not a
requirement to copy that exact environment. The build script also supports Intel
Fortran Classic (`ifort` / `mpiifort`); the latest regression runs use GNU Fortran
with Intel MPI.

The default build uses f2py directly and does not need Meson or Ninja. Selecting
`--backend meson` additionally requires Meson >=1.3 and Ninja.

Load your compiler, CGNS, and MPI environments before building or running. Their
shared libraries must be available to the dynamic linker. The MPI launcher,
mpi4py, and the Fortran MPI extension must use compatible MPI libraries. Containers
using MPI shared-memory transport also need sufficient `/dev/shm` space.

## Minimal installation

From the repository root, with the dependencies above already installed:

```bash
cd /path/to/CFL3D

# Register the package using existing dependencies.
python -m pip install --no-deps --no-build-isolation -e .

# Build the serial Fortran extension.
python tools/build_python.py --fc gfortran --jobs 8

# Build the CGNS bridge against your existing installation.
python tools/build_cgns.py --prefix /path/to/cgns
```

The CGNS prefix must contain `include/cgnslib.h` and either `lib/libcgns.so` or
`lib64/libcgns.so`. If `CGNS_HOME` already points to that prefix, the last command
can be shortened to `python tools/build_cgns.py`.

For parallel calculations, also build the MPI extension:

```bash
# mpif90 must invoke GNU Fortran and link the intended MPI implementation.
python tools/build_python.py --mpi --fc mpif90 --jobs 8
```

For Intel Fortran Classic, replace the solver build commands with:

```bash
python tools/build_python.py --compiler-family intel --fc ifort --jobs 8
python tools/build_python.py --mpi --compiler-family intel --fc mpiifort --jobs 8
```

The extensions are installed in `cfl3d/`: `_native`, `_native_mpi`, and
`_cgns_bridge.so`. Registering the Python package does not compile these libraries.
Add `--debug` to a solver build for debugging, or adjust `--jobs` for your machine.
Rebuild the affected extensions after changing the native interface, Python ABI,
NumPy major version, compiler, or MPI implementation, then restart Python.

Verify the serial extension and CGNS bridge:

```bash
python -c "from cfl3d import _native; print('native API:', _native.py_api_version())"
python -c "from cfl3d._cgns_native import library; print('CGNS has HDF5:', library().has_hdf5)"
```

The native API should be `10`. A `False` HDF5 result still permits CGNS/ADF input;
CGNS/HDF5 input requires a capable libcgns or h5py.

## Documentation

- [Python interface reference](docs/python-wrapper.md): complete case example,
  options, dynamic updates, MPI, restart behavior, loads, and output formats.
- [Known limitations and validation gaps](docs/limitations.md): implementation
  constraints, unavailable capabilities, and the scope of current test evidence.
- [Runnable case template](examples/python/standard_case.py).
- [Development checkpoints and validation records](MPI_PROGRESS.md).
- [Original NASA CFL3D documentation](https://nasa.github.io/CFL3D/).

## License

Copyright 2001 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. All Rights Reserved.

CFL3D is distributed under the [Apache License, Version 2.0](LICENSE).
