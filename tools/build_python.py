#!/usr/bin/env python3
"""Build serial/MPI real-valued f2py extensions independently of legacy makefiles."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]


def sources(makefile, *variables):
    text = makefile.read_text().replace("\\\n", " ")
    result = []
    for variable in variables:
        match = re.search(r"^" + variable + r"\s*=([^\n]*)", text, re.M)
        if match:
            result.extend(re.findall(r"\b[\w]+\.F(?:90)?\b", match[1]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fc", default=os.environ.get("FC"))
    parser.add_argument("--mpi", action="store_true", help="Use the native host/worker MPI path")
    parser.add_argument("--compiler-family", choices=("gnu", "intel"), default="gnu")
    parser.add_argument("--backend", choices=("direct", "meson"), default="direct")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--standalone", action="store_true", help="Build a serial executable for parity testing")
    parser.add_argument("--extra-fflags", default="", help="Additional Fortran compile flags, for controlled compiler comparisons")
    parser.add_argument("--extra-ldflags", default="", help="Additional final linker flags, for controlled comparisons")
    parser.add_argument("--build-dir", type=Path, help="Isolated build directory for compiler/performance comparisons")
    parser.add_argument("--no-install", action="store_true", help="Leave the extension in the build directory without replacing the installed module")
    args = parser.parse_args()
    args.fc = args.fc or (("mpiifort" if args.mpi else "ifort") if args.compiler_family == "intel"
                          else ("mpif90" if args.mpi else "gfortran"))
    if not shutil.which(args.fc):
        parser.error(f"Fortran compiler not found: {args.fc}")
    suffix = "-mpi" if args.mpi else ""
    module = "_native_mpi" if args.mpi else "_native"
    build = ROOT / (("build/python-standalone" if args.standalone else "build/python-native") + suffix)
    if args.build_dir is not None:
        build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    flags = ["-DPYTHON_API", "-DDBLE_PRECSN", "-fPIC", "-I", str(build)]
    if args.compiler_family == "gnu":
        flags += ["-cpp", "-fdefault-real-8", "-fdefault-double-8", "-ffixed-line-length-none",
                  "-fallow-argument-mismatch", "-std=legacy", "-J", str(build)]
        flags += ["-O0", "-g", "-fbacktrace"] if args.debug else ["-O2"]
    else:
        flags += ["-fpp", "-r8", "-double-size", "64", "-extend-source", "-module", str(build)]
        flags += ["-O0", "-g", "-traceback"] if args.debug else ["-O2"]
    if args.mpi:
        flags.append("-DDIST_MPI")
    if args.standalone:
        flags.remove("-DPYTHON_API")
    flags.extend(shlex.split(args.extra_fflags))
    linker_flags = shlex.split(args.extra_ldflags)
    manifest = {
        "compiler": shutil.which(args.fc),
        "compiler_version": subprocess.run([args.fc, "--version"], capture_output=True, text=True, check=True).stdout,
        "fortran_flags": flags,
        "extra_linker_flags": linker_flags,
        "mpi": args.mpi, "standalone": args.standalone,
        "backend": args.backend, "python": sys.version,
        "status": "building",
    }
    manifest_path = build / "build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def completed(target):
        manifest.update(status="complete", artifact=str(target))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def link(command):
        manifest["link_command"] = command
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.run(command, check=True)

    def compile_file(path):
        output = build / (path.stem + ".o")
        extra = (["-fcheck=bounds"] if args.compiler_family == "gnu" else ["-check", "bounds"]) \
            if args.debug and path.name == "python_state.F90" else []
        subprocess.run([args.fc, *flags, *extra, "-c", str(path), "-o", str(output)], check=True)
        return output

    libs = ROOT / "source/cfl3d/libs"
    dist = ROOT / "source/cfl3d/dist"
    objects = [] if args.standalone else [compile_file(ROOT / "source/python/python_state.F90")]
    for name in sources(ROOT / "build/cfl/libs/makefile", "F90SRC_LIBS"):
        objects.append(compile_file(libs / name))
    pending = [libs / name for name in sources(ROOT / "build/cfl/libs/makefile", "FSRC_LIBS", "FSRC_SPEC")]
    pending += [dist / name for name in sources(ROOT / f"build/cfl/{'mpi' if args.mpi else 'seq'}/makefile", "FSRC_DIST")]
    pending += [dist / "main.F", dist / "development.F"]
    pending = list(dict.fromkeys(pending))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        objects.extend(pool.map(compile_file, pending))
    if args.standalone:
        target = build / ("cfl3d_mpi" if args.mpi else "cfl3d_seq")
        link([args.fc, *flags, "-o", str(target), *map(str, objects), *linker_flags])
        completed(target)
        print(f"Built {target}")
        return
    guard = build / "python_guard.o"
    subprocess.run(["cc", "-fPIC", "-c", str(ROOT / "source/python/python_guard.c"), "-o", str(guard)], check=True)
    objects.append(guard)
    archive = build / "libcfl3d_python.a"
    subprocess.run(["ar", "rcs", str(archive), *map(str, objects)], check=True)
    signature = build / "cfl3d_native.pyf"
    signature.write_text((ROOT / "source/python/cfl3d_native.pyf").read_text().replace("_native", module))
    if args.backend == "meson":
        subprocess.run([sys.executable, "-m", "numpy.f2py", "-c", "--backend", "meson",
                        str(signature), "-L" + str(build), "-lcfl3d_python"], cwd=build, check=True,
                       env={**os.environ, "FC": args.fc,
                            "LDFLAGS": shlex.join([*shlex.split(os.environ.get("LDFLAGS", "")), *linker_flags])})
        extensions = list(build.glob(module + "*.so"))
        if len(extensions) != 1:
            raise RuntimeError("Expected one generated f2py extension")
        extension = extensions[0]
    else:
        # f2py generates the C shim; use the same Fortran linker as the core so
        # MPI and compiler runtime libraries cannot silently be omitted.
        import numpy
        import numpy.f2py
        subprocess.run([sys.executable, "-m", "numpy.f2py", str(signature)], cwd=build, check=True)
        support = Path(numpy.f2py.__file__).parent / "src"
        includes = ["-I" + sysconfig.get_path("include"), "-I" + numpy.get_include(), "-I" + str(support)]
        shim_objects = []
        for source in (build / (module + "module.c"), support / "fortranobject.c"):
            obj = build / (source.stem + ".o")
            subprocess.run([os.environ.get("CC", "cc"), "-fPIC", "-O2", *includes,
                            "-c", str(source), "-o", str(obj)], check=True)
            shim_objects.append(obj)
        for source in sorted(build.glob(module + "-f2pywrappers*.f*")):
            if source.stat().st_size:
                shim_objects.append(compile_file(source))
        extension = build / (module + sysconfig.get_config_var("EXT_SUFFIX"))
        link([args.fc, "-shared", "-o", str(extension), *map(str, shim_objects), str(archive), *linker_flags])
    if args.no_install:
        completed(extension)
        print(f"Built {extension} (not installed)")
        return
    target = ROOT / "cfl3d" / extension.name
    shutil.copy2(extension, target)
    completed(target)
    shutil.copy2(manifest_path, target.with_suffix(target.suffix + ".build.json"))
    print(f"Built {target}")


if __name__ == "__main__":
    main()
