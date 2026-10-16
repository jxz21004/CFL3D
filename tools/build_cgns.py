#!/usr/bin/env python3
"""Build the small CGNS ABI bridge with an existing CGNS installation."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prefix", default=os.environ.get("CGNS_HOME"),
                        help="CGNS installation prefix (default: CGNS_HOME)")
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()
    if not args.prefix:
        parser.error("Set CGNS_HOME or pass --prefix /path/to/cgns")
    prefix = Path(args.prefix).resolve()
    if not (prefix / "include/cgnslib.h").is_file():
        parser.error(f"Missing {prefix / 'include/cgnslib.h'}")
    libdir = next((prefix / d for d in ("lib", "lib64")
                   if (prefix / d / "libcgns.so").exists()), None)
    if libdir is None:
        parser.error("A shared libcgns.so installation is required")
    target = ROOT / "cfl3d/_cgns_bridge.so"
    # Atomic installation leaves libraries loaded by other processes intact.
    with tempfile.TemporaryDirectory(dir=target.parent) as directory:
        output = Path(directory) / target.name
        subprocess.run([*shlex.split(args.cc), "-std=c11", "-shared", "-fPIC", "-O2",
                        "-Wall", "-Wextra", "-Werror", "-I" + str(prefix / "include"),
                        str(ROOT / "source/python/cgns_bridge.c"), "-L" + str(libdir),
                        "-Wl,-rpath," + str(libdir), "-lcgns", "-o", str(output)], check=True)
        output.replace(target)
    print(f"Built {target} against {prefix}")


if __name__ == "__main__":
    main()
