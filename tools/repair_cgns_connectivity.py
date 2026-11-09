#!/usr/bin/env python3
"""Repair uniquely coordinate-matched interface directions into a CGNS copy."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfl3d.mesh import repair_cgns_connectivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Original CGNS file (never modified)')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Repaired CGNS copy')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output file')
    args = parser.parse_args()
    try:
        repairs = repair_cgns_connectivity(args.input, args.output, overwrite=args.overwrite)
    except (ValueError, OSError, ImportError, AssertionError) as exc:
        parser.exit(1, f'Repair failed: {exc}\n')
    print(json.dumps(dict(output=str(args.output), repaired_connections=len(repairs), repairs=repairs), indent=2))


if __name__ == '__main__':
    main()
