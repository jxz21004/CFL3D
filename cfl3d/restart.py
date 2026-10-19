"""Python-owned complete native restart snapshots (including turbulence/ghost state).

The legacy reader/writer supplies the solver state. Python owns its immutable
bytes between solves; no old output directory or live Fortran COMMON is needed.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

from .solution_io import _temporary_output

FORMAT_VERSION = 1
MODELS = {"euler": 0, "laminar": 1, "sa": 5, "komega": 6, "sst": 7, "gamma-theta": 40,
          "bsl": 7, "sst2003": 7, "wilcox2006": 6}


def metadata(mesh, options):
    return {
        "format": FORMAT_VERSION,
        "byteorder": sys.byteorder,
        "turbulence": options.turbulence.lower(),
        "gamma": options.gamma,
        "y_up": options.y_up,
        "zones": [
            {
                "name": z.name,
                "dimensions": list(z.dimensions),
                "boundaries": [[b.name, b.bc_type, b.point_range.tolist()] for b in z.boundaries],
            }
            for z in mesh.zones
        ],
    }


def records(payload):
    offset = 0
    blocks = []
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError("Truncated native restart record")
        length = struct.unpack_from("=i", payload, offset)[0]
        end = offset + 4 + length
        if length < 0 or end + 4 > len(payload) or struct.unpack_from("=i", payload, end)[0] != length:
            raise ValueError("Invalid restart record markers; require native-endian int32 records")
        blocks.append(memoryview(payload)[offset + 4 : end])
        offset = end + 4
    if not blocks:
        raise ValueError("Empty restart file")
    return blocks


def validate_native(payload, mesh, options):
    """Validate native real64 headers, dimensions and model before entering Fortran."""
    blocks = records(payload)
    # Header dimensions can differ by zone; find all headers through their model record.
    candidates = []
    for i, b in enumerate(blocks):
        if len(b) != 208 or struct.unpack_from("=3i", b, 168) not in {
            (z.dimensions[1], z.dimensions[2], z.dimensions[0]) for z in mesh.zones
        }:
            continue
        model_index = i + 4 if not candidates else i + 3
        if model_index < len(blocks) and len(blocks[model_index]) == 12:
            candidates.append((i, model_index))
    if len(candidates) != len(mesh.zones) or candidates[0][0] != 0:
        raise ValueError("Restart must contain one finest-grid native block per CGNS zone")
    model = MODELS[options.turbulence.lower()]
    i2d = all(z.dimensions[0] == 2 for z in mesh.zones)
    for zone, (i, model_index) in zip(mesh.zones, candidates, strict=True):
        dims = struct.unpack_from("=3i", blocks[i], 168)
        expected = (zone.dimensions[1], zone.dimensions[2], zone.dimensions[0])
        if dims != expected:
            raise ValueError(f"Restart grid dimensions differ for {zone.name}: {dims} != {expected}")
        history_count = struct.unpack_from("=i", blocks[i], 196)[0]
        if history_count < 0 or (i == 0 and len(blocks[i + 1]) != history_count * 16 * 8):
            raise ValueError("Restart convergence history does not match its header")
        q_index = i + 2 if i == 0 else i + 1
        count = int(np.prod(np.asarray(zone.dimensions) - 1))
        if len(blocks[q_index]) != count * 5 * 8:
            raise ValueError(f"Restart flow record size differs for {zone.name}")
        q = np.frombuffer(blocks[q_index], dtype=np.float64).reshape(count, 5, order="F")
        if not np.isfinite(q).all() or (q[:, 0] <= 0).any() or (q[:, 4] <= 0).any():
            raise ValueError(f"Restart contains invalid density/pressure/flow values for {zone.name}")
        ni, nj, nk = np.asarray(zone.dimensions) - 1
        ghost_size = int((nj * nk + ni * nk + ni * nj) * 5 * 4 * 8)
        if len(blocks[q_index + 1]) != ghost_size:
            raise ValueError(f"Restart requires the default ghost-cell records for {zone.name}")
        codes = struct.unpack_from("=3i", blocks[model_index])
        if codes != ((0 if i2d else model), model, model):
            raise ValueError(f"Restart turbulence model differs for {zone.name}")
    return payload


@dataclass
class FlowState:
    # native is authoritative; fields is an independent readout for Python algorithms.
    native: bytes
    metadata: dict
    fields: dict

    def validate(self, mesh, options):
        current = metadata(mesh, options)
        if self.metadata != current:
            different = [key for key in current if current[key] != self.metadata.get(key)]
            raise ValueError(f"Cached restart incompatible on {different}; call resetFlow() for a fresh start")
        validate_native(self.native, mesh, options)

    @property
    def digest(self):
        return hashlib.sha256(self.native).hexdigest()


def read_restart(path, mesh, options):
    path = Path(path)
    with path.open("rb") as stream:
        signature = stream.read(4)
    if signature[:2] != b"PK":
        payload = validate_native(path.read_bytes(), mesh, options)
        return FlowState(payload, metadata(mesh, options), {})
    with np.load(path, allow_pickle=False) as saved:
        if not {"metadata", "native", "sha256"} <= set(saved.files):
            raise ValueError("restartFile is not a CFL3D checkpoint; solution.npz is visualization output")
        state = FlowState(
            saved["native"].astype(np.uint8, copy=False).tobytes(),
            json.loads(str(saved["metadata"])),
            {name[6:]: saved[name].copy() for name in saved.files if name.startswith("field_")},
        )
        if state.digest != str(saved["sha256"]):
            raise ValueError("Restart checkpoint checksum mismatch")
        state.validate(mesh, options)
        return state


def write_restart(path, state):
    with _temporary_output(path) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=json.dumps(state.metadata, sort_keys=True),
            native=np.frombuffer(state.native, dtype=np.uint8),
            sha256=state.digest,
            **{"field_" + name: value for name, value in state.fields.items()},
        )
