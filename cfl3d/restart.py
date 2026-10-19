"""Python-owned native restart snapshots, partitioned by zone in MPI.

Every owned zone retains its complete turbulence/ghost state.

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


def validate_native(payload, mesh, options, *, zone_indices=None):
    """Validate native real64 headers, dimensions and model before entering Fortran."""
    indices = tuple(range(len(mesh.zones))) if zone_indices is None else tuple(zone_indices)
    if (len(set(indices)) != len(indices) or indices != tuple(sorted(indices))
            or any(i < 0 or i >= len(mesh.zones) for i in indices)):
        raise ValueError("Invalid restart zone ownership")
    if not indices:
        if payload:
            raise ValueError("A rank with no zones must have an empty native restart shard")
        return payload
    zones = [mesh.zones[i] for i in indices]
    blocks = records(payload)
    # Header dimensions can differ by zone; find all headers through their model record.
    candidates = []
    for i, b in enumerate(blocks):
        if len(b) != 208 or struct.unpack_from("=3i", b, 168) not in {
            (z.dimensions[1], z.dimensions[2], z.dimensions[0]) for z in zones
        }:
            continue
        model_index = i + 4 if not candidates and indices[0] == 0 else i + 3
        if model_index < len(blocks) and len(blocks[model_index]) == 12:
            candidates.append((i, model_index))
    if len(candidates) != len(zones) or candidates[0][0] != 0:
        raise ValueError("Restart must contain one finest-grid native block per CGNS zone")
    model = MODELS[options.turbulence.lower()]
    i2d = all(z.dimensions[0] == 2 for z in mesh.zones)
    for global_index, zone, (i, model_index) in zip(indices, zones, candidates, strict=True):
        dims = struct.unpack_from("=3i", blocks[i], 168)
        expected = (zone.dimensions[1], zone.dimensions[2], zone.dimensions[0])
        if dims != expected:
            raise ValueError(f"Restart grid dimensions differ for {zone.name}: {dims} != {expected}")
        history_count = struct.unpack_from("=i", blocks[i], 196)[0]
        if history_count < 0 or (global_index == 0 and len(blocks[i + 1]) != history_count * 16 * 8):
            raise ValueError("Restart convergence history does not match its header")
        q_index = i + 2 if global_index == 0 else i + 1
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
    # None denotes a portable complete state. MPI snapshots contain only these
    # global zone indices, including all native ghost and turbulence records.
    zone_indices: tuple[int, ...] | None = None
    global_digest: str | None = None

    def validate(self, mesh, options):
        current = metadata(mesh, options)
        if self.metadata != current:
            different = [key for key in current if current[key] != self.metadata.get(key)]
            raise ValueError(f"Cached restart incompatible on {different}; call resetFlow() for a fresh start")
        validate_native(self.native, mesh, options, zone_indices=self.zone_indices)

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
    if state.zone_indices is not None:
        raise ValueError("Use CFL3D.writeRestartFile() collectively to save distributed state")
    with _temporary_output(path) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=json.dumps(state.metadata, sort_keys=True),
            native=np.frombuffer(state.native, dtype=np.uint8),
            sha256=state.digest,
            **{"field_" + name: value for name, value in state.fields.items()},
        )


def zone_payloads(state, mesh):
    """Return zero-copy views of complete native per-zone record sequences."""
    indices = tuple(range(len(mesh.zones))) if state.zone_indices is None else state.zone_indices
    if not indices:
        return {}
    blocks = records(state.native)
    starts = []
    offset = 0
    for i, block in enumerate(blocks):
        if len(starts) < len(indices) and len(block) == 208:
            zone = mesh.zones[indices[len(starts)]]
            expected = (zone.dimensions[1], zone.dimensions[2], zone.dimensions[0])
            model_index = i + (4 if indices[len(starts)] == 0 else 3)
            if (struct.unpack_from("=3i", block, 168) == expected
                    and model_index < len(blocks) and len(blocks[model_index]) == 12):
                starts.append(offset)
        offset += len(block) + 8
    if len(starts) != len(indices):
        raise ValueError("Cannot locate all native restart zone records")
    ends = starts[1:] + [len(state.native)]
    view = memoryview(state.native)
    return {index: view[start:end] for index, start, end in zip(indices, starts, ends, strict=True)}


def partition_state(state, mesh, owners, size):
    """Split a complete file snapshot for scattering; never broadcast its payload."""
    if state.zone_indices is not None:
        raise ValueError("partition_state requires a complete checkpoint")
    if len(owners) != len(mesh.zones) or size < 1 or any(owner < 0 or owner >= size for owner in owners):
        raise ValueError("Invalid restart partition map")
    parts = zone_payloads(state, mesh)
    identity = state.digest
    return [FlowState(
        b"".join(parts[i] for i, owner in enumerate(owners) if owner == rank),
        state.metadata,
        {z.name: state.fields[z.name] for i, z in enumerate(mesh.zones)
         if owners[i] == rank and z.name in state.fields},
        tuple(i for i, owner in enumerate(owners) if owner == rank), identity,
    ) for rank in range(size)]


def merge_states(states, mesh):
    """Assemble only on the output rank, for a portable checkpoint writer."""
    if not states:
        raise ValueError("No distributed restart state")
    if any(state.metadata != states[0].metadata for state in states):
        raise ValueError("Distributed restart metadata differs")
    identities = {state.global_digest or state.digest for state in states}
    if len(identities) != 1:
        raise ValueError("Distributed restart states differ between completed solves")
    parts, fields = {}, {}
    for state in states:
        for index, payload in zone_payloads(state, mesh).items():
            if index in parts:
                raise ValueError("Duplicate restart zone ownership")
            parts[index] = payload
        fields.update(state.fields)
    if set(parts) != set(range(len(mesh.zones))):
        raise ValueError("Incomplete distributed restart state")
    return FlowState(b"".join(parts[i] for i in range(len(mesh.zones))), states[0].metadata, fields)
