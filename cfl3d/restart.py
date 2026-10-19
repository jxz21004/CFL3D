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


def index_native_file(stream, mesh, options):
    """Locate native zones by record metadata, without reading the flow payload."""
    stream.seek(0, 2)
    size = stream.tell()
    rows, offset = [], 0
    while offset < size:
        if offset+8 > size:
            raise ValueError('Truncated native restart record')
        stream.seek(offset)
        marker = stream.read(4)
        if len(marker) != 4:
            raise ValueError('Truncated native restart record')
        length = struct.unpack('=i', marker)[0]
        end = offset+4+length
        if length < 0 or end+4 > size:
            raise ValueError('Invalid native restart record markers')
        stream.seek(end)
        if stream.read(4) != marker:
            raise ValueError('Invalid native restart record markers')
        rows.append((offset, length))
        offset = end+4
    if not rows:
        raise ValueError('Empty restart file')
    candidates = []
    dimensions = {(z.dimensions[1], z.dimensions[2], z.dimensions[0]) for z in mesh.zones}
    for i, (start, length) in enumerate(rows):
        if length != 208:
            continue
        stream.seek(start+4)
        header = stream.read(208)
        if struct.unpack_from('=3i', header, 168) not in dimensions:
            continue
        model_index = i+(4 if not candidates else 3)
        if model_index < len(rows) and rows[model_index][1] == 12:
            candidates.append((i, model_index, header))
    if len(candidates) != len(mesh.zones) or candidates[0][0] != 0:
        raise ValueError('Restart must contain one finest-grid native block per CGNS zone')
    model = MODELS[options.turbulence.lower()]
    i2d = all(z.dimensions[0] == 2 for z in mesh.zones)
    starts = []
    for global_index, (zone, (i, model_index, header)) in enumerate(zip(mesh.zones, candidates, strict=True)):
        expected = (zone.dimensions[1], zone.dimensions[2], zone.dimensions[0])
        if struct.unpack_from('=3i', header, 168) != expected:
            raise ValueError(f'Restart grid dimensions differ for {zone.name}')
        history_count = struct.unpack_from('=i', header, 196)[0]
        if history_count < 0 or (global_index == 0 and rows[i+1][1] != history_count*16*8):
            raise ValueError('Restart convergence history does not match its header')
        q_index = i+(2 if global_index == 0 else 1)
        ni,nj,nk = np.asarray(zone.dimensions)-1
        if rows[q_index][1] != int(ni*nj*nk)*5*8:
            raise ValueError(f'Restart flow record size differs for {zone.name}')
        if rows[q_index+1][1] != int(nj*nk+ni*nk+ni*nj)*5*4*8:
            raise ValueError(f'Restart ghost records differ for {zone.name}')
        stream.seek(rows[model_index][0]+4)
        if struct.unpack('=3i', stream.read(12)) != ((0 if i2d else model), model, model):
            raise ValueError(f'Restart turbulence model differs for {zone.name}')
        starts.append(rows[i][0])
    return tuple((start, end-start) for start,end in zip(starts, starts[1:]+[size], strict=True))


def _read_npy_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError(f'Unsupported restart NPY header version {version}')


def open_native_restart(path, mesh, options):
    """Open native bytes with bounded RAM; compressed NPZ is spooled to disk.

    Returns an owned seekable file and the verified checksum for NPZ input.
    Raw files return None for the checksum: it is computed while transferring
    their bytes, avoiding a separate full-file read. Redundant NPZ mean-flow
    readouts are skipped; public serial read_restart() still returns them.
    """
    import tempfile
    import zipfile
    from ._streaming import CHUNK_BYTES
    stream = Path(path).open('rb')
    try:
        signature = stream.read(4)
        stream.seek(0)
        if signature[:2] != b'PK':
            return stream, None
        with zipfile.ZipFile(stream) as archive:
            if not {'native.npy','metadata.npy','sha256.npy'} <= set(archive.namelist()):
                raise ValueError('restartFile is not a CFL3D checkpoint; solution.npz is visualization output')
            with archive.open('metadata.npy') as member:
                saved_metadata = json.loads(str(np.load(member, allow_pickle=False)))
            if saved_metadata != metadata(mesh, options):
                raise ValueError('Cached restart incompatible; call resetFlow() for a fresh start')
            with archive.open('sha256.npy') as member:
                expected_digest = str(np.load(member, allow_pickle=False))
            temporary = tempfile.TemporaryFile(mode='w+b')
            try:
                with archive.open('native.npy') as member:
                    shape, _, dtype = _read_npy_header(member)
                    if len(shape) != 1 or dtype != np.dtype('uint8'):
                        raise ValueError('Restart native array must be one-dimensional uint8')
                    remaining = shape[0]
                    digest = hashlib.sha256()
                    while remaining:
                        chunk = member.read(min(remaining, CHUNK_BYTES))
                        if not chunk:
                            raise ValueError('Truncated restart native array')
                        temporary.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                    if member.read(1):
                        raise ValueError('Unexpected bytes after restart native array')
                if digest.hexdigest() != expected_digest:
                    raise ValueError('Restart checkpoint checksum mismatch')
                temporary.seek(0)
            except BaseException:
                temporary.close()
                raise
        stream.close()
        return temporary, expected_digest
    except BaseException:
        stream.close()
        raise


def read_restart_partitioned(path, mesh, options, owners, comm, checked):
    """Read a complete native/NPZ checkpoint into owned shards, with bounded host RAM."""
    from ._streaming import file_chunks, transfer_bytes
    opened = checked(lambda: open_native_restart(path, mesh, options) if comm.rank == 0 else None)
    stream = opened[0] if comm.rank == 0 else None
    transfer_comm = comm.Dup()
    try:
        ranges = checked(lambda: index_native_file(stream, mesh, options) if comm.rank == 0 else None)
        ranges, expected_digest = comm.bcast((ranges, opened[1]) if comm.rank == 0 else None, root=0)
        digest = hashlib.sha256()
        indices = tuple(i for i,owner in enumerate(owners) if owner == comm.rank)
        local = checked(lambda: bytearray(sum(ranges[i][1] for i in indices)))
        cursor = 0
        for i,(start,length) in enumerate(ranges):
            owner = owners[i]
            position = cursor
            def consume(chunk):
                nonlocal position
                local[position:position+len(chunk)] = chunk
                position += len(chunk)
            def source_chunks():
                for chunk in file_chunks(stream, start, length):
                    if expected_digest is None:
                        digest.update(chunk)
                    yield chunk
            chunks = source_chunks() if comm.rank == 0 else None
            transfer_bytes(transfer_comm, 0, owner, chunks, consume, length, checked)
            if comm.rank == owner:
                cursor = position
        identity = comm.bcast((expected_digest or digest.hexdigest()) if comm.rank == 0 else None, root=0)
        state = checked(lambda: FlowState(bytes(local), metadata(mesh, options), {}, indices, identity))
        checked(lambda: state.validate(mesh, options))
        return state
    finally:
        try:
            checked(lambda: stream.close() if stream is not None else None)
        finally:
            transfer_comm.Free()


def _array_header(shape, dtype, fortran_order=False):
    return dict(descr=np.lib.format.dtype_to_descr(np.dtype(dtype)),
                fortran_order=bool(fortran_order), shape=tuple(shape))


def write_restart_partitioned(path, state, mesh, comm, checked):
    """Write the existing NPZ format without gathering native/field arrays on root."""
    from contextlib import contextmanager
    import zipfile
    from ._streaming import array_chunks, byte_chunks, transfer_bytes

    def describe():
        parts = zone_payloads(state, mesh)
        names = {mesh.zones[index].name for index in parts}
        fields = {}
        for name, array in state.fields.items():
            if name not in names:
                raise ValueError(f'Checkpoint field {name} does not belong to this rank')
            array = np.asarray(array)
            zone = next(z for z in mesh.zones if z.name == name)
            if array.shape != (*tuple(n-1 for n in zone.dimensions), 5) or array.dtype.hasobject:
                raise ValueError(f'Checkpoint field layout differs for {name}')
            fields[name] = _array_header(array.shape, array.dtype,
                                        array.flags.f_contiguous and not array.flags.c_contiguous)
        return parts, fields
    parts, fields = checked(describe)
    metadata_digest = hashlib.sha256(json.dumps(state.metadata, sort_keys=True).encode()).hexdigest()
    descriptions = comm.allgather((metadata_digest, state.global_digest or state.digest,
                                  {i:len(part) for i,part in parts.items()}, fields))
    def check_layout():
        if any(item[:2] != descriptions[0][:2] for item in descriptions):
            raise ValueError('Distributed restart states differ between completed solves')
        owners, sizes, field_owners, field_headers = {}, {}, {}, {}
        for rank,(_,_,native,headers) in enumerate(descriptions):
            for index, size in native.items():
                if index in owners:
                    raise ValueError('Duplicate restart zone ownership')
                owners[index], sizes[index] = rank, size
            for name, header in headers.items():
                if name in field_owners:
                    raise ValueError('Duplicate checkpoint field ownership')
                field_owners[name], field_headers[name] = rank, header
        if set(owners) != set(range(len(mesh.zones))):
            raise ValueError('Incomplete distributed restart state')
        return owners, sizes, field_owners, field_headers
    owners, sizes, field_owners, headers = checked(check_layout)

    @contextmanager
    def output():
        with _temporary_output(path) as temporary:
            with zipfile.ZipFile(temporary, mode='w', compression=zipfile.ZIP_DEFLATED,
                                 allowZip64=True) as archive:
                yield archive
    manager = output() if comm.rank == 0 else None
    archive = checked(lambda: manager.__enter__() if manager is not None else None)
    member = None
    transfer_comm = comm.Dup()
    try:
        def open_array(name, header):
            if comm.rank != 0:
                return None
            result = archive.open(name+'.npy', mode='w', force_zip64=True)
            try:
                np.lib.format.write_array_header_1_0(result, header)
            except BaseException:
                result.close()
                raise
            return result
        member = checked(lambda: open_array('native', _array_header((sum(sizes.values()),), 'uint8')))
        digest = hashlib.sha256()
        def write_native(chunk):
            member.write(chunk)
            digest.update(chunk)
        for index in range(len(mesh.zones)):
            owner = owners[index]
            chunks = byte_chunks(parts[index]) if comm.rank == owner else None
            transfer_bytes(transfer_comm, owner, 0, chunks, write_native, sizes[index], checked)
        checked(lambda: member.close() if member is not None else None)
        member = None
        # Store only small scalar metadata on the root; native/field data above
        # and below never become whole-case Python arrays on the output rank.
        def write_scalar(name, value):
            if comm.rank == 0:
                with archive.open(name+'.npy', mode='w', force_zip64=True) as scalar:
                    np.lib.format.write_array(scalar, np.asarray(value), allow_pickle=False)
        checked(lambda: write_scalar('metadata', json.dumps(state.metadata, sort_keys=True)))
        checked(lambda: write_scalar('sha256', digest.hexdigest()))
        for name in sorted(headers):
            header = headers[name]
            member = checked(lambda: open_array('field_'+name, header))
            chunks = (array_chunks(np.asarray(state.fields[name]), fortran_order=header['fortran_order'])
                      if comm.rank == field_owners[name] else None)
            size = int(np.prod(header['shape']))*np.dtype(header['descr']).itemsize
            transfer_bytes(transfer_comm, field_owners[name], 0, chunks,
                           lambda chunk: member.write(chunk), size, checked)
            checked(lambda: member.close() if member is not None else None)
            member = None
    except BaseException as exc:
        try:
            checked(lambda: member.close() if member is not None else None)
        finally:
            checked(lambda: manager.__exit__(type(exc), exc, exc.__traceback__) if manager is not None else None)
        raise
    else:
        checked(lambda: manager.__exit__(None, None, None) if manager is not None else None)
    finally:
        transfer_comm.Free()
