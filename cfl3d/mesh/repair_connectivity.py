"""Explicit, coordinate-verified repair of structured interface orientations."""
from itertools import permutations, product
import numpy as np


def _face_axis(bounds, dimensions):
    bounds = np.asarray(bounds)
    if bounds.shape != (3, 2) or not np.issubdtype(bounds.dtype, np.integer):
        raise ValueError('Interface ranges must be integer (3, 2) arrays')
    if np.any(bounds < 1) or np.any(bounds > np.asarray(dimensions)[:, None]):
        raise ValueError('Interface range is outside its zone')
    fixed = np.flatnonzero(bounds[:, 0] == bounds[:, 1])
    if len(fixed) != 1 or bounds[fixed[0], 0] not in (1, dimensions[fixed[0]]):
        raise ValueError('Only nondegenerate exterior rectangular faces are supported')
    return int(fixed[0])


def repair_interface_orientations(mesh, *, rtol=1e-10, atol=1e-10):
    """Repair uniquely identifiable PointRangeDonor/Transform orientations in place.

    Call on a complete CGNSMesh before constructing a solver. Coordinates, source
    ranges, donor zone names, and donor face extents are never changed. Every face
    node must match; ambiguous or nonmatching faces raise ValueError. All changes
    are staged, so a failure leaves every connection unchanged. Returns a list of
    dictionaries describing the old/new mappings. Does not write the CGNS file.

    This explicit preprocessing helper requires local coordinates for every zone;
    it is not a collective operation or an automatic distributed loading step.
    """
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError('Tolerances must be finite and nonnegative')
    zones = {z.name: z for z in mesh.zones}
    if len(zones) != len(mesh.zones):
        raise ValueError('Duplicate zone names')
    if any(z.coordinates is None for z in mesh.zones):
        raise ValueError('Repair requires a complete mesh with local coordinates')
    staged, report = [], []
    for zone in mesh.zones:
        for link in zone.connections:
            donor = zones[link.donor]
            p, d = np.asarray(link.point_range), np.asarray(link.donor_range)
            normal = _face_axis(p, zone.dimensions)
            dn = _face_axis(d, donor.dimensions)
            axes = [np.arange(r[0], r[1] + (1 if r[1] >= r[0] else -1),
                              1 if r[1] >= r[0] else -1) for r in p]
            points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
            values = zone.coordinates[tuple((points-1).T)]

            def matches(bounds, transform):
                if sorted(abs(np.asarray(transform))) != [1, 2, 3]:
                    return False
                q = np.tile(bounds[:, 0], (len(points), 1))
                end = bounds[:, 0].copy()
                for a, t in enumerate(transform):
                    q[:, abs(t)-1] += np.sign(t)*(points[:, a]-p[a, 0])
                    end[abs(t)-1] += np.sign(t)*(p[a, 1]-p[a, 0])
                if not np.array_equal(end, bounds[:, 1]):
                    return False
                if np.any(q < 1) or np.any(q > np.asarray(donor.dimensions)):
                    return False
                return np.allclose(values, donor.coordinates[tuple((q-1).T)], rtol=rtol, atol=atol)

            if matches(d, link.transform):
                continue
            tangents = [a for a in range(3) if a != normal]
            dt = [a for a in range(3) if a != dn]
            candidates = []
            for mapped in permutations(dt):
                if any(abs(p[a, 1]-p[a, 0]) != abs(d[b, 1]-d[b, 0])
                       for a, b in zip(tangents, mapped)):
                    continue
                for signs in product((-1, 1), repeat=2):
                    bounds = np.sort(d, axis=1)
                    transform = np.zeros(3, dtype=int)
                    # Extend index directions across the two exterior faces.
                    side = -1 if p[normal, 0] == 1 else 1
                    donor_side = -1 if d[dn, 0] == 1 else 1
                    transform[normal] = -side*donor_side*(dn+1)
                    for a, b, sign in zip(tangents, mapped, signs):
                        transform[a] = sign*(b+1)
                        if sign*(p[a, 1]-p[a, 0]) < 0:
                            bounds[b] = bounds[b, ::-1]
                    if matches(bounds, transform):
                        candidates.append((bounds, transform))
            if len(candidates) != 1:
                raise ValueError(f'{zone.name} -> {donor.name}: expected one matching orientation, '
                                 f'found {len(candidates)}; no connections changed')
            bounds, transform = candidates[0]
            staged.append((link, bounds, transform))
            report.append(dict(zone=zone.name, donor=donor.name,
                               old_donor_range=d.tolist(), new_donor_range=bounds.tolist(),
                               old_transform=np.asarray(link.transform).tolist(),
                               new_transform=transform.tolist()))
    for link, bounds, transform in staged:
        link.donor_range = bounds
        link.transform = transform
    return report


def repair_cgns_connectivity(input_file, output_file, *, overwrite=False):
    """Write a verified repaired CGNS copy and return old/new mapping records.

    Serial preprocessing: run once before MPI solver/warper initialization.
    The input is never overwritten. Existing output requires overwrite=True.
    A temporary sibling is validated before atomic replacement. Requires libcgns
    support for the input format, one base, and a complete mesh in local memory.
    """
    import ctypes as C
    import os
    from pathlib import Path
    import shutil
    import tempfile
    from ..cgns import CGNSMesh
    from .._cgns_native import library

    source, destination = Path(input_file).resolve(), Path(output_file).resolve()
    if source == destination or (destination.exists() and os.path.samefile(source, destination)):
        raise ValueError('Input and output must be different files')
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    api = library()
    with api.open(source) as fn:
        if api.count('cg_nbases', fn) != 1:
            raise ValueError('File repair currently requires exactly one CGNS base')
    mesh = CGNSMesh.from_file(source, backend='cgns')
    report = repair_interface_orientations(mesh)
    fd, path = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.cgns', dir=destination.parent)
    os.close(fd)
    temporary = Path(path)
    try:
        shutil.copyfile(source, temporary)
        integer = C.c_int
        ip = C.POINTER(integer)
        sp = C.POINTER(api.index)
        api.lib.cg_1to1_write.argtypes = [integer, integer, integer, C.c_char_p,
                                        C.c_char_p, sp, sp, ip, ip]
        api.lib.cg_1to1_write.restype = integer
        with api.open(temporary, mode=api.constant('CG_MODE_MODIFY')) as fn:
            # Capture names before writes: libcgns may reorder overwritten nodes.
            updates = []
            for zi, zone in enumerate(mesh.zones, 1):
                for ci, link in enumerate(zone.connections, 1):
                    name, donor = C.create_string_buffer(33), C.create_string_buffer(33)
                    p, d, t = (api.index*6)(), (api.index*6)(), (integer*3)()
                    api.call('cg_1to1_read', fn, 1, zi, ci, name, donor, p, d, t)
                    if (list(d) != link.donor_range.ravel(order='F').tolist()
                            or list(t) != link.transform.tolist()):
                        updates.append((zi, name.value, donor.value, link))
            for zi, name, donor, link in updates:
                p = (api.index*6)(*link.point_range.ravel(order='F'))
                d = (api.index*6)(*link.donor_range.ravel(order='F'))
                t = (integer*3)(*link.transform)
                index = integer()
                api.call('cg_1to1_write', fn, 1, zi, name, donor, p, d, t, C.byref(index))
        verified = CGNSMesh.from_file(temporary, backend='cgns')
        if repair_interface_orientations(verified):
            raise ValueError('Written file still requires interface repair')
        # Verify contents independently of node order after CGNS modification.
        if len(mesh.zones) != len(verified.zones):
            raise ValueError('Zone count changed during write')
        for a, b in zip(mesh.zones, verified.zones):
            if (a.name, a.family, a.additional_families) != (b.name, b.family, b.additional_families):
                raise ValueError('Zone metadata changed during write')
            np.testing.assert_array_equal(a.coordinates, b.coordinates)
            if len(a.boundaries) != len(b.boundaries) or len(a.connections) != len(b.connections):
                raise ValueError('Boundary or connection count changed during write')
            for u, v in zip(sorted(a.boundaries, key=lambda x: x.name),
                            sorted(b.boundaries, key=lambda x: x.name)):
                if (u.name, u.bc_type, u.family, u.additional_families) != (v.name, v.bc_type, v.family, v.additional_families):
                    raise ValueError('Boundary metadata changed during write')
                np.testing.assert_array_equal(u.point_range, v.point_range)
                if u.data.keys() != v.data.keys():
                    raise ValueError('Boundary data changed during write')
                for key in u.data:
                    np.testing.assert_array_equal(u.data[key], v.data[key])
            key = lambda c: (c.donor, tuple(c.point_range.ravel()))
            for u, v in zip(sorted(a.connections, key=key), sorted(b.connections, key=key)):
                if u.donor != v.donor:
                    raise ValueError('Donor changed during write')
                for field in ('point_range', 'donor_range', 'transform'):
                    np.testing.assert_array_equal(getattr(u, field), getattr(v, field))
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Atomic publication without overwriting a concurrently created file.
            os.link(temporary, destination)
        return report
    finally:
        temporary.unlink(missing_ok=True)
