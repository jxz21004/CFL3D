"""Write solver snapshots without re-entering the CFD core or MPI collectives."""
from contextlib import contextmanager
import ctypes as C
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np


@contextmanager
def _temporary_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="."+path.name+"-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_volume(path, mesh, fields, options):
    """CGNS/ADF: original vertices/BC/connectivity, cell-centered primitive fields."""
    from ._cgns_native import library, _LOCK
    api = library()
    integer, ip, sp, char = C.c_int, C.POINTER(C.c_int), C.POINTER(api.index), C.c_char_p
    signatures = {
        "cg_set_file_type": [integer],
        "cg_base_write": [integer, char, integer, integer, ip],
        "cg_zone_write": [integer, integer, char, sp, integer, ip],
        "cg_coord_write": [integer, integer, integer, integer, char, C.c_void_p, ip],
        "cg_boco_write": [integer, integer, integer, char, integer, integer, api.index, sp, ip],
        "cg_1to1_write": [integer, integer, integer, char, char, sp, sp, ip, ip],
        "cg_family_write": [integer, integer, char, ip],
        "cg_famname_write": [char], "cg_multifam_write": [char, char],
        "cg_sol_write": [integer, integer, integer, char, integer, ip],
        "cg_field_write": [integer, integer, integer, integer, integer, char, C.c_void_p, ip],
        "cg_descriptor_write": [char, char],
    }
    with _LOCK, _temporary_output(path) as temporary:
        for name, arguments in signatures.items():
            api.bind(name, arguments)
        api.call("cg_set_file_type", api.constant("CG_FILE_ADF"))
        with api.open(temporary, api.constant("CG_MODE_WRITE")) as fn:
            base = integer()
            api.call("cg_base_write", fn, b"Base", 3, 3, C.byref(base))
            # Families in output refer to this output base, including originally qualified names.
            families = {name for zone in mesh.zones for item in (zone, *zone.boundaries)
                        for name in (item.family, *item.additional_families) if name}
            for family in sorted(families):
                index = integer()
                api.call("cg_family_write", fn, base, family.rsplit("/",1)[-1].encode(), C.byref(index))

            def family_nodes(item):
                if item.family:
                    api.call("cg_famname_write", item.family.rsplit("/",1)[-1].encode())
                for number, family in enumerate(item.additional_families, 1):
                    api.call("cg_multifam_write", f"Family{number}".encode(), family.rsplit("/",1)[-1].encode())

            bc_types = {api.lib.cg_BCTypeName(i).decode(): i
                        for i in range(api.constant("NofValidBCTypes"))}
            for zone in mesh.zones:
                z = integer()
                dims = zone.dimensions
                sizes = (api.index*9)(*dims, *(n-1 for n in dims), 0, 0, 0)
                api.call("cg_zone_write", fn, base, zone.name.encode(), sizes,
                         api.constant("Structured"), C.byref(z))
                api.navigate(fn, base.value, ("Zone_t", z.value))
                family_nodes(zone)
                for axis, name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
                    values = np.asfortranarray(zone.coordinates[...,axis], dtype=np.float64)
                    index = integer()
                    api.call("cg_coord_write", fn, base, z, api.constant("RealDouble"),
                             name.encode(), values.ctypes.data, C.byref(index))
                for boundary in zone.boundaries:
                    index = integer()
                    points = (api.index*6)(*boundary.point_range.flatten(order="F"))
                    api.call("cg_boco_write", fn, base, z, boundary.name.encode(), bc_types[boundary.bc_type],
                             api.constant("PointRange"), 2, points, C.byref(index))
                    api.navigate(fn, base.value, ("Zone_t",z.value), ("ZoneBC_t",1), ("BC_t",index.value))
                    family_nodes(boundary)
                for number, connection in enumerate(zone.connections, 1):
                    index = integer()
                    left = (api.index*6)(*connection.point_range.flatten(order="F"))
                    right = (api.index*6)(*connection.donor_range.flatten(order="F"))
                    transform = (integer*3)(*connection.transform)
                    api.call("cg_1to1_write", fn, base, z, f"Connection{number}".encode(),
                             connection.donor.encode(), left, right, transform, C.byref(index))
                solution = integer()
                api.call("cg_sol_write", fn, base, z, b"FlowSolution", api.constant("CellCenter"), C.byref(solution))
                q = fields[zone.name]
                values = {name: q[...,i] for i,name in enumerate(
                    ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"))}
                values["Temperature"] = options.temperature * options.gamma * q[...,4]/q[...,0]
                for name, array in values.items():
                    array = np.asfortranarray(array, dtype=np.float64)
                    index = integer()
                    api.call("cg_field_write", fn, base, z, solution, api.constant("RealDouble"),
                             name.encode(), array.ctypes.data, C.byref(index))
                api.navigate(fn, base.value, ("Zone_t",z.value), ("FlowSolution_t",solution.value))
                description = {
                    "coordinates": "m", "Density": "rho/rho_inf",
                    "VelocityX/Y/Z": "velocity/a_inf in input mesh axes",
                    "Pressure": "p/(rho_inf*a_inf^2)", "Temperature": "K",
                    "Mach": options.mach, "gamma": options.gamma,
                    "T_inf_K": options.temperature, "p_inf_Pa": options.freestream_pressure,
                    "purpose": "Visualization output, not a CFL3D restart file",
                }
                api.call("cg_descriptor_write", b"CFL3DUnits", json.dumps(description).encode())


def wall_vertices(zone, boundary):
    ranges = np.sort(boundary.point_range, axis=1)
    axis = int(np.flatnonzero(ranges[:,0] == ranges[:,1])[0])
    slices = tuple(int(r[0]-1) if i == axis else slice(int(r[0]-1), int(r[1]))
                   for i,r in enumerate(ranges))
    return zone.coordinates[slices]


def write_surface(path, mesh, walls, members):
    """Tecplot FEQUADRILATERAL: exact wall vertices, cell-centered native wall values."""
    with _temporary_output(path) as temporary, temporary.open("w", encoding="utf-8") as stream:
        stream.write('TITLE = "CFL3D wall solution"\n')
        stream.write('VARIABLES = "X", "Y", "Z", "Cp", "T", "Cfx", "Cfy", "Cfz", "y+"\n')
        for zone in mesh.zones:
            for boundary in zone.boundaries:
                key = zone.name, boundary.name
                if key not in walls or key not in members:
                    continue
                xyz = wall_vertices(zone, boundary)
                a,b = xyz.shape[:2]
                values = walls[key]
                # JSON string escaping is also valid for the simple CGNS names in Tecplot titles.
                stream.write(f'ZONE T={json.dumps("/".join(key))}, N={a*b}, E={(a-1)*(b-1)}, '
                             'ZONETYPE=FEQUADRILATERAL, DATAPACKING=BLOCK, VARLOCATION=([4-9]=CELLCENTERED)\n')
                for column in range(3):
                    np.savetxt(stream, xyz[...,column].ravel(order="F"), fmt="%.16e")
                for column in range(6):
                    np.savetxt(stream, values[...,column].ravel(order="F"), fmt="%.16e")
                for j in range(b-1):
                    for i in range(a-1):
                        p = i+a*j+1
                        stream.write(f"{p} {p+1} {p+1+a} {p+a}\n")


def _record(stream, values, dtype):
    data = np.asarray(values, dtype=dtype).tobytes(order="F")
    _record_bytes(stream, data)


def _record_bytes(stream, data):
    if len(data) > 2**31-1:
        raise ValueError("PLOT3D record exceeds 32-bit record markers; use CGNS output")
    marker = struct.pack("=i", len(data))
    stream.write(marker); stream.write(data); stream.write(marker)


def write_flow(prefix, mesh, fields, options, final_alpha):
    """CFL3D IPtype=1 PLOT3D: int32 record markers/IBLANK, float64 coordinates/Q.

    Serialize cached values using plot3d.F's packing. No Fortran COMMON state
    is accessed, so delayed output remains valid after another solver runs.
    """
    x_path, q_path = Path(str(prefix)+".x"), Path(str(prefix)+".q")
    i2d = all(z.dimensions[0] == 2 for z in mesh.zones)
    dimensions = np.array([np.array(z.dimensions)[1 if i2d else 0:]-1 for z in mesh.zones], dtype=np.int32).T
    with _temporary_output(x_path) as tx, _temporary_output(q_path) as tq:
        with tx.open("wb") as xfile, tq.open("wb") as qfile:
            for stream in (xfile,qfile):
                _record(stream, [len(mesh.zones)], np.int32)
                _record(stream, dimensions, np.int32)
            for zone in mesh.zones:
                xyz = zone.coordinates
                centers = sum(xyz[i:xyz.shape[0]-1+i,j:xyz.shape[1]-1+j,k:xyz.shape[2]-1+k]
                              for i,j,k in np.ndindex(2,2,2))/8.
                q = fields[zone.name]
                conservative = q.copy()
                conservative[...,1:4] *= q[...,0,None]
                conservative[...,4] = q[...,4]/(options.gamma-1) + .5*q[...,0]*np.sum(q[...,1:4]**2, axis=-1)
                if i2d:
                    # Original CFL3D writes the x-z plane, or x-y for y-up input.
                    vertical = 1 if options.y_up else 2
                    centers = centers[0][...,[0,vertical]]
                    conservative = conservative[0][...,[0,1,vertical+1,4]]
                # CFL3D's default IBLNK=1 packs integer blanking after xyz in the same record.
                coordinate_data = np.asarray(centers,dtype=np.float64).tobytes(order="F")
                blank = np.ones(centers.shape[:-1],dtype=np.int32).tobytes(order="F")
                _record_bytes(xfile, coordinate_data+blank)
                _record(qfile, [options.mach,final_alpha,options.reynolds,0.0], np.float64)
                _record(qfile, conservative, np.float64)
    return x_path, q_path
