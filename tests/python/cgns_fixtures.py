"""Write test files with the official CGNS mid-level API, independently of the reader."""
import ctypes as C
from pathlib import Path

import numpy as np
from cfl3d._cgns_native import library


def writer():
    api = library()
    integer, ip, sp = C.c_int, C.POINTER(C.c_int), C.POINTER(api.index)
    signatures = {
        "cg_set_file_type": [integer], "cg_get_file_type": [integer, ip],
        "cg_base_write": [integer, C.c_char_p, integer, integer, ip],
        "cg_family_write": [integer, integer, C.c_char_p, ip],
        "cg_fambc_write": [integer, integer, integer, C.c_char_p, integer, ip],
        "cg_famname_write": [C.c_char_p],
        "cg_multifam_write": [C.c_char_p, C.c_char_p],
        "cg_zone_write": [integer, integer, C.c_char_p, sp, integer, ip],
        "cg_coord_write": [integer, integer, integer, integer, C.c_char_p, C.c_void_p, ip],
        "cg_boco_write": [integer, integer, integer, C.c_char_p, integer, integer, api.index, sp, ip],
        "cg_boco_gridlocation_write": [integer, integer, integer, integer, integer],
        "cg_1to1_write": [integer, integer, integer, C.c_char_p, C.c_char_p, sp, sp, ip, ip],
        "cg_1to1_periodic_write": [integer, integer, integer, integer, C.c_void_p, C.c_void_p, C.c_void_p],
        "cg_dataset_write": [integer, integer, integer, integer, C.c_char_p, integer, ip],
        "cg_bcdata_write": [integer, integer, integer, integer, integer, integer],
        "cg_array_write": [C.c_char_p, integer, integer, sp, C.c_void_p],
    }
    for name, arguments in signatures.items():
        api.bind(name, arguments)
    return api


def write_native(mesh, filename, *, base="Base", file_type="ADF", family_bc_types=None):
    api = writer()
    bc_types = {api.lib.cg_BCTypeName(i).decode(): i
                for i in range(api.constant("NofValidBCTypes"))}
    path = Path(filename)
    api.call("cg_set_file_type", api.constant("CG_FILE_" + file_type))
    with api.open(path, api.constant("CG_MODE_WRITE")) as fn:
        b = C.c_int()
        api.call("cg_base_write", fn, base.encode(), 3, 3, C.byref(b))
        families = set()
        for zone in mesh.zones:
            for item in (zone, *zone.boundaries):
                families.update(f for f in (item.family, *item.additional_families) if f)
        for family in sorted(families):
            f = C.c_int()
            api.call("cg_family_write", fn, b, family.encode(), C.byref(f))
            if family in (family_bc_types or {}):
                bc = C.c_int()
                api.call("cg_fambc_write", fn, b, f, b"FamilyBC",
                         bc_types[family_bc_types[family]], C.byref(bc))

        def write_families(item):
            if item.family:
                api.call("cg_famname_write", item.family.encode())
            for number, name in enumerate(item.additional_families, 1):
                api.call("cg_multifam_write", f"Additional{number}".encode(), name.encode())

        for zone in mesh.zones:
            dims = zone.dimensions
            sizes = (api.index * 9)(*dims, *(n-1 for n in dims), 0, 0, 0)
            z = C.c_int()
            api.call("cg_zone_write", fn, b, zone.name.encode(), sizes,
                     api.constant("Structured"), C.byref(z))
            api.navigate(fn, b.value, ("Zone_t", z.value))
            write_families(zone)
            for axis, name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
                values = np.asfortranarray(zone.coordinates[..., axis], dtype=np.float64)
                index = C.c_int()
                api.call("cg_coord_write", fn, b, z, api.constant("RealDouble"), name.encode(),
                         values.ctypes.data, C.byref(index))
            for bc in zone.boundaries:
                points = (api.index * 6)(*bc.point_range.flatten(order="F"))
                index = C.c_int()
                api.call("cg_boco_write", fn, b, z, bc.name.encode(), bc_types[bc.bc_type],
                         api.constant("PointRange"), 2, points, C.byref(index))
                api.navigate(fn, b.value, ("Zone_t", z.value), ("ZoneBC_t", 1), ("BC_t", index.value))
                write_families(bc)
                if bc.data:
                    ds = C.c_int()
                    api.call("cg_dataset_write", fn, b, z, index, b"Data", bc_types[bc.bc_type], C.byref(ds))
                    api.call("cg_bcdata_write", fn, b, z, index, ds, api.constant("Neumann"))
                    api.navigate(fn, b.value, ("Zone_t", z.value), ("ZoneBC_t", 1),
                                 ("BC_t", index.value), ("BCDataSet_t", ds.value),
                                 ("BCData_t", api.constant("Neumann")))
                    for name, values in bc.data.items():
                        values = np.asfortranarray(np.atleast_1d(values), dtype=np.float64)
                        shape = (api.index * values.ndim)(*values.shape)
                        api.call("cg_array_write", name.encode(), api.constant("RealDouble"),
                                 values.ndim, shape, values.ctypes.data)
            for i, cn in enumerate(zone.connections):
                points = (api.index * 6)(*cn.point_range.flatten(order="F"))
                donor = (api.index * 6)(*cn.donor_range.flatten(order="F"))
                transform = (C.c_int * 3)(*cn.transform)
                index = C.c_int()
                api.call("cg_1to1_write", fn, b, z, f"connect{i}".encode(), cn.donor.encode(),
                         points, donor, transform, C.byref(index))
    return path


def write_adf(mesh, filename, *, base="Base", family_bc_types=None):
    return write_native(mesh, filename, base=base, family_bc_types=family_bc_types)
