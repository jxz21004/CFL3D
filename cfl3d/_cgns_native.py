"""CGNS mid-level API reader, using the ABI of a locally compiled bridge."""
from contextlib import contextmanager
import ctypes as C
from functools import lru_cache
from pathlib import Path
import threading

import numpy as np

_LOCK = threading.RLock()  # CGNS has process-global file/navigation state.
_INT = C.c_int
_IP = C.POINTER(_INT)
_VOID = C.c_void_p
_CHAR = C.c_char_p


class CGNSLibrary:
    def __init__(self):
        path = Path(__file__).with_name("_cgns_bridge.so")
        try:
            self.lib = C.CDLL(str(path))
        except OSError as exc:
            raise ImportError("Native CGNS reader unavailable. Load the CGNS environment and run "
                              "python tools/build_cgns.py (CGNS_HOME or --prefix required). "
                              f"Loader detail: {exc}") from exc
        self.lib.cfl3d_cgns_index_bytes.restype = _INT
        self.lib.cfl3d_cgns_has_hdf5.restype = _INT
        self.lib.cfl3d_cgns_constant.argtypes = [_CHAR]
        self.lib.cfl3d_cgns_constant.restype = _INT
        self.lib.cfl3d_cgns_error.restype = _CHAR
        # cgsize_t 可为 32 或 64 位。猜错宽度会让 C 读写越界，必须由编译后的桥返回。
        width = self.lib.cfl3d_cgns_index_bytes()
        if width not in (4, 8):
            raise RuntimeError(f"Unsupported CGNS index width: {width}")
        self.index = C.c_int32 if width == 4 else C.c_int64
        self.dtype = np.dtype(np.int32 if width == 4 else np.int64)
        self.has_hdf5 = bool(self.lib.cfl3d_cgns_has_hdf5())
        sp = C.POINTER(self.index)
        signatures = {
            "cg_open": [_CHAR, _INT, _IP], "cg_close": [_INT],
            "cg_nbases": [_INT, _IP], "cg_base_read": [_INT, _INT, _CHAR, _IP, _IP],
            "cg_nzones": [_INT, _INT, _IP], "cg_zone_type": [_INT, _INT, _INT, _IP],
            "cg_nfamilies": [_INT, _INT, _IP],
            "cg_family_read": [_INT, _INT, _INT, _CHAR, _IP, _IP],
            "cg_fambc_read": [_INT, _INT, _INT, _INT, _CHAR, _IP],
            "cg_famname_read": [_CHAR], "cg_nmultifam": [_IP],
            "cg_multifam_read": [_INT, _CHAR, _CHAR],
            "cg_zone_read": [_INT, _INT, _INT, _CHAR, sp],
            "cg_ngrids": [_INT, _INT, _INT, _IP],
            "cg_grid_read": [_INT, _INT, _INT, _INT, _CHAR],
            "cg_coord_read": [_INT, _INT, _INT, _CHAR, _INT, sp, sp, _VOID],
            "cg_gopath": [_INT, _CHAR], "cg_rind_read": [_IP],
            "cg_golist": [_INT, _INT, _INT, C.POINTER(_CHAR), _IP],
            "cg_nbocos": [_INT, _INT, _INT, _IP],
            "cg_boco_info": [_INT, _INT, _INT, _INT, _CHAR, _IP, _IP, sp, _IP, sp, _IP, _IP],
            "cg_boco_read": [_INT, _INT, _INT, _INT, sp, _VOID],
            "cg_boco_gridlocation_read": [_INT, _INT, _INT, _INT, _IP],
            "cg_dataset_read": [_INT, _INT, _INT, _INT, _INT, _CHAR, _IP, _IP, _IP],
            "cg_narrays": [_IP], "cg_array_info": [_INT, _CHAR, _IP, _IP, sp],
            "cg_array_read_as": [_INT, _INT, _VOID],
            "cg_nzconns": [_INT, _INT, _INT, _IP],
            "cg_nconns": [_INT, _INT, _INT, _IP], "cg_nholes": [_INT, _INT, _INT, _IP],
            "cg_n1to1": [_INT, _INT, _INT, _IP],
            "cg_1to1_read": [_INT, _INT, _INT, _INT, _CHAR, _CHAR, sp, sp, _IP],
            "cg_1to1_periodic_read": [_INT, _INT, _INT, _INT, _VOID, _VOID, _VOID],
            "cg_1to1_average_read": [_INT, _INT, _INT, _INT, _IP],
        }
        for name, args in signatures.items():
            self.bind(name, args)
        self.lib.cg_BCTypeName.argtypes = [_INT]
        self.lib.cg_BCTypeName.restype = _CHAR

    def bind(self, name, arguments):
        function = getattr(self.lib, name)
        function.argtypes = arguments
        function.restype = _INT

    def constant(self, name):
        value = self.lib.cfl3d_cgns_constant(name.encode())
        if value < 0:
            raise ValueError(f"Unknown CGNS constant {name}")
        return value

    def check(self, status, operation):
        if status:
            message = self.lib.cfl3d_cgns_error().decode(errors="replace")
            raise ValueError(f"{operation}: {message}")

    def call(self, name, *args):
        self.check(getattr(self.lib, name)(*args), name)

    def count(self, name, *args):
        count = _INT()
        self.call(name, *args, C.byref(count))
        return count.value

    @contextmanager
    def open(self, filename, mode=None):
        with _LOCK:
            fn = _INT()
            self.call("cg_open", str(filename).encode(),
                      self.constant("CG_MODE_READ") if mode is None else mode, C.byref(fn))
            try:
                yield fn.value
            finally:
                self.call("cg_close", fn)

    def goto(self, fn, *names):
        self.call("cg_gopath", fn, ("/" + "/".join(names)).encode())

    def navigate(self, fn, base, *nodes):
        labels = (_CHAR * len(nodes))(*(label.encode() for label, _ in nodes))
        indices = (_INT * len(nodes))(*(index for _, index in nodes))
        self.call("cg_golist", fn, base, len(nodes), labels, indices)

    def optional(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status == self.constant("CG_NODE_NOT_FOUND"):
            return False
        self.check(status, name)
        return True


@lru_cache(maxsize=1)
def library():
    return CGNSLibrary()


def _name():
    return C.create_string_buffer(33)  # CGNS names contain at most 32 bytes.


def _data_arrays(api):
    arrays = {}
    numeric = {api.constant(t) for t in ("Integer", "LongInteger", "RealSingle", "RealDouble")}
    for a in range(1, api.count("cg_narrays") + 1):
        name, typ, ndim = _name(), _INT(), _INT()
        dims = (api.index * 12)()
        api.call("cg_array_info", a, name, C.byref(typ), C.byref(ndim), dims)
        key = name.value.decode()
        if typ.value not in numeric or not 1 <= ndim.value <= 12:
            raise NotImplementedError(f"{key}: only numeric BC data arrays are supported")
        shape = tuple(dims[:ndim.value])
        if any(n < 1 for n in shape):
            raise ValueError(f"{key}: invalid BC data dimensions")
        values = np.empty(shape, dtype=np.float64, order="F")
        api.call("cg_array_read_as", a, api.constant("RealDouble"), values.ctypes.data)
        arrays[key] = values
    return arrays


def _families(api):
    primary = C.create_string_buffer(1025)
    family = primary.value.decode() if api.optional("cg_famname_read", primary) else None
    additional = []
    for i in range(1, api.count("cg_nmultifam") + 1):
        name, value = _name(), C.create_string_buffer(1025)
        api.call("cg_multifam_read", i, name, value)
        additional.append(value.value.decode())
    return family, tuple(additional)


def read_mesh(filename, *, base=None):
    from .cgns import Boundary, CGNSMesh, Connection, Zone

    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(path)
    api = library()
    zones = []
    with api.open(path) as fn:
        bases = []
        for b in range(1, api.count("cg_nbases", fn) + 1):
            name, cell, physical = _name(), _INT(), _INT()
            api.call("cg_base_read", fn, b, name, C.byref(cell), C.byref(physical))
            if base is None or name.value.decode() == base:
                bases.append((b, name.value.decode(), cell.value, physical.value))
        if len(bases) != 1:
            raise ValueError("Select exactly one CGNS base using base=...")
        b, basename, cell, physical = bases[0]
        if (cell, physical) != (3, 3):
            raise NotImplementedError("Native wrapper requires cell/physical dimension 3 (including extruded 2D grids)")
        family_types = {}
        for f in range(1, api.count("cg_nfamilies", fn, b) + 1):
            name, nbc, ngeom = _name(), _INT(), _INT()
            api.call("cg_family_read", fn, b, f, name, C.byref(nbc), C.byref(ngeom))
            if nbc.value == 1:
                bcname, bctype = _name(), _INT()
                api.call("cg_fambc_read", fn, b, f, 1, bcname, C.byref(bctype))
                family_types[name.value.decode()] = api.lib.cg_BCTypeName(bctype).decode()
        for z in range(1, api.count("cg_nzones", fn, b) + 1):
            typ = _INT()
            api.call("cg_zone_type", fn, b, z, C.byref(typ))
            if typ.value != api.constant("Structured"):
                raise NotImplementedError("Only structured CGNS zones are supported")
            name, sizes = _name(), (api.index * 9)()
            api.call("cg_zone_read", fn, b, z, name, sizes)
            zonename = name.value.decode()
            dims = tuple(sizes[:3])
            if any(n < 2 for n in dims) or tuple(sizes[3:6]) != tuple(n-1 for n in dims):
                raise ValueError(f"{zonename}: invalid vertex/cell dimensions")
            if api.count("cg_ngrids", fn, b, z) != 1:
                raise ValueError(f"{zonename}: expected one GridCoordinates node")
            grid = _name()
            api.call("cg_grid_read", fn, b, z, 1, grid)
            # cg_coord_read addresses the standard GridCoordinates node.
            if grid.value != b"GridCoordinates":
                raise NotImplementedError(f"{zonename}: coordinates must use the GridCoordinates node")
            api.goto(fn, basename, zonename, grid.value.decode())
            rind = (_INT * 6)()
            if api.optional("cg_rind_read", rind) and any(rind):
                raise NotImplementedError("CGNS coordinate rind layers are not supported")
            # CGNS C API 的结构化坐标按 i 索引最快存放；F 顺序可直接读入，无需转置。
            coordinates = np.empty((*dims, 3), dtype=np.float64, order="F")
            lo, hi = (api.index * 3)(1, 1, 1), (api.index * 3)(*dims)
            for axis, coord in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
                api.call("cg_coord_read", fn, b, z, coord.encode(), api.constant("RealDouble"),
                         lo, hi, coordinates[..., axis].ctypes.data)
            if not np.isfinite(coordinates).all():
                raise ValueError(f"{zonename}: nonfinite coordinate")
            zone = Zone(zonename, coordinates)
            api.navigate(fn, b, ("Zone_t", z))
            zone.family, zone.additional_families = _families(api)
            for bc in range(1, api.count("cg_nbocos", fn, b, z) + 1):
                bcname, bctype, pointset = _name(), _INT(), _INT()
                count, normal_size = api.index(), api.index()
                normal_index, normal_type, datasets = (_INT * 3)(), _INT(), _INT()
                api.call("cg_boco_info", fn, b, z, bc, bcname, C.byref(bctype), C.byref(pointset),
                         C.byref(count), normal_index, C.byref(normal_size), C.byref(normal_type),
                         C.byref(datasets))
                if pointset.value != api.constant("PointRange") or count.value != 2:
                    raise NotImplementedError(f"{zonename}/{bcname.value.decode()}: requires PointRange; PointList is not supported")
                location = _INT()
                api.call("cg_boco_gridlocation_read", fn, b, z, bc, C.byref(location))
                if location.value != api.constant("Vertex"):
                    raise NotImplementedError(f"{bcname.value.decode()}: boundary GridLocation must be Vertex")
                points = (api.index * 6)()
                api.call("cg_boco_read", fn, b, z, bc, points, None)
                api.navigate(fn, b, ("Zone_t", z), ("ZoneBC_t", 1), ("BC_t", bc))
                family, additional = _families(api)
                bc_type = api.lib.cg_BCTypeName(bctype).decode()
                if bc_type == "FamilySpecified":
                    # 只解析本 base 的 FamilyBC；跨 base/多义 Family 保留给显式覆盖。
                    parts = (family or zone.family or "").strip("/").split("/")
                    if len(parts) == 1 or (len(parts) == 2 and parts[0] == basename):
                        bc_type = family_types.get(parts[-1], bc_type)
                data = {}
                for ds in range(1, datasets.value + 1):
                    dsname, dstype, dirichlet, neumann = _name(), _INT(), _INT(), _INT()
                    api.call("cg_dataset_read", fn, b, z, bc, ds, dsname,
                             C.byref(dstype), C.byref(dirichlet), C.byref(neumann))
                    for flag, kind in ((dirichlet, "Dirichlet"), (neumann, "Neumann")):
                        if flag.value:
                            api.navigate(fn, b, ("Zone_t", z), ("ZoneBC_t", 1), ("BC_t", bc),
                                         ("BCDataSet_t", ds), ("BCData_t", api.constant(kind)))
                            arrays = _data_arrays(api)
                            if data.keys() & arrays.keys():
                                raise ValueError(f"{bcname.value.decode()}: ambiguous BC data")
                            data.update(arrays)
                # PointRange 的六个数是起点(i,j,k)、终点(i,j,k)，并保留 CGNS 的 1-based 索引。
                zone.boundaries.append(Boundary(bcname.value.decode(), bc_type,
                                               np.array(points, dtype=np.int64).reshape(3, 2, order="F"),
                                               data, family, additional))
            if api.count("cg_nzconns", fn, b, z) > 1:
                raise NotImplementedError(f"{zonename}: multiple ZoneGridConnectivity nodes are not supported")
            if api.count("cg_nconns", fn, b, z) or api.count("cg_nholes", fn, b, z):
                raise NotImplementedError(f"{zonename}: general/overset connectivity is not supported")
            for cn in range(1, api.count("cg_n1to1", fn, b, z) + 1):
                cnname, donor = _name(), _name()
                points, donor_points, transform = (api.index * 6)(), (api.index * 6)(), (_INT * 3)()
                api.call("cg_1to1_read", fn, b, z, cn, cnname, donor, points, donor_points, transform)
                floats = (C.c_float * 3)()
                average = _INT()
                if (api.optional("cg_1to1_periodic_read", fn, b, z, cn, floats, floats, floats)
                        or api.optional("cg_1to1_average_read", fn, b, z, cn, C.byref(average))):
                    raise NotImplementedError(f"{cnname.value.decode()}: periodic/averaged connections are not supported")
                if sorted(abs(t) for t in transform) != [1, 2, 3]:
                    raise ValueError(f"{cnname.value.decode()}: Transform must be a signed permutation of 1,2,3")
                zone.connections.append(Connection(donor.value.decode(),
                    np.array(points, dtype=np.int64).reshape(3, 2, order="F"),
                    np.array(donor_points, dtype=np.int64).reshape(3, 2, order="F"), np.array(transform)))
            zones.append(zone)
    if not zones:
        raise ValueError("CGNS base contains no structured zones")
    if len({z.name for z in zones}) != len(zones):
        raise ValueError("Duplicate zone names")
    return CGNSMesh(zones, str(path.resolve()))
