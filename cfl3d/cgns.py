"""Structured CGNS readers; independent of CFL3D's legacy CGNS API."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class Boundary:
    name: str
    bc_type: str
    point_range: np.ndarray
    data: dict[str, np.ndarray] = field(default_factory=dict)
    family: str | None = None
    additional_families: tuple[str, ...] = ()


@dataclass
class Connection:
    donor: str
    point_range: np.ndarray
    donor_range: np.ndarray
    transform: np.ndarray


@dataclass
class Zone:
    name: str
    coordinates: np.ndarray | None  # None for non-owned MPI zones; otherwise (ni,nj,nk,3)
    boundaries: list[Boundary] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    family: str | None = None
    additional_families: tuple[str, ...] = ()

    coordinate_dimensions: tuple[int, int, int] | None = field(default=None, repr=False)

    @property
    def dimensions(self):
        if self.coordinates is not None:
            return self.coordinates.shape[:3]
        if self.coordinate_dimensions is None:
            raise ValueError(f"Zone {self.name} has neither coordinates nor dimensions")
        return self.coordinate_dimensions


def _text(value):
    if isinstance(value, (str, bytes, np.bytes_)):
        return value.decode().rstrip("\x00") if isinstance(value, bytes) else str(value).rstrip("\x00")
    a = np.asarray(value)
    return a.tobytes().decode().rstrip("\x00")


def _children(node, label):
    import h5py
    return [v for k, v in node.items() if not k.startswith(" ")
            and isinstance(v, h5py.Group) and _text(v.attrs.get("label", "")) == label]


def _data(node):
    if " data" not in node:
        raise ValueError(f"Missing CGNS data at {node.name}")
    return np.asarray(node[" data"])


def _range(node, name):
    if name not in node:
        raise NotImplementedError(f"{node.name}: requires {name}; PointList is not supported")
    value = _data(node[name])
    # CGNS HDF5 stores the SIDS (IndexDimension,2) dimensions reversed.
    if value.shape != (2, 3):
        raise ValueError(f"{node[name].name}: expected a 3D PointRange")
    return np.asarray(value.T, dtype=np.int64)


def _families(node):
    primary = _children(node, "FamilyName_t")
    if len(primary) > 1:
        raise ValueError(f"{node.name}: multiple primary FamilyName nodes")
    return (_text(_data(primary[0])) if primary else None,
            tuple(_text(_data(child)) for child in _children(node, "AdditionalFamilyName_t")))


def _is_hdf5(path):
    # HDF5 signatures may follow a user block of 512 * 2**n bytes.
    size = path.stat().st_size
    with path.open("rb") as stream:
        offset = 0
        while offset + 8 <= size:
            stream.seek(offset)
            if stream.read(8) == b"\x89HDF\r\n\x1a\n":
                return True
            offset = 512 if offset == 0 else 2 * offset
    return False


@dataclass
class CGNSMesh:
    zones: list[Zone]
    source: str | None = None

    @classmethod
    def from_file(cls, filename, *, base=None, backend="auto"):
        """Read ADF with libcgns, or HDF5 with a capable libcgns/h5py.

        backend can be 'auto', 'cgns', or 'h5py'. ADF needs the locally
        compiled CGNS bridge; the h5py dependency is optional.
        """
        if backend not in ("auto", "cgns", "h5py"):
            raise ValueError("CGNS backend must be auto, cgns, or h5py")
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        if backend == "h5py":
            return cls._from_hdf5(path, base=base)
        from ._cgns_native import library, read_mesh
        try:
            api = library()
        except ImportError:
            if backend == "cgns" or not _is_hdf5(path):
                raise
            return cls._from_hdf5(path, base=base)
        if _is_hdf5(path) and not api.has_hdf5:
            if backend == "cgns":
                raise NotImplementedError("The linked CGNS library was built without HDF5 support")
            # 原生 CGNS 库可能只编译了 ADF；仅在能力不足时回退，解析错误仍直接报告。
            return cls._from_hdf5(path, base=base)
        return read_mesh(path, base=base)

    @classmethod
    def _from_hdf5(cls, filename, *, base=None):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Reading CGNS/HDF5 requires h5py or a CGNS library built with HDF5. "
                              "ADF files do not require h5py.") from exc

        path = Path(filename)
        if not h5py.is_hdf5(path):
            raise NotImplementedError("Expected CGNS/HDF5 for backend='h5py'; use backend='cgns' for ADF")
        zones = []
        with h5py.File(path, "r") as file:
            bases = _children(file, "CGNSBase_t")
            if base is not None:
                bases = [b for b in bases if b.name.rsplit("/", 1)[-1] == base]
            if len(bases) != 1:
                raise ValueError("Select exactly one CGNS base using base=...")
            b = bases[0]
            if tuple(_data(b).flat) != (3, 3):
                raise NotImplementedError("Native wrapper currently requires cell/physical dimension 3 (including extruded 2D grids)")
            family_types = {}
            for family in _children(b, "Family_t"):
                bcs = _children(family, "FamilyBC_t")
                if len(bcs) == 1:
                    family_types[family.name.rsplit("/", 1)[-1]] = _text(_data(bcs[0]))
            for z in _children(b, "Zone_t"):
                if "ZoneType" not in z or _text(_data(z["ZoneType"])) != "Structured":
                    raise NotImplementedError(f"{z.name}: only structured zones are supported")
                sizes = _data(z)
                if sizes.shape != (3, 3):
                    raise ValueError(f"{z.name}: invalid structured ZoneSize")
                dims = tuple(map(int, sizes[0]))
                if any(n < 2 for n in dims) or not np.array_equal(sizes[1], np.asarray(dims) - 1):
                    raise ValueError(f"{z.name}: invalid vertex/cell dimensions")
                grids = _children(z, "GridCoordinates_t")
                if len(grids) != 1:
                    raise ValueError(f"{z.name}: expected one GridCoordinates node")
                if _children(grids[0], "Rind_t"):
                    raise NotImplementedError("CGNS coordinate rind layers are not supported")
                coordinates = np.empty((*dims, 3), dtype=np.float64, order="F")
                for axis, name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
                    if name not in grids[0]:
                        raise ValueError(f"{z.name}: missing {name}")
                    values = _data(grids[0][name])
                    if values.shape != dims[::-1]:
                        raise ValueError(f"{z.name}: {name} shape does not match ZoneSize")
                    coordinates[..., axis] = values.transpose(2, 1, 0)
                if not np.isfinite(coordinates).all():
                    raise ValueError(f"{z.name}: nonfinite coordinate")
                zone = Zone(z.name.rsplit("/", 1)[-1], coordinates)
                zone.family, zone.additional_families = _families(z)
                for zb in _children(z, "ZoneBC_t"):
                    for bc in _children(zb, "BC_t"):
                        if "GridLocation" in bc and _text(_data(bc["GridLocation"])) != "Vertex":
                            raise NotImplementedError(f"{bc.name}: boundary GridLocation must be Vertex")
                        data = {}
                        for dataset in _children(bc, "BCDataSet_t"):
                            for bcdata in _children(dataset, "BCData_t"):
                                for array in _children(bcdata, "DataArray_t"):
                                    name = array.name.rsplit("/", 1)[-1]
                                    if name in data:
                                        raise ValueError(f"{bc.name}: ambiguous BC data {name}")
                                    data[name] = _data(array)
                        family, additional = _families(bc)
                        bc_type = _text(_data(bc))
                        if bc_type == "FamilySpecified":
                            parts = (family or zone.family or "").strip("/").split("/")
                            if len(parts) == 1 or (len(parts) == 2 and parts[0] == b.name.rsplit("/", 1)[-1]):
                                bc_type = family_types.get(parts[-1], bc_type)
                        zone.boundaries.append(Boundary(bc.name.rsplit("/", 1)[-1], bc_type,
                                                        _range(bc, "PointRange"), data, family, additional))
                for zg in _children(z, "ZoneGridConnectivity_t"):
                    if _children(zg, "GridConnectivity_t"):
                        raise NotImplementedError(f"{z.name}: general/overset connectivity is not supported")
                    for cn in _children(zg, "GridConnectivity1to1_t"):
                        if _children(cn, "GridConnectivityProperty_t"):
                            raise NotImplementedError(f"{cn.name}: periodic/averaged connections are not supported")
                        transform = _data(cn["Transform"]).ravel() if "Transform" in cn else np.arange(1, 4)
                        if sorted(abs(transform)) != [1, 2, 3]:
                            raise ValueError(f"{cn.name}: Transform must be a signed permutation of 1,2,3")
                        zone.connections.append(Connection(_text(_data(cn)), _range(cn, "PointRange"),
                                                           _range(cn, "PointRangeDonor"), transform.astype(int)))
                zones.append(zone)
        if not zones:
            raise ValueError("CGNS base contains no structured zones")
        if len({z.name for z in zones}) != len(zones):
            raise ValueError("Duplicate zone names")
        return cls(zones, str(path.resolve()))
