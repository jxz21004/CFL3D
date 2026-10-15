"""Small CGNS fixtures plus the real NASA flat-plate mesh shipped in this repo."""
from pathlib import Path
import tarfile
import numpy as np
from cfl3d import Boundary, CGNSMesh, Connection, Zone

ROOT = Path(__file__).resolve().parents[2]


def box_mesh(multiblock=False):
    zones = []
    for g in range(2 if multiblock else 1):
        coordinates = np.empty((2, 9, 9, 3))
        for i, j, k in np.ndindex(coordinates.shape[:3]):
            coordinates[i, j, k] = [g + j / 8, -i, k / 8]
        z = Zone(f"zone{g}", coordinates)
        for axis in range(3):
            for side in (0, 1):
                r = np.array([[1, n] for n in z.dimensions])
                r[axis] = [1 if side == 0 else z.dimensions[axis]] * 2
                if multiblock and axis == 1 and side == 1-g:
                    donor = r.copy(); donor[axis] = [1 if side else 9] * 2
                    z.connections.append(Connection(f"zone{1-g}", r, donor, np.array([1, 2, 3])))
                else:
                    bc_type = "BCSymmetryPlane" if axis == 0 and side == 0 else "BCOutflowSupersonic" if axis == 0 else "BCFarfield"
                    z.boundaries.append(Boundary(f"face{axis}{side}", bc_type, r))
        zones.append(z)
    return CGNSMesh(zones)


def flatplate_mesh():
    with tarfile.open(ROOT / "docs/Cfl3dv6/2DTestcases/Flatplateyplus/Flatplateyplus.tar.gz") as tar:
        values = np.fromstring(tar.extractfile("Flatplateyplus/grid_y+4.6.fmt").read().decode(), sep=" ")
    dims = tuple(values[1:4].astype(int))
    xyz = values[4:].reshape((*dims, 3), order="F")
    z = Zone("flatplate", xyz)
    for axis in range(3):
        for side in (0, 1):
            r = np.array([[1, n] for n in dims]); r[axis] = [1 if not side else dims[axis]] * 2
            if axis == 2 and side == 0:
                r[1] = [1, 17]; z.boundaries.append(Boundary("symmetry", "BCSymmetryPlane", r.copy()))
                r[1] = [17, 65]; z.boundaries.append(Boundary("plate", "BCWallViscous", r.copy()))
            else:
                bc = "BCSymmetryPlane" if (axis, side) == (0, 0) else "BCOutflowSupersonic" if side and axis < 2 else "BCFarfield"
                z.boundaries.append(Boundary(f"face{axis}{side}", bc, r))
    return CGNSMesh([z])


def write_cgns(mesh, filename):
    """CGNS/HDF5 mapping fixture, checked separately with the official CGNS library."""
    import h5py
    with h5py.File(filename, "w") as f:
        f.attrs.create("name", np.bytes_("HDF5 MotherNode"), dtype="S33")
        f.attrs.create("label", np.bytes_("Root Node of HDF5 File"), dtype="S33")
        f.attrs.create("type", np.bytes_("MT"), dtype="S3")
        f.create_dataset(" format", data=np.frombuffer(b"IEEE_LITTLE_64\x00", dtype=np.int8))
        f.create_dataset(" hdf5version", data=np.frombuffer(b"HDF5 Version 1.10.10\x00", dtype=np.int8))

        def node(parent, name, label, data=None):
            group = parent.create_group(name, track_order=True)
            group.attrs.create("name", np.bytes_(name), dtype="S33")
            group.attrs.create("label", np.bytes_(label), dtype="S33")
            if data is None:
                group.attrs.create("type", np.bytes_("MT"), dtype="S3")
            else:
                if isinstance(data, str):
                    data = np.frombuffer(data.encode(), dtype=np.int8); typ = "C1"
                else:
                    data = np.asarray(data)
                    typ = "R8" if data.dtype == np.float64 else "R4" if data.dtype == np.float32 else "I8" if data.dtype == np.int64 else "I4"
                group.attrs.create("type", np.bytes_(typ), dtype="S3")
                group.create_dataset(" data", data=data)
            group.attrs["flags"] = np.array([1], dtype=np.int32)
            return group
        node(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t", np.array([4.5], np.float32))
        base = node(f, "Base", "CGNSBase_t", np.array([3, 3], np.int32))
        for z in mesh.zones:
            dims = np.array(z.dimensions, np.int64)
            zone = node(base, z.name, "Zone_t", np.array([dims, dims-1, [0, 0, 0]], np.int64))
            node(zone, "ZoneType", "ZoneType_t", "Structured")
            grid = node(zone, "GridCoordinates", "GridCoordinates_t")
            for a, name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
                node(grid, name, "DataArray_t", z.coordinates[..., a].transpose(2, 1, 0))
            zb = node(zone, "ZoneBC", "ZoneBC_t")
            for bc in z.boundaries:
                bn = node(zb, bc.name, "BC_t", bc.bc_type)
                node(bn, "PointRange", "IndexRange_t", bc.point_range.T.astype(np.int64))
            if z.connections:
                zg = node(zone, "ZoneGridConnectivity", "ZoneGridConnectivity_t")
                for i, cn in enumerate(z.connections):
                    c = node(zg, f"connect{i}", "GridConnectivity1to1_t", cn.donor)
                    node(c, "PointRange", "IndexRange_t", cn.point_range.T.astype(np.int64))
                    node(c, "PointRangeDonor", "IndexRange_t", cn.donor_range.T.astype(np.int64))
                    node(c, "Transform", "int[IndexDimension]", cn.transform.astype(np.int32))
    return Path(filename)
