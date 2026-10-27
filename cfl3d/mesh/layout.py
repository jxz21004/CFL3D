"""Local structured volume coordinates with explicit CGNS vertex identities."""
import numpy as np


class VolumeLayout:
    """Describe one rank's owned zones without retaining coordinate copies.

    Zones follow CGNS file order, vertices use i-fastest order, and each vertex
    has interleaved x,y,z components. Shared block vertices remain separate:
    surface coupling IDs and common-volume indices are different namespaces.
    Recreate this layout if native load balancing changes zone ownership.
    """

    def __init__(self, mesh):
        self.topology = tuple((z.name, tuple(z.dimensions)) for z in mesh.zones)
        self.owned = tuple(i for i, z in enumerate(mesh.zones) if z.coordinates is not None)
        counts = [int(np.prod(d)) for _, d in self.topology]
        self.offsets = np.cumsum([0] + counts, dtype=np.int64)
        self.size = sum(counts[i] for i in self.owned)

    def coordinates(self, mesh):
        """Return an independent (N,3) array in metres; never save x0 implicitly."""
        if (tuple((z.name, tuple(z.dimensions)) for z in mesh.zones) != self.topology
                or tuple(i for i, z in enumerate(mesh.zones)
                         if z.coordinates is not None) != self.owned):
            raise ValueError("Mesh topology or ownership changed; rebuild the volume layout")
        arrays = [mesh.zones[i].coordinates.reshape(-1, 3, order="F") for i in self.owned]
        return np.concatenate(arrays) if arrays else np.empty((0, 3))

    def component_indices(self):
        """Zero-based common CGNS coordinate DOFs required by IDWarp."""
        arrays = [np.arange(3*self.offsets[i], 3*self.offsets[i+1], dtype=np.int64)
                  for i in self.owned]
        return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)

    def unpack(self, coordinates):
        """Validate absolute coordinates and return detached per-zone arrays.

        This stages data only. It does not modify the mesh or native solver;
        installing coordinates requires a separate, validated geometry refresh.
        """
        xyz = np.asarray(coordinates)
        if xyz.shape != (self.size, 3) or xyz.dtype.kind not in 'fiu' or not np.isfinite(xyz).all():
            raise ValueError(f"Expected finite real coordinates with shape ({self.size}, 3)")
        result, start = {}, 0
        for i in self.owned:
            name, dimensions = self.topology[i]
            count = int(self.offsets[i+1]-self.offsets[i])
            result[name] = np.array(xyz[start:start+count].reshape((*dimensions, 3), order="F"),
                                    dtype=np.float64, copy=True)
            start += count
        return result
