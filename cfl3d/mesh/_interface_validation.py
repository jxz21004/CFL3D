"""Update-only checks: no work is added to the native iteration loop."""
import numpy as np


def validate_interfaces(mesh, pending, owners, comm, checked):
    """Exchange point-matched interface coordinates, not global volume grids."""
    size = 1 if comm is None else comm.size
    names = {z.name: i for i, z in enumerate(mesh.zones)}

    def requests():
        messages = [[] for _ in range(size)]
        for zone in mesh.zones:
            if zone.name not in pending:
                continue
            for link in zone.connections:
                lo = np.min(link.point_range, axis=1)-1
                hi = np.max(link.point_range, axis=1)-1
                points = np.indices(tuple(hi-lo+1)).reshape(3, -1)+lo[:, None]
                donor = np.repeat((link.donor_range[:, 0]-1)[:, None], points.shape[1], axis=1)
                for axis, transform in enumerate(link.transform):
                    donor[abs(transform)-1] += np.sign(transform)*(points[axis]-(link.point_range[axis, 0]-1))
                values = pending[zone.name][tuple(points)]
                messages[owners[names[link.donor]]].append((link.donor, donor, values))
        return messages

    messages = checked(requests)
    received = messages if comm is None else comm.alltoall(messages)

    def compare():
        for part in received:
            for name, points, values in part:
                if not np.allclose(pending[name][tuple(points)], values, rtol=1e-10, atol=1e-10):
                    raise ValueError("Updated point-matched interface coordinates do not coincide")
    checked(compare)
