import unittest
import numpy as np
from cfl3d.cgns import CGNSMesh, Zone, Boundary, Connection
from cfl3d.mesh._block_split import split_metadata, _pack, _range


def cube(name='cube', n=9):
    dims = (n, n, n)
    return Zone(name, None, [Boundary('face'+str(f), 'BCFarfield',
                _range(f, [1,n,1,n], dims)) for f in range(1,7)],
                coordinate_dimensions=dims)


class TestBlockSplit(unittest.TestCase):
    def test_metadata_and_owned_surface_signatures_match(self):
        from copy import deepcopy
        from cfl3d.surface_mesh import SurfaceTopology
        result = split_metadata(CGNSMesh([cube()]), 3, 2)
        owned = deepcopy(result)
        for z in owned.zones:
            z.coordinates = np.zeros((*z.dimensions, 3))
        self.assertEqual(SurfaceTopology(result).signature, SurfaceTopology(owned).signature)

    def test_native_cuts_preserve_cells_and_boundaries(self):
        mesh = CGNSMesh([cube()])
        result = split_metadata(mesh, 6, 2)
        self.assertEqual(len(result.zones), 6)
        self.assertEqual(sum(np.prod(np.array(z.dimensions)-1) for z in result.zones), 512)
        areas = {}
        for z in result.zones:
            for b in z.boundaries:
                ext = np.abs(np.diff(b.point_range, axis=1)).ravel()
                areas[b.name] = areas.get(b.name, 0) + np.prod(ext[ext>0])
        self.assertEqual(areas, {f'face{f}':64 for f in range(1,7)})
        _pack(result)  # checks reciprocal ranges and transforms after repeated cuts
        self.assertEqual(mesh.zones[0].dimensions, (9,9,9))

    def test_levels_never_downgraded(self):
        with self.assertRaisesRegex(ValueError, 'Cannot create'):
            split_metadata(CGNSMesh([cube()]), 2, 3)
        self.assertIsInstance(split_metadata(CGNSMesh([cube(n=17)]), 2, 3), CGNSMesh)

    def test_noop(self):
        mesh=CGNSMesh([cube()])
        self.assertIs(split_metadata(mesh, 1, 3), mesh)

    def test_swapped_reversed_connection(self):
        a,b=cube('a',17),cube('b',17)
        a.boundaries=a.boundaries[1:]; b.boundaries=b.boundaries[:-1]
        p=_range(1,[1,17,1,17],a.dimensions)
        q=_range(6,[17,1,1,17],b.dimensions)
        a.connections=[Connection('b',p,q,np.array([3,-1,2]))]
        mesh=CGNSMesh([a,b])
        result=split_metadata(mesh,7,2)
        _pack(result)
        # Check actual coordinates independently of the encode/decode roundtrip.
        from cfl3d.mesh._interface_validation import validate_interfaces
        ijk = np.indices((17,17,17))
        xyz = {'a':np.moveaxis(ijk,0,-1),
               'b':np.stack((ijk[2]-16,16-ijk[0],ijk[1]),axis=-1)}
        pending={}
        for z in result.zones:
            origin, _, _, lo, hi = result.split_sources[z.name]
            pending[z.name] = xyz[origin][tuple(slice(i,j) for i,j in zip(lo,hi))]
        validate_interfaces(result,pending,(0,)*len(result.zones),None,lambda op:op())
        self.assertEqual(sum(np.prod(np.array(z.dimensions)-1) for z in result.zones),8192)

if __name__=='__main__': unittest.main()
