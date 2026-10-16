import ctypes as C
from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from cfl3d import CFL3D, CGNSMesh, SolverOptions
from cfl3d._cgns_native import library
from cfl3d.solution_io import wall_vertices
from cgns_fixtures import write_adf
from fixtures import box_mesh, flatplate_mesh
from test_surface_forces import grouped_plate


def record(stream, dtype):
    size = struct.unpack('=i', stream.read(4))[0]
    data = stream.read(size)
    assert struct.unpack('=i', stream.read(4))[0] == size
    return np.frombuffer(data, dtype=dtype)


class OptionsDictionaryTests(unittest.TestCase):
    def test_normalization_validation_and_copy(self):
        values = {'multigrid_levels': 2, 'limiter': [0,2,4],
                  'multigrid_stages': [{'iterations': 4, 'levels': 2, 'smoothing_steps': [1,2]}]}
        options = SolverOptions.from_dict(values)
        options.validate()
        self.assertEqual(options.limiter, (0,2,4))
        self.assertEqual(values['limiter'], [0,2,4])
        self.assertEqual(options.multigrid_stages[0].smoothing_steps, (1,2))
        for opts in ({'unknownOption': False}, {'writevolumesolution': 'false'},
                     {'writeTecplotSurfaceSolution': 0}, {'writeflowsolution': None}):
            with self.subTest(options=opts), self.assertRaises(ValueError):
                CFL3D(box_mesh(), opts)
        with self.assertRaisesRegex(ValueError, 'input_grid'):
            CFL3D(options={})
        with self.assertRaisesRegex(ValueError, 'not both'):
            CFL3D(box_mesh(), {'input_grid': 'another.cgns'})
        for method in ('writeVolumeSolution', 'writeTecplotSurfaceSolution', 'writeFlowSolution'):
            with self.assertRaisesRegex(RuntimeError, 'solve'):
                getattr(CFL3D(box_mesh()), method)()


class SolutionIOTests(unittest.TestCase):
    def test_disabled_output_deferred_export_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_adf(grouped_plate(), root/'input.cgns')
            options = dict(input_grid=path, output_directory=root/'disabled', turbulence='sst',
                           mach=.2, alpha=2., iterations=6, cfl=1., multigrid_levels=2,
                           freestream_pressure=90000., writevolumesolution=False,
                           writeTecplotSurfaceSolution=False, writeflowsolution=False)
            solver = CFL3D(options=options)
            result = solver.solve()
            output = result.output_directory
            for name in ('volume.cgns','surface.dat','flow.x','flow.q','restart.bin','plot3dg.bin','plot3dq.bin'):
                self.assertFalse((output/name).exists(), name)
            with np.load(output/'solution.npz') as saved:
                self.assertFalse(any(name.startswith('zone_') for name in saved.files))
                self.assertIn('history', saved.files)
            snapshot = {k:v.copy() for k,v in result.fields.items()}
            solver.writeVolumeSolution()
            solver.writeTecplotSurfaceSolution()
            solver.writeFlowSolution()
            self.check_volume(output/'volume.cgns', solver.mesh, snapshot, solver.options)
            self.check_surface(output/'surface.dat', solver, result)
            self.check_flow(output/'flow', solver.mesh, snapshot, solver.options, result.final_alpha)
            original = (output/'surface.dat').read_bytes()
            original_volume = (output/'flow.q').read_bytes()
            # A different core run and caller mutations cannot change the deferred snapshot.
            CFL3D(box_mesh(), dict(turbulence='euler', iterations=1,
                 writevolumesolution=False, writeTecplotSurfaceSolution=False)).solve(output_directory=root/'other')
            for field in result.fields.values():
                field[:] = 0
            result.final_alpha = 123.
            result.output_directory = root/'changed-by-caller'
            solver.mesh.zones[0].coordinates[:] = 0
            solver.writeTecplotSurfaceSolution()
            solver.writeFlowSolution()
            self.assertEqual((output/'surface.dat').read_bytes(), original)
            self.assertEqual((output/'flow.q').read_bytes(), original_volume)
            self.assertEqual(options['writevolumesolution'], False)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                solver.writeVolumeSolution(path)

    def check_volume(self, path, mesh, fields, options):
        loaded = CGNSMesh.from_file(path)
        for a,b in zip(mesh.zones,loaded.zones):
            np.testing.assert_array_equal(a.coordinates,b.coordinates)
            self.assertEqual([x.name for x in a.boundaries], [x.name for x in b.boundaries])
            self.assertEqual(len(a.connections),len(b.connections))
        api = library()
        api.bind('cg_field_read', [C.c_int]*4+[C.c_char_p,C.c_int,C.POINTER(api.index),C.POINTER(api.index),C.c_void_p])
        with api.open(path) as fn:
            for z, zone in enumerate(mesh.zones,1):
                q = fields[zone.name]
                lo,hi = (api.index*3)(1,1,1), (api.index*3)(*(n-1 for n in zone.dimensions))
                for i,name in enumerate(('Density','VelocityX','VelocityY','VelocityZ','Pressure','Temperature')):
                    array = np.empty(q.shape[:3],order='F')
                    api.call('cg_field_read',fn,1,z,1,name.encode(),api.constant('RealDouble'),lo,hi,array.ctypes.data)
                    expected = q[...,i] if i<5 else options.temperature*options.gamma*q[...,4]/q[...,0]
                    np.testing.assert_array_equal(array,expected)

    def check_surface(self, path, solver, result):
        text = path.read_text()
        self.assertIn('"y+"',text)
        self.assertEqual(text.count('ZONE T='),2)
        self.assertIn('VARLOCATION=([4-9]=CELLCENTERED)', text)
        self.assertNotIn('symmetry',text)
        integrated = np.zeros(3)
        pressure = np.zeros(3)
        for zone in solver.mesh.zones:
            bc = next(b for b in zone.boundaries if b.name=='plate')
            xyz = wall_vertices(zone,bc)
            normals = .5*np.cross(xyz[1:,1:]-xyz[:-1,:-1],xyz[:-1,1:]-xyz[1:,:-1])
            areas = np.linalg.norm(normals,axis=-1)
            wall = result.wall_fields[(zone.name,bc.name)]
            self.assertTrue((wall[...,1]>0).all())
            self.assertTrue((wall[...,5]>0).all())
            integrated += np.sum(wall[...,2:5]*areas[...,None],axis=(0,1))
            pressure -= np.sum(wall[...,0,None]*normals,axis=(0,1))
            # Validate first-cell y+ from the exported wall state, rather than just positivity.
            q = result.fields[zone.name][:,int(bc.point_range[1,0])-1:int(bc.point_range[1,1])-1,0]
            top = zone.coordinates[:,int(bc.point_range[1,0])-1:int(bc.point_range[1,1]),1]
            distance = np.abs((top[...,2]-xyz[...,2])[:-1,:-1])/2
            tr = wall[...,1]/solver.options.temperature
            rho = (1+wall[...,0]*.5*solver.options.gamma*solver.options.mach**2)/tr
            cbar = 198.6/(solver.options.temperature*1.8)
            mu = tr**1.5*(1+cbar)/(tr+cbar)
            expected = distance*solver.options.reynolds*np.sqrt(.5*np.linalg.norm(wall[...,2:5],axis=-1)*rho)/mu
            np.testing.assert_allclose(wall[...,5],expected,rtol=2e-12)
        wind = np.array([np.cos(np.deg2rad(solver.options.alpha)),0,np.sin(np.deg2rad(solver.options.alpha))])
        self.assertAlmostEqual(integrated@wind/solver.options.sref,result.coefficients['CDv'],places=12)
        self.assertAlmostEqual(pressure@wind/solver.options.sref,result.coefficients['CDp'],places=12)
        solver.writeTecplotSurfaceSolution(path.parent/'forward.dat', 'Forward')
        self.assertEqual((path.parent/'forward.dat').read_text().count('ZONE T='),1)

    def check_flow(self, prefix, mesh, fields, options, alpha):
        i2d = all(z.dimensions[0]==2 for z in mesh.zones)
        for suffix in ('.x','.q'):
            with Path(str(prefix)+suffix).open('rb') as stream:
                self.assertEqual(record(stream,np.int32).tolist(),[len(mesh.zones)])
                dims = record(stream,np.int32).reshape(-1,2 if i2d else 3)
                for index, zone in enumerate(mesh.zones):
                    np.testing.assert_array_equal(dims[index],np.array(zone.dimensions)[1 if i2d else 0:]-1)
                    if suffix=='.q':
                        np.testing.assert_array_equal(record(stream,np.float64),[options.mach,alpha,options.reynolds,0])
                    raw = record(stream,np.uint8).tobytes()
                    npoints = int(np.prod(dims[index]))
                    if suffix=='.x':
                        np.testing.assert_array_equal(np.frombuffer(raw[-4*npoints:],dtype=np.int32),np.ones(npoints,dtype=np.int32))
                        raw=raw[:-4*npoints]
                    array = np.frombuffer(raw,dtype=np.float64)
                    self.assertTrue(np.isfinite(array).all())
                    self.assertEqual(len(array),np.prod(dims[index])*((4 if i2d else 5) if suffix=='.q' else (2 if i2d else 3)))
                    if suffix=='.q':
                        q = fields[zone.name]
                        energy = q[...,4]/(options.gamma-1)+.5*q[...,0]*np.sum(q[...,1:4]**2,axis=-1)
                        np.testing.assert_array_equal(array[-energy.size:],energy.ravel(order='F'))
                self.assertEqual(stream.read(),b'')

    def test_independent_flags_and_isothermal_wall(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = grouped_plate()
            options = SolverOptions(turbulence='sst',mach=.2,iterations=3,cfl=1.,
                writevolumesolution=False, writeflowsolution=True, temperature=280.)
            solver = CFL3D(mesh,options,boundary_overrides={(z.name,'plate'):(2004,(1.1,0.)) for z in mesh.zones})
            result=solver.solve(output_directory=root/'flow')
            self.assertFalse((root/'flow/volume.cgns').exists())
            self.assertTrue((root/'flow/flow.q').exists())
            self.assertTrue((root/'flow/surface.dat').exists())
            for wall in result.wall_fields.values():
                np.testing.assert_allclose(wall[...,1],308.,atol=1e-12)
            solver=CFL3D(mesh,replace(options,writevolumesolution=True,writeflowsolution=False,writeTecplotSurfaceSolution=False))
            solver.solve(output_directory=root/'volume')
            self.assertTrue((root/'volume/volume.cgns').exists())
            self.assertFalse((root/'volume/surface.dat').exists())
            self.assertFalse((root/'volume/flow.q').exists())

    def test_three_dimensional_i_j_k_wall_faces(self):
        with tempfile.TemporaryDirectory() as directory:
            for order in ((0,1,2),(1,2,0),(2,0,1)):
                mesh = flatplate_mesh()
                z = mesh.zones[0]
                z.coordinates = np.linspace(z.coordinates[0],z.coordinates[-1],5)
                for boundary in z.boundaries:
                    boundary.point_range[0][boundary.point_range[0]==2]=5
                    if boundary.point_range[0,0]==boundary.point_range[0,1]:
                        boundary.bc_type='BCSymmetryPlane'
                    boundary.point_range=boundary.point_range[list(order)].copy()
                z.coordinates=z.coordinates.transpose((*order,3)).copy()
                options = SolverOptions(turbulence='sst',mach=.2,alpha=2.,iterations=2,cfl=1.,
                    writevolumesolution=False,writeTecplotSurfaceSolution=False)
                s = CFL3D(mesh,options,boundary_overrides={('flatplate','plate'):(2004,(1.05,0.))})
                result=s.solve(output_directory=Path(directory)/''.join(map(str,order)))
                wall=result.wall_fields[('flatplate','plate')]
                np.testing.assert_allclose(wall[...,1],options.temperature*1.05,atol=1e-12)
                self.assertTrue((wall[...,5]>0).all())
                xyz=wall_vertices(z,next(b for b in z.boundaries if b.name=='plate'))
                area=np.linalg.norm(.5*np.cross(xyz[1:,1:]-xyz[:-1,:-1],xyz[:-1,1:]-xyz[1:,:-1]),axis=-1)
                integrated=np.sum(wall[...,2:5]*area[...,None],axis=(0,1))
                wind=np.array([np.cos(np.deg2rad(options.alpha)),0,np.sin(np.deg2rad(options.alpha))])
                self.assertAlmostEqual(integrated@wind,result.coefficients['CDv'],places=12)
                s.writeFlowSolution()
                self.check_flow(result.output_directory/'flow',mesh,result.fields,options,result.final_alpha)


if __name__ == '__main__':
    unittest.main()
