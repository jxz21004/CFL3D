"""Final component loads survive close without stale or mutable cache exposure."""
import importlib.util
from unittest.mock import Mock
import unittest

if any(importlib.util.find_spec(name) is None for name in ('openmdao', 'mphys')):
    raise unittest.SkipTest('Optional OpenMDAO/MPhys dependencies are unavailable')

from cfl3d.mphys.mphys_cfl3d import CFL3DBuilder


class MPhysForceSummaryTests(unittest.TestCase):
    def session(self):
        builder = CFL3DBuilder({'input_grid': 'unused.cgns'})
        builder.surface = {}
        session = builder._session('cruise')
        session.functions = {'cl': 1., 'cd': .1, 'lift': 100., 'drag': 10.}
        solver = Mock()
        solver.getGroupNames.return_value = ['Wing', 'Aircraft']
        # These groups overlap. Total must not be their sum.
        def solution(name=None):
            prefix = '' if name is None else name + '_'
            return {prefix+'L': 100., prefix+'D': 10.,
                    'units': {prefix+'L': 'N', prefix+'D': 'N'}}
        solver.getSolution.side_effect = solution
        session.solver = solver
        builder._active = session
        return builder, session, solver

    def test_final_loads_survive_close_without_double_counting(self):
        builder, session, solver = self.session()
        expected = session.functions.copy()
        self.assertEqual(session.finish(), expected)
        solver.finalizeSolution.assert_called_once()
        solver.close.assert_called_once()
        self.assertIsNone(session.solver)
        summary = builder.get_force_summary('cruise')
        self.assertEqual(summary['total']['L'], 100.)
        self.assertEqual(summary['components']['Wing']['L'], 100.)
        self.assertEqual(summary['components']['Aircraft']['units']['D'], 'N')
        summary['components']['Wing']['L'] = -1.
        self.assertEqual(builder.get_force_summary('cruise')['components']['Wing']['L'], 100.)
        self.assertEqual(session.finish(), expected)
        solver.close.assert_called_once()

    def test_new_attempt_invalidates_previous_summary(self):
        builder, session, _ = self.session()
        session.finish()
        builder._active = object()  # Force an early failed attempt, before native work.
        with self.assertRaises(RuntimeError):
            session.advance([], 2.)
        with self.assertRaisesRegex(RuntimeError, 'successful scenario'):
            builder.get_force_summary('cruise')

    def test_unfinished_summary_rejected(self):
        builder, _, _ = self.session()
        with self.assertRaisesRegex(RuntimeError, 'successful scenario'):
            builder.get_force_summary('cruise')


if __name__ == '__main__':
    unittest.main()
