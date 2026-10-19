"""Deterministic wall-time partitions using fake models; no Gurobi calls."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from intercity_delivery.v1.flexible_configuration import V1FConfig, flexible_profile
from intercity_delivery.v1.flexible_runner import run_flexible, run_flexible_rolling
from intercity_delivery.v1.flexible_rolling import RollingOptions, PHASES, PhaseTimer, mother_domain_sha256
from intercity_delivery.v1.flexible_validation import GROUPS, validate_flexible_schedule


def config():
    return V1FConfig(T=2, t0_minutes=60, tau_periods=1,
        N_manual={1: 1, 2: 1}, N_auto={1: 1, 2: 1},
        a={1: 60., 2: 60.}, b={1: 0., 2: 0.}, rho={1: 0., 2: 0.},
        M=1., M_auto=1., c_manual_hour=1., c_auto_hour=1., c_direct_hour=1.)


class Clock:
    value = 0.
    def now(self):
        return self.value
    def advance(self, amount):
        self.value += amount


class FakeOptimizer:
    def __init__(self, clock, incumbent=True, build_seconds=.3):
        self.clock = clock
        self.build_seconds = build_seconds
        self.groups = {n: {} for n in GROUPS}
        self.model = SimpleNamespace(Params=SimpleNamespace(), Status=2, SolCount=int(incumbent),
            ObjVal=0., ObjBound=0., MIPGap=0., Runtime=.4, NumVars=10, NumBinVars=2,
            NumIntVars=4, NumConstrs=20, optimize=lambda: clock.advance(.4),
            dispose=lambda: clock.advance(.01), setParam=lambda *args: None)
    def build_model(self):
        self.clock.advance(self.build_seconds)
        return self
    def configure_window(self, *args):
        self.clock.advance(.1)
    def extract_schedule(self):
        self.clock.advance(.05)
        return {name: {} for name in GROUPS}


class InstrumentationTests(unittest.TestCase):
    def assert_partition(self, row):
        self.assertEqual(set(row['phase_timing_sec']), set(PHASES))
        self.assertTrue(all(v >= 0 for v in row['phase_timing_sec'].values()))
        self.assertAlmostEqual(sum(row['phase_timing_sec'].values()), row['elapsed_sec'])

    def test_full_partitions_counts_and_legacy_fields(self):
        clock = Clock()
        def metadata(*args):
            clock.advance(.2)
            return {}
        fake = FakeOptimizer(clock)
        with patch('time.monotonic', clock.now), patch('intercity_delivery.v1.flexible_runner._metadata', metadata), \
                patch('intercity_delivery.v1.flexible_optimizer.V1FlexibleOptimizer', return_value=fake):
            result = run_flexible(config(), (), flexible_profile(), 10)
        self.assert_partition(result)
        self.assertAlmostEqual(result['phase_timing_sec']['build'], .3)
        self.assertAlmostEqual(result['phase_timing_sec']['solve'], .4)
        self.assertAlmostEqual(result['phase_timing_sec']['setup_provenance'], .2)
        self.assertAlmostEqual(result['phase_timing_sec']['extract_commit'], .05)
        self.assertAlmostEqual(result['phase_timing_sec']['other'], .01)
        self.assertEqual(result['continuous_variable_count'], 6)
        self.assertEqual(result['binary_variable_count'], 2)
        self.assertEqual(result['constraint_count'], 20)
        self.assertEqual(result['total_cost'], 0)
        self.assertEqual(result['solver_runtime_sec'], .4)
        self.assertEqual(result['solver_optimize_attempts'], 1)

    def test_rolling_window_and_run_partitions_are_not_double_counted(self):
        clock = Clock()
        def candidate(*args):
            clock.advance(.2)
            return {'candidates': {}, 'statistics': {}}
        with patch('time.monotonic', clock.now), patch('intercity_delivery.v1.flexible_runner._metadata', return_value={}), \
                patch('intercity_delivery.v1.flexible_rolling.prune_candidates', candidate), \
                patch('intercity_delivery.v1.flexible_rolling.V1FlexibleOptimizer', side_effect=lambda *a, **kw: FakeOptimizer(clock)):
            result = run_flexible_rolling(config(), (), RollingOptions(2, 1, 0, True), 10)
        self.assertTrue(result['completed'])
        self.assert_partition(result)
        self.assertEqual(len(result['windows']), 2)
        self.assertEqual(result['solver_optimize_attempts'], 2)
        for window in result['windows']:
            self.assert_partition(window)
            self.assertEqual(window['continuous_variable_count'], 6)
            self.assertEqual(window['binary_variable_count'], 2)
            self.assertEqual(window['bound_scope'], 'window_local_only')
            self.assertAlmostEqual(window['phase_timing_sec']['candidate'], .2)
        self.assertAlmostEqual(result['phase_timing_sec']['candidate'], .4)
        self.assertAlmostEqual(result['phase_timing_sec']['build'], .6)
        self.assertAlmostEqual(result['phase_timing_sec']['solve'], .8)
        self.assertAlmostEqual(result['elapsed_sec'], sum(w['elapsed_sec'] for w in result['windows']))
        self.assertIsNone(result['best_bound'])

    def test_no_incumbent_retains_partial_status_and_timing(self):
        clock = Clock()
        with patch('time.monotonic', clock.now), patch('intercity_delivery.v1.flexible_runner._metadata', return_value={}), \
                patch('intercity_delivery.v1.flexible_rolling.V1FlexibleOptimizer', side_effect=lambda *a, **kw: FakeOptimizer(clock, False)):
            result = run_flexible_rolling(config(), (), RollingOptions(2, 1, 0), 10)
        self.assertFalse(result['completed'])
        self.assertEqual(result['status_name'], 'PARTIAL_NO_INCUMBENT')
        self.assertIsNone(result['total_cost'])
        self.assert_partition(result)
        self.assert_partition(result['windows'][0])

    def test_build_exhaustion_keeps_elapsed_build_phase(self):
        clock = Clock()
        with patch('time.monotonic', clock.now), patch('intercity_delivery.v1.flexible_runner._metadata', return_value={}), \
                patch('intercity_delivery.v1.flexible_rolling.V1FlexibleOptimizer', side_effect=lambda *a, **kw: FakeOptimizer(clock, build_seconds=2)):
            result = run_flexible_rolling(config(), (), RollingOptions(2, 1, 0), 1)
        self.assertEqual(result['status_name'], 'PARTIAL_TIME_LIMIT')
        self.assertFalse(result['completed'])
        self.assertTrue(result['budget_exceeded'])
        self.assertAlmostEqual(result['phase_timing_sec']['build'], 2)
        self.assertEqual(result['phase_timing_sec']['solve'], 0)
        self.assertEqual(result['solver_optimize_attempts'], 0)
        self.assert_partition(result)

    def test_phase_exception_still_records_consumed_time(self):
        clock = Clock()
        timer = PhaseTimer(clock.now())
        with patch('time.monotonic', clock.now):
            with self.assertRaises(ValueError):
                with timer.measure('build'):
                    clock.advance(.3)
                    raise ValueError('test')
            elapsed, phases = timer.finish()
        self.assertAlmostEqual(elapsed, .3)
        self.assertAlmostEqual(phases['build'], .3)

    def test_residual_zero_for_slack_and_exact_constraints(self):
        evidence = validate_flexible_schedule(config(), (), flexible_profile(),
            {name: {} for name in GROUPS}, reported_objective=0.)
        self.assertEqual(evidence['max_constraint_violation'], 0.)
        self.assertEqual(evidence['constraint_violation_by_check']['manual city inventory'], 0.)
        self.assertGreater(evidence['constraint_comparison_count_by_check']['manual city inventory'], 0)
        self.assertEqual(evidence['constraint_violation_units_by_check']['objective mismatch'], 'currency')
        self.assertIn('mixed-units', evidence['constraint_violation_scope'])

    def test_equality_residual_records_either_signed_direction_below_tolerance(self):
        for reported in (-5e-7, 5e-7):
            with self.subTest(reported=reported):
                evidence = validate_flexible_schedule(config(), (), flexible_profile(),
                    {name: {} for name in GROUPS}, reported_objective=reported)
                self.assertEqual(evidence['constraint_violation_by_check']['objective mismatch'], 5e-7)
                self.assertEqual(evidence['max_constraint_violation'], 5e-7)
                self.assertEqual(evidence['constraint_comparison_count_by_check']['objective mismatch'], 2)

    def test_residual_does_not_mask_over_tolerance_or_nonfinite(self):
        for reported in (2e-6, -2e-6, float('nan'), float('inf')):
            with self.subTest(reported=reported), self.assertRaises(ValueError):
                validate_flexible_schedule(config(), (), flexible_profile(),
                    {name: {} for name in GROUPS}, reported_objective=reported)

    def test_mother_domain_signature_excludes_cargo_values_and_order(self):
        groups = {name: {} for name in GROUPS}
        groups['x_manual'] = {(0, 1, 1, '+'): object(), (1, 2, 2, '-'): object()}
        signature = mother_domain_sha256(groups)
        groups['g_manual'][(0, 1, 1, '+', 'one')] = object()
        groups['x_manual'] = dict(reversed(list(groups['x_manual'].items())))
        groups['x_manual'][0, 1, 1, '+'] = None
        self.assertEqual(signature, mother_domain_sha256(groups))
        groups['g_manual'].clear()
        self.assertEqual(signature, mother_domain_sha256(groups))
        groups['w_direct'][0, 1, '+'] = None
        self.assertNotEqual(signature['vehicle_selector_domain_sha256'], mother_domain_sha256(groups)['vehicle_selector_domain_sha256'])
        self.assertEqual(signature['vehicle_selector_domain_counts']['x_manual'], 2)


if __name__ == '__main__':
    unittest.main()
