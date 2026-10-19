"""Pure mechanism policy checks; no Gurobi environment or optimization."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import unittest

from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.flexible_configuration import V1FConfig, flexible_profile, validate_f_instance
from intercity_delivery.v1.flexible_optimizer import V1FlexibleOptimizer
from intercity_delivery.v1.flexible_runner import _metadata, _source_binding
from intercity_delivery.v1.flexible_validation import GROUPS, validate_flexible_schedule


def config(**kwargs):
    values = dict(T=6, t0_minutes=60., tau_periods=1,
                  N_manual={1: 1, 2: 1}, N_auto={1: 1, 2: 0},
                  a={1: 60., 2: 60.}, b={1: 0., 2: 0.}, rho={1: 0., 2: 0.},
                  M=10., M_auto=10., c_manual_hour=1., c_auto_hour=1., c_direct_hour=2.)
    values.update(kwargs)
    return V1FConfig(**values)


def blank():
    return {name: {} for name in GROUPS}


def direct(load=1., count=1):
    schedule = blank()
    end = 3 if load else 1
    schedule['w_direct'][0, end, '+'] = count
    schedule['u_direct'][0, end, '+', count] = 1
    if load:
        schedule['h_direct'][0, end, '+', 'one'] = load
    return schedule


class MechanismTests(unittest.TestCase):
    def replay(self, cfg, schedule, orders=(), partial=False):
        return validate_flexible_schedule(cfg, orders, flexible_profile(), schedule,
            completed=not partial, completed_horizon=1 if partial else cfg.T)

    def test_default_profile_preserves_zero_cargo_direct(self):
        evidence = self.replay(config(), direct(0))
        self.assertTrue(evidence['schedule_valid'])

    def test_mechanism_domain(self):
        for value in (True, False, None, 0, [], 't0', 'T4'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_f_instance(config(service_mechanism=value), (), flexible_profile())

    def test_minimum_domain(self):
        for value in (True, False, float('nan'), float('inf'), -.1, 10.1, '0.1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_f_instance(config(min_direct_load_per_vehicle=value), (), flexible_profile())
        validate_f_instance(config(min_direct_load_per_vehicle=10), (), flexible_profile())

    def test_t2_requires_explicit_positive_minimum(self):
        with self.assertRaisesRegex(ValueError, 'explicit positive'):
            validate_f_instance(config(service_mechanism='T2'), (), flexible_profile())
        validate_f_instance(config(service_mechanism='T2', min_direct_load_per_vehicle=.1), (), flexible_profile())

    def test_disabled_direct_ratio_conflict(self):
        for mechanism in ('T0', 'T1'):
            with self.subTest(mechanism=mechanism), self.assertRaisesRegex(ValueError, 'conflicts'):
                validate_f_instance(config(service_mechanism=mechanism, direct_ratio_min=.1), (), flexible_profile())

    def test_empty_matrix_full_and_partial(self):
        schedule = blank()
        schedule['empty_vehicles'][0, 1, '+'] = 1
        for mechanism in ('T0', 'T1', 'T2', 'T3'):
            cfg = config(service_mechanism=mechanism, min_direct_load_per_vehicle=.1 if mechanism == 'T2' else 0)
            for partial in (False, True):
                with self.subTest(mechanism=mechanism, partial=partial):
                    if mechanism in ('T0', 'T2'):
                        with self.assertRaisesRegex(ValueError, 'disables independent empty'):
                            self.replay(cfg, schedule, partial=partial)
                    else:
                        self.assertTrue(self.replay(cfg, schedule, partial=partial)['schedule_valid'])

    def test_disabled_direct_full_and_partial(self):
        for mechanism in ('T0', 'T1'):
            for partial in (False, True):
                with self.subTest(mechanism=mechanism, partial=partial), self.assertRaisesRegex(ValueError, 'disables direct'):
                    self.replay(config(service_mechanism=mechanism), direct(0), partial=partial)

    def test_minimum_rejects_zero_cargo_direct_full_and_partial(self):
        for mechanism in ('T2', 'T3'):
            for partial in (False, True):
                with self.subTest(mechanism=mechanism, partial=partial), self.assertRaisesRegex(ValueError, 'minimum direct load'):
                    self.replay(config(service_mechanism=mechanism, min_direct_load_per_vehicle=.1), direct(0), partial=partial)

    def test_positive_direct_and_minimum_per_vehicle(self):
        order = V1Order('one', '+', 1., 0, 6, 100.)
        cfg = config(service_mechanism='T2', min_direct_load_per_vehicle=.5)
        self.assertTrue(self.replay(cfg, direct(), (order,))['schedule_valid'])
        # Two vehicles need aggregate load >= 2 * minimum, including in-flight commitments.
        cfg = replace(cfg, N_manual={1: 2, 2: 0}, min_direct_load_per_vehicle=.6)
        for partial in (False, True):
            with self.subTest(partial=partial), self.assertRaisesRegex(ValueError, 'minimum direct load'):
                self.replay(cfg, direct(count=2), (order,), partial=partial)

    def test_tiny_positive_minimum_never_allows_zero_cargo(self):
        for partial in (False, True):
            for mechanism in ('T2', 'T3'):
                with self.subTest(partial=partial, mechanism=mechanism), self.assertRaisesRegex(ValueError, 'strictly positive'):
                    self.replay(config(service_mechanism=mechanism, min_direct_load_per_vehicle=1e-9),
                                direct(0), partial=partial)

    def test_source_binding_detects_drift_and_outside_paths(self):
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / 'experiments/specs/v1_source_manifest.json'
        contents = {root / 'source.tex': b'original'}
        entries = {'source.tex': {'sha256': hashlib.sha256(b'original').hexdigest(), 'size_bytes': 8}}
        def save():
            fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            contents[manifest_path] = json.dumps({'schema_version': 'v1_p0_source_manifest/1',
                'files': entries, 'content_sha256': fingerprint}).encode()
        with patch.object(Path, 'is_file', return_value=True), patch.object(Path, 'read_bytes', autospec=True,
                side_effect=lambda path: contents[path]):
            save()
            self.assertTrue(_source_binding(root)['current_sources_match'])
            contents[root / 'source.tex'] = b'changed!'
            binding = _source_binding(root)
            self.assertFalse(binding['current_sources_match'])
            self.assertTrue(any('bytes differ' in e for e in binding['verification_errors']))
            entries['../outside.tex'] = {'sha256': 'no', 'size_bytes': 0}
            save()
            binding = _source_binding(root)
            self.assertFalse(binding['current_sources_match'])
            self.assertTrue(any('outside project' in e for e in binding['verification_errors']))
            contents[manifest_path] = b'invalid'
            self.assertFalse(_source_binding(root)['current_sources_match'])

    def test_bounds_keep_mother_domain_keys(self):
        for mechanism in ('T0', 'T1', 'T2', 'T3'):
            obj = object.__new__(V1FlexibleOptimizer)
            obj.config = config(service_mechanism=mechanism, min_direct_load_per_vehicle=.1 if mechanism == 'T2' else 0)
            obj.groups = {name: {(0,): SimpleNamespace(UB=100.)} for name in GROUPS}
            obj._apply_mechanism_bounds()
            disabled = set()
            if mechanism in ('T0', 'T1'):
                disabled.update(('w_direct', 'u_direct', 'h_direct'))
            if mechanism in ('T0', 'T2'):
                disabled.add('empty_vehicles')
            for name, group in obj.groups.items():
                self.assertEqual(set(group), {(0,)})
                self.assertEqual(group[0,].UB, 0 if name in disabled else 100)

    def test_metadata_discloses_policy_and_source_scope(self):
        data = _metadata(config(), (), flexible_profile(), 'full')
        self.assertEqual(data['service_mechanism'], 'T3')
        self.assertTrue(data['mechanism_policy']['zero_cargo_direct_allowed'])
        self.assertIn('not certified', data['provenance']['source_binding']['scope'])
        self.assertTrue(any(p.endswith('final_manuscript/main.tex') for p in data['provenance']['source_sha256']))


if __name__ == '__main__':
    unittest.main()
