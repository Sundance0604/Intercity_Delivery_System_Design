"""Pure standard-library source oracles; no optimization or V0 imports."""

from decimal import Decimal, localcontext
import math
import subprocess
import sys
import unittest

from intercity_delivery.v1 import (
    active_arcs, av_arcs, bhh_inverse, bhh_time, is_active,
    local_arcs, local_capacity,
)


class CapacityTests(unittest.TestCase):
    def test_hand_calculated_f_h(self):
        for q, theta in ((0, 0), (1, 2), (4, 6)):
            self.assertEqual(bhh_time(q, 1, 1), theta)
            self.assertEqual(bhh_inverse(theta, 1, 1), q)

    def test_inverse_linear(self):
        self.assertEqual(bhh_inverse(3, 2, 0), 1.5)
        self.assertEqual(bhh_inverse(0, 2, 0), 0)

    def test_inverse_substitution_original_formula(self):
        for a, b in ((0.2, 3), (2, 0), (1.5, 0.1)):
            for theta in (1e-20, 1e-8, 0.1, 2, 10000):
                with self.subTest(a=a, b=b, theta=theta):
                    q = bhh_inverse(theta, a, b)
                    # Independently substitute into the source f, not the API.
                    self.assertTrue(math.isclose(a*q+b*math.sqrt(q), theta,
                                                 rel_tol=1e-14))

    def test_small_theta_decimal_oracle(self):
        for theta in (1e-10, 1e-20, 1e-100):
            with self.subTest(theta=theta), localcontext() as context:
                context.prec = 250
                t = Decimal.from_float(theta)
                # High precision original (unrationalized) source expression.
                oracle = (((Decimal(1)+4*t).sqrt()-1)/2) ** 2
                self.assertTrue(math.isclose(bhh_inverse(theta, 1, 1),
                                            float(oracle), rel_tol=1e-14))
                self.assertGreater(bhh_inverse(theta, 1, 1), 0)

    def test_zero_tour_budget_and_vehicle_extension(self):
        for vehicles in (1, 2, 5):
            self.assertEqual(local_capacity(2, vehicles, 1, 1, 1, 10), 0)
            self.assertEqual(local_capacity(1, vehicles, 1, 1, 1, 10), 0)
        self.assertEqual(local_capacity(5, 0, 1, 1, 1, 10), 0)

    def test_cap_boundary_hand_oracle(self):
        self.assertAlmostEqual(local_capacity(1, 1, 1, 1, 0, 0.5),
                               (3-math.sqrt(5))/2, places=14)
        self.assertEqual(local_capacity(1, 2, 1, 1, 0, 0.5), 1)
        self.assertEqual(local_capacity(1, 3, 1, 1, 0, 0.5), 1.5)

    def test_linear_capacity_subtracts_stem(self):
        for v in (1, 2, 3):
            self.assertEqual(local_capacity(3, v, 2, 0, 0.5, 10), v)

    def test_capacity_never_exceeds_cap(self):
        for v in (0, 1, 3, 10):
            for duration in (0, 0.5, 1, 3, 100):
                result = local_capacity(duration, v, 1, 1, 0.5, 0.5)
                self.assertGreaterEqual(result, 0)
                self.assertLessEqual(result, v*0.5)

    def test_units_covariance(self):
        for factor in (60, 0.01):
            self.assertAlmostEqual(bhh_inverse(6*factor, factor, factor), 4)
            self.assertAlmostEqual(local_capacity(2*factor, 2, factor, factor,
                                                 0.5*factor, 10), 1)
            self.assertEqual(local_arcs(5, factor, factor, factor,
                                        0.5*factor, 1),
                             ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
                              (1, 4), (2, 3), (2, 4), (2, 5),
                              (3, 4), (3, 5), (4, 5)))

    def test_scalar_domains(self):
        functions = [(bhh_time, [1, 1, 1]), (bhh_inverse, [1, 1, 1]),
                     (local_capacity, [2, 1, 1, 1, 0, 1]),
                     (local_arcs, [3, 1, 1, 1, 0, 1])]
        for function, baseline in functions:
            for position in range(len(baseline)):
                for invalid in (True, False, math.nan, math.inf, -math.inf, -1,
                                "1", None):
                    args = baseline.copy()
                    args[position] = invalid
                    with self.subTest(fn=function.__name__, arg=position,
                                      invalid=invalid):
                        with self.assertRaises(ValueError):
                            function(*args)

    def test_strict_positive_and_integer_domains(self):
        calls = [(bhh_time, (1, 0, 1)), (bhh_inverse, (1, 0, 1)),
                 (local_capacity, (1, 1, 1, 1, 0, 0)),
                 (local_capacity, (1, 1.0, 1, 1, 0, 1)),
                 (local_capacity, (1, 0, 0, 1, 0, 1)),
                 (local_arcs, (1, 0, 1, 1, 0, 1)),
                 (local_arcs, (1, 1, 1, 1, 0, 0)),
                 (local_arcs, (1.0, 1, 1, 1, 0, 1))]
        for function, args in calls:
            with self.subTest(fn=function.__name__, args=args):
                with self.assertRaises(ValueError):
                    function(*args)

    def test_finite_intermediate_overflow_is_explicit(self):
        calls = [(bhh_time, (1e308, 2, 0)),
                 (bhh_time, (1, 1e308, 1e308)),
                 (bhh_inverse, (1e308, 1e308, 1)),
                 (bhh_inverse, (1e308, 1e-308, 0)),
                 (local_capacity, (1, 2, 1, 0, 0, 1e308)),
                 (local_capacity, (1, 1, 1, 0, 1e308, 1)),
                 (local_capacity, (1e308, 2, 1, 0, 0, 1)),
                 (local_arcs, (4, 1e-308, 1e308, 0, 0, 1))]
        for function, args in calls:
            with self.subTest(fn=function.__name__, args=args):
                with self.assertRaises(ValueError):
                    function(*args)
        with self.assertRaises(ValueError):
            bhh_time(10**1000, 1, 0)


class NetworkTests(unittest.TestCase):
    def test_local_hand_arcs_retain_zero_capacity(self):
        arcs = local_arcs(4, 1, 1, 0, 0.5, 1)
        self.assertEqual(arcs, ((0, 1), (0, 2), (1, 2), (1, 3),
                                (2, 3), (2, 4), (3, 4)))
        self.assertEqual(local_capacity(1, 1, 1, 0, 0.5, 1), 0)
        self.assertIn((0, 1), arcs)

    def test_ceil_noninteger(self):
        self.assertIn((0, 3), local_arcs(4, 1, 1, 0, 0.5, 1.25))

    def test_ceil_float_neighbors(self):
        for cap, expected in ((3, 3), (math.nextafter(3, math.inf), 4),
                              (math.nextafter(3, 0), 3)):
            with self.subTest(cap=cap):
                arcs = local_arcs(5, 1, 1, 0, 0, cap)
                self.assertEqual(max(j-i for i, j in arcs), expected)

    def test_av_hand_arcs_and_horizon_boundary(self):
        self.assertEqual(av_arcs(4, 2), ((0, 2), (1, 3), (2, 4)))
        self.assertEqual(av_arcs(4, 5), ())
        self.assertEqual(av_arcs(4, 4), ((0, 4),))

    def test_horizon_zero_extension(self):
        self.assertEqual(local_arcs(0, 1, 1, 0, 0, 1), ())
        self.assertEqual(av_arcs(0, 1), ())

    def test_half_open_and_same_time_release(self):
        self.assertEqual([is_active((1, 3), t) for t in range(4)],
                         [False, True, True, False])
        self.assertEqual(active_arcs([(3, 4), (1, 3)], 3), ((3, 4),))

    def test_deterministic_sorted_unique_output(self):
        first = local_arcs(4, 1, 1, 0, 0.5, 1)
        self.assertEqual(first, local_arcs(4, 1, 1, 0, 0.5, 1))
        self.assertEqual(first, tuple(sorted(set(first))))
        self.assertEqual(active_arcs([(1, 3), (0, 2), (1, 3)], 1),
                         ((0, 2), (1, 3)))

    def test_network_domains(self):
        for invalid in (True, False, math.nan, math.inf, -1, 1.5, "1", None):
            for args in ((invalid, 1), (1, invalid)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    av_arcs(*args)
            with self.assertRaises(ValueError):
                is_active((0, 1), invalid)
        with self.assertRaises(ValueError):
            av_arcs(1, 0)

    def test_arc_domains_validate_inactive_arcs_too(self):
        for arc in ((1, 1), (2, 1), (-1, 2), (True, 2), (1.0, 2),
                    (0, math.inf), (0,), (0, 1, 2), None):
            with self.subTest(arc=arc):
                with self.assertRaises(ValueError):
                    is_active(arc, 0)
                with self.assertRaises(ValueError):
                    active_arcs([arc], 0)

    def test_import_has_no_solver_or_old_config_dependency(self):
        script = ("import sys; import intercity_delivery.v1; "
                  "assert 'gurobipy' not in sys.modules; "
                  "assert 'intercity_delivery.configuration' not in sys.modules; "
                  "assert 'intercity_delivery.models' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
