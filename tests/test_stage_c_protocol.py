import argparse
import unittest

from simulation.run_protocol import build_specs, parse_float_list


def stage_args(**overrides):
    values = {
        "stage": "stage_c",
        "order_counts": [100],
        "seeds": [7],
        "penalty_lost": 10.0,
        "penalty_values": [1.0, 2.0],
        "mechanisms": ["transshipment_only", "flexible", "direct_only"],
        "fleet_scales": [0.75, 1.25],
        "planning_periods": 24,
        "period_minutes": 60.0,
        "travel_time": 9,
        "prediction_horizon": 8,
        "rolling_step": 2,
        "extension_horizon": 6,
        "buffer_min": 0,
        "buffer_max": 5,
        "time_limit": 120,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class StageCProtocolTests(unittest.TestCase):
    def test_float_list_requires_positive_values(self):
        self.assertEqual(parse_float_list("0.75,1,1.25"), [0.75, 1.0, 1.25])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_float_list("0,1")

    def test_stage_c_builds_full_factorial_with_mechanism_bounds(self):
        specs = build_specs(stage_args())
        self.assertEqual(len(specs), 12)

        by_mechanism = {
            spec.scenario.split("stage_c_", 1)[1].split("_fleet", 1)[0]: spec
            for spec in specs
        }
        self.assertEqual(
            (
                by_mechanism["transshipment_only"].config.direct_ratio_min,
                by_mechanism["transshipment_only"].config.direct_ratio_max,
            ),
            (0.0, 0.0),
        )
        self.assertEqual(
            (
                by_mechanism["flexible"].config.direct_ratio_min,
                by_mechanism["flexible"].config.direct_ratio_max,
            ),
            (0.0, 1.0),
        )
        self.assertEqual(
            (
                by_mechanism["direct_only"].config.direct_ratio_min,
                by_mechanism["direct_only"].config.direct_ratio_max,
            ),
            (1.0, 1.0),
        )
        self.assertIn("C_N100_S7_P1p0_MT_F75", {spec.experiment_id for spec in specs})
        scaled = next(spec for spec in specs if spec.experiment_id.endswith("MF_F75"))
        self.assertEqual(scaled.config.N_manual, {1: 22, 2: 22})
        self.assertEqual(scaled.config.N_auto, {1: 11, 2: 11})

    def test_legacy_stage_keeps_legacy_identifiers(self):
        specs = build_specs(
            stage_args(
                stage="stage_a",
                penalty_values=None,
                mechanisms=["flexible"],
                fleet_scales=[1.0],
            )
        )
        self.assertEqual(specs[0].experiment_id, "STAGE_A_REAL_N100_S7")
        self.assertEqual(specs[0].scenario, "simulation_stage_a")


if __name__ == "__main__":
    unittest.main()
