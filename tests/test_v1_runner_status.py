"""Runner failure boundaries using fakes only: no Gurobi model or solve."""

from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from intercity_delivery.v1.runner import run_mandatory
from tests.test_v1_mandatory import explicit_profile, order, tiny_config


def fake_optimizer(status=9, incumbent=False):
    model = SimpleNamespace(
        Params=SimpleNamespace(), Status=status, SolCount=int(incumbent),
        ObjBound=7.0, Runtime=0.0, NumVars=0, NumConstrs=0,
        ObjVal=12.0, MIPGap=0.0, optimize=Mock(), dispose=Mock(),
    )
    optimizer = SimpleNamespace(
        model=model, model_id="v1_mandatory_arc_wave_v1",
        semantic_status="author_confirmed_profile", traceability={}, groups={},
        build_model=Mock(), extract_schedule=Mock(return_value={}),
    )
    optimizer.build_model.return_value = optimizer
    return optimizer


@contextmanager
def isolated(optimizer):
    with patch("intercity_delivery.v1.runner.V1MandatoryOptimizer", return_value=optimizer), \
         patch("intercity_delivery.v1.runner._provenance", return_value={}):
        yield


class RunnerStatusTests(unittest.TestCase):
    def invoke(self):
        return run_mandatory(tiny_config(), [order()], explicit_profile(), 2)

    def assert_no_plan(self, result):
        self.assertFalse(result["has_incumbent"])
        self.assertFalse(result["optimal"])
        self.assertFalse(result["schedule_valid"])
        self.assertIsNone(result["total_cost"])
        self.assertIsNone(result["schedule"])
        self.assertIsNone(result["validation"])

    def test_build_exhausts_budget_without_optimize(self):
        optimizer = fake_optimizer(status=1)
        with isolated(optimizer), patch("intercity_delivery.v1.runner.time.monotonic",
                                       side_effect=[0.0, 3.0, 3.0, 3.0]):
            result = self.invoke()
        self.assertEqual(result["status_name"], "BUDGET_EXHAUSTED_DURING_BUILD")
        self.assertEqual(result["solver_time_limit_sec"], 0)
        self.assertTrue(result["budget_exceeded"])
        optimizer.model.optimize.assert_not_called()
        optimizer.model.dispose.assert_called_once()
        self.assert_no_plan(result)

    def test_timeout_without_incumbent_keeps_bound_only(self):
        optimizer = fake_optimizer()
        with isolated(optimizer):
            result = self.invoke()
        self.assertEqual(result["status_name"], "TIME_LIMIT")
        self.assertEqual(result["best_bound"], 7)
        self.assertIsNone(result["mip_gap"])
        optimizer.model.optimize.assert_called_once()
        optimizer.extract_schedule.assert_not_called()
        self.assert_no_plan(result)

    def test_missing_dependency_never_fabricates_solution(self):
        optimizer = fake_optimizer()
        optimizer.model = None
        optimizer.build_model.side_effect = ImportError("fake dependency unavailable")
        with isolated(optimizer):
            result = self.invoke()
        self.assertEqual(result["status_name"], "DEPENDENCY_UNAVAILABLE")
        self.assertTrue(any("fake dependency" in item for item in result["warnings"]))
        self.assert_no_plan(result)

    def test_validation_error_propagates_and_disposes(self):
        optimizer = fake_optimizer(status=2, incumbent=True)
        with isolated(optimizer), patch(
            "intercity_delivery.v1.validation.validate_schedule",
            side_effect=ValueError("deliberate independent verification failure"),
        ), self.assertRaisesRegex(ValueError, "independent verification failure"):
            self.invoke()
        optimizer.model.dispose.assert_called_once()


if __name__ == "__main__":
    unittest.main()
