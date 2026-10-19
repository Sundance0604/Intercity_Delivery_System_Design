"""Pure follow-up plan/identity/certificate tests; no solver imports."""

from dataclasses import asdict
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.specs import run_v1_followup_20261019 as follow


class FollowupTests(unittest.TestCase):
    def test_exact_30_jobs_and_references(self):
        jobs = follow.jobs()
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({j["id"] for j in jobs}), 30)
        self.assertEqual({j["seed"] for j in jobs}, set(range(4101, 4106)))
        self.assertEqual({j["n"] for j in jobs}, {40})
        self.assertTrue(all(j["algorithm"] == "full" for j in jobs))
        self.assertTrue(all(follow.reference_id(s) not in {j["id"] for j in jobs} for s in follow.SEEDS))

    def test_only_mechanism_minimum_changes(self):
        manifest = follow.old.freeze_snapshot()
        baseline = asdict(follow.config_for(manifest, {"mechanism": "T3", "minimum": 0.}))
        for job in follow.jobs():
            cfg = asdict(follow.config_for(manifest, job))
            self.assertEqual({k: v for k, v in cfg.items() if k not in ("service_mechanism", "min_direct_load_per_vehicle")},
                             {k: v for k, v in baseline.items() if k not in ("service_mechanism", "min_direct_load_per_vehicle")})

    def test_exact_budgets(self):
        self.assertEqual(follow.BUDGET, {"maximum_attempts": 30, "total_wall_sec": 300.,
            "logical_per_run_sec": 5., "hard_process_sec": 15., "Threads": 1, "Seed": 0})

    def test_difference_interval_uses_bounds_and_incumbents(self):
        left = {"status": "TIME_LIMIT", "metrics": {"completed_feasible": True, "total_cost": 100., "full_horizon_best_bound": 90.}}
        right = {"status": "TIME_LIMIT", "metrics": {"completed_feasible": True, "total_cost": 95., "full_horizon_best_bound": 85.}}
        self.assertEqual(follow.cost_difference_interval(left, right), [-15., 5.])
        left["metrics"]["completed_feasible"] = False
        self.assertIsNone(follow.cost_difference_interval(left, right))

    def test_no_certificate_for_missing_nonfinite_bound(self):
        left = {"status": "TIME_LIMIT", "metrics": {"completed_feasible": True, "total_cost": 100., "full_horizon_best_bound": None}}
        for missing in (None, float("nan"), float("inf"), True):
            left["metrics"]["full_horizon_best_bound"] = missing
            self.assertIsNone(follow.cost_difference_interval(left, left))

    def test_seven_predeclared_comparisons(self):
        self.assertEqual(len(follow.COMPARISONS), 7)
        self.assertIn(("T3_m01_minus_T2_m01", ("T2", .1), ("T3", .1)), follow.COMPARISONS)

    def test_readonly_analysis_missing_keeps_planned_denominators(self):
        manifest = {"source_content_sha256": "x", "followup_harness_sha256": "h", "prepare_wall_sec": 0}
        records = {j["id"]: {"job": j, "status": "NOT_RUN"} for j in follow.jobs()}
        records.update({follow.reference_id(s): {"status": "OPTIMAL"} for s in follow.SEEDS})
        with patch.object(follow, "records_for", return_value=(follow.ROOT / "nonexistent", manifest, records)):
            result = follow.analyze("test")
        for pair in result["pairs"]:
            self.assertEqual(len(pair["seeds"]), 5)
            self.assertEqual(pair["incumbent_difference_summary"]["n"], 0)
            self.assertEqual(pair["incumbent_difference_summary"]["planned_n"], 5)

    def test_additional_source_drift_fails_before_worker_access(self):
        manifest = {"source_content_sha256": "x", "source_manifest_sha256": "y",
            "followup_harness_sha256": "s", "additional_sources": {"protocol": "old"}}
        with patch.object(follow, "check_sources", return_value=manifest), \
             patch.object(follow.old, "read_json", return_value=manifest), \
             patch.object(follow, "sha", return_value="s"), \
             patch.object(follow, "additional_identities", return_value={"protocol": "changed"}):
            with self.assertRaisesRegex(ValueError, "script/test/protocol"):
                follow.load("test")


class FakeExecutionTests(unittest.TestCase):
    """Real exclusive filesystem bookkeeping, fake process and physical validator."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="followup_test_", dir=follow.ROOT / "simulation")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.manifest = follow.old.freeze_snapshot()
        self.manifest.update(prepare_wall_sec=0, environment={})
        self.planned = follow.jobs()[:2]
        self.instance = {"input_sha256": "input", "orders_sha256": "orders", "identity": {"orders": []}}
        self.patches = [patch.object(follow, "load", return_value=(self.directory, self.manifest)),
            patch.object(follow, "jobs", return_value=self.planned),
            patch.object(follow.old, "environment", return_value={}),
            patch.object(follow.old, "load_instance", return_value=self.instance),
            patch.object(follow, "check_sources", return_value=self.manifest)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def process(self, raw):
        def fake(command, timeout):
            identifier = command[command.index("--worker")+1]
            follow.old.write_new(self.directory / "jobs" / identifier / "raw.json", raw)
            return {"worker_exit_status": 0, "process_status": "EXITED", "stdout": "", "stderr": "", "parent_wall_sec": .01}
        return fake

    def test_time_limit_without_incumbent_continues_and_retains_no_cost(self):
        raw = {"status_name": "TIME_LIMIT", "completed": False, "schedule_valid": False}
        with patch.object(follow.old, "execute_attempt", side_effect=self.process(raw)) as process:
            result = follow.run("fake")
        self.assertEqual(process.call_count, 2)
        self.assertIsNone(result["stop_reason"])
        for job in self.planned:
            record = follow.old.read_json(self.directory / "jobs" / job["id"] / "record.json")
            self.assertFalse(record["metrics"]["completed_feasible"])
            self.assertIsNone(record["metrics"]["total_cost"])

    def test_independent_replay_error_stops_following_jobs(self):
        raw = {"status_name": "TIME_LIMIT", "completed": False, "schedule_valid": False, "schedule": {}}
        with patch.object(follow.old, "execute_attempt", side_effect=self.process(raw)) as process, \
             patch.object(follow, "replay_one", side_effect=ValueError("inventory invalid")):
            result = follow.run("fake")
        self.assertEqual(process.call_count, 1)
        self.assertIn("REPLAY_ERROR", result["stop_reason"])
        self.assertEqual(result["not_run"], [self.planned[1]["id"]])

    def test_budget_exhaustion_preserves_all_planned_without_process(self):
        self.manifest["prepare_wall_sec"] = 299.99
        with patch.object(follow.old, "execute_attempt") as process:
            result = follow.run("fake")
        process.assert_not_called()
        self.assertEqual(len(result["not_run"]), 2)
        self.assertEqual(result["attempted"], [])

    def test_existing_interrupted_attempt_is_never_retried(self):
        target = self.directory / "jobs" / self.planned[0]["id"]
        target.mkdir(parents=True)
        follow.old.write_new(target / "worker_claim.json", {"pid": 123})
        raw = {"status_name": "TIME_LIMIT", "completed": False, "schedule_valid": False}
        with patch.object(follow.old, "execute_attempt", side_effect=self.process(raw)) as process:
            result = follow.run("fake")
        self.assertEqual(process.call_count, 1)
        self.assertEqual(result["existing_retained"], [self.planned[0]["id"]])
        self.assertEqual(follow.old.read_json(target / "worker_claim.json"), {"pid": 123})

    def test_worker_claim_reentry_fails_before_solver_import(self):
        job = self.planned[0]
        target = self.directory / "jobs" / job["id"]
        target.mkdir(parents=True)
        follow.old.write_new(self.directory / "run.lock", {"pid": 123})
        request = {"job": job, "config_sha256": follow.old.fingerprint(asdict(follow.config_for(self.manifest, job))),
            "input_sha256": "input", "parent_pid": 123, "budget": follow.BUDGET,
            "followup_harness_sha256": follow.sha(follow.__file__)}
        follow.old.write_new(target / "request.json", request)
        follow.old.write_new(target / "worker_claim.json", {"pid": 123})
        with self.assertRaises(FileExistsError):
            follow.worker("fake", job["id"])


if __name__ == "__main__":
    unittest.main()
