"""Bounded pilot bookkeeping and flow metrics; never import/initialize Gurobi."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.specs import run_v1_p1 as harness
from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.experiment_metrics import mean_ci95, summarize_result, weighted_quantile


def workspace_temp():
    # Windows-managed sandbox may deny files inside the OS's temporary folder.
    # Keep disposable fixtures under the explicitly writable workspace instead.
    temporary = tempfile.TemporaryDirectory(prefix="v1_p1_test_", dir=harness.ROOT / "simulation")
    assert Path(temporary.name).resolve().is_relative_to((harness.ROOT / "simulation").resolve())
    return temporary


def decisions():
    return {"schema_version": "v1_p0_decisions/1",
            "unit": {"status": "AUTHOR_CONFIRMED", "choice": "one_shipment_one_synthetic_job_equivalent"},
            "direct_endpoint": {"status": "AUTHOR_CONFIRMED", "choice": "aggregate_destination_service_node_original_formula"},
            "t2_policy": {"status": "AUTHOR_CONFIRMED", "choice": "minimum_average_direct_load", "minimum_per_vehicle": .1}}


def manifest():
    return {"design": harness.read_json(harness.ROOT / "experiments/specs/v1_p0_capacity_design.json"),
            "decisions": decisions(), "source_content_sha256": "content", "source_manifest_sha256": "manifest",
            "sampling_seconds": 1., "budget": harness.BUDGET}


class PilotPlanTests(unittest.TestCase):
    def test_counts_and_reused_references(self):
        self.assertEqual(len(harness.job_list("pilot")), 45)
        self.assertEqual(len(harness.job_list("mechanism")), 20)
        self.assertEqual(len(harness.job_list("sensitivity")), 20)
        jobs = harness.job_list("all")
        self.assertEqual(len(jobs), 75)
        self.assertEqual(len({j["id"] for j in jobs}), 75)
        self.assertEqual({j["seed"] for j in jobs}, set(range(4101, 4106)))
        self.assertLessEqual(max(j["n"] for j in jobs), 40)

    def test_job_budgets(self):
        for job in harness.job_list("all"):
            expected = 5 if job in harness.job_list("pilot") else 3
            self.assertEqual(harness.job_budget(job), expected)

    def test_job_config_changes_only_disclosed_policy(self):
        config = harness.job_config(manifest(), {"mechanism": "T2", "minimum": .05})
        self.assertEqual(config.min_direct_load_per_vehicle, .05)
        self.assertEqual(config.tau_periods, 9)
        self.assertEqual(config.service_mechanism, "T2")

    def test_path_escape_and_exclusive_creation(self):
        for invalid in ("../old", "x/y", "x\\y", "", "."):
            with self.assertRaises(ValueError):
                harness.batch_path(invalid)
        with workspace_temp() as temp:
            path = Path(temp) / "fixed.json"
            harness.write_new(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                harness.write_new(path, {"a": 2})
            self.assertEqual(harness.read_json(path), {"a": 1})

    def test_instance_mutation_is_rejected(self):
        with workspace_temp() as temp:
            directory = Path(temp)
            path = directory / "input.json"
            identity = {"orders": [asdict(V1Order("x", "+", 1, 0, 24, 100))]}
            payload = {"identity": identity, "input_sha256": harness.fingerprint(identity),
                       "orders_sha256": harness.fingerprint(identity["orders"])}
            harness.write_new(path, payload)
            entries = {"instances": {"one": {"path": "input.json", "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "input_sha256": payload["input_sha256"]}}}
            self.assertEqual(harness.load_instance(directory, entries, "one"), payload)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                harness.load_instance(directory, entries, "one")

    def test_micro_requires_five_real_pass_flags_and_matching_sources(self):
        with workspace_temp() as temp:
            path = Path(temp) / "micro.json"
            good = {"all_passed": True, "runs": [{"passed": True} for _ in range(5)],
                    "source_binding": {"manifest_sha256": "manifest", "manifest": {"content_sha256": "content"}}, "elapsed_sec": 1.}
            for override, accepted in (({}, True), ({"runs": [{"passed": True}]*4}, False),
                                       ({"runs": [{"passed": False}]*5}, False),
                                       ({"elapsed_sec": None}, False), ({"source_content_sha256": "wrong"}, False)):
                path.write_text(json.dumps({**good, **override}), encoding="utf-8")
                if accepted:
                    self.assertEqual(harness.require_micro(path, manifest())["evidence"]["elapsed_sec"], 1.)
                else:
                    with self.assertRaises(ValueError):
                        harness.require_micro(path, manifest())


class ProcessTests(unittest.TestCase):
    def test_timeout_rechecks_time_spent_collecting_metadata(self):
        with patch.object(harness.time, "monotonic", return_value=106.):
            # Initial 16-second allowance lost 6 seconds to registration.
            self.assertAlmostEqual(harness.remaining_process_timeout(16., 100.), 9.9)
        with patch.object(harness.time, "monotonic", return_value=117.):
            self.assertEqual(harness.remaining_process_timeout(16., 100.), 0.)
        with patch.object(harness.time, "monotonic", return_value=100.):
            self.assertEqual(harness.remaining_process_timeout(600., 100.), 15.)

    def test_hard_timeout_retains_logs_and_has_no_cost(self):
        def timeout(*args, **kwargs):
            self.assertEqual(kwargs["timeout"], 15)
            raise subprocess.TimeoutExpired(args[0], 15, output=b"progress", stderr=b"warning")
        output = harness.execute_attempt(["fake"], 15, process_run=timeout)
        self.assertEqual(output["process_status"], "TIMEOUT")
        self.assertEqual(output["stdout"], "progress")
        self.assertEqual(output["stderr"], "warning")
        self.assertNotIn("total_cost", output)

    def test_failed_process_retains_exit_status(self):
        output = harness.execute_attempt(["fake"], 15,
            process_run=lambda *a, **k: subprocess.CompletedProcess(a[0], 4, "out", "license failed"))
        self.assertEqual(output["worker_exit_status"], 4)
        self.assertEqual(output["stderr"], "license failed")

    def test_spawn_error_is_explicit(self):
        def missing(*args, **kwargs):
            raise FileNotFoundError("missing interpreter")
        self.assertEqual(harness.execute_attempt(["missing"], 15, process_run=missing)["process_status"], "SPAWN_ERROR")

    def run_fixture(self, directory):
        jobs = harness.job_list("pilot")[:2]
        identity = {"orders": [asdict(V1Order("x", "+", 1, 0, 24, 100))]}
        instance = {"input_sha256": "input", "orders_sha256": "orders", "identity": identity}
        micro = {"path": "micro", "sha256": "microhash", "evidence": {"elapsed_sec": 1.}}
        return jobs, instance, micro

    def test_budget_exhaustion_keeps_not_run_and_no_solve(self):
        with workspace_temp() as temp:
            directory = Path(temp)
            jobs, instance, micro = self.run_fixture(directory)
            (directory / "sessions").mkdir()
            harness.write_new(directory / "sessions/old.json", {"wall_sec": 599, "micro_sha256": "microhash"})
            with patch.object(harness, "load_batch", return_value=(directory, manifest())), \
                    patch.object(harness, "require_micro", return_value=micro), \
                    patch.object(harness, "job_list", return_value=jobs), \
                    patch.object(harness, "execute_attempt") as execute:
                result = harness.run_batch("fake", "pilot", 600, "micro")
            execute.assert_not_called()
            self.assertEqual(len(result["budget_not_run"]), 2)
            self.assertEqual(result["cumulative_wall_before_sec"], 601)

    def test_worker_error_stops_later_jobs_without_retry(self):
        with workspace_temp() as temp:
            directory = Path(temp)
            jobs, instance, micro = self.run_fixture(directory)
            outcome = {"process_status": "EXITED", "worker_exit_status": 1,
                       "stdout": "", "stderr": "validator error", "parent_wall_sec": .1}
            with patch.object(harness, "load_batch", return_value=(directory, manifest())), \
                    patch.object(harness, "require_micro", return_value=micro), \
                    patch.object(harness, "job_list", return_value=jobs), \
                    patch.object(harness, "load_instance", return_value=instance), \
                    patch.object(harness, "environment", return_value={}), \
                    patch.object(harness, "execute_attempt", return_value=outcome) as execute:
                result = harness.run_batch("fake", "pilot", 600, "micro")
                self.assertEqual(execute.call_count, 1)
                self.assertIn("WORKER_ERROR", result["stop_reason"])
                with self.assertRaisesRegex(ValueError, "stopped"):
                    harness.run_batch("fake", "pilot", 600, "micro")
            record = harness.read_json(directory / "jobs" / jobs[0]["id"] / "record.json")
            self.assertNotIn("metrics", record)
            self.assertTrue((directory / "jobs" / jobs[0]["id"] / "stderr.log").is_file())

    def test_claim_refuses_second_worker_before_solver_import(self):
        with workspace_temp() as temp:
            directory = Path(temp)
            jobs, instance, micro = self.run_fixture(directory)
            job = jobs[0]
            target = directory / "jobs" / job["id"]
            target.mkdir(parents=True)
            config = harness.job_config(manifest(), job)
            micro_path = directory / "micro.json"
            micro_path.write_text("{}", encoding="utf-8")
            micro["path"] = str(micro_path)
            micro["sha256"] = hashlib.sha256(micro_path.read_bytes()).hexdigest()
            harness.write_new(directory / "run.lock", {"pid": 77})
            harness.write_new(target / "request.json", {"job": job, "input_sha256": "input",
                "config_sha256": harness.fingerprint(asdict(config)), "parent_pid": 77,
                "micro_evidence": micro, "per_run_budget_sec": 5., "source_content_sha256": "content"})
            harness.write_new(target / "worker_claim.json", {"pid": 11})
            with patch.object(harness, "load_batch", return_value=(directory, manifest())), \
                    patch.object(harness, "load_instance", return_value=instance), \
                    patch.object(harness, "require_micro", return_value=micro):
                with self.assertRaises(FileExistsError):
                    harness.worker("fake", job["id"])


class MetricsTests(unittest.TestCase):
    def fixture(self, completed=True):
        config = asdict(harness.job_config(manifest(), {"mechanism": "T3", "minimum": 0.}))
        orders = [asdict(V1Order("a", "+", 1, 0, 24, 100)),
                  asdict(V1Order("b", "-", 1, 2, 24, 100))]
        result = {"schedule_valid": True, "completed": completed, "total_cost": 99,
                  "status_name": "OPTIMAL", "optimal": True, "metadata": {"run_mode": "full"},
                  "validation": {"served_quantity": 1, "direct_quantity": .25, "unserved_quantity": 1,
                                 "empty_trip_count": 1, "costs": {"empty": 10}, "completed_horizon": 24},
                  "schedule": {"h_direct": {str((0, 10, "+", "a")): .25},
                               "g_manual": {str((0, 1, 1, "+", "a")): .75,
                                            str((10, 12, 2, "+", "a")): .75},
                               "w_direct": {str((0, 10, "+")): 1, str((10, 19, "-")): 2},
                               "x_manual": {str((0, 1, 1, "+")): 1, str((10, 12, 2, "+")): 1},
                               "empty_vehicles": {str((0, 9, "-")): 1},
                               "y_auto": {str((1, 10, "+")): 1}}}
        return result, config, orders

    def test_delivered_flow_weighted_lead_time_not_order_average(self):
        result, config, orders = self.fixture()
        metrics = summarize_result(result, config, orders)
        self.assertEqual(metrics["delivered_flow_weight"], 1.)
        self.assertEqual(metrics["lead_time_mean_minutes"], .25*600+.75*720)
        self.assertEqual(metrics["lead_time_p50_minutes"], 720)
        self.assertEqual(metrics["lead_time_p95_minutes"], 720)
        self.assertEqual(metrics["zero_cargo_direct_vehicle_tasks"], 2)
        self.assertEqual(metrics["manual_busy_vehicle_minutes"], (10+18+3+9)*60)
        self.assertEqual(metrics["manual_busy_fraction"], 40/(12*24))

    def test_partial_never_gets_full_cost_or_oracle(self):
        result, config, orders = self.fixture(False)
        metrics = summarize_result(result, config, orders)
        self.assertIsNone(metrics["total_cost"])
        self.assertFalse(metrics["global_optimal_oracle"])
        self.assertIn("residual", metrics["unserved_scope"])

    def test_failed_and_zero_served_metrics_have_nulls(self):
        _, config, orders = self.fixture()
        metrics = summarize_result({"status_name": "SOLVER_ERROR"}, config, orders)
        self.assertIsNone(metrics["served_rate"])
        self.assertIsNone(metrics["direct_share"])
        self.assertIsNone(metrics["manual_busy_vehicle_minutes"])

    def test_quantiles_and_ci_missingness(self):
        self.assertEqual(weighted_quantile([(1, .1), (2, .9)], .5), 2)
        self.assertIsNone(weighted_quantile([], .95))
        self.assertIsNone(mean_ci95([1, 2, 3, 4])["ci95"])
        summary = mean_ci95([1, 2, 3, 4, 5])
        self.assertEqual(summary["median"], 3)
        self.assertEqual(summary["min"], 1)
        self.assertEqual(summary["max"], 5)
        self.assertLess(summary["ci95"][0], 3)
        self.assertGreater(summary["ci95"][1], 3)


if __name__ == "__main__":
    unittest.main()
