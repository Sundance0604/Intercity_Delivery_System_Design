"""No solver: exercise immutable planning and real controller fake-process paths."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.fixed_n import round as r


class FixedNTests(unittest.TestCase):
    def setUp(self):
        core = r.freeze_snapshot()
        self.config = r.config_from_design(core["design"], core["decisions"])
        self.design = r.read_json(r.DEFAULT_DESIGN)

    def plan(self, seeds=None, n=4):
        return r.make_plan(n, seeds or [4101], self.design, self.config, 3., 10.)

    def test_four_directions_share_one_baseline(self):
        plan = self.plan([4101, 4102])
        self.assertEqual(plan["unique_jobs"], 28)
        self.assertEqual(len(set(c["id"] for c in plan["cases"])), 28)
        self.assertEqual([c["stage"] for c in plan["cases"]], sorted([c["stage"] for c in plan["cases"]], key=r.STAGES.index))
        refs = plan["stage_references"]
        for stage in r.STAGES:
            self.assertIn("n4_s4101_T3", refs[stage])
        self.assertEqual([len(refs[s]) for s in r.STAGES], [10, 6, 18, 4])
        self.assertEqual(len(r.comparisons(plan)), 15)
        self.assertIn(("mechanism", "T2_m0.1", "T3_m0.1"), r.comparisons(plan))
        self.assertIn(("mechanism", "T3", "T3_m0.1"), r.comparisons(plan))

    def test_one_factor_direct_price_holds_empty_fixed(self):
        cases = {c["label"]: c for c in self.plan()["cases"]}
        base = cases["T3"]["config"]
        for label in ("direct_rate18.75", "direct_rate31.25"):
            variant = cases[label]["config"]
            self.assertEqual([k for k in base if base[k] != variant[k]], ["c_direct_hour"])
            self.assertEqual(variant["c_empty_hour"], 25.)
        self.assertEqual(cases["RH_H8"]["E"], cases["RH_pruned"]["E"])
        self.assertEqual(cases["separate"]["config"], base)

    def test_custom_float_labels_cannot_collide(self):
        design = copy.deepcopy(self.design)
        design["direct_hourly_rates"] = [18.75, 18.75000001]
        with self.assertRaisesRegex(ValueError, "collide"):
            r.make_plan(4, [1], design, self.config, 3., 10.)

    def test_arbitrary_N_and_seed_count(self):
        self.assertEqual(self.plan(n=3)["n"], 3)
        for value in (0, 1, True, 4.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.plan(n=value)
        with self.assertRaises(ValueError):
            self.plan([1, 1])
        for value in (float("nan"), float("inf"), -1, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.positive(value, "budget")

    def test_plan_is_pure_and_run_requires_explicit_budget(self):
        with patch.object(r, "run_round") as run, patch("builtins.print"):
            r.main(["--orders", "4", "--seeds", "1", "--batch", "unit_plan"])
            run.assert_not_called()
            with self.assertRaisesRegex(ValueError, "explicit"):
                r.main(["--run", "--orders", "4", "--seeds", "1", "--batch", "unit_plan"])
            run.assert_not_called()

    def test_path_and_semantic_identity_bind_every_control(self):
        for name in ("../oops", "a/b", "a\\b", "", "."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                r.batch_path(name)
        case = self.plan()["cases"][0]
        instance = {"input_sha256": "input", "orders_sha256": "orders"}
        manifest = {"core": {"source_content_sha256": "core"}, "suite": {"sha256": "suite"}}
        first = r.semantic_identity(case, instance, manifest, {"cumulative_wall_sec": 180})
        for key, value in (("H", 99), ("pooling_model", "separate"), ("logical_budget_sec", 8)):
            changed = copy.deepcopy(case)
            changed[key] = value
            self.assertNotEqual(r.fingerprint(first), r.fingerprint(r.semantic_identity(changed, instance, manifest, {"cumulative_wall_sec": 180})))

    def test_remaining_budget_rechecked_after_overhead(self):
        self.assertAlmostEqual(r.remaining_timeout(100, 20, 10, 15, clock=lambda: 11), 15)
        self.assertAlmostEqual(r.remaining_timeout(100, 90, 10, 15, clock=lambda: 14), 5.9)
        self.assertEqual(r.remaining_timeout(100, 90, 10, 15, clock=lambda: 20), 0.)

    def test_no_incumbent_is_retained_but_error_stops(self):
        process = {"process_status": "EXITED", "worker_exit_status": 0}
        self.assertIsNone(r.stop_reason({"status_name": "TIME_LIMIT", "schedule": None}, process))
        for status in ("SOLVER_ERROR", "DEPENDENCY_UNAVAILABLE", "INFEASIBLE"):
            with self.subTest(status=status):
                self.assertIsNotNone(r.stop_reason({"status_name": status}, process))
        self.assertEqual(r.stop_reason({}, {"process_status": "TIMEOUT"}), "WORKER_ERROR")

    def test_child_timeout_keeps_partial_logs(self):
        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output=b"partial", stderr=b"license")
        result = r.execute_attempt(["fake"], 1., process_run=timeout)
        self.assertEqual(result["process_status"], "TIMEOUT")
        self.assertEqual(result["stdout"], "partial")
        self.assertEqual(result["stderr"], "license")

    def fixture(self, directory):
        plan = self.plan()
        manifest = {"n": 4, "seeds": [4101], "prepare_wall_sec": .1, "budget_scope": "test",
            "per_run_sec": 3., "worker_timeout_sec": 10., "design": self.design,
            "core": {"source_content_sha256": "core", "source_manifest_sha256": "manifest"}, "suite": {"sha256": "suite"},
            "instances": {"n4_s4101": {"orders_sha256": "orders"}}}
        instance = {"input_sha256": "input", "orders_sha256": "orders", "identity": {"orders": []}}
        args = SimpleNamespace(batch="unit", orders=4, seeds=[4101], budget_sec=180.,
            per_run_sec=3., worker_timeout_sec=10., design=None)
        return plan, manifest, instance, args

    def fake_raw(self, case, manifest, instance, status="TIME_LIMIT"):
        inputs = {"config": case["config"], "orders": instance["identity"]["orders"],
            "profile": asdict(r.flexible_profile(case["information"]))}
        return {"status_name": status, "schedule": None, "completed": False,
            "model_id": "v1_comparison_local_separate_v1" if case["pooling_model"] == "separate" else "v1_flexible_arc_wave_empty_v1",
            "requested_time_limit_sec": case["logical_budget_sec"], "metadata": {
                "threads": 1, "solver_seed": 0, "run_mode": "full" if case["algorithm"] == "full" else "rolling",
                "rolling_options": {"prediction_horizon": case["H"], "rolling_step": case["h"], "extension_horizon": case["E"], "prune": case["algorithm"] == "pruned"},
                "provenance": {"input": inputs, "input_sha256": r.fingerprint(inputs),
                    "source_binding": {"current_sources_match": True, "manifest_sha256": manifest["core"]["source_manifest_sha256"]}}}}

    def test_raw_identity_rejects_wrong_source_model_and_horizon(self):
        case = next(c for c in self.plan()["cases"] if c["label"] == "RH_pruned")
        manifest = {"core": {"source_manifest_sha256": "manifest"}}
        instance = {"identity": {"orders": []}}
        raw = self.fake_raw(case, manifest, instance)
        r.validate_raw_identity(case, instance, manifest, raw)
        for field in ("source", "model", "H"):
            other = copy.deepcopy(raw)
            if field == "source":
                other["metadata"]["provenance"]["source_binding"]["current_sources_match"] = False
            elif field == "model":
                other["model_id"] = "wrong"
            else:
                other["metadata"]["rolling_options"]["prediction_horizon"] = 99
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.validate_raw_identity(case, instance, manifest, other)

    def test_full_fake_round_resume_and_no_recharge(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            plan, manifest, instance, args = self.fixture(directory)
            calls = []
            def process(command, **kwargs):
                case_id = command[command.index("--worker")+1]
                case = next(c for c in plan["cases"] if c["id"] == case_id)
                calls.append(case_id)
                r.write_new(directory / "jobs" / case_id / "raw.json", self.fake_raw(case, manifest, instance))
                return SimpleNamespace(returncode=0, stdout="kept log", stderr="")
            with patch.object(r, "batch_path", return_value=directory), patch.object(r, "load_round", return_value=(directory, manifest, plan)), patch.object(r, "instance_for", return_value=instance):
                first = r.run_round(args, process_run=process)
                self.assertEqual(first["new_attempts"], 14)
                self.assertEqual(len(calls), 14)
                second = r.run_round(args, process_run=process)
                self.assertEqual(second["new_attempts"], 0)
                self.assertGreaterEqual(second["cumulative_wall_sec"], first["cumulative_wall_sec"])
                args.budget_sec = 200
                with self.assertRaisesRegex(ValueError, "recharged"):
                    r.run_round(args, process_run=process)
                self.assertEqual(len(calls), 14)
                self.assertEqual((directory / "jobs" / calls[0] / "stdout.log").read_text(), "kept log")

    def test_worker_failure_halts_and_failed_attempt_never_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            plan, manifest, instance, args = self.fixture(directory)
            calls = []
            def process(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=9, stdout="failed", stderr="bad license")
            with patch.object(r, "batch_path", return_value=directory), patch.object(r, "load_round", return_value=(directory, manifest, plan)), patch.object(r, "instance_for", return_value=instance):
                result = r.run_round(args, process_run=process)
                self.assertEqual(result["stop_reason"], "WORKER_ERROR")
                self.assertEqual(len(calls), 1)
                self.assertTrue((directory / "stop.json").is_file())
                with self.assertRaisesRegex(ValueError, "stopped"):
                    r.run_round(args, process_run=process)
                self.assertEqual(len(calls), 1)

    def test_budget_exhaustion_spawns_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            plan, manifest, instance, args = self.fixture(directory)
            manifest["prepare_wall_sec"] = 181
            process = unittest.mock.Mock()
            with patch.object(r, "batch_path", return_value=directory), patch.object(r, "load_round", return_value=(directory, manifest, plan)), patch.object(r, "instance_for", return_value=instance):
                result = r.run_round(args, process_run=process)
            process.assert_not_called()
            self.assertEqual(result["stop_reason"], "BUDGET_EXHAUSTED")
            self.assertEqual(result["new_attempts"], 0)

    def test_single_seed_and_missing_pairs_have_no_false_CI(self):
        plan = self.plan()
        rows = [{"seed": c["seed"], "label": c["label"], "orders_sha256": "same",
            "completed_feasible": c["label"] != "RH_dense", "cost": 10., "served_rate": .5,
            "parent_wall_sec": 1., "information": c["information"], "status": "TIME_LIMIT"} for c in plan["cases"]]
        pairs, aggregates = r.paired_rows(plan, rows)
        self.assertEqual(len(pairs), 15)
        self.assertTrue(all(a["cost_delta"]["ci95"] is None for a in aggregates))
        self.assertTrue(all(a["cost_delta"]["planned_n"] == 1 for a in aggregates))
        self.assertEqual(sum(not p["eligible"] for p in pairs), 2)

    def test_full_model_bound_interval_excludes_RH(self):
        plan = self.plan()
        rows = [{"seed": c["seed"], "label": c["label"], "orders_sha256": "same", "completed_feasible": True,
            "cost": 10., "global_bound": 8. if c["algorithm"] == "full" else None,
            "served_rate": .5, "parent_wall_sec": 1., "information": c["information"], "status": "TIME_LIMIT"} for c in plan["cases"]]
        pairs, _ = r.paired_rows(plan, rows)
        for pair in pairs:
            expected = [-2., 2.] if pair["information_comparison"] == "same" and pair["baseline"] != "RH_dense" and not pair["treatment"].startswith("RH") else None
            self.assertEqual(pair["optimal_cost_delta_interval_from_solver_bounds"], expected)


if __name__ == "__main__":
    unittest.main()
