"""Run the bounded conditional V1-M acceptance suite and save fresh evidence.

Evidence goes to stdout by default. With --output the file is opened exclusively
before testing, so a permission failure cannot waste the eight-run budget.
Existing experiment evidence is never overwritten.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new evidence file; default: stdout")
    args = parser.parse_args()
    output = None
    if args.output is not None:
        try:
            output = args.output.open("x", encoding="utf-8")
        except OSError as error:
            parser.error(f"cannot create evidence file before testing: {error}")
    from tests import test_v1_mandatory as cases
    from intercity_delivery.v1.configuration import PROFILE_SEMANTIC_STATUS

    actual_run = cases.run_mandatory
    attempts = 0
    payloads = []

    def capture(*arguments, **keywords):
        nonlocal attempts
        if attempts >= 8:
            raise RuntimeError("eight-run first-pass budget exhausted")
        if keywords.get("time_limit") != 2:
            raise RuntimeError("each micro run must use the two-second budget")
        attempts += 1
        payload = actual_run(*arguments, **keywords)
        payloads.append(payload)
        return payload

    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "tests.test_v1_mandatory.ConfigurationTests",
        "tests.test_v1_mandatory.MandatoryMicroSolveTests",
        "tests.test_v1_validation", "tests.test_v1_kernel", "tests.test_v1_runner_status",
    ])
    started = time.monotonic()
    with patch.object(cases, "run_mandatory", side_effect=capture):
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_status": PROFILE_SEMANTIC_STATUS,
        "budget": {"maximum_runs": 8, "per_run_seconds": 2,
                   "threads": 1, "solver_seed": 0},
        "tests_run": outcome.testsRun,
        "successful": outcome.wasSuccessful(),
        "failures": [(str(case), detail) for case, detail in outcome.failures],
        "errors": [(str(case), detail) for case, detail in outcome.errors],
        "skipped": [(str(case), reason) for case, reason in outcome.skipped],
        "suite_elapsed_sec": time.monotonic() - started,
        "solver_run_attempts": attempts,
        "runs": payloads,
    }
    destination = output if output is not None else sys.stdout
    try:
        json.dump(evidence, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
    finally:
        if output is not None:
            output.close()
    print(json.dumps({"tests_run": outcome.testsRun,
                      "successful": outcome.wasSuccessful(),
                      "solver_run_attempts": attempts,
                      "costs": [item["total_cost"] for item in payloads],
                      "evidence_path": str(args.output.resolve()) if args.output else "stdout"}),
          file=sys.stderr)
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
