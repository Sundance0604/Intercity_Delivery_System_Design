"""Registry/CLI integration, without constructing or solving a model."""

import unittest
from unittest.mock import patch

from intercity_delivery.experiments.solvers import SOLVER_REGISTRY


class RepositioningIntegrationTests(unittest.TestCase):
    def test_existing_names_still_resolve_to_original_classes(self):
        self.assertEqual(type(SOLVER_REGISTRY["exact_mip"]).__name__, "ExactMIPSolver")
        self.assertEqual(
            type(SOLVER_REGISTRY["flexible_direct_mip"]).__name__,
            "FlexibleDirectMIPSolver",
        )

    def test_new_names_are_explicit_opt_in(self):
        import main

        for name in ("flexible_direct_empty_mip", "flexible_direct_empty_rolling"):
            with self.subTest(name=name), patch("sys.argv", ["main.py", "--solver", name]):
                self.assertEqual(main.parse_args().solver, name)
        with patch("sys.argv", ["main.py"]):
            self.assertEqual(main.parse_args().solver, "exact_mip")

    def test_direct_parameters_apply_to_new_names(self):
        from intercity_delivery.configuration import DIRECT_MODEL_SOLVERS

        self.assertIn("flexible_direct_empty_mip", DIRECT_MODEL_SOLVERS)
        self.assertIn("flexible_direct_empty_rolling", DIRECT_MODEL_SOLVERS)

    def test_rolling_extension_is_not_silently_ignored(self):
        from dataclasses import fields
        from intercity_delivery.configuration import RollingHorizonConfig

        field = next(f for f in fields(RollingHorizonConfig) if f.name == "extension_horizon")
        self.assertIn("flexible_direct_empty_rolling", field.metadata["solvers"])
        self.assertNotIn("flexible_direct_empty_mip", field.metadata["solvers"])


if __name__ == "__main__":
    unittest.main()
