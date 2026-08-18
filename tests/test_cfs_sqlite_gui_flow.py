import tempfile
import unittest
from pathlib import Path

import pandas as pd

from intercity_delivery.configuration import DeliveryConfig, OrderGenerationConfig
from intercity_delivery.data.cfs_catalog import cfs_area_name, inspect_cfs_sqlite
from intercity_delivery.data.sqlite_store import build_sqlite_store
from intercity_delivery.experiments.core import (
    build_result_context_tag,
    load_real_orders_with_metadata,
)


class CfsSqliteGuiFlowTests(unittest.TestCase):
    @staticmethod
    def _rows():
        return [
            {
                "SHIPMT_ID": f"S{index}",
                "ORIG_CFS_AREA": "06-348" if index % 2 else "06-488",
                "DEST_CFS_AREA": "06-488" if index % 2 else "06-348",
                "MODE": "111",
                "SCTG": "35",
                "SHIPMT_VALUE": 1000 + index,
                "SHIPMT_WGHT": 1000 + index,
                "SHIPMT_DIST_GC": 100 + index,
                "TEMP_CNTL_YN": "N",
                "EXPORT_YN": "N",
                "HAZMAT": "N",
                "WGT_FACTOR": 1 + index,
            }
            for index in range(1, 7)
        ]

    def test_catalog_exposes_columns_and_bidirectional_city_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sample.csv"
            database = root / "sample.sqlite"
            pd.DataFrame(self._rows()).to_csv(source, index=False)
            build_sqlite_store(source, database, chunksize=2)

            catalog = inspect_cfs_sqlite(database)

            self.assertIn("ORIG_CFS_AREA", {name for name, _ in catalog.columns})
            self.assertEqual(len(catalog.city_pairs), 1)
            pair = catalog.city_pairs[0]
            self.assertEqual((pair.city_a, pair.city_b), ("06-348", "06-488"))
            self.assertEqual((pair.records_a_to_b, pair.records_b_to_a), (3, 3))
            self.assertIn("Los Angeles-Long Beach", pair.display_label)

    def test_sqlite_pair_can_feed_experiment_order_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sample.csv"
            database = root / "sample.sqlite"
            pd.DataFrame(self._rows()).to_csv(source, index=False)
            build_sqlite_store(source, database, chunksize=3)

            orders, metadata = load_real_orders_with_metadata(
                str(database),
                DeliveryConfig(),
                OrderGenerationConfig(num_orders=4),
                seed=42,
                city_pair=("06-348", "06-488"),
            )

            self.assertEqual(len(orders[2]), 4)
            self.assertEqual({order.flow for order in orders[2].values()}, {"+", "-"})
            self.assertEqual(metadata["source_kind"], "sqlite")
            self.assertEqual(metadata["city_pair"]["city_1"], "06-348")

    def test_result_filename_tag_contains_pair_and_both_approaches(self):
        tag = build_result_context_tag(
            "real",
            ["paper_candidate_mip", "paper_priority_heuristic"],
            ("06-348", "06-488"),
        )

        self.assertIn("Los_Angeles-Long_Beach_06-348", tag)
        self.assertIn("San_Jose-San_Francisco-Oakland_06-488", tag)
        self.assertIn("RH_pruning+RH_pruning_solution", tag)
        self.assertEqual(cfs_area_name("36-408").split(",", 1)[0], "New York-Newark")


if __name__ == "__main__":
    unittest.main()
