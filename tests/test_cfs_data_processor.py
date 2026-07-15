import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from intercity_delivery.data.cfs_processor import ProcessorConfig, load_processed_orders, main


class CfsDataProcessorTests(unittest.TestCase):
    def test_explicit_pair_produces_orderbatch_compatible_output(self):
        rows = []
        for index in range(12):
            plus = index % 2 == 0
            rows.append(
                {
                    "SHIPMT_ID": f"{index + 1:08d}",
                    "ORIG_CFS_AREA": "06-348" if plus else "06-488",
                    "DEST_CFS_AREA": "06-488" if plus else "06-348",
                    "MODE": "111",
                    "SCTG": "35",
                    "SHIPMT_VALUE": 10000 + index,
                    "SHIPMT_WGHT": 1000 + 100 * index,
                    "SHIPMT_DIST_GC": 300 + index,
                    "TEMP_CNTL_YN": "N",
                    "EXPORT_YN": "N",
                    "HAZMAT": "N",
                    "WGT_FACTOR": 1 + index,
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sample.csv"
            output = root / "output"
            pd.DataFrame(rows).to_csv(source, index=False)

            paths = main(
                [
                    "--input",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--city-a",
                    "06-348",
                    "--city-b",
                    "06-488",
                    "--num-orders",
                    "6",
                    "--planning-periods",
                    "24",
                    "--min-distance-miles",
                    "50",
                    "--chunksize",
                    "4",
                ]
            )

            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(len(payload["orders"]), 6)
            self.assertEqual(
                {item["flow"] for item in payload["orders"]}, {"+", "-"}
            )
            orders_tuple = load_processed_orders(paths["json"])
            self.assertEqual(len(orders_tuple[2]), 6)
            for order in orders_tuple[2].values():
                self.assertLess(order.earliest_start, order.latest_completion)

    def test_configuration_rejects_short_planning_horizon(self):
        config = ProcessorConfig(planning_periods=1)
        config.validate()


if __name__ == "__main__":
    unittest.main()
