import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "revision_materials" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase4_aggregate_robustness import aggregate
from phase4_eval_robustness import SCHEMA_VERSION, severity_spec


class Phase4RobustnessTest(unittest.TestCase):
    def test_severity_spec_is_deterministic_and_serializable(self):
        first = severity_spec(3, 123)
        second = severity_spec(3, 123)

        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 123)
        self.assertEqual(first["noise_std"], 0.15)
        self.assertGreater(first["blur_kernel"], 0)

    def test_aggregate_requires_complete_paired_manifest(self):
        rows = []
        for dataset in ["fgvc", "eurosat", "food101", "oxford_pets", "oxford_flowers", "caltech101", "dtd", "ucf101"]:
            for shot in [1, 4, 16]:
                for seed in [1, 2, 3]:
                    for method in ["lora", "ohsinglora"]:
                        for severity in [0, 1, 2, 3]:
                            base = 80.0 if method == "lora" else 81.0
                            rows.append(
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "status": "completed",
                                    "dataset": dataset,
                                    "shot": shot,
                                    "seed": seed,
                                    "method": method,
                                    "severity": severity,
                                    "metrics": {"accuracy": base - severity},
                                }
                            )

        pair_rows, audit = aggregate(rows)

        self.assertEqual(audit["manifest_rows"], 576)
        self.assertEqual(len(pair_rows), 288)
        self.assertTrue(all(row["delta_oh_minus_lora"] == 1.0 for row in pair_rows))

    def test_aggregate_fails_when_manifest_is_incomplete(self):
        with self.assertRaisesRegex(SystemExit, "Expected 576"):
            aggregate([])


if __name__ == "__main__":
    unittest.main()
