import unittest

import reviewer4_analyze_evidence as analysis


def rows(offset):
    output = []
    for dataset_index in range(8):
        for shot in (1, 4, 16):
            for seed in (1, 2, 3):
                output.append(
                    {
                        "dataset": f"d{dataset_index}",
                        "shot": shot,
                        "seed": seed,
                        "accuracy": 70.0 + dataset_index + shot / 100 + seed / 1000 + offset,
                        "split_hash": f"hash-{dataset_index}-{shot}-{seed}",
                        "trainable_parameters": 460800,
                    }
                )
    return output


class Reviewer4AnalysisTest(unittest.TestCase):
    def test_pairing_produces_72_dynamic_minus_static_deltas(self):
        paired = analysis.pair_accuracy_rows(rows(0.25), rows(0.0))
        self.assertEqual(len(paired), 72)
        self.assertTrue(all(abs(row["delta"] - 0.25) < 1e-10 for row in paired))

        summary = analysis.summarize_pairs(paired)
        self.assertAlmostEqual(summary["overall"]["mean_delta"], 0.25)
        self.assertEqual(summary["overall"]["wins_ties_losses"], "72/0/0")
        self.assertEqual(len(summary["by_shot"]), 3)
        self.assertEqual(len(summary["by_dataset"]), 8)

    def test_pairing_hard_fails_on_hash_parameter_or_coverage_mismatch(self):
        static = rows(0.0)
        with self.assertRaises(ValueError):
            analysis.pair_accuracy_rows(rows(0.25), static[:-1])
        bad_hash = rows(0.0)
        bad_hash[0]["split_hash"] = "wrong"
        with self.assertRaises(ValueError):
            analysis.pair_accuracy_rows(rows(0.25), bad_hash)
        bad_params = rows(0.0)
        bad_params[0]["trainable_parameters"] = 1
        with self.assertRaises(ValueError):
            analysis.pair_accuracy_rows(rows(0.25), bad_params)

    def test_hierarchical_bootstrap_is_reproducible(self):
        paired = analysis.pair_accuracy_rows(rows(0.25), rows(0.0))
        first = analysis.dataset_cluster_bootstrap(paired, replicates=1000, seed=20261007)
        second = analysis.dataset_cluster_bootstrap(paired, replicates=1000, seed=20261007)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_delta"], 0.25)
        self.assertAlmostEqual(first["ci95_low"], 0.25)
        self.assertAlmostEqual(first["ci95_high"], 0.25)

    def test_ties_use_explicit_tolerance(self):
        paired = analysis.pair_accuracy_rows(rows(0.0), rows(0.0))
        summary = analysis.summarize_pairs(paired, tie_tolerance=1e-12)
        self.assertEqual(summary["overall"]["wins_ties_losses"], "0/72/0")


if __name__ == "__main__":
    unittest.main()
