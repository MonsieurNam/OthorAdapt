import unittest

import reviewer4_profile_compute as profile


class Reviewer4ProfileTest(unittest.TestCase):
    def test_training_profile_matrix_has_24_unique_runs(self):
        specs = profile.build_training_profile_specs()
        self.assertEqual(len(specs), 24)
        self.assertEqual(len({spec.run_key for spec in specs}), 24)
        self.assertEqual({spec.method for spec in specs}, {"lora", "singlora", "static", "dynamic"})
        commands = profile.generate_training_profile_commands(
            data_root="${DATA_ROOT}",
            split_reference="revision_materials/reviewer4/raw/frozen_dynamic_reference.jsonl",
        )
        self.assertEqual(len(commands), 24)
        self.assertIn("--profile_repetition 1", commands[0])
        self.assertTrue(all("--batch_size 32" in command for command in commands))
        self.assertTrue(all("--compute_profile_only" in command for command in commands))
        self.assertTrue(all("--report_test" not in command for command in commands))
        benchmark = profile.generate_benchmark_commands(data_root="${DATA_ROOT}")
        self.assertEqual(len(benchmark), 24)
        self.assertTrue(all(" benchmark " in command for command in benchmark))

    def test_latency_summary_uses_five_repetitions(self):
        summary = profile.summarize_inference_timings(
            repetition_batch_seconds=[
                [0.10, 0.11, 0.09], [0.20, 0.19, 0.21], [0.15, 0.14, 0.16],
                [0.12, 0.13, 0.11], [0.18, 0.17, 0.19],
            ],
            batch_size=256,
        )
        self.assertEqual(summary["timing_repetitions"], 5)
        self.assertEqual(summary["measured_batches_per_repetition"], 3)
        self.assertAlmostEqual(summary["median_batch_latency_seconds"], 0.15)
        self.assertGreater(summary["images_per_second"], 0)
        self.assertGreaterEqual(summary["p95_batch_latency_seconds"], 0.20)

    def test_gating_flops_uses_two_flops_per_mac(self):
        breakdown = profile.gating_flops(batch_tokens=257, width=768, heads=2)
        self.assertEqual(breakdown["linear_flops"], 257 * 768 * 2 * 2)
        self.assertEqual(breakdown["mac_to_flop_multiplier"], 2)

    def test_profile_row_requires_all_metrics(self):
        row = {field: 1 for field in profile.REQUIRED_PROFILE_METRICS}
        profile.validate_profile_row(row)
        del row["inference_p95_latency_seconds"]
        with self.assertRaises(ValueError):
            profile.validate_profile_row(row)

    def test_method_identity_uses_adapter_and_routing(self):
        self.assertEqual(profile.method_from_config({"adapter": "lora"}), "lora")
        self.assertEqual(profile.method_from_config({"adapter": "singlora"}), "singlora")
        self.assertEqual(
            profile.method_from_config({"adapter": "ohsinglora", "routing_mode": "input_independent"}),
            "static",
        )
        self.assertEqual(
            profile.method_from_config({"adapter": "ohsinglora", "routing_mode": "input_conditioned"}),
            "dynamic",
        )

    def test_dynamic_static_overhead_is_computed_from_same_gpu_summary(self):
        summary = [
            {"method": "static", "mean_inference_median_latency_seconds": 0.10,
             "mean_seconds_per_optimizer_step": 0.20},
            {"method": "dynamic", "mean_inference_median_latency_seconds": 0.11,
             "mean_seconds_per_optimizer_step": 0.22},
        ]
        rows = profile.dynamic_static_overhead(summary)
        self.assertAlmostEqual(rows[0]["absolute_overhead_seconds"], 0.01)
        self.assertAlmostEqual(rows[0]["relative_overhead_percent"], 10.0)


if __name__ == "__main__":
    unittest.main()
