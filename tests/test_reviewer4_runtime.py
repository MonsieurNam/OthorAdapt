import unittest

from reviewer4_runtime import summarize_training_runtime


class Reviewer4RuntimeTest(unittest.TestCase):
    def test_summary_excludes_five_complete_warmup_epochs(self):
        summary = summarize_training_runtime(
            complete_epoch_seconds=[1, 2, 3, 4, 5, 10, 12],
            partial_epoch_seconds=[0.5],
            total_steps=100,
            total_seconds=50,
            peak_allocated_bytes=1024,
            peak_reserved_bytes=2048,
            warmup_epochs=5,
        )
        self.assertEqual(summary["complete_epochs"], 7)
        self.assertEqual(summary["partial_epochs"], 1)
        self.assertEqual(summary["steady_epoch_count"], 2)
        self.assertEqual(summary["mean_complete_epoch_seconds_after_warmup"], 11.0)
        self.assertEqual(summary["median_complete_epoch_seconds_after_warmup"], 11.0)
        self.assertEqual(summary["seconds_per_optimizer_step"], 0.5)
        self.assertEqual(summary["peak_gpu_allocated_bytes"], 1024)
        self.assertEqual(summary["peak_gpu_reserved_bytes"], 2048)

    def test_too_few_epochs_yields_null_steady_statistics(self):
        summary = summarize_training_runtime([1, 2], [], 4, 2.0, 0, 0)
        self.assertIsNone(summary["mean_complete_epoch_seconds_after_warmup"])
        self.assertIsNone(summary["median_complete_epoch_seconds_after_warmup"])

    def test_invalid_timings_hard_fail(self):
        with self.assertRaises(ValueError):
            summarize_training_runtime([1, -1], [], 4, 2.0, 0, 0)
        with self.assertRaises(ValueError):
            summarize_training_runtime([], [], 0, 2.0, 0, 0)


if __name__ == "__main__":
    unittest.main()
