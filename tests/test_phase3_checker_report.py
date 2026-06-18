import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from revision_materials.scripts import phase3_checker_report as report


class Phase3CheckerReportTest(unittest.TestCase):
    def test_build_report_counts_pending_and_estimates_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = root / "phase3.sh"
            manifest = root / "manifest.jsonl"
            runtime_source = root / "runtime.jsonl"
            commands.write_text(
                "$PYTHON main.py --n_iters 500 --shots 1 --filename run_a\n"
                "$PYTHON main.py --n_iters 500 --shots 4 --filename run_b\n",
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "config": {"filename": "run_a"},
                        "metrics": {"runtime_seconds": 5, "train_total_iterations": 500},
                        "checkpoint": {"sha256": "A" * 64},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runtime_source.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "config": {"filename": "history"},
                        "metrics": {"runtime_seconds": 20, "train_total_iterations": 10},
                        "checkpoint": {"sha256": "B" * 64},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            old_commands = report.runner.DEFAULT_COMMANDS
            old_manifest = report.runner.DEFAULT_MANIFEST
            old_runtime = report.runner.DEFAULT_RUNTIME_SOURCE
            try:
                report.runner.DEFAULT_COMMANDS = commands
                report.runner.DEFAULT_MANIFEST = manifest
                report.runner.DEFAULT_RUNTIME_SOURCE = runtime_source

                data = report.build_report(
                    cost_per_hour_vnd=5000,
                    now=datetime(2026, 6, 18, 0, 0, tzinfo=timezone.utc),
                )
            finally:
                report.runner.DEFAULT_COMMANDS = old_commands
                report.runner.DEFAULT_MANIFEST = old_manifest
                report.runner.DEFAULT_RUNTIME_SOURCE = old_runtime

        self.assertEqual(data["done"], 1)
        self.assertEqual(data["total"], 2)
        self.assertEqual(data["pending"], 1)
        self.assertEqual(data["rate"], "1.0050s/iter")
        self.assertEqual(data["pending_iterations"], 2000)
        self.assertEqual(data["eta_seconds"], 2010)
        self.assertEqual(data["eta_human"], "33m30s")
        self.assertEqual(data["estimated_finish_utc"], "2026-06-18 00:33 UTC")
        self.assertEqual(data["estimated_finish_vn"], "2026-06-18 07:33 VN")
        self.assertEqual(data["cost_remaining_vnd"], 5000)


if __name__ == "__main__":
    unittest.main()
