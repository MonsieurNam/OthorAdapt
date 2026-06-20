import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from revision_materials.scripts.phase3b_checker_report import build_report


def manifest_row(filename, runtime=20.0, iterations=10):
    return {
        "schema_version": "ecr3.run.v1",
        "status": "completed",
        "config": {"filename": filename},
        "metrics": {
            "runtime_seconds": runtime,
            "fine_tuning_seconds": runtime,
            "train_total_iterations": iterations,
        },
        "checkpoint": {"sha256": "A" * 64},
    }


class Phase3BAuxiliaryToolsTest(unittest.TestCase):
    def test_checker_report_uses_phase3b_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = root / "phase3b_commands.sh"
            manifest = root / "phase3b_results.jsonl"
            commands.write_text(
                "#!/usr/bin/env bash\n"
                "mkdir -p logs && $PYTHON main.py --n_iters 500 --shots 1 --filename run_a\n"
                "mkdir -p logs && $PYTHON main.py --n_iters 500 --shots 4 --filename run_b\n",
                encoding="utf-8",
            )
            manifest.write_text(json.dumps(manifest_row("run_a")) + "\n", encoding="utf-8")

            report = build_report(
                commands_path=commands,
                manifest_path=manifest,
                runtime_sources=[manifest],
                now=datetime(2026, 6, 20, tzinfo=timezone.utc),
            )

        self.assertEqual(report["done"], 1)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["pending"], 1)
        self.assertEqual(report["pending_iterations"], 2000)
        self.assertEqual(report["rate"], "2.0000s/iter")


if __name__ == "__main__":
    unittest.main()
