import io
import json
import tempfile
import unittest
from pathlib import Path

from revision_materials.scripts.phase3_resumable_runner import (
    RunResult,
    command_iterations,
    completed_filenames,
    extract_filename,
    load_phase3_commands,
    run_pending_commands,
    seconds_per_iteration,
)


def manifest_row(filename, runtime=10.0, status="completed", sha="A" * 64):
    return {
        "schema_version": "ecr3.run.v1",
        "status": status,
        "config": {"filename": filename},
        "metrics": {"runtime_seconds": runtime},
        "checkpoint": {"sha256": sha},
    }


class Phase3ResumableRunnerTest(unittest.TestCase):
    def test_extract_filename_from_phase3_command(self):
        command = (
            "mkdir -p logs && $PYTHON main.py --dataset eurosat "
            "--filename eurosat_4shot_seed1_test_lora_r4 --report_test"
        )

        self.assertEqual(extract_filename(command), "eurosat_4shot_seed1_test_lora_r4")

    def test_completed_filenames_accepts_hashed_train_and_eval_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "results.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps(manifest_row("done")),
                        json.dumps(manifest_row("evaluated", status="eval_only")),
                        json.dumps(manifest_row("failed", status="failed")),
                        json.dumps(manifest_row("missing_hash", sha="")),
                        json.dumps(manifest_row("eval_missing_hash", status="eval_only", sha="")),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed, runtimes = completed_filenames(manifest)

        self.assertEqual(completed, {"done", "evaluated"})
        self.assertEqual(runtimes, [10.0, 10.0])

    def test_load_phase3_commands_skips_headers_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "phase3.sh"
            script.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "\n"
                "mkdir -p logs && $PYTHON main.py --filename run_a\n"
                "mkdir -p logs && $PYTHON main.py --filename run_b\n",
                encoding="utf-8",
            )

            commands = load_phase3_commands(script)

        self.assertEqual(len(commands), 2)
        self.assertTrue(all("--filename run_" in command for command in commands))

    def test_run_pending_commands_skips_completed_and_reports_eta(self):
        commands = [
            "$PYTHON main.py --filename run_a",
            "$PYTHON main.py --filename run_b",
            "$PYTHON main.py --filename run_c",
        ]
        calls = []

        def fake_executor(command):
            calls.append(command)
            return RunResult(returncode=0, elapsed_seconds=4.0)

        output = io.StringIO()
        summary = run_pending_commands(
            commands,
            completed={"run_a"},
            historical_runtimes=[10.0],
            executor=fake_executor,
            max_runs=1,
            output=output,
        )

        self.assertEqual(calls, ["$PYTHON main.py --filename run_b"])
        self.assertEqual(summary["completed_before"], 1)
        self.assertEqual(summary["executed"], 1)
        self.assertEqual(summary["remaining_after"], 1)
        self.assertIn("ETA", output.getvalue())

    def test_seconds_per_iteration_uses_runtime_and_train_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "results.jsonl"
            first = manifest_row("a", runtime=20.0)
            first["metrics"]["train_total_iterations"] = 10
            second = manifest_row("b", runtime=40.0)
            second["metrics"]["train_total_iterations"] = 20
            manifest.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )

            estimate = seconds_per_iteration([manifest])

        self.assertEqual(estimate, 2.0)

    def test_command_iterations_scales_n_iters_by_shots(self):
        command = "$PYTHON main.py --n_iters 500 --shots 16 --filename run_a"

        self.assertEqual(command_iterations(command), 8000)

    def test_command_iterations_does_not_count_eval_only_as_training(self):
        command = "$PYTHON main.py --n_iters 500 --shots 16 --eval_only --filename run_a"

        self.assertIsNone(command_iterations(command))


if __name__ == "__main__":
    unittest.main()
