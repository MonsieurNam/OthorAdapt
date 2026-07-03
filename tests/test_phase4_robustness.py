import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "revision_materials" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase4_aggregate_robustness import aggregate
from phase4_eval_robustness import SCHEMA_VERSION, severity_spec
from phase4_generate_robustness_commands import command_for
from phase4_make_resume_commands import make_resume


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

    def test_phase4_commands_use_large_eval_loader(self):
        row = {
            "checkpoint": {"path": "revision_materials/checkpoints/phase3_main/example.pt"},
            "config": {
                "adapter": "lora",
                "dataset": "eurosat",
                "shots": 16,
                "seed": 1,
                "filename": "eurosat_16shot_seed1_test_lora_r8",
                "backbone": "ViT-B/16",
                "r": 8,
                "alpha": 1,
                "position": "all",
                "encoder": "both",
                "params": ["q", "k", "v"],
            },
        }

        command = command_for(row)

        self.assertIn("--batch_size 256", command)
        self.assertIn("--num_workers 8", command)

    def test_resume_skips_jobs_completed_in_existing_logs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            commands = tmp_path / "commands.sh"
            manifest = tmp_path / "missing_manifest.jsonl"
            out = tmp_path / "resume.sh"
            log_dir = tmp_path / "logs"
            log_dir.mkdir()
            commands.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        "python phase4_eval.py --filename completed_job",
                        "python phase4_eval.py --filename pending_job",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (log_dir / "completed_job_20260703_010101.log").write_text(
                '{"job_id":"completed_job","rows":4,"runtime_seconds":12.3}\n',
                encoding="utf-8",
            )

            completed, skipped, pending = make_resume(commands, manifest, out, log_dir=log_dir)

            self.assertEqual(completed, 1)
            self.assertEqual(skipped, 1)
            self.assertEqual(pending, 1)
            resume_text = out.read_text(encoding="utf-8")
            self.assertNotIn("--filename completed_job", resume_text)
            self.assertIn("--filename pending_job", resume_text)


if __name__ == "__main__":
    unittest.main()
