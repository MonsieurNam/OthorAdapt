import re
import tempfile
import unittest
from pathlib import Path

from phase3_reviewer2_lambda0_experiment import (
    generate_commands,
    load_protocol,
    validate_protocol,
    write_commands,
)


PROTOCOL_PATH = "revision_materials/plan/phase3_reviewer2_lambda0_protocol.yaml"


def _flag(command, name):
    match = re.search(rf"(?:^|\s){re.escape(name)}\s+(\S+)", command)
    return match.group(1) if match else None


class Reviewer2Lambda0ExperimentTest(unittest.TestCase):
    def test_frozen_protocol_generates_complete_train_and_eval_matrices(self):
        protocol = load_protocol(PROTOCOL_PATH)

        train_commands = generate_commands(protocol, mode="train")
        eval_commands = generate_commands(protocol, mode="eval")

        self.assertEqual(len(train_commands), 72)
        self.assertEqual(len(eval_commands), 72)
        expected = {
            (dataset, shot, seed)
            for dataset in protocol["datasets"]
            for shot in protocol["shots"]
            for seed in protocol["seeds"]
        }
        for commands in (train_commands, eval_commands):
            observed = {
                (_flag(command, "--dataset"), int(_flag(command, "--shots")), int(_flag(command, "--seed")))
                for command in commands
            }
            self.assertEqual(observed, expected)
            self.assertTrue(all("--adapter ohsinglora" in command for command in commands))
            self.assertTrue(all("--num_heads 2" in command for command in commands))
            self.assertTrue(all("--r 8" in command for command in commands))
            self.assertTrue(all("--alpha 1" in command for command in commands))
            self.assertTrue(all("--lambda_o 0" in command for command in commands))
            self.assertTrue(all("--ramp_up_steps 100" in command for command in commands))
            self.assertTrue(all('--root_path "${DATA_ROOT}"' in command for command in commands))
            self.assertTrue(all("--adapter singlora" not in command for command in commands))

        self.assertTrue(all("--selection_split val" in command for command in train_commands))
        self.assertTrue(all("--report_test" not in command for command in train_commands))
        self.assertTrue(all("--eval_only" not in command for command in train_commands))
        self.assertTrue(all("phase3_reviewer2_lambda0_train_results.jsonl" in command for command in train_commands))

        self.assertTrue(all("--selection_split test" in command for command in eval_commands))
        self.assertTrue(all("--report_test" in command for command in eval_commands))
        self.assertTrue(all("--eval_only" in command for command in eval_commands))
        self.assertTrue(all("phase3_reviewer2_lambda0_eval_results.jsonl" in command for command in eval_commands))

        train_filenames = {_flag(command, "--filename") for command in train_commands}
        eval_filenames = {_flag(command, "--filename") for command in eval_commands}
        self.assertEqual(train_filenames, eval_filenames)
        self.assertTrue(all(name.endswith("_test_ohsinglora_h2_r8_lo0_ramp100") for name in train_filenames))

    def test_validate_protocol_rejects_method_drift(self):
        replacements = {
            "adapter": "singlora",
            "num_heads": 4,
            "r": 4,
            "alpha": 2,
            "lambda_o": 0.03,
            "ramp_up_steps": 1000,
        }
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                protocol = load_protocol(PROTOCOL_PATH)
                protocol["method"][field] = replacement
                with self.assertRaisesRegex(ValueError, field):
                    validate_protocol(protocol)

    def test_validate_protocol_rejects_drift_from_frozen_training_config(self):
        replacements = {
            "backbone": "ViT-B/32",
            "encoder": "vision",
            "position": "top3",
            "params": ["q", "v"],
            "n_iters": 501,
            "batch_size": 16,
            "loss_fn": "focal",
            "ortho_reduction": "sum",
        }
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                protocol = load_protocol(PROTOCOL_PATH)
                protocol["base_command"][field] = replacement
                with self.assertRaisesRegex(ValueError, field):
                    validate_protocol(protocol)

    def test_write_commands_adds_resume_safe_shell_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "commands.sh"
            write_commands(["echo ok"], output)

            text = output.read_text(encoding="utf-8")
            raw = output.read_bytes()

        self.assertIn("set -euo pipefail", text)
        self.assertIn(': "${DATA_ROOT:?Set DATA_ROOT=/path/to/datasets}"', text)
        self.assertIn(': "${RUN_STAMP:=$(date +%Y%m%d_%H%M%S)}"', text)
        self.assertNotIn(b"\r\n", raw)

    def test_checked_in_shell_artifacts_match_the_frozen_generator(self):
        protocol = load_protocol(PROTOCOL_PATH)
        artifacts = {
            "train": "revision_materials/scripts/phase3_reviewer2_lambda0_train_commands.sh",
            "eval": "revision_materials/scripts/phase3_reviewer2_lambda0_eval_commands.sh",
        }
        for mode, path in artifacts.items():
            with self.subTest(mode=mode):
                checked_in = [
                    line
                    for line in Path(path).read_text(encoding="utf-8").splitlines()
                    if "--filename" in line
                ]
                self.assertEqual(checked_in, generate_commands(protocol, mode=mode))

    def test_launcher_uses_resume_safe_runner_for_train_and_eval(self):
        text = Path(
            "revision_materials/scripts/phase3_reviewer2_lambda0_run.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('MODE="${1:-all}"', text)
        self.assertIn(': "${DATA_ROOT:?Set DATA_ROOT=/path/to/datasets}"', text)
        self.assertIn("phase3_resumable_runner.py", text)
        self.assertIn("phase3_reviewer2_lambda0_train_commands.sh", text)
        self.assertIn("phase3_reviewer2_lambda0_train_results.jsonl", text)
        self.assertIn("phase3_reviewer2_lambda0_eval_commands.sh", text)
        self.assertIn("phase3_reviewer2_lambda0_eval_results.jsonl", text)
        self.assertIn('train|eval|all', text)


if __name__ == "__main__":
    unittest.main()
