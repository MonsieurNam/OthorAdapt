import json
import tempfile
import unittest
from pathlib import Path

from phase3_main_experiment import (
    generate_main_commands,
    load_protocol,
    validate_protocol,
    write_commands,
)


def protocol():
    return {
        "schema_version": "phase3.main.v1",
        "datasets": ["eurosat", "caltech101"],
        "shots": [4],
        "seeds": [1, 2],
        "methods": [
            {"adapter": "lora", "r": 4, "alpha": 1, "num_heads": 2, "lambda_o": 0.0},
            {"adapter": "ohsinglora", "r": 4, "alpha": 1, "num_heads": 2, "lambda_o": 0.0},
        ],
        "base_command": {
            "python": "$PYTHON",
            "entrypoint": "main.py",
            "root_path": "${DATA_ROOT}",
            "run_manifest": "revision_materials/results/phase3_main_results.jsonl",
            "save_path": "revision_materials/checkpoints/phase3_main",
            "log_dir": "revision_materials/logs/phase3_main",
            "backbone": "ViT-B/16",
            "encoder": "both",
            "position": "all",
            "params": ["q", "k", "v"],
            "n_iters": 500,
            "batch_size": 32,
            "loss_fn": "ce",
            "ortho_reduction": "mean",
        },
    }


class Phase3MainExperimentTest(unittest.TestCase):
    def test_generate_main_commands_are_test_only_for_selected_config(self):
        commands = generate_main_commands(protocol())

        self.assertEqual(len(commands), 8)
        self.assertTrue(all(command.startswith("mkdir -p revision_materials/logs/phase3_main && $PYTHON main.py ") for command in commands))
        self.assertTrue(all("--selection_split test" in command for command in commands))
        self.assertTrue(all("--report_test" in command for command in commands))
        self.assertTrue(all("--sweep_mode" not in command for command in commands))
        self.assertTrue(all("--run_manifest revision_materials/results/phase3_main_results.jsonl" in command for command in commands))
        self.assertTrue(any("--adapter ohsinglora" in command and "--num_heads 2" in command and "--r 4" in command and "--lambda_o 0.0" in command for command in commands))
        self.assertTrue(any("--filename eurosat_4shot_seed1_test_ohsinglora_h2_r4_lo0p0" in command for command in commands))
        self.assertTrue(any("eurosat_4shot_seed1_test_ohsinglora_h2_r4_lo0p0_${RUN_STAMP}.log" in command for command in commands))

    def test_frozen_phase3_protocol_generates_144_runs(self):
        frozen = load_protocol("revision_materials/plan/phase3_main_protocol.yaml")
        commands = generate_main_commands(frozen)

        self.assertEqual(len(commands), 144)
        self.assertTrue(all("--selection_split test" in command for command in commands))
        self.assertTrue(all("--report_test" in command for command in commands))
        self.assertTrue(all("--sweep_mode" not in command for command in commands))
        self.assertEqual(sum("--adapter ohsinglora" in command for command in commands), 72)
        self.assertEqual(sum("--adapter lora" in command for command in commands), 72)
        self.assertTrue(all("--adapter singlora" not in command for command in commands))
        self.assertTrue(all("--lambda_o 0.03" not in command for command in commands if "--adapter ohsinglora" in command))

    def test_validate_protocol_rejects_unselected_orthoadapt_config(self):
        bad = protocol()
        bad["methods"][1]["lambda_o"] = 0.03

        with self.assertRaisesRegex(ValueError, "selected OrthoAdapt"):
            validate_protocol(bad)

    def test_write_commands_records_run_stamp_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "phase3.sh"
            write_commands(["echo ok"], out)

            text = out.read_text(encoding="utf-8")

        self.assertIn(': "${RUN_STAMP:=$(date +%Y%m%d_%H%M%S)}"', text)

    def test_load_protocol_accepts_json_compatible_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase3.yaml"
            path.write_text(json.dumps(protocol(), indent=2), encoding="utf-8")

            loaded = load_protocol(path)

        self.assertEqual(loaded["schema_version"], "phase3.main.v1")


if __name__ == "__main__":
    unittest.main()
