import json
import tempfile
import unittest
from pathlib import Path

from phase3b_same_param_experiment import (
    generate_phase3b_commands,
    load_protocol,
    validate_protocol,
    write_commands,
)


def protocol():
    return {
        "schema_version": "phase3b.same_param.v1",
        "datasets": ["eurosat", "caltech101"],
        "shots": [4],
        "seeds": [1, 2],
        "methods": [
            {"adapter": "lora", "r": 2, "alpha": 1},
            {"adapter": "ohsinglora", "r": 2, "alpha": 1, "num_heads": 2, "lambda_o": 0.03},
        ],
        "base_command": {
            "python": "$PYTHON",
            "entrypoint": "main.py",
            "root_path": "${DATA_ROOT}",
            "run_manifest": "revision_materials/results/phase3b_same_param_results.jsonl",
            "save_path": "revision_materials/checkpoints/phase3b_same_param",
            "log_dir": "revision_materials/logs/phase3b_same_param",
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


class Phase3BSameParamExperimentTest(unittest.TestCase):
    def test_generate_same_parameter_commands(self):
        commands = generate_phase3b_commands(protocol())

        self.assertEqual(len(commands), 8)
        self.assertTrue(all("--selection_split test" in command for command in commands))
        self.assertTrue(all("--report_test" in command for command in commands))
        self.assertTrue(all("--sweep_mode" not in command for command in commands))
        self.assertEqual(sum("--adapter lora" in command for command in commands), 4)
        self.assertEqual(sum("--adapter ohsinglora" in command for command in commands), 4)
        self.assertTrue(all("--adapter lora --r 2 --alpha 1" in command for command in commands if "--adapter lora" in command))
        self.assertTrue(all("--num_heads" not in command and "--lambda_o" not in command for command in commands if "--adapter lora" in command))
        self.assertTrue(any("--adapter ohsinglora --num_heads 2 --r 2 --alpha 1 --lambda_o 0.03" in command for command in commands))
        self.assertTrue(any("--filename eurosat_4shot_seed1_test_lora_r2" in command for command in commands))
        self.assertTrue(any("--filename eurosat_4shot_seed1_test_ohsinglora_h2_r2_lo0p03" in command for command in commands))

    def test_frozen_protocol_generates_144_runs(self):
        frozen = load_protocol("revision_materials/plan/phase3b_same_param_protocol.yaml")
        commands = generate_phase3b_commands(frozen)

        self.assertEqual(len(commands), 144)
        self.assertEqual(sum("--adapter lora" in command for command in commands), 72)
        self.assertEqual(sum("--adapter ohsinglora" in command for command in commands), 72)
        self.assertTrue(all("--adapter lora --r 2 --alpha 1" in command for command in commands if "--adapter lora" in command))
        self.assertTrue(all("--num_heads" not in command and "--lambda_o" not in command for command in commands if "--adapter lora" in command))
        self.assertTrue(all("--adapter singlora" not in command for command in commands))

    def test_validate_protocol_rejects_non_legacy_claim_config(self):
        bad = protocol()
        bad["methods"][1]["lambda_o"] = 0.0

        with self.assertRaisesRegex(ValueError, "Phase 3B OrthoAdapt"):
            validate_protocol(bad)

    def test_write_commands_records_run_stamp_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "phase3b.sh"
            write_commands(["echo ok"], out)

            text = out.read_text(encoding="utf-8")

        self.assertIn(': "${RUN_STAMP:=$(date +%Y%m%d_%H%M%S)}"', text)

    def test_load_protocol_accepts_json_compatible_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase3b.yaml"
            path.write_text(json.dumps(protocol(), indent=2), encoding="utf-8")

            loaded = load_protocol(path)

        self.assertEqual(loaded["schema_version"], "phase3b.same_param.v1")


if __name__ == "__main__":
    unittest.main()
