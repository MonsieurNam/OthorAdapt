import json
import tempfile
import unittest
from pathlib import Path

import reviewer4_gating_experiment as experiment


def protocol():
    return {
        "schema_version": "reviewer4.gating.v1",
        "datasets": [
            "caltech101", "dtd", "eurosat", "fgvc",
            "oxford_flowers", "food101", "oxford_pets", "ucf101",
        ],
        "shots": [1, 4, 16],
        "seeds": [1, 2, 3],
        "sentinel_datasets": ["eurosat", "caltech101"],
        "sentinel_seed": 1,
        "expected_trainable_parameters": 460800,
        "method": {
            "adapter": "ohsinglora", "routing_mode": "input_independent",
            "num_heads": 2, "r": 8, "alpha": 1, "lambda_o": 0.03,
            "ramp_up_steps": 100, "ortho_reduction": "mean",
        },
        "base_command": {
            "python": "${PYTHON}", "entrypoint": "main.py",
            "root_path": "${DATA_ROOT}", "save_path": "artifacts/checkpoints",
            "run_manifest": "artifacts/static_train.jsonl", "log_dir": "artifacts/logs",
            "sentinel_run_manifest": "artifacts/dynamic_sentinel_train.jsonl",
            "dynamic_run_manifest": "artifacts/dynamic_sentinel_train.jsonl",
            "split_reference_manifest": "artifacts/frozen_dynamic_reference.jsonl",
            "backbone": "ViT-B/16", "encoder": "both", "position": "all",
            "params": ["q", "k", "v"], "n_iters": 500, "batch_size": 32,
            "loss_fn": "ce",
        },
    }


class Reviewer4ProtocolTest(unittest.TestCase):
    def test_static_matrix_has_72_unique_configs(self):
        specs = experiment.build_specs(protocol(), routing_mode="input_independent")
        self.assertEqual(len(specs), 72)
        self.assertEqual(len({spec.run_key for spec in specs}), 72)
        self.assertTrue(all(spec.expected_trainable_parameters == 460800 for spec in specs))

    def test_dynamic_sentinel_has_six_configs(self):
        specs = experiment.build_sentinel_specs(protocol())
        self.assertEqual(len(specs), 6)
        self.assertTrue(all(spec.routing_mode == "input_conditioned" for spec in specs))
        command = experiment.command_for_spec(protocol(), specs[0], stage="train")
        self.assertIn("--run_manifest artifacts/dynamic_sentinel_train.jsonl", command)

    def test_commands_lock_main_protocol_and_eval_only(self):
        spec = experiment.build_specs(protocol(), routing_mode="input_independent")[0]
        train = experiment.command_for_spec(protocol(), spec, stage="train")
        evaluate = experiment.command_for_spec(protocol(), spec, stage="eval")
        for required in (
            "--routing_mode input_independent", "--num_heads 2", "--r 8",
            "--alpha 1", "--lambda_o 0.03", "--ramp_up_steps 100",
            "--ortho_reduction mean", "--encoder both", "--position all",
            "--params q k v", "--n_iters 500", "--batch_size 32",
            "--selection_split test", "--report_test",
            "--split_reference_manifest artifacts/frozen_dynamic_reference.jsonl",
        ):
            self.assertIn(required, train)
        self.assertNotIn("--eval_only", train)
        self.assertIn("--eval_only", evaluate)

    def test_smoke_is_static_eurosat_four_shot_seed_one(self):
        spec = experiment.build_smoke_spec(protocol())
        self.assertEqual((spec.dataset, spec.shot, spec.seed), ("eurosat", 4, 1))
        self.assertEqual(spec.routing_mode, "input_independent")

    def test_resume_filters_completed_without_duplicates(self):
        specs = experiment.build_specs(protocol(), routing_mode="input_independent")
        remaining = experiment.filter_completed(specs, {specs[0].run_key, specs[1].run_key})
        self.assertEqual(len(remaining), 70)
        self.assertEqual(len({spec.run_key for spec in remaining}), 70)

    def test_protocol_rejects_scientific_drift(self):
        for path, value in (
            (("method", "r"), 4),
            (("method", "lambda_o"), 0.0),
            (("base_command", "batch_size"), 64),
        ):
            bad = protocol()
            bad[path[0]][path[1]] = value
            with self.assertRaises(ValueError):
                experiment.validate_protocol(bad)

        missing_reference = protocol()
        del missing_reference["base_command"]["split_reference_manifest"]
        with self.assertRaises(ValueError):
            experiment.validate_protocol(missing_reference)

    def test_completed_keys_are_read_from_manifest(self):
        spec = experiment.build_specs(protocol(), routing_mode="input_independent")[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text(json.dumps({"reviewer4_run_key": spec.run_key, "status": "completed"}) + "\n")
            self.assertEqual(experiment.completed_keys(path, "completed"), {spec.run_key})

            path.write_text(json.dumps({
                "status": "completed",
                "config": {
                    "dataset": spec.dataset, "shots": spec.shot, "seed": spec.seed,
                    "routing_mode": spec.routing_mode,
                },
            }) + "\n")
            self.assertEqual(experiment.completed_keys(path, "completed"), {spec.run_key})


if __name__ == "__main__":
    unittest.main()
