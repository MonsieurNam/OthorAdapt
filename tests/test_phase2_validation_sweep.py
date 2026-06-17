import json
import tempfile
import unittest
from pathlib import Path

from phase2_validation_sweep import (
    candidate_grid,
    generate_sweep_commands,
    load_protocol,
    select_winner,
    write_selected_config,
)


def protocol():
    return {
        "schema_version": "phase2.selection.v1",
        "selection_split": "val",
        "selection_metric": {
            "datasets": ["eurosat", "caltech101"],
            "shots": [4],
            "seeds": [1, 2],
        },
        "candidate_grid": {
            "num_heads": [2, 4],
            "r": [2, 4, 8],
            "lambda_o": [0.0, 0.01, 0.03, 0.05],
        },
        "base_command": {
            "python": "$PYTHON",
            "entrypoint": "main.py",
            "root_path": "${DATA_ROOT}",
            "run_manifest": "revision_materials/results/validation_sweep_results.jsonl",
            "save_path": "revision_materials/checkpoints/validation_sweep",
            "log_dir": "revision_materials/logs/validation_sweep",
            "adapter": "ohsinglora",
            "backbone": "ViT-B/16",
            "encoder": "both",
            "position": "all",
            "params": ["q", "k", "v"],
            "n_iters": 500,
            "batch_size": 32,
            "loss_fn": "ce",
            "ortho_reduction": "mean",
        },
        "tie_break": ["parameter_count_asc", "lambda_o_asc"],
        "require_all_candidates": False,
    }


def row(dataset, seed, num_heads, r, lambda_o, accuracy, parameter_count=1000):
    return {
        "schema_version": "phase1b.run.v1",
        "dataset": dataset,
        "method": "ohsinglora",
        "shot": 4,
        "seed": seed,
        "split": "val",
        "accuracy": accuracy,
        "runtime_seconds": 10.0,
        "parameter_count": parameter_count,
        "checkpoint_sha256": f"{seed}{num_heads}{r}".ljust(64, "A"),
        "git_revision": "TESTREV",
        "config": {
            "num_heads": num_heads,
            "r": r,
            "lambda_o": lambda_o,
            "selection_split": "val",
        },
    }


class Phase2ValidationSweepTest(unittest.TestCase):
    def test_candidate_grid_keeps_only_rank_divisible_by_head_count(self):
        candidates = candidate_grid(protocol())

        self.assertEqual(len(candidates), 20)
        self.assertIn({"num_heads": 4, "r": 4, "lambda_o": 0.03}, candidates)
        self.assertIn({"num_heads": 2, "r": 2, "lambda_o": 0.0}, candidates)
        self.assertNotIn({"num_heads": 4, "r": 2, "lambda_o": 0.01}, candidates)
        self.assertNotIn({"num_heads": 1, "r": 2, "lambda_o": 0.01}, candidates)

    def test_generate_sweep_commands_are_validation_only_and_manifested(self):
        commands = generate_sweep_commands(protocol())

        self.assertEqual(len(commands), 80)
        self.assertTrue(all(command.startswith("mkdir -p revision_materials/logs/validation_sweep && $PYTHON main.py ") for command in commands))
        self.assertTrue(all("--selection_split val" in command for command in commands))
        self.assertTrue(all("--sweep_mode" in command for command in commands))
        self.assertTrue(all("--run_manifest revision_materials/results/validation_sweep_results.jsonl" in command for command in commands))
        self.assertTrue(all("2>&1 | tee revision_materials/logs/validation_sweep/" in command for command in commands))
        self.assertTrue(any("--filename eurosat_4shot_seed1_val_ohsinglora_h2_r2_lo0p0" in command for command in commands))
        self.assertTrue(any("eurosat_4shot_seed1_val_ohsinglora_h2_r2_lo0p0_${RUN_STAMP}.log" in command for command in commands))
        self.assertTrue(all("--report_test" not in command for command in commands))

    def test_frozen_protocol_generates_120_runs(self):
        frozen = load_protocol("revision_materials/plan/selection_protocol.yaml")
        commands = generate_sweep_commands(frozen)

        self.assertEqual(len(candidate_grid(frozen)), 20)
        self.assertEqual(len(commands), 120)
        self.assertTrue(all("--num_heads 1" not in command for command in commands))
        self.assertTrue(all("--lambda_o 0.1" not in command for command in commands))
        self.assertTrue(any("--lambda_o 0.0" in command for command in commands))

    def test_select_winner_uses_unweighted_mean_then_tie_breaks(self):
        rows = [
            row("eurosat", 1, 2, 2, 0.03, 80.0, parameter_count=900),
            row("eurosat", 2, 2, 2, 0.03, 82.0, parameter_count=900),
            row("caltech101", 1, 2, 2, 0.03, 84.0, parameter_count=900),
            row("caltech101", 2, 2, 2, 0.03, 86.0, parameter_count=900),
            row("eurosat", 1, 4, 4, 0.01, 81.0, parameter_count=800),
            row("eurosat", 2, 4, 4, 0.01, 82.0, parameter_count=800),
            row("caltech101", 1, 4, 4, 0.01, 84.0, parameter_count=800),
            row("caltech101", 2, 4, 4, 0.01, 85.0, parameter_count=800),
        ]

        winner = select_winner(rows, protocol())

        self.assertEqual(winner["config"], {"num_heads": 4, "r": 4, "lambda_o": 0.01})
        self.assertAlmostEqual(winner["mean_validation_accuracy"], 83.0)
        self.assertEqual(winner["n"], 4)

    def test_select_winner_rejects_test_split_rows(self):
        bad = row("eurosat", 1, 2, 2, 0.03, 80.0)
        bad["split"] = "test"

        with self.assertRaisesRegex(ValueError, "non-validation"):
            select_winner([bad], protocol())

    def test_select_winner_can_require_all_candidates(self):
        strict = protocol()
        strict["require_all_candidates"] = True
        rows = [
            row("eurosat", 1, 2, 2, 0.03, 80.0),
            row("eurosat", 2, 2, 2, 0.03, 82.0),
            row("caltech101", 1, 2, 2, 0.03, 84.0),
            row("caltech101", 2, 2, 2, 0.03, 86.0),
        ]

        with self.assertRaisesRegex(ValueError, "Missing complete validation coverage"):
            select_winner(rows, strict)

    def test_write_selected_config_records_hash_sidecar(self):
        winner = {
            "config": {"num_heads": 2, "r": 2, "lambda_o": 0.03},
            "mean_validation_accuracy": 83.0,
            "n": 4,
            "covered_datasets": ["caltech101", "eurosat"],
            "covered_seeds": [1, 2],
            "covered_shots": [4],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "selected_config.md"
            digest = write_selected_config(winner, protocol(), out)

            self.assertTrue(out.exists())
            self.assertTrue((Path(str(out) + ".sha256")).exists())
            self.assertIn(digest, (Path(str(out) + ".sha256")).read_text(encoding="utf-8"))

    def test_load_protocol_accepts_json_compatible_yaml_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_protocol.yaml"
            path.write_text(json.dumps(protocol(), indent=2), encoding="utf-8")

            loaded = load_protocol(path)

        self.assertEqual(loaded["schema_version"], "phase2.selection.v1")


if __name__ == "__main__":
    unittest.main()
