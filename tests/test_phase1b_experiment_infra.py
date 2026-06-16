import json
import tempfile
import unittest
from pathlib import Path

import torch

from aggregate_results import aggregate_manifest, load_manifest_jsonl
from experiment_manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    count_named_adapter_parameters,
    normalize_run_record,
    validate_run_record,
)


def run_record(
    *,
    dataset="eurosat",
    method="ohsinglora",
    shot=4,
    seed=1,
    split="val",
    accuracy=88.0,
    runtime_seconds=12.5,
    parameter_count=1234,
    checkpoint_sha256="A" * 64,
    git_revision="abc123",
):
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_utc": "2026-06-16T00:00:00+00:00",
        "git_revision": git_revision,
        "command": ["python", "main.py"],
        "config": {
            "dataset": dataset,
            "adapter": method,
            "shots": shot,
            "seed": seed,
            "selection_split": split,
        },
        "metrics": {
            f"{split}_accuracy": accuracy,
            "runtime_seconds": runtime_seconds,
            "trainable_parameters": parameter_count,
        },
        "checkpoint": {
            "path": f"checkpoints/{dataset}_{method}_{shot}_{seed}.pt",
            "sha256": checkpoint_sha256,
        },
        "dataset": {
            "splits": {
                split: {
                    "count": 10,
                    "sha256": "B" * 64,
                }
            }
        },
        "status": "completed",
    }


class Phase1BExperimentInfrastructureTest(unittest.TestCase):
    def test_count_adapter_parameters_excludes_frozen_backbone_weights(self):
        module = torch.nn.Module()
        module.backbone = torch.nn.Linear(5, 7)
        module.lora_A = torch.nn.Linear(5, 2, bias=False)
        module.scaler = torch.nn.Parameter(torch.ones(3))

        self.assertEqual(count_named_adapter_parameters(module.named_parameters()), 13)

    def test_validate_run_record_requires_phase1b_fields(self):
        record = run_record()

        normalized = validate_run_record(record)

        self.assertEqual(normalized["schema_version"], RUN_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(normalized["dataset"], "eurosat")
        self.assertEqual(normalized["method"], "ohsinglora")
        self.assertEqual(normalized["shot"], 4)
        self.assertEqual(normalized["seed"], 1)
        self.assertEqual(normalized["split"], "val")
        self.assertEqual(normalized["accuracy"], 88.0)
        self.assertEqual(normalized["runtime_seconds"], 12.5)
        self.assertEqual(normalized["parameter_count"], 1234)
        self.assertEqual(normalized["checkpoint_sha256"], "A" * 64)
        self.assertEqual(normalized["git_revision"], "abc123")

    def test_validate_run_record_fails_closed_when_required_field_is_missing(self):
        record = run_record()
        del record["checkpoint"]["sha256"]

        with self.assertRaisesRegex(ValueError, "checkpoint.sha256"):
            validate_run_record(record)

    def test_normalize_run_record_accepts_flat_records_for_dry_run_fixtures(self):
        flat = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "dataset": "dtd",
            "method": "lora",
            "shot": 16,
            "seed": 3,
            "split": "test",
            "accuracy": 77.5,
            "runtime_seconds": 8.0,
            "parameter_count": 555,
            "checkpoint_sha256": "C" * 64,
            "git_revision": "def456",
        }

        self.assertEqual(normalize_run_record(flat), flat)

    def test_aggregate_manifest_reports_mean_std_ci_and_paired_effect_size(self):
        rows = [
            run_record(method="lora", seed=1, accuracy=80.0, checkpoint_sha256="1" * 64),
            run_record(method="lora", seed=2, accuracy=82.0, checkpoint_sha256="2" * 64),
            run_record(method="lora", seed=3, accuracy=84.0, checkpoint_sha256="3" * 64),
            run_record(method="ohsinglora", seed=1, accuracy=83.0, checkpoint_sha256="4" * 64),
            run_record(method="ohsinglora", seed=2, accuracy=85.0, checkpoint_sha256="5" * 64),
            run_record(method="ohsinglora", seed=3, accuracy=87.0, checkpoint_sha256="6" * 64),
        ]

        report = aggregate_manifest(rows, baseline_method="lora")

        key = ("eurosat", 4, "ohsinglora", "val")
        self.assertEqual(report["groups"][key]["n"], 3)
        self.assertAlmostEqual(report["groups"][key]["mean_accuracy"], 85.0)
        self.assertAlmostEqual(report["groups"][key]["std_accuracy"], 2.0)
        self.assertGreater(report["groups"][key]["ci95_half_width"], 0.0)
        self.assertEqual(report["paired"][key]["baseline_method"], "lora")
        self.assertEqual(report["paired"][key]["paired_n"], 3)
        self.assertAlmostEqual(report["paired"][key]["mean_delta"], 3.0)
        self.assertGreater(report["paired"][key]["effect_size_dz"], 0.0)

    def test_load_manifest_jsonl_rejects_invalid_rows_before_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            path.write_text(json.dumps({"dataset": "eurosat"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1"):
                load_manifest_jsonl(path)


if __name__ == "__main__":
    unittest.main()
