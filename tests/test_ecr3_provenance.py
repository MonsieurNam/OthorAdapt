import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from ecr3_provenance import (
    args_to_config,
    build_run_record,
    dataset_provenance,
    file_sha256,
    validate_reporting_policy,
    write_jsonl_record,
)


class Item:
    def __init__(self, impath, label, classname):
        self.impath = impath
        self.label = label
        self.classname = classname
        self.domain = -1


class Dataset:
    def __init__(self):
        self.train_x = [
            Item("b.jpg", 1, "beta"),
            Item("a.jpg", 0, "alpha"),
        ]
        self.val = [Item("v.jpg", 0, "alpha")]
        self.test = [Item("t.jpg", 1, "beta")]
        self.classnames = ["alpha", "beta"]


class ECR3ProvenanceTest(unittest.TestCase):
    def test_dataset_provenance_hashes_splits_stably(self):
        first = dataset_provenance(Dataset())
        second = dataset_provenance(Dataset())

        self.assertEqual(first["splits"]["train"]["count"], 2)
        self.assertEqual(first["splits"]["val"]["class_counts"], {"0": 1})
        self.assertEqual(first["splits"]["test"]["class_counts"], {"1": 1})
        self.assertEqual(first["splits"]["train"]["sha256"], second["splits"]["train"]["sha256"])
        self.assertNotEqual(first["splits"]["train"]["sha256"], first["splits"]["test"]["sha256"])

    def test_build_run_record_includes_config_split_metrics_and_gate_status(self):
        args = Namespace(
            seed=3,
            dataset="eurosat",
            shots=4,
            adapter="ohsinglora",
            r=2,
            num_heads=2,
            lambda_o=0.03,
            params=["q", "k", "v"],
            run_manifest="out.jsonl",
            git_revision="TESTREV",
        )
        record = build_run_record(
            args,
            dataset_provenance(Dataset()),
            metrics={"val_accuracy": 88.1, "test_accuracy": 87.9, "trainable_parameters": 42},
            checkpoint_path="checkpoints/model.pt",
            checkpoint_sha256="ABC123",
            status="completed",
        )

        self.assertEqual(record["schema_version"], "ecr3.run.v1")
        self.assertEqual(record["config"]["seed"], 3)
        self.assertEqual(record["config"]["adapter"], "ohsinglora")
        self.assertEqual(record["metrics"]["val_accuracy"], 88.1)
        self.assertEqual(record["checkpoint"]["sha256"], "ABC123")
        self.assertEqual(record["git_revision"], "TESTREV")
        self.assertEqual(record["metrics"]["trainable_parameters"], 42)
        self.assertEqual(record["evidence_gate"]["seed_status"], "row_seed_recorded")
        self.assertEqual(record["evidence_gate"]["split_status"], "split_hash_recorded")

    def test_write_jsonl_record_appends_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            write_jsonl_record(path, {"b": 2, "a": 1})
            write_jsonl_record(path, {"c": 3})

            rows = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(rows, [{"a": 1, "b": 2}, {"c": 3}])

    def test_validate_reporting_policy_blocks_test_set_tuning(self):
        validate_reporting_policy("val", sweep_mode=True, report_test=False)

        with self.assertRaisesRegex(ValueError, "Sweep mode cannot use the test split"):
            validate_reporting_policy("test", sweep_mode=True, report_test=True)

        with self.assertRaisesRegex(ValueError, "requires --report_test"):
            validate_reporting_policy("test", sweep_mode=False, report_test=False)

    def test_config_excludes_internal_provenance_payloads(self):
        args = Namespace(
            seed=1,
            dataset="dtd",
            checkpoint_extra_metadata={"large": "payload"},
            dataset_provenance={"splits": {}},
            run_command=["main.py"],
        )

        config = args_to_config(args)

        self.assertEqual(config, {"dataset": "dtd", "seed": 1})

    def test_file_sha256_records_checkpoint_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter_weights.pt"
            path.write_bytes(b"checkpoint")

            self.assertEqual(
                file_sha256(path),
                "47320987F9A49D5B00119B960F247A956773F57543982B8BFCB6DA5BB3AFD9EF",
            )


if __name__ == "__main__":
    unittest.main()
