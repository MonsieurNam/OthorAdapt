import unittest

from reviewer4_provenance import split_identity, validate_split_reference, validate_target_environment


def dataset_payload(tag="same"):
    return {
        "classnames_sha256": f"classes-{tag}",
        "splits": {
            "train": {"sha256": f"train-{tag}"},
            "val": {"sha256": f"val-{tag}"},
            "test": {"sha256": f"test-{tag}"},
        },
    }


class Reviewer4ProvenanceTest(unittest.TestCase):
    def test_target_environment_requires_one_visible_3090(self):
        environment = {
            "gpu_name": "NVIDIA GeForce RTX 3090", "gpu_uuid": "GPU-1",
            "visible_gpu_count": 1, "os": "Linux-x86_64",
            "os_distribution": "Ubuntu 22.04.5 LTS",
            "python": "3.11.10", "pytorch": "2.5.1+cu124",
            "cuda_runtime": "12.4", "cudnn": 90100,
        }
        self.assertEqual(validate_target_environment(environment), environment)
        with self.assertRaises(ValueError):
            validate_target_environment({**environment, "visible_gpu_count": 2})

    def test_split_identity_covers_classes_and_all_splits(self):
        first = split_identity(dataset_payload())
        changed = dataset_payload()
        changed["splits"]["val"]["sha256"] = "changed"
        self.assertNotEqual(first, split_identity(changed))

    def test_reference_gate_accepts_exact_key_and_hashes(self):
        reference = [{
            "config": {"dataset": "eurosat", "shots": 4, "seed": 1},
            "dataset": dataset_payload(),
        }]
        result = validate_split_reference("eurosat", 4, 1, dataset_payload(), reference)
        self.assertTrue(result["passed"])

    def test_reference_gate_rejects_missing_duplicate_or_mismatch(self):
        row = {
            "config": {"dataset": "eurosat", "shots": 4, "seed": 1},
            "dataset": dataset_payload(),
        }
        with self.assertRaises(ValueError):
            validate_split_reference("eurosat", 4, 1, dataset_payload(), [])
        with self.assertRaises(ValueError):
            validate_split_reference("eurosat", 4, 1, dataset_payload(), [row, row])
        with self.assertRaises(ValueError):
            validate_split_reference("eurosat", 4, 1, dataset_payload("new"), [row])


if __name__ == "__main__":
    unittest.main()
