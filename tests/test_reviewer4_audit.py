import unittest

import reviewer4_audit as audit


def records(status, routing="input_independent", accuracy_offset=0.0):
    output = []
    for dataset_index in range(8):
        for shot in (1, 4, 16):
            for seed in (1, 2, 3):
                output.append({
                    "status": status,
                    "config": {
                        "dataset": f"d{dataset_index}", "shots": shot, "seed": seed,
                        "adapter": "ohsinglora", "routing_mode": routing,
                    },
                    "dataset": {
                        "classnames_sha256": f"classes-{dataset_index}",
                        "splits": {
                            "train": {"sha256": f"train-{dataset_index}-{shot}-{seed}"},
                            "val": {"sha256": f"val-{dataset_index}-{shot}-{seed}"},
                            "test": {"sha256": f"test-{dataset_index}"},
                        },
                    },
                    "metrics": {
                        "test_accuracy": 70 + dataset_index + accuracy_offset,
                        "trainable_parameters": 460800,
                    },
                    "checkpoint": {"path": "unused.pt", "sha256": "A" * 64},
                })
    return output


class Reviewer4AuditTest(unittest.TestCase):
    def test_smoke_and_sentinel_manifest_gates(self):
        smoke_train = records("completed")[0]
        smoke_eval = records("eval_only")[0]
        for row in (smoke_train, smoke_eval):
            row["config"].update({"dataset": "eurosat", "shots": 4, "seed": 1})
        smoke_eval["metrics"]["test_accuracy"] += 0.02
        self.assertTrue(audit.audit_smoke([smoke_train], [smoke_eval])["passed"])

        train, evaluate, reference = [], [], []
        for dataset in ("eurosat", "caltech101"):
            for shot in (1, 4, 16):
                base = records("completed")[0]
                base["config"].update({
                    "dataset": dataset, "shots": shot, "seed": 1,
                    "routing_mode": "input_conditioned",
                })
                reload = {**base, "status": "eval_only", "metrics": dict(base["metrics"])}
                old = {**base, "metrics": dict(base["metrics"])}
                base["metrics"]["test_accuracy"] = 80.2
                reload["metrics"]["test_accuracy"] = 80.2
                old["metrics"]["test_accuracy"] = 80.0
                train.append(base)
                evaluate.append(reload)
                reference.append(old)
        self.assertTrue(audit.audit_sentinel_manifests(train, evaluate, reference)["passed"])

    def test_freeze_reference_selects_only_72_selected_dynamic_rows(self):
        dynamic = records("completed", routing="input_conditioned")
        for row in dynamic:
            row["config"].update({"num_heads": 2, "r": 8, "lambda_o": 0.03, "ramp_up_steps": 100})
        lora = records("completed")
        for row in lora:
            row["config"]["adapter"] = "lora"
        frozen = audit.freeze_dynamic_reference(dynamic + lora)
        self.assertEqual(len(frozen), 72)

    def test_static_train_eval_matrix_passes(self):
        report = audit.audit_accuracy_matrix(records("completed"), records("eval_only", accuracy_offset=0.02))
        self.assertTrue(report["passed"])
        self.assertEqual(report["unique_configurations"], 72)
        self.assertAlmostEqual(report["max_train_eval_accuracy_difference_pp"], 0.02)

        dynamic_train = records("completed", routing="input_conditioned")
        dynamic_eval = records("eval_only", routing="input_conditioned", accuracy_offset=0.02)
        dynamic_report = audit.audit_accuracy_matrix(
            dynamic_train, dynamic_eval, routing_mode="input_conditioned",
        )
        self.assertTrue(dynamic_report["passed"])

    def test_duplicate_missing_bad_parameter_and_reload_drift_fail(self):
        train = records("completed")
        evaluate = records("eval_only")
        with self.assertRaises(ValueError):
            audit.audit_accuracy_matrix(train + [train[0]], evaluate)
        with self.assertRaises(ValueError):
            audit.audit_accuracy_matrix(train[:-1], evaluate)
        bad = records("completed")
        bad[0]["metrics"]["trainable_parameters"] = 1
        with self.assertRaises(ValueError):
            audit.audit_accuracy_matrix(bad, evaluate)
        drift = records("eval_only")
        drift[0]["metrics"]["test_accuracy"] += 0.051
        with self.assertRaises(ValueError):
            audit.audit_accuracy_matrix(train, drift)

    def test_sentinel_gate_uses_mean_and_max_absolute_difference(self):
        old = [
            {"dataset": d, "shot": s, "seed": 1, "accuracy": 80.0}
            for d in ("eurosat", "caltech101") for s in (1, 4, 16)
        ]
        new = [dict(row, accuracy=80.4) for row in old]
        report = audit.audit_dynamic_sentinel(new, old)
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["mean_absolute_difference_pp"], 0.4)
        failing = [dict(row, accuracy=(81.01 if index == 0 else 80.0)) for index, row in enumerate(old)]
        with self.assertRaises(ValueError):
            audit.audit_dynamic_sentinel(failing, old)

    def test_compute_environment_requires_one_physical_3090(self):
        profiles = [
            {"gpu_name": "NVIDIA GeForce RTX 3090", "gpu_uuid": "GPU-1", "software_stack_id": "stack-a", "power_limit_watts": "350.00"}
            for _ in range(24)
        ]
        self.assertTrue(audit.audit_profile_environment(profiles)["passed"])
        profiles[-1]["gpu_uuid"] = "GPU-2"
        with self.assertRaises(ValueError):
            audit.audit_profile_environment(profiles)


if __name__ == "__main__":
    unittest.main()
