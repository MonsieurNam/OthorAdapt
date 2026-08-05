import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

from adapter_checkpoint_v2 import (
    CHECKPOINT_SCHEMA_VERSION,
    adapter_metadata,
    validate_adapter_metadata,
)
from loralib.layers_singlora import LinearSingLoRA
from loralib.layers_OH_singlora import LinearOHsingLoRA
from loralib.utils import load_adapter, save_adapter


def args_for(save_path, **overrides):
    values = {
        "save_path": str(save_path),
        "adapter": "singlora",
        "backbone": "ViT-B/16",
        "dataset": "eurosat",
        "shots": 4,
        "seed": 1,
        "filename": "eurosat_4shot_seed1_test_singlora_r8_ramp100",
        "r": 2,
        "alpha": 1,
        "num_heads": 2,
        "ramp_up_steps": 2,
        "params": ["v", "q", "k"],
        "position": "all",
        "encoder": "both",
        "lambda_o": 0.0,
        "ortho_reduction": "mean",
        "checkpoint_extra_metadata": {"ecr3": {"schema_version": "ecr3.checkpoint.v1"}},
    }
    values.update(overrides)
    return Namespace(**values)


class TinyAdapterModel(nn.Module):
    def __init__(self, ramp_up_steps=2):
        super().__init__()
        base = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            base.weight.copy_(torch.eye(4))
        self.adapter = LinearSingLoRA(base, r=2, lora_alpha=1, ramp_up_steps=ramp_up_steps)

    def forward(self, inputs):
        return self.adapter(inputs)


class TinyOHAdapterModel(nn.Module):
    def __init__(self, ramp_up_steps=2):
        super().__init__()
        base = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            base.weight.copy_(torch.eye(4))
        self.adapter = LinearOHsingLoRA(
            base,
            r=2,
            lora_alpha=1,
            ramp_up_steps=ramp_up_steps,
            num_heads=2,
        )

    def forward(self, inputs):
        return self.adapter(inputs)


class AdapterCheckpointV2Test(unittest.TestCase):
    def test_metadata_is_canonical_and_singlora_has_one_effective_head(self):
        metadata = adapter_metadata(args_for("unused"))

        self.assertEqual(metadata["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(metadata["adapter"], "singlora")
        self.assertEqual(metadata["num_heads"], 1)
        self.assertEqual(metadata["params"], ["q", "k", "v"])
        self.assertEqual(metadata["ramp_up_steps"], 2)

    def test_metadata_validation_reports_every_mismatched_compatibility_field(self):
        args = args_for("unused")
        replacements = {
            "adapter": "ohsinglora",
            "r": 4,
            "num_heads": 2,
            "alpha": 2,
            "ramp_up_steps": 100,
            "params": ["q", "v"],
            "position": "top3",
            "encoder": "vision",
            "backbone": "ViT-B/32",
        }
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                metadata = adapter_metadata(args)
                metadata[field] = replacement
                with self.assertRaisesRegex(ValueError, field):
                    validate_adapter_metadata(metadata, args)

    def test_checkpoint_roundtrip_preserves_output_and_training_step(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp)
            trained = TinyAdapterModel()
            with torch.no_grad():
                trained.adapter.lora_A.copy_(torch.randn_like(trained.adapter.lora_A))
                trained.adapter.training_step.fill_(2)
            trained.eval()
            inputs = torch.randn(3, 4)
            expected = trained(inputs)

            checkpoint_path = save_adapter(args, trained)
            fresh = TinyAdapterModel()
            fresh.eval()
            loaded_path = load_adapter(args, fresh)

            self.assertEqual(Path(loaded_path), Path(checkpoint_path))
            self.assertEqual(fresh.adapter.training_step.item(), 2)
            torch.testing.assert_close(fresh(inputs), expected)
            payload = torch.load(checkpoint_path, map_location="cpu")
            self.assertIn("adapter.training_step", payload["weights"])
            self.assertEqual(payload["metadata"]["schema_version"], CHECKPOINT_SCHEMA_VERSION)
            self.assertIn("ecr3", payload["metadata"])

    def test_orthoadapt_roundtrip_preserves_gate_weights_and_training_step(self):
        torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp, adapter="ohsinglora", num_heads=2)
            trained = TinyOHAdapterModel()
            with torch.no_grad():
                trained.adapter.lora_A_heads.copy_(torch.randn_like(trained.adapter.lora_A_heads))
                trained.adapter.gating_network[0].weight.copy_(
                    torch.randn_like(trained.adapter.gating_network[0].weight)
                )
                trained.adapter.training_step.fill_(2)
            trained.eval()
            inputs = torch.randn(3, 4)
            expected = trained(inputs)

            checkpoint_path = save_adapter(args, trained)
            fresh = TinyOHAdapterModel()
            fresh.eval()
            load_adapter(args, fresh)

            self.assertEqual(fresh.adapter.training_step.item(), 2)
            torch.testing.assert_close(
                fresh.adapter.gating_network[0].weight,
                trained.adapter.gating_network[0].weight,
            )
            torch.testing.assert_close(fresh(inputs), expected)
            payload = torch.load(checkpoint_path, map_location="cpu")
            self.assertIn("adapter.gating_network.0.weight", payload["weights"])

    def test_load_rejects_unsaturated_ramp(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp)
            model = TinyAdapterModel()
            model.adapter.training_step.fill_(1)
            save_adapter(args, model)

            with self.assertRaisesRegex(ValueError, "not saturated"):
                load_adapter(args, TinyAdapterModel())

    def test_load_rejects_missing_unexpected_and_shape_mismatched_state(self):
        mutations = {
            "missing": lambda weights: weights.pop("adapter.training_step"),
            "unexpected": lambda weights: weights.__setitem__("adapter.unexpected_lora_key", torch.zeros(1)),
            "shape": lambda weights: weights.__setitem__("adapter.lora_A", torch.zeros(1, 1)),
        }
        patterns = {"missing": "missing", "unexpected": "unexpected", "shape": "shape"}

        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                args = args_for(tmp)
                model = TinyAdapterModel()
                model.adapter.training_step.fill_(2)
                checkpoint_path = save_adapter(args, model)
                payload = torch.load(checkpoint_path, map_location="cpu")
                mutate(payload["weights"])
                torch.save(payload, checkpoint_path)

                with self.assertRaisesRegex(ValueError, patterns[label]):
                    load_adapter(args, TinyAdapterModel())


if __name__ == "__main__":
    unittest.main()
