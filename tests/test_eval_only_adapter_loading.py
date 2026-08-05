import unittest
from argparse import Namespace
from unittest.mock import patch
import sys
import types

import torch

clip_stub = types.ModuleType("clip")
clip_stub.tokenize = lambda texts: texts
sys.modules.setdefault("clip", clip_stub)

import lora


class FakeClipModel:
    def cuda(self):
        return self

    def named_parameters(self):
        return []


class FakeDataset:
    classnames = ["class_a", "class_b"]
    template = ["a photo of {}"]


class EvalOnlyAdapterLoadingTest(unittest.TestCase):
    def test_eval_only_loads_into_clip_model_and_records_checkpoint_hash(self):
        args = Namespace(
            adapter="singlora",
            eval_only=True,
            selection_split="test",
            report_test=True,
            run_manifest="eval.jsonl",
            dataset_provenance={"splits": {"test": {"sha256": "DATA"}}},
        )
        clip_model = FakeClipModel()
        features = torch.eye(2)
        labels = torch.tensor([0, 1])

        with (
            patch.object(torch.Tensor, "cuda", lambda self: self),
            patch.object(lora, "clip_classifier", return_value=features),
            patch.object(lora, "pre_load_features", return_value=(features, labels)),
            patch.object(lora, "cls_acc", return_value=50.0),
            patch.object(lora, "apply_adapter", return_value=[]),
            patch.object(lora, "count_named_adapter_parameters", return_value=123),
            patch.object(lora, "load_adapter", return_value="checkpoint.pt") as loader,
            patch.object(lora, "evaluate", return_value=81.25) as evaluator,
            patch.object(lora, "file_sha256", return_value="CHECKPOINT_SHA"),
            patch.object(lora, "build_run_record", return_value={"record": True}) as build_record,
            patch.object(lora, "write_jsonl_record") as write_record,
        ):
            lora.run_lora(args, clip_model, 100, FakeDataset(), None, object(), object())

        loader.assert_called_once_with(args, clip_model)
        self.assertEqual(evaluator.call_count, 1)
        self.assertEqual(build_record.call_args.kwargs["checkpoint_path"], "checkpoint.pt")
        self.assertEqual(build_record.call_args.kwargs["checkpoint_sha256"], "CHECKPOINT_SHA")
        self.assertEqual(build_record.call_args.kwargs["status"], "eval_only")
        write_record.assert_called_once_with("eval.jsonl", {"record": True})


if __name__ == "__main__":
    unittest.main()
