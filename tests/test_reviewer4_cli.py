import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch  # Keep the real module registered while run_utils is loaded with a fake lora.


def load_run_utils():
    fake_lora = types.ModuleType("lora")
    fake_lora.run_lora = object()
    path = Path(__file__).parents[1] / "run_utils.py"
    spec = importlib.util.spec_from_file_location("reviewer4_run_utils", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"lora": fake_lora}):
        spec.loader.exec_module(module)
    return module


RUN_UTILS = load_run_utils()


class Reviewer4CliTest(unittest.TestCase):
    def test_routing_mode_defaults_to_input_conditioned(self):
        with patch.object(sys, "argv", ["main.py"]):
            args = RUN_UTILS.get_arguments()

        self.assertEqual(args.routing_mode, "input_conditioned")
        self.assertIsNone(args.split_reference_manifest)
        self.assertEqual(args.profile_repetition, 0)
        self.assertFalse(args.compute_profile_only)

    def test_split_reference_manifest_is_accepted(self):
        with patch.object(sys, "argv", ["main.py", "--split_reference_manifest", "reference.jsonl"]):
            args = RUN_UTILS.get_arguments()
        self.assertEqual(args.split_reference_manifest, "reference.jsonl")

    def test_input_independent_is_available_for_orthoadapt(self):
        with patch.object(
            sys,
            "argv",
            ["main.py", "--adapter", "ohsinglora", "--routing_mode", "input_independent"],
        ):
            args = RUN_UTILS.get_arguments()

        self.assertEqual(args.routing_mode, "input_independent")

    def test_input_independent_is_rejected_for_singlora(self):
        with patch.object(
            sys,
            "argv",
            ["main.py", "--adapter", "singlora", "--routing_mode", "input_independent"],
        ):
            with patch.object(sys, "stderr", new=io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    RUN_UTILS.get_arguments()

        self.assertIn(
            "--routing_mode input_independent requires --adapter ohsinglora",
            stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
