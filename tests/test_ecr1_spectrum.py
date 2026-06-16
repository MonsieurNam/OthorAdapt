import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import analyze_spectrum


class ECR1SpectrumTest(unittest.TestCase):
    def test_validate_spectrum_args_rejects_demo_mode(self):
        args = Namespace(lora_path="", oh_path="", output="spectrum.png", report_path="report.json")

        with self.assertRaisesRegex(analyze_spectrum.SpectrumError, "requires --lora_path and --oh_path"):
            analyze_spectrum.validate_spectrum_args(args)

    def test_validate_spectrum_args_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.pt")
            existing = Path(tmp) / "existing.pt"
            existing.write_bytes(b"not a torch checkpoint")
            args = Namespace(lora_path=missing, oh_path=str(existing), output="spectrum.png", report_path="report.json")

            with self.assertRaisesRegex(analyze_spectrum.SpectrumError, "Missing checkpoint"):
                analyze_spectrum.validate_spectrum_args(args)

    def test_checkpoint_metadata_is_required(self):
        with self.assertRaisesRegex(analyze_spectrum.SpectrumError, "metadata"):
            analyze_spectrum.require_checkpoint_metadata({"weights": {}})

    def test_matrix_definition_records_adapter_layer_param_and_formula(self):
        definition = analyze_spectrum.matrix_definition("lora", layer_idx=6, param_name="q_proj")

        self.assertEqual(definition["adapter"], "lora")
        self.assertEqual(definition["layer_idx"], 6)
        self.assertEqual(definition["param"], "q_proj")
        self.assertIn("B @ A", definition["formula"])


if __name__ == "__main__":
    unittest.main()
