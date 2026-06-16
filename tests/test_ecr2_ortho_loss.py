import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed in this environment")
class ECR2OrthoLossTest(unittest.TestCase):
    def test_mean_reduction_normalizes_pair_count(self):
        import torch
        from loralib.layers_OH_singlora import calculate_ortho_loss

        h2 = torch.ones(2, 3, 1)
        h4 = torch.ones(4, 3, 1)

        self.assertEqual(calculate_ortho_loss(h2, reduction="sum").item(), 9.0)
        self.assertEqual(calculate_ortho_loss(h4, reduction="sum").item(), 54.0)
        self.assertEqual(calculate_ortho_loss(h2, reduction="mean").item(), 9.0)
        self.assertEqual(calculate_ortho_loss(h4, reduction="mean").item(), 9.0)

    def test_invalid_reduction_fails_closed(self):
        import torch
        from loralib.layers_OH_singlora import calculate_ortho_loss

        with self.assertRaisesRegex(ValueError, "reduction"):
            calculate_ortho_loss(torch.ones(2, 3, 1), reduction="median")


if __name__ == "__main__":
    unittest.main()
