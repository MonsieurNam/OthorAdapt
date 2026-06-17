import unittest
from argparse import Namespace

from loralib.utils import adapter_checkpoint_path


class AdapterCheckpointPathTest(unittest.TestCase):
    def test_adapter_checkpoint_path_uses_config_specific_filename(self):
        args = Namespace(
            save_path="revision_materials/checkpoints/validation_sweep",
            adapter="ohsinglora",
            backbone="ViT-B/16",
            dataset="eurosat",
            shots=4,
            seed=1,
            filename="eurosat_4shot_seed1_val_ohsinglora_h2_r4_lo0p03",
        )

        path = adapter_checkpoint_path(args)

        self.assertTrue(path.endswith("eurosat_4shot_seed1_val_ohsinglora_h2_r4_lo0p03.pt"))
        self.assertIn("ohsinglora", path)
        self.assertIn("eurosat", path)
        self.assertIn("seed1", path)


if __name__ == "__main__":
    unittest.main()
