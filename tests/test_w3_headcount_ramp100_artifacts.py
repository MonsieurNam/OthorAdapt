import re
import unittest
from pathlib import Path


SCRIPT = Path("revision_materials/scripts/w3_headcount_h1_ramp100_commands.sh")
RUNBOOK = Path("revision_materials/plan/w3_headcount_h1_ramp100_runbook.md")


class W3HeadcountRamp100ArtifactTest(unittest.TestCase):
    def test_h1_ramp100_script_has_exact_12_validation_runs(self):
        text = SCRIPT.read_text(encoding="utf-8")
        commands = [line for line in text.splitlines() if " main.py " in line]

        self.assertEqual(len(commands), 12)
        self.assertTrue(all("--selection_split val" in command for command in commands))
        self.assertTrue(all("--sweep_mode" in command for command in commands))
        self.assertTrue(all("--report_test" not in command for command in commands))
        self.assertTrue(all("--ramp_up_steps 100" in command for command in commands))
        self.assertTrue(all("--adapter ohsinglora" in command for command in commands))
        self.assertTrue(all("--num_heads 1" in command for command in commands))
        self.assertTrue(all("--lambda_o 0.0" in command for command in commands))
        self.assertTrue(all("--run_manifest revision_materials/results/w3_headcount_h1_ramp100_results.jsonl" in command for command in commands))

        observed = set()
        for command in commands:
            dataset = re.search(r"--dataset (\S+)", command).group(1)
            seed = int(re.search(r"--seed (\d+)", command).group(1))
            rank = int(re.search(r"--r (\d+)", command).group(1))
            observed.add((dataset, seed, rank))

        expected = {
            (dataset, seed, rank)
            for dataset in ("eurosat", "caltech101")
            for seed in (1, 2, 3)
            for rank in (2, 4)
        }
        self.assertEqual(observed, expected)

    def test_h1_ramp100_runbook_documents_outputs_and_merge_path(self):
        text = RUNBOOK.read_text(encoding="utf-8")

        self.assertIn("12 validation runs", text)
        self.assertIn("w3_headcount_h1_ramp100_results.jsonl", text)
        self.assertIn("validation_sweep_ramp100_results.jsonl", text)
        self.assertIn("fixed total rank", text)
        self.assertIn("fixed per-head rank", text)


if __name__ == "__main__":
    unittest.main()
