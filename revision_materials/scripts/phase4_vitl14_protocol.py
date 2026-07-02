"""Generate Phase 4 ViT-L/14 pilot and subset commands."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "revision_materials" / "scripts"
RESULTS = ROOT / "revision_materials" / "results"
PILOT_COMMANDS = SCRIPT_DIR / "phase4_vitl14_pilot_commands.sh"
SUBSET_COMMANDS = SCRIPT_DIR / "phase4_vitl14_commands.sh"
REPORT = RESULTS / "phase4_backbone_scaling_report.md"

DATASETS = ["eurosat", "caltech101"]
SEEDS = [1, 2, 3]
METHODS = [
    {"adapter": "lora", "r": 8, "alpha": 1},
    {"adapter": "ohsinglora", "r": 8, "alpha": 1, "num_heads": 2, "lambda_o": 0.03, "ramp_up_steps": 100},
]


def lambda_tag(value: float) -> str:
    return str(value).replace(".", "p")


def command(dataset: str, seed: int, method: dict, manifest: str, log_dir: str, save_path: str) -> str:
    adapter = method["adapter"]
    filename = f"{dataset}_4shot_seed{seed}_vitl14_{adapter}_r{method['r']}"
    if adapter == "ohsinglora":
        filename += f"_h{method['num_heads']}_lo{lambda_tag(method['lambda_o'])}_ramp100"
    flags = [
        "$PYTHON main.py",
        "--root_path ${DATA_ROOT}",
        "--backbone ViT-L/14",
        "--encoder both",
        "--position all",
        "--params q k v",
        "--n_iters 500",
        "--batch_size 32",
        "--loss_fn ce",
        "--ortho_reduction mean",
        f"--dataset {dataset}",
        "--shots 4",
        f"--seed {seed}",
        f"--adapter {adapter}",
        f"--r {method['r']}",
        f"--alpha {method['alpha']}",
        f"--save_path {save_path}",
        f"--filename {filename}",
        f"--run_manifest {manifest}",
        "--selection_split test",
        "--report_test",
    ]
    if adapter == "ohsinglora":
        flags.insert(13, f"--num_heads {method['num_heads']}")
        flags.insert(15, f"--lambda_o {method['lambda_o']}")
        flags.insert(16, f"--ramp_up_steps {method['ramp_up_steps']}")
    return f"mkdir -p {log_dir} && {' '.join(flags)} 2>&1 | tee {log_dir}/{filename}_${{RUN_STAMP}}.log"


def write_script(path: Path, commands: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "PYTHON=${PYTHON:-python3}",
                "DATA_ROOT=${DATA_ROOT:-/root/DATA}",
                "RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}",
                "",
                *commands,
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_report() -> None:
    manifest = RESULTS / "phase4_vitl14_results.jsonl"
    if manifest.exists():
        status = "pending aggregation; manifest exists"
    else:
        status = "pending pilot; no ViT-L/14 results have been run yet"
    REPORT.write_text(
        "\n".join(
            [
                "# Phase 4 Backbone Scaling Report",
                "",
                f"Status: {status}.",
                "",
                "Protocol:",
                "- Pilot: EuroSAT 4-shot seed1, CLIP-LoRA r=8 and OrthoAdapt H=2,r=8,lambda_o=0.03,ramp100.",
                "- If pilot completes within available compute, run EuroSAT + Caltech101, 4-shot, seeds 1/2/3, both methods.",
                "",
                "Claim gate:",
                "- Do not claim backbone scaling until `phase4_vitl14_results.jsonl` has 12 completed rows and the summary is generated.",
                "- If pilot is too slow or fails, mark ViT-L/14 as deferred in limitations.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    pilot = [
        command("eurosat", 1, method, "revision_materials/results/phase4_vitl14_pilot_results.jsonl", "revision_materials/logs/phase4_vitl14_pilot", "revision_materials/checkpoints/phase4_vitl14_pilot")
        for method in METHODS
    ]
    subset = [
        command(dataset, seed, method, "revision_materials/results/phase4_vitl14_results.jsonl", "revision_materials/logs/phase4_vitl14", "revision_materials/checkpoints/phase4_vitl14")
        for dataset in DATASETS
        for seed in SEEDS
        for method in METHODS
    ]
    write_script(PILOT_COMMANDS, pilot)
    write_script(SUBSET_COMMANDS, subset)
    write_report()
    print(PILOT_COMMANDS.relative_to(ROOT).as_posix())
    print(SUBSET_COMMANDS.relative_to(ROOT).as_posix())
    print(REPORT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
