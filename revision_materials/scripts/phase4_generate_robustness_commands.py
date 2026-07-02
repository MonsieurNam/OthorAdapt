"""Generate server commands for full Phase 4 paired robustness evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "revision_materials" / "results" / "phase3_main_ramp100_results.jsonl"
OUT_COMMANDS = ROOT / "revision_materials" / "scripts" / "phase4_robustness_commands.sh"
ROBUST_MANIFEST = "revision_materials/results/phase4_robustness_manifest.jsonl"
LOG_DIR = "revision_materials/logs/phase4_robustness"


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sh_value(value) -> str:
    text = str(value)
    if any(ch in text for ch in " /\\:"):
        return "'" + text.replace("'", "'\"'\"'") + "'"
    return text


def command_for(row: dict) -> str:
    cfg = row["config"]
    method = cfg["adapter"]
    flags = [
        "$PYTHON revision_materials/scripts/phase4_eval_robustness.py",
        f"--root_path ${{DATA_ROOT}}",
        f"--dataset {cfg['dataset']}",
        f"--shots {cfg['shots']}",
        f"--seed {cfg['seed']}",
        f"--adapter {method}",
        f"--checkpoint {sh_value(row['checkpoint']['path'])}",
        f"--filename {cfg['filename']}",
        f"--run_manifest {ROBUST_MANIFEST}",
        f"--backbone {sh_value(cfg['backbone'])}",
        f"--r {cfg['r']}",
        f"--alpha {cfg['alpha']}",
        f"--position {cfg['position']}",
        f"--encoder {cfg['encoder']}",
        "--params " + " ".join(cfg["params"]),
        "--batch_size 128",
        "--num_workers 0",
    ]
    if method == "ohsinglora":
        flags += [
            f"--num_heads {cfg['num_heads']}",
            f"--lambda_o {cfg['lambda_o']}",
            f"--ramp_up_steps {cfg['ramp_up_steps']}",
        ]
    log_file = f"{LOG_DIR}/{cfg['filename']}_${{RUN_STAMP}}.log"
    return "mkdir -p {log_dir} && {cmd} 2>&1 | tee {log}".format(
        log_dir=LOG_DIR,
        cmd=" ".join(flags),
        log=log_file,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(OUT_COMMANDS))
    args = parser.parse_args(argv)
    rows = load_manifest(Path(args.manifest))
    expected = 144
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} Phase 3 rows, found {len(rows)}")
    commands = [command_for(row) for row in rows]
    out = Path(args.out)
    out.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "PYTHON=${PYTHON:-python3}",
                "DATA_ROOT=${DATA_ROOT:-/root/DATA}",
                "RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}",
                "",
                "# Generated from phase3_main_ramp100_results.jsonl.",
                "# Each command evaluates one checkpoint over severity 0,1,2,3 and appends four JSONL rows.",
                "",
                *commands,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(out.relative_to(ROOT).as_posix())
    print(f"commands={len(commands)}")


if __name__ == "__main__":
    main()
