"""Generate the fixed Reviewer-2 three-seed OrthoAdapt lambda=0 matrix."""

import argparse
import json
from pathlib import Path


PROTOCOL_SCHEMA_VERSION = "phase3.reviewer2.lambda0.v1"
EXPECTED_DATASETS = [
    "fgvc",
    "eurosat",
    "food101",
    "oxford_pets",
    "oxford_flowers",
    "caltech101",
    "dtd",
    "ucf101",
]
EXPECTED_SHOTS = [1, 4, 16]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_METHOD = {
    "adapter": "ohsinglora",
    "num_heads": 2,
    "r": 8,
    "alpha": 1,
    "lambda_o": 0,
    "ramp_up_steps": 100,
}
EXPECTED_TRAINING_CONFIG = {
    "backbone": "ViT-B/16",
    "encoder": "both",
    "position": "all",
    "params": ["q", "k", "v"],
    "n_iters": 500,
    "batch_size": 32,
    "loss_fn": "ce",
    "ortho_reduction": "mean",
}


def load_protocol(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Reviewer-2 lambda=0 protocol must be JSON-compatible YAML/JSON") from exc


def _quote(value):
    text = str(value)
    if text.startswith("${"):
        return f'"{text}"'
    if text.startswith("$"):
        return text
    if any(character.isspace() for character in text):
        return json.dumps(text)
    return text


def validate_protocol(protocol):
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {PROTOCOL_SCHEMA_VERSION}")
    if protocol.get("datasets") != EXPECTED_DATASETS:
        raise ValueError(f"datasets must exactly match the frozen Phase 3 order: {EXPECTED_DATASETS}")
    if protocol.get("shots") != EXPECTED_SHOTS:
        raise ValueError(f"shots must be {EXPECTED_SHOTS}")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise ValueError(f"seeds must be {EXPECTED_SEEDS}")
    method = protocol.get("method")
    if not isinstance(method, dict):
        raise ValueError("method must be a mapping")
    for field, expected in EXPECTED_METHOD.items():
        if method.get(field) != expected:
            raise ValueError(
                f"Reviewer-2 lambda=0 {field} mismatch: expected {expected!r}, "
                f"got {method.get(field)!r}"
            )
    base = protocol.get("base_command", {})
    required = (
        "python",
        "entrypoint",
        "root_path",
        "save_path",
        "train_manifest",
        "eval_manifest",
        "train_log_dir",
        "eval_log_dir",
    )
    for field in required:
        if not base.get(field):
            raise ValueError(f"base_command.{field} is required")
    for field, expected in EXPECTED_TRAINING_CONFIG.items():
        if base.get(field) != expected:
            raise ValueError(
                f"Frozen Phase 3 {field} mismatch: expected {expected!r}, "
                f"got {base.get(field)!r}"
            )


def _base_flags(base):
    excluded = {
        "python",
        "entrypoint",
        "root_path",
        "save_path",
        "train_manifest",
        "eval_manifest",
        "train_log_dir",
        "eval_log_dir",
    }
    flags = [f"--root_path {_quote(base['root_path'])}"]
    for key, value in base.items():
        if key in excluded:
            continue
        if isinstance(value, bool):
            if value:
                flags.append(f"--{key}")
        elif isinstance(value, list):
            flags.append(f"--{key} " + " ".join(_quote(item) for item in value))
        else:
            flags.append(f"--{key} {_quote(value)}")
    return flags


def _run_name(dataset, shot, seed):
    return f"{dataset}_{int(shot)}shot_seed{int(seed)}_test_ohsinglora_h2_r8_lo0_ramp100"


def generate_commands(protocol, mode):
    validate_protocol(protocol)
    if mode not in {"train", "eval"}:
        raise ValueError("mode must be 'train' or 'eval'")
    base = protocol["base_command"]
    method = protocol["method"]
    manifest = base[f"{mode}_manifest"]
    log_dir = base[f"{mode}_log_dir"]
    commands = []
    for dataset in protocol["datasets"]:
        for shot in protocol["shots"]:
            for seed in protocol["seeds"]:
                filename = _run_name(dataset, shot, seed)
                flags = [
                    f"--dataset {_quote(dataset)}",
                    f"--shots {int(shot)}",
                    f"--seed {int(seed)}",
                    "--adapter ohsinglora",
                    f"--num_heads {int(method['num_heads'])}",
                    f"--r {int(method['r'])}",
                    f"--alpha {int(method['alpha'])}",
                    f"--lambda_o {method['lambda_o']}",
                    f"--ramp_up_steps {int(method['ramp_up_steps'])}",
                    f"--save_path {_quote(base['save_path'])}",
                    f"--filename {_quote(filename)}",
                    f"--run_manifest {_quote(manifest)}",
                ]
                if mode == "train":
                    flags.append("--selection_split val")
                else:
                    flags.extend(["--selection_split test", "--report_test", "--eval_only"])
                command = " ".join(
                    [
                        _quote(base["python"]),
                        _quote(base["entrypoint"]),
                        *_base_flags(base),
                        *flags,
                    ]
                )
                log_path = f"{log_dir}/{filename}_{mode}_${{RUN_STAMP}}.log"
                commands.append(
                    f"mkdir -p {_quote(log_dir)} && {command} 2>&1 | tee {_quote(log_path)}"
                )
    return commands


def write_commands(commands, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        ': "${DATA_ROOT:?Set DATA_ROOT=/path/to/datasets}"',
        ': "${PYTHON:=python3}"',
        ': "${RUN_STAMP:=$(date +%Y%m%d_%H%M%S)}"',
        "",
    ]
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(header + list(commands)) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--mode", required=True, choices=["train", "eval"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    write_commands(generate_commands(protocol, args.mode), args.out)


if __name__ == "__main__":
    main()
