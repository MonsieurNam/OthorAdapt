"""Phase 2 validation-only sweep tooling."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from experiment_manifest import validate_run_record


PROTOCOL_SCHEMA_VERSION = "phase2.selection.v1"


def load_protocol(path):
    """Load JSON-compatible YAML without requiring a YAML dependency."""
    with Path(path).open("r", encoding="utf-8") as f:
        protocol = json.load(f)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"selection protocol schema_version must be {PROTOCOL_SCHEMA_VERSION}")
    if protocol.get("selection_split") != "val":
        raise ValueError("Phase 2 selection_protocol must use selection_split=val")
    metric = protocol.get("selection_metric", {})
    for field in ("datasets", "shots", "seeds"):
        if not metric.get(field):
            raise ValueError(f"selection_metric.{field} is required")
    grid = protocol.get("candidate_grid", {})
    for field in ("num_heads", "r", "lambda_o"):
        if not grid.get(field):
            raise ValueError(f"candidate_grid.{field} is required")
    base = protocol.get("base_command", {})
    for field in ("python", "entrypoint", "run_manifest", "save_path", "adapter"):
        if not base.get(field):
            raise ValueError(f"base_command.{field} is required")
    if "require_all_candidates" not in protocol:
        protocol["require_all_candidates"] = True


def candidate_grid(protocol):
    grid = protocol["candidate_grid"]
    candidates = []
    for num_heads in grid["num_heads"]:
        for rank in grid["r"]:
            if rank % num_heads != 0:
                continue
            for lambda_o in grid["lambda_o"]:
                candidates.append(
                    {
                        "num_heads": int(num_heads),
                        "r": int(rank),
                        "lambda_o": float(lambda_o),
                    }
                )
    return candidates


def _quote(value):
    text = str(value)
    if not text or any(ch.isspace() for ch in text):
        return json.dumps(text)
    return text


def _lambda_tag(value):
    return str(value).replace(".", "p").replace("-", "m")


def _base_flags(base):
    flags = []
    skip = {"python", "entrypoint", "run_manifest", "save_path", "log_dir", "adapter"}
    for key, value in base.items():
        if key in skip:
            continue
        if isinstance(value, bool):
            if value:
                flags.append(f"--{key}")
        elif isinstance(value, list):
            flags.append(f"--{key} " + " ".join(_quote(v) for v in value))
        else:
            flags.append(f"--{key} {_quote(value)}")
    return flags


def generate_sweep_commands(protocol):
    validate_protocol(protocol)
    metric = protocol["selection_metric"]
    base = protocol["base_command"]
    commands = []
    base_flags = _base_flags(base)
    log_dir = base.get("log_dir", "revision_materials/logs/validation_sweep")

    for candidate in candidate_grid(protocol):
        for dataset in metric["datasets"]:
            for shot in metric["shots"]:
                for seed in metric["seeds"]:
                    filename = (
                        f"{dataset}_{int(shot)}shot_seed{int(seed)}_val_{base['adapter']}"
                        f"_h{candidate['num_heads']}_r{candidate['r']}"
                        f"_lo{_lambda_tag(candidate['lambda_o'])}"
                    )
                    flags = [
                        f"--dataset {_quote(dataset)}",
                        f"--shots {int(shot)}",
                        f"--seed {int(seed)}",
                        f"--adapter {_quote(base['adapter'])}",
                        f"--num_heads {candidate['num_heads']}",
                        f"--r {candidate['r']}",
                        f"--lambda_o {candidate['lambda_o']}",
                        f"--save_path {_quote(base['save_path'])}",
                        f"--filename {_quote(filename)}",
                        f"--run_manifest {_quote(base['run_manifest'])}",
                        "--selection_split val",
                        "--sweep_mode",
                    ]
                    log_path = f"{log_dir}/{filename}_${{RUN_STAMP}}.log"
                    run_command = " ".join(
                        [
                            _quote(base["python"]),
                            _quote(base["entrypoint"]),
                            *base_flags,
                            *flags,
                        ]
                    )
                    commands.append(
                        f"mkdir -p {_quote(log_dir)} && {run_command} 2>&1 | tee {_quote(log_path)}"
                    )
    return commands


def write_commands(commands, path):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        ': "${DATA_ROOT:?Set DATA_ROOT=/path/to/datasets}"',
        ': "${PYTHON:=python3}"',
        ': "${RUN_STAMP:=$(date +%Y%m%d_%H%M%S)}"',
        "",
    ]
    out_path.write_text("\n".join(header + commands) + "\n", encoding="utf-8")


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no}: {exc}") from exc
    return rows


def _config_from_record(record):
    config = record.get("config", {})
    return {
        "num_heads": int(config.get("num_heads", record.get("num_heads"))),
        "r": int(config.get("r", record.get("r"))),
        "lambda_o": float(config.get("lambda_o", record.get("lambda_o"))),
    }


def _candidate_key(config):
    return (config["num_heads"], config["r"], config["lambda_o"])


def select_winner(rows, protocol):
    validate_protocol(protocol)
    expected_split = protocol["selection_split"]
    metric = protocol["selection_metric"]
    expected_datasets = set(metric["datasets"])
    expected_shots = set(int(v) for v in metric["shots"])
    expected_seeds = set(int(v) for v in metric["seeds"])
    allowed_candidates = {_candidate_key(candidate) for candidate in candidate_grid(protocol)}

    grouped = defaultdict(list)
    parameter_counts = defaultdict(set)
    for raw in rows:
        row = validate_run_record(raw)
        if row["split"] != expected_split:
            raise ValueError("selection manifest contains non-validation split rows")
        if row["dataset"] not in expected_datasets:
            continue
        if row["shot"] not in expected_shots:
            continue
        if row["seed"] not in expected_seeds:
            continue
        config = _config_from_record(raw)
        key = _candidate_key(config)
        if key not in allowed_candidates:
            raise ValueError(f"selection manifest contains candidate outside protocol: {config}")
        grouped[key].append(row)
        parameter_counts[key].add(row["parameter_count"])

    if not grouped:
        raise ValueError("No validation rows matched the selection protocol")

    complete_rows = len(expected_datasets) * len(expected_shots) * len(expected_seeds)
    summaries = []
    for key, group_rows in grouped.items():
        if len(group_rows) != complete_rows:
            continue
        accuracies = [row["accuracy"] for row in group_rows]
        config = {"num_heads": key[0], "r": key[1], "lambda_o": key[2]}
        summaries.append(
            {
                "config": config,
                "mean_validation_accuracy": sum(accuracies) / len(accuracies),
                "n": len(accuracies),
                "parameter_count": min(parameter_counts[key]),
                "covered_datasets": sorted({row["dataset"] for row in group_rows}),
                "covered_shots": sorted({row["shot"] for row in group_rows}),
                "covered_seeds": sorted({row["seed"] for row in group_rows}),
            }
        )

    if not summaries:
        raise ValueError("No candidate has complete validation coverage for the protocol")
    if protocol.get("require_all_candidates", True) and len(summaries) != len(allowed_candidates):
        missing = len(allowed_candidates) - len(summaries)
        raise ValueError(f"Missing complete validation coverage for {missing} pre-registered candidates")

    summaries.sort(
        key=lambda item: (
            -item["mean_validation_accuracy"],
            item["parameter_count"],
            item["config"]["lambda_o"],
            item["config"]["num_heads"],
            item["config"]["r"],
        )
    )
    winner = summaries[0]
    winner["candidate_count_complete"] = len(summaries)
    return winner


def _selected_config_markdown(winner, protocol):
    config = winner["config"]
    lines = [
        "# Selected Validation Configuration",
        "",
        "Status: FROZEN_FROM_VALIDATION_ONLY_SWEEP",
        "",
        "## Selection Rule",
        "",
        "- Split: val only",
        "- Metric: unweighted mean validation accuracy across the pre-registered datasets, shots, and seeds",
        "- Tie-break: lower parameter count, then lower lambda_o",
        "",
        "## Winner",
        "",
        f"- num_heads: {config['num_heads']}",
        f"- r: {config['r']}",
        f"- lambda_o: {config['lambda_o']}",
        f"- mean_validation_accuracy: {winner['mean_validation_accuracy']:.6f}",
        f"- n: {winner['n']}",
        f"- covered_datasets: {', '.join(winner['covered_datasets'])}",
        f"- covered_shots: {', '.join(str(v) for v in winner['covered_shots'])}",
        f"- covered_seeds: {', '.join(str(v) for v in winner['covered_seeds'])}",
        "",
        "## Protocol",
        "",
        f"- schema_version: {protocol['schema_version']}",
        f"- selection_split: {protocol['selection_split']}",
    ]
    return "\n".join(lines) + "\n"


def write_selected_config(winner, protocol, path):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_selected_config_markdown(winner, protocol), encoding="utf-8")
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest().upper()
    Path(str(out_path) + ".sha256").write_text(f"{digest}  {out_path.name}\n", encoding="utf-8")
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--protocol", required=True)
    generate_parser.add_argument("--out", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--protocol", required=True)
    select_parser.add_argument("--manifest", required=True)
    select_parser.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)

    if args.command == "generate":
        write_commands(generate_sweep_commands(protocol), args.out)
    elif args.command == "select":
        winner = select_winner(load_jsonl(args.manifest), protocol)
        write_selected_config(winner, protocol, args.out)


if __name__ == "__main__":
    main()
