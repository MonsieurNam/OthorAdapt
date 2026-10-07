"""Generate and resume the frozen Reviewer #4 routing-control experiment."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "reviewer4.gating.v1"
EXPECTED_DATASETS = 8
EXPECTED_SHOTS = (1, 4, 16)
EXPECTED_SEEDS = (1, 2, 3)
LOCKED_METHOD = {
    "adapter": "ohsinglora",
    "num_heads": 2,
    "r": 8,
    "alpha": 1,
    "lambda_o": 0.03,
    "ramp_up_steps": 100,
    "ortho_reduction": "mean",
}
LOCKED_BASE = {
    "backbone": "ViT-B/16",
    "encoder": "both",
    "position": "all",
    "params": ["q", "k", "v"],
    "n_iters": 500,
    "batch_size": 32,
    "loss_fn": "ce",
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    shot: int
    seed: int
    routing_mode: str
    expected_trainable_parameters: int = 460800
    cohort: str = "static"

    @property
    def run_key(self) -> str:
        return f"{self.dataset}|{self.shot}|{self.seed}|{self.routing_mode}"

    @property
    def filename(self) -> str:
        route = "static" if self.routing_mode == "input_independent" else "dynamic"
        return f"reviewer4_{self.dataset}_{self.shot}shot_seed{self.seed}_{route}_h2_r8_lo0p03_ramp100"


def load_protocol(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Protocol must be JSON-compatible YAML/JSON") from exc


def validate_protocol(protocol: dict) -> None:
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    datasets = protocol.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != EXPECTED_DATASETS or len(set(datasets)) != EXPECTED_DATASETS:
        raise ValueError("datasets must contain exactly eight unique names")
    if tuple(protocol.get("shots", ())) != EXPECTED_SHOTS:
        raise ValueError(f"shots must be {list(EXPECTED_SHOTS)}")
    if tuple(protocol.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError(f"seeds must be {list(EXPECTED_SEEDS)}")
    if int(protocol.get("expected_trainable_parameters", -1)) != 460800:
        raise ValueError("expected_trainable_parameters must be 460800")
    method = protocol.get("method", {})
    for key, expected in LOCKED_METHOD.items():
        if method.get(key) != expected:
            raise ValueError(f"method.{key} must be {expected!r}")
    if method.get("routing_mode") != "input_independent":
        raise ValueError("method.routing_mode must be input_independent")
    base = protocol.get("base_command", {})
    for key in (
        "python", "entrypoint", "root_path", "save_path", "run_manifest", "log_dir",
        "split_reference_manifest", "sentinel_run_manifest", "dynamic_run_manifest",
    ):
        if not base.get(key):
            raise ValueError(f"base_command.{key} is required")
    for key, expected in LOCKED_BASE.items():
        if base.get(key) != expected:
            raise ValueError(f"base_command.{key} must be {expected!r}")
    sentinel = protocol.get("sentinel_datasets")
    if not isinstance(sentinel, list) or len(sentinel) != 2 or not set(sentinel).issubset(datasets):
        raise ValueError("sentinel_datasets must contain two main-matrix datasets")
    if protocol.get("sentinel_seed") not in protocol["seeds"]:
        raise ValueError("sentinel_seed must be one of seeds")


def build_specs(protocol: dict, routing_mode: str = "input_independent") -> list[RunSpec]:
    validate_protocol(protocol)
    if routing_mode not in {"input_conditioned", "input_independent"}:
        raise ValueError(f"unsupported routing_mode: {routing_mode}")
    specs = [
        RunSpec(
            dataset, int(shot), int(seed), routing_mode, 460800,
            "static" if routing_mode == "input_independent" else "dynamic",
        )
        for dataset in protocol["datasets"]
        for shot in protocol["shots"]
        for seed in protocol["seeds"]
    ]
    if len(specs) != 72 or len({spec.run_key for spec in specs}) != 72:
        raise ValueError("protocol did not produce 72 unique configurations")
    return specs


def build_sentinel_specs(protocol: dict) -> list[RunSpec]:
    validate_protocol(protocol)
    specs = [
        RunSpec(dataset, int(shot), int(protocol["sentinel_seed"]), "input_conditioned", 460800, "sentinel")
        for dataset in protocol["sentinel_datasets"]
        for shot in protocol["shots"]
    ]
    if len(specs) != 6 or len({spec.run_key for spec in specs}) != 6:
        raise ValueError("sentinel protocol did not produce six unique configurations")
    return specs


def build_smoke_spec(protocol: dict) -> RunSpec:
    validate_protocol(protocol)
    if "eurosat" not in protocol["datasets"]:
        raise ValueError("smoke protocol requires eurosat")
    return RunSpec("eurosat", 4, 1, "input_independent", 460800, "static")


def completed_keys(path: str | Path, status: str) -> set[str]:
    manifest = Path(path)
    if not manifest.exists():
        return set()
    keys: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {manifest}:{line_number}") from exc
        if record.get("status") == status:
            run_key = record.get("reviewer4_run_key")
            if not run_key:
                config = record.get("config", {})
                if all(config.get(field) is not None for field in ("dataset", "shots", "seed")):
                    mode = config.get("routing_mode", "input_conditioned")
                    run_key = (
                        f"{config['dataset']}|{int(config['shots'])}|{int(config['seed'])}|{mode}"
                    )
            if run_key:
                keys.add(run_key)
    return keys


def filter_completed(specs: Iterable[RunSpec], done: set[str]) -> list[RunSpec]:
    remaining = [spec for spec in specs if spec.run_key not in done]
    if len({spec.run_key for spec in remaining}) != len(remaining):
        raise ValueError("duplicate run specifications")
    return remaining


def _quote(value: object) -> str:
    text = str(value)
    if text.startswith("${") and text.endswith("}"):
        return text
    return shlex.quote(text)


def _base_flags(base: dict) -> list[str]:
    excluded = {
        "python", "entrypoint", "root_path", "save_path", "run_manifest", "log_dir",
        "sentinel_run_manifest", "dynamic_run_manifest",
    }
    flags = ["--root_path", _quote(base["root_path"])]
    for key, value in base.items():
        if key in excluded:
            continue
        flags.extend([f"--{key}"])
        if isinstance(value, list):
            flags.extend(_quote(item) for item in value)
        elif isinstance(value, bool):
            if not value:
                flags.pop()
        else:
            flags.append(_quote(value))
    return flags


def command_for_spec(protocol: dict, spec: RunSpec, stage: str) -> str:
    validate_protocol(protocol)
    if stage not in {"train", "eval"}:
        raise ValueError("stage must be train or eval")
    base = protocol["base_command"]
    method = protocol["method"]
    manifest = {
        "static": base["run_manifest"],
        "sentinel": base["sentinel_run_manifest"],
        "dynamic": base["dynamic_run_manifest"],
    }[spec.cohort]
    if stage == "eval":
        manifest = str(manifest).replace("train", "eval")
    argv = [
        _quote(base["python"]), _quote(base["entrypoint"]), *_base_flags(base),
        "--dataset", _quote(spec.dataset), "--shots", str(spec.shot), "--seed", str(spec.seed),
        "--adapter", method["adapter"], "--routing_mode", spec.routing_mode,
        "--num_heads", str(method["num_heads"]), "--r", str(method["r"]),
        "--alpha", str(method["alpha"]), "--lambda_o", str(method["lambda_o"]),
        "--ramp_up_steps", str(method["ramp_up_steps"]),
        "--ortho_reduction", method["ortho_reduction"],
        "--save_path", _quote(base["save_path"]), "--filename", _quote(spec.filename),
        "--run_manifest", _quote(manifest), "--selection_split", "test", "--report_test",
    ]
    if stage == "eval":
        argv.append("--eval_only")
    command = " ".join(argv)
    log_path = f"{base['log_dir']}/{spec.filename}_{stage}.log"
    return f"mkdir -p {_quote(base['log_dir'])} && {command} 2>&1 | tee {_quote(log_path)}"


def write_bash(protocol: dict, specs: Iterable[RunSpec], stage: str, path: str | Path) -> None:
    commands = [command_for_spec(protocol, spec, stage) for spec in specs]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "#!/usr/bin/env bash", "set -euo pipefail",
        ': "${DATA_ROOT:?Set DATA_ROOT=/path/to/eight-dataset-root}"',
        ': "${PYTHON:=python3}"', "",
    ]
    out.write_text("\n".join(header + commands) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--matrix", choices=["smoke", "static", "sentinel", "dynamic"], required=True)
    parser.add_argument("--stage", choices=["train", "eval"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume-manifest")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    if args.matrix == "smoke":
        specs = [build_smoke_spec(protocol)]
    elif args.matrix == "sentinel":
        specs = build_sentinel_specs(protocol)
    else:
        mode = "input_independent" if args.matrix == "static" else "input_conditioned"
        specs = build_specs(protocol, routing_mode=mode)
    if args.resume_manifest:
        status = "completed" if args.stage == "train" else "eval_only"
        specs = filter_completed(specs, completed_keys(args.resume_manifest, status))
    write_bash(protocol, specs, args.stage, args.out)


if __name__ == "__main__":
    main()
