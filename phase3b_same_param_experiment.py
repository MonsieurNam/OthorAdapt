"""Generate Phase 3B same-parameter commands for the legacy claim check."""

import argparse
import json
from pathlib import Path


PROTOCOL_SCHEMA_VERSION = "phase3b.same_param.v1"
EXPECTED_LORA = {"adapter": "lora", "r": 2}
EXPECTED_ORTHOADAPT = {
    "adapter": "ohsinglora",
    "num_heads": 2,
    "r": 2,
    "lambda_o": 0.03,
    "ramp_up_steps": 100,
}


def load_protocol(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Phase 3B protocol must be JSON-compatible YAML/JSON; avoid comments"
        ) from exc


def _quote(value):
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value)
    if text.startswith("$") or text.startswith("${"):
        return text
    if any(ch.isspace() for ch in text):
        return json.dumps(text)
    return text


def _lambda_tag(value):
    return str(value).replace(".", "p")


def _require_list(protocol, key):
    value = protocol.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return value


def _assert_method(method, expected, label):
    for key, value in expected.items():
        if method.get(key) != value:
            raise ValueError(
                f"Phase 3B {label} config mismatch for {key}: "
                f"expected {value}, got {method.get(key)}"
            )


def validate_protocol(protocol):
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {PROTOCOL_SCHEMA_VERSION}")
    _require_list(protocol, "datasets")
    _require_list(protocol, "shots")
    _require_list(protocol, "seeds")
    methods = _require_list(protocol, "methods")

    base = protocol.get("base_command", {})
    for field in ("python", "entrypoint", "root_path", "run_manifest", "save_path", "log_dir"):
        if not base.get(field):
            raise ValueError(f"base_command.{field} is required")

    lora_methods = [method for method in methods if method.get("adapter") == "lora"]
    ortho_methods = [method for method in methods if method.get("adapter") == "ohsinglora"]
    if len(lora_methods) != 1:
        raise ValueError("Phase 3B must include exactly one CLIP-LoRA method")
    if len(ortho_methods) != 1:
        raise ValueError("Phase 3B must include exactly one OrthoAdapt method")
    _assert_method(lora_methods[0], EXPECTED_LORA, "CLIP-LoRA")
    _assert_method(ortho_methods[0], EXPECTED_ORTHOADAPT, "OrthoAdapt")
    if int(ortho_methods[0]["ramp_up_steps"]) != 100:
        raise ValueError("Phase 3B OrthoAdapt must use ramp_up_steps=100")


def _base_flags(base):
    excluded = {"python", "entrypoint", "root_path", "run_manifest", "save_path", "log_dir"}
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


def _run_name(dataset, shot, seed, method):
    adapter = method["adapter"]
    if adapter == "ohsinglora":
        ramp_tag = ""
        if "ramp_up_steps" in method:
            ramp_tag = f"_ramp{int(method['ramp_up_steps'])}"
        return (
            f"{dataset}_{int(shot)}shot_seed{int(seed)}_test_{adapter}"
            f"_h{int(method['num_heads'])}_r{int(method['r'])}"
            f"_lo{_lambda_tag(method['lambda_o'])}"
            f"{ramp_tag}"
        )
    return f"{dataset}_{int(shot)}shot_seed{int(seed)}_test_{adapter}_r{int(method['r'])}"


def generate_phase3b_commands(protocol, adapter_filter=None):
    validate_protocol(protocol)
    base = protocol["base_command"]
    base_flags = _base_flags(base)
    log_dir = base["log_dir"]
    commands = []

    for method in protocol["methods"]:
        if adapter_filter and method["adapter"] != adapter_filter:
            continue
        for dataset in protocol["datasets"]:
            for shot in protocol["shots"]:
                for seed in protocol["seeds"]:
                    filename = _run_name(dataset, shot, seed, method)
                    flags = [
                        f"--dataset {_quote(dataset)}",
                        f"--shots {int(shot)}",
                        f"--seed {int(seed)}",
                        f"--adapter {_quote(method['adapter'])}",
                        f"--r {int(method['r'])}",
                        f"--alpha {int(method.get('alpha', 1))}",
                        f"--save_path {_quote(base['save_path'])}",
                        f"--filename {_quote(filename)}",
                        f"--run_manifest {_quote(base['run_manifest'])}",
                        "--selection_split test",
                        "--report_test",
                    ]
                    if method["adapter"] == "ohsinglora":
                        flags.insert(4, f"--num_heads {int(method['num_heads'])}")
                        flags.insert(7, f"--lambda_o {float(method['lambda_o'])}")
                        flags.insert(8, f"--ramp_up_steps {int(method['ramp_up_steps'])}")
                    run_command = " ".join(
                        [_quote(base["python"]), _quote(base["entrypoint"]), *base_flags, *flags]
                    )
                    log_path = f"{log_dir}/{filename}_${{RUN_STAMP}}.log"
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate Phase 3B bash commands")
    generate.add_argument("--protocol", required=True)
    generate.add_argument("--out", required=True)
    generate.add_argument(
        "--adapter-filter",
        choices=["lora", "ohsinglora"],
        default=None,
        help="Generate commands only for one adapter",
    )

    args = parser.parse_args(argv)
    if args.command == "generate":
        protocol = load_protocol(args.protocol)
        write_commands(generate_phase3b_commands(protocol, args.adapter_filter), args.out)


if __name__ == "__main__":
    main()
