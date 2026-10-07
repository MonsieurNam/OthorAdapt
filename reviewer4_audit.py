"""Fail-closed coverage, provenance, and consistency audit for Reviewer #4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")


def freeze_dynamic_reference(records):
    selected = []
    for record in records:
        config = record.get("config", {})
        matches = (
            config.get("adapter") == "ohsinglora"
            and int(config.get("num_heads", -1)) == 2
            and int(config.get("r", -1)) == 8
            and float(config.get("lambda_o", -1)) == 0.03
            and int(config.get("ramp_up_steps", -1)) == 100
            and config.get("routing_mode", "input_conditioned") == "input_conditioned"
            and record.get("status") == "completed"
        )
        if matches:
            selected.append(record)
    keys = [_key(record) for record in selected]
    if len(selected) != 72 or len(set(keys)) != 72:
        raise ValueError(
            f"selected dynamic reference must contain 72 unique completed rows; "
            f"found rows={len(selected)}, unique={len(set(keys))}"
        )
    return sorted(selected, key=_key)


def load_jsonl(path):
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_checksum_inventory(path):
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("checksum inventory must be a non-empty list")
    for entry in entries:
        artifact = Path(entry["path"])
        if not artifact.is_file():
            raise ValueError(f"frozen artifact is missing: {artifact}")
        actual = file_sha256(artifact)
        if actual != str(entry["sha256"]).upper():
            raise ValueError(f"frozen artifact checksum mismatch: {artifact}")
    return {"passed": True, "artifacts": len(entries)}


def _key(record):
    config = record.get("config", record)
    return (str(config.get("dataset")), int(config.get("shots", config.get("shot"))), int(config.get("seed")))


def _combined_split_hash(record):
    dataset = record.get("dataset", {})
    payload = {
        "classnames_sha256": dataset.get("classnames_sha256"),
        "splits": {
            name: dataset.get("splits", {}).get(name, {}).get("sha256")
            for name in ("train", "val", "test")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def _index(records, label, expected_status, routing_mode="input_independent"):
    if len(records) != 72:
        raise ValueError(f"{label} must contain exactly 72 rows; found {len(records)}")
    indexed = {}
    for record in records:
        key = _key(record)
        if key in indexed:
            raise ValueError(f"duplicate {label} configuration: {key}")
        if record.get("status") != expected_status:
            raise ValueError(f"{label} {key} has status={record.get('status')!r}")
        config = record.get("config", {})
        if config.get("adapter") != "ohsinglora" or config.get("routing_mode") != routing_mode:
            raise ValueError(f"{label} {key} is not the frozen {routing_mode} OrthoAdapt arm")
        metrics = record.get("metrics", {})
        if int(metrics.get("trainable_parameters", -1)) != 460800:
            raise ValueError(f"{label} {key} does not have 460800 trainable parameters")
        checkpoint = record.get("checkpoint", {})
        if not SHA256_RE.fullmatch(str(checkpoint.get("sha256", ""))):
            raise ValueError(f"{label} {key} has no valid checkpoint SHA-256")
        indexed[key] = record
    return indexed


def _verify_checkpoint_file(record, key):
    checkpoint = record["checkpoint"]
    path = Path(checkpoint["path"])
    if not path.is_file():
        raise ValueError(f"checkpoint file missing for {key}: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if digest != checkpoint["sha256"].upper():
        raise ValueError(f"checkpoint SHA-256 mismatch for {key}")


def audit_accuracy_matrix(
    train_records, eval_records, verify_checkpoint_files=False,
    routing_mode="input_independent",
):
    train = _index(train_records, "train", "completed", routing_mode=routing_mode)
    evaluate = _index(eval_records, "eval", "eval_only", routing_mode=routing_mode)
    if set(train) != set(evaluate):
        raise ValueError("train and eval manifests do not contain identical 72 configurations")
    differences = []
    for key in sorted(train):
        train_record, eval_record = train[key], evaluate[key]
        if _combined_split_hash(train_record) != _combined_split_hash(eval_record):
            raise ValueError(f"train/eval split provenance mismatch for {key}")
        if train_record["checkpoint"]["sha256"].upper() != eval_record["checkpoint"]["sha256"].upper():
            raise ValueError(f"train/eval checkpoint SHA mismatch for {key}")
        difference = abs(
            float(train_record["metrics"]["test_accuracy"])
            - float(eval_record["metrics"]["test_accuracy"])
        )
        if difference > 0.05 + 1e-12:
            raise ValueError(f"train/eval accuracy drift {difference:.6f} pp exceeds 0.05 for {key}")
        differences.append(difference)
        if verify_checkpoint_files:
            _verify_checkpoint_file(train_record, key)
    return {
        "passed": True,
        "train_rows": len(train_records), "eval_rows": len(eval_records),
        "unique_configurations": len(train),
        "max_train_eval_accuracy_difference_pp": max(differences),
        "checkpoint_files_verified": bool(verify_checkpoint_files),
    }


def checkpoint_inventory(train_records, routing_mode="input_independent"):
    indexed = _index(train_records, "train", "completed", routing_mode=routing_mode)
    return [
        {
            "dataset": key[0], "shot": key[1], "seed": key[2],
            "path": indexed[key]["checkpoint"]["path"],
            "sha256": indexed[key]["checkpoint"]["sha256"].upper(),
        }
        for key in sorted(indexed)
    ]


def _accuracy_row(record):
    config = record.get("config", {})
    return {
        "dataset": config.get("dataset"),
        "shot": config.get("shots"),
        "seed": config.get("seed"),
        "accuracy": record.get("metrics", {}).get("test_accuracy"),
        "split_hash": _combined_split_hash(record),
        "trainable_parameters": record.get("metrics", {}).get("trainable_parameters"),
    }


def audit_smoke(train_records, eval_records):
    key = ("eurosat", 4, 1)
    train = [record for record in train_records if _key(record) == key]
    evaluate = [record for record in eval_records if _key(record) == key]
    if len(train) != 1 or len(evaluate) != 1:
        raise ValueError("smoke manifests must contain exactly one EuroSAT 4-shot seed-1 row")
    for label, record, status in (("train", train[0], "completed"), ("eval", evaluate[0], "eval_only")):
        config = record.get("config", {})
        if record.get("status") != status:
            raise ValueError(f"smoke {label} status is not {status}")
        if config.get("routing_mode") != "input_independent":
            raise ValueError("smoke is not input-independent")
        if int(record.get("metrics", {}).get("trainable_parameters", -1)) != 460800:
            raise ValueError("smoke parameter count is not 460800")
    if _combined_split_hash(train[0]) != _combined_split_hash(evaluate[0]):
        raise ValueError("smoke train/eval split mismatch")
    difference = abs(float(train[0]["metrics"]["test_accuracy"]) - float(evaluate[0]["metrics"]["test_accuracy"]))
    if not math.isfinite(difference) or difference > 0.05 + 1e-12:
        raise ValueError(f"smoke reload drift {difference} exceeds 0.05 pp")
    return {"passed": True, "configuration": key, "reload_difference_pp": difference}


def audit_sentinel_manifests(new_train_records, new_eval_records, reference_records):
    expected_keys = {(dataset, shot, 1) for dataset in ("eurosat", "caltech101") for shot in (1, 4, 16)}
    def select(records, status, label):
        selected = [record for record in records if _key(record) in expected_keys]
        if len(selected) != 6 or {_key(record) for record in selected} != expected_keys:
            raise ValueError(f"{label} does not contain the six sentinel configurations")
        if any(record.get("status") != status for record in selected):
            raise ValueError(f"{label} contains invalid status rows")
        return selected
    train = select(new_train_records, "completed", "sentinel train")
    evaluate = select(new_eval_records, "eval_only", "sentinel eval")
    reference = select(reference_records, "completed", "sentinel reference")
    eval_index = {_key(record): record for record in evaluate}
    for record in train:
        key = _key(record)
        reload_record = eval_index[key]
        if record["checkpoint"]["sha256"].upper() != reload_record["checkpoint"]["sha256"].upper():
            raise ValueError(f"sentinel checkpoint mismatch for {key}")
        drift = abs(float(record["metrics"]["test_accuracy"]) - float(reload_record["metrics"]["test_accuracy"]))
        if drift > 0.05 + 1e-12:
            raise ValueError(f"sentinel reload drift exceeds 0.05 pp for {key}")
    return audit_dynamic_sentinel(
        [_accuracy_row(record) for record in train],
        [_accuracy_row(record) for record in reference],
    )


def audit_dynamic_sentinel(new_rows, reference_rows):
    def index(rows, label):
        values = {}
        for row in rows:
            key = (str(row["dataset"]), int(row.get("shot", row.get("shots"))), int(row["seed"]))
            if key in values:
                raise ValueError(f"duplicate {label} sentinel key {key}")
            values[key] = row
        if len(values) != 6:
            raise ValueError(f"{label} sentinel must contain six rows")
        return values
    new, reference = index(new_rows, "new"), index(reference_rows, "reference")
    if set(new) != set(reference):
        raise ValueError("sentinel keys do not match")
    differences = []
    for key in sorted(new):
        for field in ("split_hash", "trainable_parameters"):
            if field in new[key] or field in reference[key]:
                if new[key].get(field) != reference[key].get(field):
                    raise ValueError(f"sentinel {field} mismatch for {key}")
        differences.append(abs(float(new[key]["accuracy"]) - float(reference[key]["accuracy"])))
    mean_difference = sum(differences) / len(differences)
    max_difference = max(differences)
    if mean_difference > 0.5 + 1e-12 or max_difference > 1.0 + 1e-12:
        raise ValueError(
            f"dynamic sentinel failed: mean absolute difference={mean_difference:.6f} pp, "
            f"maximum={max_difference:.6f} pp"
        )
    return {
        "passed": True, "pairs": 6,
        "mean_absolute_difference_pp": mean_difference,
        "maximum_absolute_difference_pp": max_difference,
    }


def audit_profile_environment(records):
    if len(records) != 24:
        raise ValueError(f"compute profiling requires 24 training rows; found {len(records)}")
    def environment(record):
        return record.get("environment", record)
    gpu_names = {environment(record).get("gpu_name") for record in records}
    gpu_uuids = {environment(record).get("gpu_uuid") for record in records}
    stacks = {environment(record).get("software_stack_id") for record in records}
    power_limits = {environment(record).get("power_limit_watts") for record in records}
    if len(gpu_names) != 1 or "3090" not in next(iter(gpu_names), ""):
        raise ValueError(f"all profiling rows must use an RTX 3090; found {gpu_names}")
    if len(gpu_uuids) != 1 or None in gpu_uuids:
        raise ValueError(f"profiling spans multiple or unidentified physical GPUs: {gpu_uuids}")
    if len(stacks) != 1 or None in stacks:
        raise ValueError(f"profiling spans multiple or unidentified software stacks: {stacks}")
    if len(power_limits) != 1 or None in power_limits:
        raise ValueError(f"profiling power limit changed or was not recorded: {power_limits}")
    return {
        "passed": True, "rows": 24,
        "gpu_name": next(iter(gpu_names)), "gpu_uuid": next(iter(gpu_uuids)),
        "software_stack_id": next(iter(stacks)),
        "power_limit_watts": next(iter(power_limits)),
    }


def audit_dataset_root(data_root, reference_records):
    import random
    import numpy as np
    import torch
    import clip
    from datasets import build_dataset
    from ecr3_provenance import dataset_provenance
    from reviewer4_provenance import split_identity, validate_split_reference

    if len(reference_records) != 72:
        raise ValueError("dataset audit requires the frozen 72-row dynamic reference")
    model, preprocess = clip.load("ViT-B/16", device="cpu")
    del model
    observed = []
    for reference in sorted(reference_records, key=_key):
        config = reference["config"]
        seed = int(config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dataset = build_dataset(config["dataset"], data_root, int(config["shots"]), preprocess)
        provenance = dataset_provenance(dataset)
        result = validate_split_reference(
            config["dataset"], config["shots"], seed, provenance, reference_records,
        )
        observed.append({
            "config": {"dataset": config["dataset"], "shots": int(config["shots"]), "seed": seed},
            "dataset": provenance, "split_identity": split_identity(provenance), "passed": result["passed"],
        })
    if len(observed) != 72 or len({_key(row) for row in observed}) != 72:
        raise ValueError("dataset audit did not produce 72 unique identities")
    return observed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--train-manifest", required=True)
    matrix.add_argument("--eval-manifest", required=True)
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--verify-checkpoint-files", action="store_true")
    matrix.add_argument("--checkpoint-inventory")
    matrix.add_argument(
        "--routing-mode",
        choices=("input_conditioned", "input_independent"),
        default="input_independent",
    )
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--train-manifest", required=True)
    smoke.add_argument("--eval-manifest", required=True)
    smoke.add_argument("--output", required=True)
    sentinel = subparsers.add_parser("sentinel")
    sentinel.add_argument("--train-manifest", required=True)
    sentinel.add_argument("--eval-manifest", required=True)
    sentinel.add_argument("--reference-manifest", required=True)
    sentinel.add_argument("--output", required=True)
    profile = subparsers.add_parser("profile-environment")
    profile.add_argument("--profile-jsonl", required=True)
    profile.add_argument("--output", required=True)
    freeze = subparsers.add_parser("freeze-reference")
    freeze.add_argument("--source-manifest", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--checksum-inventory", required=True)
    freeze.add_argument("--checksum-path", action="append", default=[])
    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--data-root", required=True)
    dataset.add_argument("--reference-manifest", required=True)
    dataset.add_argument("--observed-output", required=True)
    dataset.add_argument("--output", required=True)
    environment = subparsers.add_parser("environment")
    environment.add_argument("--output", required=True)
    checksums = subparsers.add_parser("checksums")
    checksums.add_argument("--inventory", required=True)
    checksums.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "matrix":
        train_records = load_jsonl(args.train_manifest)
        report = audit_accuracy_matrix(
            train_records, load_jsonl(args.eval_manifest),
            verify_checkpoint_files=args.verify_checkpoint_files,
            routing_mode=args.routing_mode,
        )
        if args.checkpoint_inventory:
            Path(args.checkpoint_inventory).write_text(
                json.dumps(checkpoint_inventory(train_records, args.routing_mode), indent=2) + "\n",
                encoding="utf-8",
            )
    elif args.command == "smoke":
        report = audit_smoke(load_jsonl(args.train_manifest), load_jsonl(args.eval_manifest))
    elif args.command == "sentinel":
        report = audit_sentinel_manifests(
            load_jsonl(args.train_manifest), load_jsonl(args.eval_manifest),
            load_jsonl(args.reference_manifest),
        )
    elif args.command == "profile-environment":
        report = audit_profile_environment(load_jsonl(args.profile_jsonl))
    elif args.command == "freeze-reference":
        frozen = freeze_dynamic_reference(load_jsonl(args.source_manifest))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in frozen),
            encoding="utf-8",
        )
        paths = [args.source_manifest, args.output, *args.checksum_path]
        inventory = [{"path": Path(path).as_posix(), "sha256": file_sha256(path)} for path in paths]
        Path(args.checksum_inventory).write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        return
    elif args.command == "dataset":
        observed = audit_dataset_root(args.data_root, load_jsonl(args.reference_manifest))
        Path(args.observed_output).write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in observed),
            encoding="utf-8",
        )
        report = {"passed": True, "rows": len(observed), "unique_configurations": len({_key(row) for row in observed})}
    elif args.command == "environment":
        from reviewer4_provenance import capture_environment, validate_target_environment
        report = validate_target_environment(capture_environment())
    else:
        report = verify_checksum_inventory(args.inventory)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
