"""Run manifest schema utilities for revision experiments."""

from copy import deepcopy


RUN_MANIFEST_SCHEMA_VERSION = "phase1b.run.v1"
ADAPTER_PARAMETER_NAME_TOKENS = ("lora_", "scaler", "gating_network")


def _get_path(record, path):
    current = record
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _require(value, name):
    if value is None or value == "":
        raise ValueError(f"Missing required run manifest field: {name}")
    return value


def _coerce_int(value, name):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer run manifest field: {name}") from exc


def _coerce_float(value, name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric run manifest field: {name}") from exc


def count_named_adapter_parameters(named_parameters):
    total = 0
    for name, parameter in named_parameters:
        if any(token in name for token in ADAPTER_PARAMETER_NAME_TOKENS):
            total += parameter.numel()
    return total


def normalize_run_record(record):
    """Return a flat Phase 1B row from either flat or ECR3 nested records."""
    if not isinstance(record, dict):
        raise ValueError("Run manifest row must be a JSON object")

    if "dataset" in record and "method" in record and "accuracy" in record:
        flat = deepcopy(record)
    else:
        split = (
            _get_path(record, "config.selection_split")
            or _get_path(record, "metrics.selection_split")
        )
        accuracy = _get_path(record, f"metrics.{split}_accuracy") if split else None
        if accuracy is None:
            accuracy = _get_path(record, "metrics.selection_accuracy")

        flat = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "dataset": _get_path(record, "config.dataset"),
            "method": _get_path(record, "config.adapter"),
            "shot": _get_path(record, "config.shots"),
            "seed": _get_path(record, "config.seed"),
            "split": split,
            "accuracy": accuracy,
            "runtime_seconds": (
                _get_path(record, "metrics.runtime_seconds")
                or _get_path(record, "metrics.fine_tuning_seconds")
            ),
            "parameter_count": (
                _get_path(record, "metrics.trainable_parameters")
                or _get_path(record, "metrics.parameter_count")
            ),
            "checkpoint_sha256": _get_path(record, "checkpoint.sha256"),
            "git_revision": record.get("git_revision"),
        }

    return flat


def validate_run_record(record):
    """Validate and normalize one run-manifest row, failing closed on gaps."""
    flat = normalize_run_record(record)

    schema_version = _require(flat.get("schema_version"), "schema_version")
    if schema_version != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run manifest schema_version: {schema_version}; "
            f"expected {RUN_MANIFEST_SCHEMA_VERSION}"
        )

    normalized = {
        "schema_version": schema_version,
        "dataset": str(_require(flat.get("dataset"), "dataset")),
        "method": str(_require(flat.get("method"), "method")),
        "shot": _coerce_int(_require(flat.get("shot"), "shot"), "shot"),
        "seed": _coerce_int(_require(flat.get("seed"), "seed"), "seed"),
        "split": str(_require(flat.get("split"), "split")),
        "accuracy": _coerce_float(_require(flat.get("accuracy"), "accuracy"), "accuracy"),
        "runtime_seconds": _coerce_float(
            _require(flat.get("runtime_seconds"), "runtime_seconds"),
            "runtime_seconds",
        ),
        "parameter_count": _coerce_int(
            _require(flat.get("parameter_count"), "parameter_count"),
            "parameter_count",
        ),
        "checkpoint_sha256": str(
            _require(flat.get("checkpoint_sha256"), "checkpoint.sha256")
        ),
        "git_revision": str(_require(flat.get("git_revision"), "git_revision")),
    }

    if normalized["split"] not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    if normalized["runtime_seconds"] < 0:
        raise ValueError("runtime_seconds must be non-negative")
    if normalized["parameter_count"] < 0:
        raise ValueError("parameter_count must be non-negative")

    return normalized
