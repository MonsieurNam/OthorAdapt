"""Validation helpers for portable SingLoRA-family adapter checkpoints."""

from collections.abc import Mapping


CHECKPOINT_SCHEMA_VERSION = "orthadapt.adapter_checkpoint.v3"
LEGACY_CHECKPOINT_SCHEMA_VERSION = "orthadapt.adapter_checkpoint.v2"
_CANONICAL_PARAMS = ("q", "k", "v", "o", "mlp")
_GATED_ADAPTERS = {"gmhsinglora", "ohsinglora"}
_REQUIRED_METADATA_FIELDS = (
    "schema_version",
    "adapter",
    "r",
    "num_heads",
    "alpha",
    "ramp_up_steps",
    "params",
    "position",
    "encoder",
    "backbone",
    "routing_mode",
    "routing_context_id",
)
_LEGACY_REQUIRED_METADATA_FIELDS = tuple(
    field
    for field in _REQUIRED_METADATA_FIELDS
    if field not in {"routing_mode", "routing_context_id"}
)


def _value(args, name, default=None):
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _normalized_text(value, field):
    text = str(value).strip().lower()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _canonical_params(values):
    raw = [_normalized_text(value, "params") for value in values]
    if len(raw) != len(set(raw)):
        raise ValueError("params must not contain duplicates")
    unknown = sorted(set(raw).difference(_CANONICAL_PARAMS))
    if unknown:
        raise ValueError(f"Unsupported adapter params: {unknown}")
    return [name for name in _CANONICAL_PARAMS if name in raw]


def effective_num_heads(args):
    adapter = _normalized_text(_value(args, "adapter"), "adapter")
    if adapter == "singlora":
        return 1
    return int(_value(args, "num_heads"))


def effective_routing_mode(args):
    adapter = _normalized_text(_value(args, "adapter"), "adapter")
    requested = _normalized_text(
        _value(args, "routing_mode", "input_conditioned"),
        "routing_mode",
    )
    if requested not in {"input_conditioned", "input_independent"}:
        raise ValueError(
            "routing_mode must be either 'input_conditioned' or 'input_independent'"
        )
    if adapter not in _GATED_ADAPTERS:
        if requested == "input_independent":
            raise ValueError(
                f"routing_mode=input_independent is not valid for adapter={adapter}"
            )
        return "not_applicable"
    return requested


def adapter_metadata(args):
    """Build canonical compatibility metadata for a checkpoint v3 payload."""
    backbone = str(_value(args, "backbone")).strip()
    if not backbone:
        raise ValueError("backbone must be non-empty")
    routing_mode = effective_routing_mode(args)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "adapter": _normalized_text(_value(args, "adapter"), "adapter"),
        "r": int(_value(args, "r")),
        "num_heads": effective_num_heads(args),
        "alpha": int(_value(args, "alpha")),
        "ramp_up_steps": int(_value(args, "ramp_up_steps")),
        "params": _canonical_params(_value(args, "params", [])),
        "position": _normalized_text(_value(args, "position"), "position"),
        "encoder": _normalized_text(_value(args, "encoder"), "encoder"),
        "backbone": backbone,
        "routing_mode": routing_mode,
        "routing_context_id": (
            "normalized_ones_v1"
            if routing_mode == "input_independent"
            else "not_applicable"
        ),
    }


def validate_adapter_metadata(metadata, args):
    if not isinstance(metadata, Mapping):
        raise ValueError("Adapter checkpoint metadata must be a mapping")
    actual_schema = metadata.get("schema_version")
    if actual_schema not in {
        CHECKPOINT_SCHEMA_VERSION,
        LEGACY_CHECKPOINT_SCHEMA_VERSION,
    }:
        raise ValueError(
            "Adapter checkpoint v2 or v3 metadata is required; "
            f"got schema_version={actual_schema!r}. "
            "Legacy checkpoints without saved ramp state cannot be evaluated safely."
        )
    expected = adapter_metadata(args)
    if actual_schema == LEGACY_CHECKPOINT_SCHEMA_VERSION:
        if expected["routing_mode"] == "input_independent":
            raise ValueError(
                "Checkpoint v2 cannot establish routing_mode=input_independent; "
                "a checkpoint v3 payload is required"
            )
        required_fields = _LEGACY_REQUIRED_METADATA_FIELDS
    else:
        required_fields = _REQUIRED_METADATA_FIELDS

    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Checkpoint metadata is missing required field '{field}'")
        expected_value = (
            LEGACY_CHECKPOINT_SCHEMA_VERSION
            if field == "schema_version" and actual_schema == LEGACY_CHECKPOINT_SCHEMA_VERSION
            else expected[field]
        )
        if metadata[field] != expected_value:
            raise ValueError(
                f"Checkpoint metadata mismatch for '{field}': "
                f"expected {expected_value!r}, found {metadata[field]!r}"
            )
    return expected


def is_adapter_state_key(name):
    return (
        "lora_" in name
        or "gating_network" in name
        or name.endswith("training_step")
        or name.endswith("routing_context")
    )


def adapter_state_dict(model):
    return {
        name: value
        for name, value in model.state_dict().items()
        if is_adapter_state_key(name)
    }


def validate_adapter_state_dict(expected_state, checkpoint_weights):
    if not isinstance(checkpoint_weights, Mapping):
        raise ValueError("Checkpoint weights must be a mapping")
    expected_keys = set(expected_state)
    actual_keys = set(checkpoint_weights)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError(f"Checkpoint is missing adapter state keys: {missing[:5]}")
    if unexpected:
        raise ValueError(f"Checkpoint contains unexpected adapter state keys: {unexpected[:5]}")
    for name in sorted(expected_keys):
        expected_shape = tuple(expected_state[name].shape)
        actual_shape = tuple(checkpoint_weights[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Checkpoint tensor shape mismatch for '{name}': "
                f"expected {expected_shape}, found {actual_shape}"
            )


def validate_saturated_ramp_state(state, ramp_up_steps):
    ramp_up_steps = int(ramp_up_steps)
    step_keys = sorted(name for name in state if name.endswith("training_step"))
    if not step_keys:
        raise ValueError("Checkpoint is missing all training_step ramp-state buffers")
    unsaturated = []
    for name in step_keys:
        value = int(state[name].item())
        if value < ramp_up_steps:
            unsaturated.append((name, value))
    if unsaturated:
        preview = ", ".join(f"{name}={value}" for name, value in unsaturated[:5])
        raise ValueError(
            f"Checkpoint ramp is not saturated for ramp_up_steps={ramp_up_steps}: {preview}"
        )
