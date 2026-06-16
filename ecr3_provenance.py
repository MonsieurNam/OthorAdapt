import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _stable_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def file_sha256(path):
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _value(obj, name, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _item_record(item):
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return {
            "impath": str(item[0]),
            "label": int(item[1]),
            "domain": -1,
            "classname": "",
        }
    impath = _value(item, "impath", _value(item, "_impath", ""))
    label = _value(item, "label", _value(item, "_label", -1))
    domain = _value(item, "domain", _value(item, "_domain", -1))
    classname = _value(item, "classname", _value(item, "_classname", ""))
    return {
        "impath": str(impath),
        "label": int(label) if label is not None else -1,
        "domain": int(domain) if domain is not None else -1,
        "classname": str(classname),
    }


def _iter_data_source(data_source):
    if data_source is None:
        return []
    if hasattr(data_source, "imgs"):
        return list(data_source.imgs)
    if hasattr(data_source, "samples"):
        return list(data_source.samples)
    return list(data_source)


def split_provenance(data_source):
    records = [_item_record(item) for item in _iter_data_source(data_source)]
    records = sorted(records, key=lambda x: (x["label"], x["impath"], x["classname"], x["domain"]))
    class_counts = {}
    for record in records:
        key = str(record["label"])
        class_counts[key] = class_counts.get(key, 0) + 1
    return {
        "count": len(records),
        "class_counts": dict(sorted(class_counts.items(), key=lambda kv: int(kv[0]))),
        "sha256": _sha256_text(_stable_json(records)),
    }


def dataset_provenance(dataset):
    classnames = list(getattr(dataset, "classnames", []))
    return {
        "classnames": classnames,
        "classnames_sha256": _sha256_text(_stable_json(classnames)),
        "splits": {
            "train": split_provenance(getattr(dataset, "train_x", None)),
            "val": split_provenance(getattr(dataset, "val", None)),
            "test": split_provenance(getattr(dataset, "test", None)),
        },
    }


def args_to_config(args):
    out = {}
    for key, value in vars(args).items():
        if key.startswith("_") or key in {"dataset_provenance", "run_command", "checkpoint_extra_metadata"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        else:
            out[key] = str(value)
    return dict(sorted(out.items()))


def validate_reporting_policy(selection_split, sweep_mode=False, report_test=False):
    if selection_split not in {"val", "test"}:
        raise ValueError("selection_split must be one of: val, test")
    if sweep_mode and selection_split == "test":
        raise ValueError("Sweep mode cannot use the test split for selection")
    if sweep_mode and report_test:
        raise ValueError("Sweep mode cannot report the test split")
    if selection_split == "test" and not report_test:
        raise ValueError("--selection_split test requires --report_test")


def build_run_record(
    args,
    dataset_info,
    metrics,
    checkpoint_path="",
    checkpoint_sha256="",
    status="completed",
):
    command = getattr(args, "run_command", None) or sys.argv
    config = args_to_config(args)
    seed = config.get("seed")
    split_hashes = dataset_info.get("splits", {}) if dataset_info else {}
    return {
        "schema_version": "ecr3.run.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": list(command),
        "cwd": os.getcwd(),
        "config": config,
        "dataset": dataset_info,
        "metrics": metrics,
        "checkpoint": {
            "path": checkpoint_path or "",
            "sha256": checkpoint_sha256 or file_sha256(checkpoint_path),
        },
        "status": status,
        "evidence_gate": {
            "seed_status": "single_seed" if seed is not None else "seed_missing",
            "split_status": "split_hash_recorded" if split_hashes else "split_hash_missing",
            "verified_tier_a": False,
            "reason": "Phase 1 run record; requires seed2/seed3 and validation protocol before Tier-A verification",
        },
    }


def write_jsonl_record(path, record):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(_stable_json(record) + "\n")
