"""Environment and split-provenance gates for the Reviewer #4 rerun."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path


def _stable_hash(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def split_identity(dataset_payload):
    payload = {
        "classnames_sha256": dataset_payload.get("classnames_sha256"),
        "splits": {
            split: dataset_payload.get("splits", {}).get(split, {}).get("sha256")
            for split in ("train", "val", "test")
        },
    }
    if not payload["classnames_sha256"] or any(not value for value in payload["splits"].values()):
        raise ValueError("dataset provenance must contain class-name and train/val/test hashes")
    return _stable_hash(payload)


def validate_split_reference(dataset, shots, seed, dataset_payload, reference_records):
    key = (str(dataset), int(shots), int(seed))
    matches = []
    for record in reference_records:
        config = record.get("config", {})
        candidate = (str(config.get("dataset")), int(config.get("shots", -1)), int(config.get("seed", -1)))
        if candidate == key:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"split reference must contain exactly one row for {key}; found {len(matches)}")
    expected = split_identity(matches[0].get("dataset", {}))
    actual = split_identity(dataset_payload)
    if actual != expected:
        raise ValueError(f"dataset split identity mismatch for {key}: expected {expected}, got {actual}")
    return {"passed": True, "dataset": key[0], "shots": key[1], "seed": key[2], "split_identity": actual}


def load_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def capture_environment():
    import torch
    try:
        os_release = platform.freedesktop_os_release()
        os_distribution = os_release.get("PRETTY_NAME", os_release.get("NAME", "UNKNOWN"))
    except (OSError, AttributeError):
        os_distribution = "UNKNOWN"
    query = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,power.limit,clocks.current.graphics,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_lines = subprocess.run(query, check=True, capture_output=True, text=True).stdout.splitlines()
        fields = [part.strip() for part in gpu_lines[0].split(",")]
        gpu_count = len(gpu_lines)
    except (OSError, subprocess.CalledProcessError, IndexError):
        fields = ["UNAVAILABLE"] * 6
        gpu_count = 0
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        pip_freeze_sha256 = _stable_hash(sorted(line.strip() for line in freeze if line.strip()))
    except (OSError, subprocess.CalledProcessError):
        pip_freeze_sha256 = "UNAVAILABLE"
    environment = {
        "hostname": socket.gethostname(), "os": platform.platform(),
        "os_distribution": os_distribution,
        "python": platform.python_version(), "python_executable": sys.executable,
        "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda or "NONE",
        "cudnn": torch.backends.cudnn.version(),
        "pip_freeze_sha256": pip_freeze_sha256,
        "gpu_name": fields[0], "gpu_uuid": fields[1], "driver": fields[2],
        "visible_gpu_count": gpu_count,
        "power_limit_watts": fields[3], "graphics_clock_mhz": fields[4], "memory_clock_mhz": fields[5],
    }
    stack = {
        key: environment[key]
        for key in (
            "os", "os_distribution", "python", "pytorch", "cuda_runtime", "cudnn", "driver",
            "pip_freeze_sha256",
        )
    }
    environment["software_stack_id"] = _stable_hash(stack)
    return environment


def validate_target_environment(environment):
    if "3090" not in str(environment.get("gpu_name", "")):
        raise ValueError(f"Reviewer #4 runs require an RTX 3090; got {environment.get('gpu_name')!r}")
    if not environment.get("gpu_uuid") or environment["gpu_uuid"] == "UNAVAILABLE":
        raise ValueError("GPU UUID is unavailable")
    if int(environment.get("visible_gpu_count", 0)) != 1:
        raise ValueError(
            f"Reviewer #4 runs require exactly one visible physical GPU; "
            f"found {environment.get('visible_gpu_count')!r}"
        )
    expected = {
        "python": "3.11.10",
        "pytorch": "2.5.1+cu124",
        "cuda_runtime": "12.4",
        "cudnn": 90100,
    }
    mismatches = {
        key: {"expected": value, "actual": environment.get(key)}
        for key, value in expected.items()
        if environment.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Reviewer #4 software stack mismatch: {mismatches}")
    if "ubuntu 22.04" not in str(environment.get("os_distribution", "")).lower():
        raise ValueError(
            f"Reviewer #4 runs require Ubuntu 22.04; "
            f"got {environment.get('os_distribution')!r}"
        )
    return environment
