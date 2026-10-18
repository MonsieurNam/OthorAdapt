"""Locked bounded protocol and strict evidence gates (no GPU dependencies)."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

VERSION = 'orthadapt.optimization.v1'
ARMS = {
    'original': ('dense', 'manual'),
    'factorized': ('factorized', 'manual'),
    'sdpa': ('dense', 'sdpa'),
    'combined': ('factorized', 'sdpa'),
}
DATASETS = ('eurosat', 'caltech101')
GPU_UUID = 'GPU-6a3242b5-13dc-58ea-be01-e96440e63771'


def bounded_cells():
    return [dict(dataset=d, shots=s, seed=1, routing=r)
            for d in DATASETS for r in ('dynamic', 'static')
            for s in ((1, 4, 16) if r == 'dynamic' else (4,))]


def compare_predictions(reference, candidate, labels):
    a, b, y = np.asarray(reference), np.asarray(candidate), np.asarray(labels)
    if a.ndim != 2 or a.shape != b.shape or y.shape != (a.shape[0],) or not len(y):
        raise ValueError('logit/label shapes mismatch or empty evidence')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite logits')
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= a.shape[1]).any():
        raise ValueError('invalid class labels')
    pa, pb = a.argmax(1), b.argmax(1)
    accuracy_a, accuracy_b = float(np.mean(pa == y) * 100), float(np.mean(pb == y) * 100)
    agreement = float(np.mean(pa == pb) * 100)
    delta = accuracy_b - accuracy_a
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(samples=len(y), reference_accuracy=accuracy_a, accuracy=accuracy_b,
                accuracy_delta_pp=delta, agreement_percent=agreement,
                disagreements=int(np.count_nonzero(pa != pb)), max_abs_logit_error=float(difference.max()),
                rms_logit_error=float(np.sqrt(np.mean(difference ** 2))),
                passed=bool(abs(delta) <= .05 + 1e-10 and agreement >= 99.9 - 1e-10))


def paired_speed_gate(reference, candidate):
    if len(reference) != 5 or len(candidate) != 5:
        raise ValueError('exactly five matched timing repetitions required')
    if any(not math.isfinite(x) or x <= 0 for x in [*reference, *candidate]):
        raise ValueError('nonpositive/nonfinite latency')
    reductions = [(a-b)/a*100 for a, b in zip(reference, candidate)]
    faster = sum(b < a for a, b in zip(reference, candidate))
    median = statistics.median(reductions)
    return dict(paired_reductions_percent=reductions, median_reduction_percent=median,
                faster_repetitions=faster, passed=bool(median >= 10 and faster >= 4))


def validate_unique_coverage(rows, expected, fields):
    keys = [tuple(row[f] for f in fields) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError('coverage duplicate/missing/unexpected cells')


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False, default=str)
        stream.write('\n')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('empty CSV evidence')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()
