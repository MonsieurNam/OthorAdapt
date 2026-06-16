"""Aggregate Phase 1B run manifests into reproducible statistics."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from experiment_manifest import validate_run_record


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def _mean(values):
    return sum(values) / len(values)


def _sample_std(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _t_critical_95(df):
    if df <= 0:
        return 0.0
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    if df < 25:
        return T_CRITICAL_95[max(k for k in T_CRITICAL_95 if k < df)]
    if df < 30:
        return T_CRITICAL_95[25]
    return 1.96


def _summary(values):
    n = len(values)
    mean = _mean(values)
    std = _sample_std(values)
    ci95_half_width = 0.0 if n < 2 else _t_critical_95(n - 1) * std / math.sqrt(n)
    return {
        "n": n,
        "mean_accuracy": mean,
        "std_accuracy": std,
        "ci95_half_width": ci95_half_width,
    }


def _paired_effect_size(mean_delta, std_delta):
    if std_delta != 0:
        return mean_delta / std_delta
    if mean_delta > 0:
        return math.inf
    if mean_delta < 0:
        return -math.inf
    return 0.0


def load_manifest_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                rows.append(validate_run_record(raw))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}: line {line_no}: {exc}") from exc
    return rows


def aggregate_manifest(rows, baseline_method=None):
    normalized = [validate_run_record(row) for row in rows]
    grouped = defaultdict(list)
    by_seed = {}

    for row in normalized:
        key = (row["dataset"], row["shot"], row["method"], row["split"])
        grouped[key].append(row)
        by_seed[(row["dataset"], row["shot"], row["split"], row["method"], row["seed"])] = row

    groups = {}
    for key, group_rows in grouped.items():
        accuracies = [row["accuracy"] for row in group_rows]
        summary = _summary(accuracies)
        summary.update(
            {
                "seeds": sorted(row["seed"] for row in group_rows),
                "checkpoint_sha256": sorted(row["checkpoint_sha256"] for row in group_rows),
                "git_revisions": sorted({row["git_revision"] for row in group_rows}),
                "parameter_count": sorted({row["parameter_count"] for row in group_rows}),
                "runtime_seconds_total": sum(row["runtime_seconds"] for row in group_rows),
            }
        )
        groups[key] = summary

    paired = {}
    if baseline_method:
        for key, group_rows in grouped.items():
            dataset, shot, method, split = key
            if method == baseline_method:
                continue
            deltas = []
            seeds = []
            for row in group_rows:
                baseline = by_seed.get((dataset, shot, split, baseline_method, row["seed"]))
                if not baseline:
                    continue
                deltas.append(row["accuracy"] - baseline["accuracy"])
                seeds.append(row["seed"])
            if not deltas:
                continue
            std_delta = _sample_std(deltas)
            mean_delta = _mean(deltas)
            paired[key] = {
                "baseline_method": baseline_method,
                "paired_n": len(deltas),
                "paired_seeds": sorted(seeds),
                "mean_delta": mean_delta,
                "std_delta": std_delta,
                "ci95_delta_half_width": (
                    0.0
                    if len(deltas) < 2
                    else _t_critical_95(len(deltas) - 1) * std_delta / math.sqrt(len(deltas))
                ),
                "effect_size_dz": _paired_effect_size(mean_delta, std_delta),
            }

    return {"groups": groups, "paired": paired}


def write_group_csv(report, path):
    fieldnames = [
        "dataset",
        "shot",
        "method",
        "split",
        "n",
        "mean_accuracy",
        "std_accuracy",
        "ci95_half_width",
        "seeds",
        "parameter_count",
        "runtime_seconds_total",
        "git_revisions",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (dataset, shot, method, split), summary in sorted(report["groups"].items()):
            writer.writerow(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "method": method,
                    "split": split,
                    "n": summary["n"],
                    "mean_accuracy": f"{summary['mean_accuracy']:.6f}",
                    "std_accuracy": f"{summary['std_accuracy']:.6f}",
                    "ci95_half_width": f"{summary['ci95_half_width']:.6f}",
                    "seeds": " ".join(str(seed) for seed in summary["seeds"]),
                    "parameter_count": " ".join(str(v) for v in summary["parameter_count"]),
                    "runtime_seconds_total": f"{summary['runtime_seconds_total']:.6f}",
                    "git_revisions": " ".join(summary["git_revisions"]),
                }
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input JSONL run manifest")
    parser.add_argument("--out_csv", required=True, help="Output CSV summary")
    parser.add_argument("--baseline_method", default=None, help="Optional paired baseline method")
    args = parser.parse_args(argv)

    rows = load_manifest_jsonl(args.manifest)
    report = aggregate_manifest(rows, baseline_method=args.baseline_method)
    write_group_csv(report, args.out_csv)


if __name__ == "__main__":
    main()
