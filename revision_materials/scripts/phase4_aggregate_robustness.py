"""Aggregate Phase 4 robustness manifest into paired summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "revision_materials" / "results"
DEFAULT_MANIFEST = RESULTS / "phase4_robustness_manifest.jsonl"
DEFAULT_REFERENCE = RESULTS / "phase3_main_ramp100_results.jsonl"
OUT_CSV = RESULTS / "phase4_robustness_summary.csv"
OUT_REPORT = RESULTS / "phase4_robustness_report.md"

DATASETS = ["fgvc", "eurosat", "food101", "oxford_pets", "oxford_flowers", "caltech101", "dtd", "ucf101"]
SHOTS = [1, 4, 16]
SEEDS = [1, 2, 3]
SEVERITIES = [0, 1, 2, 3]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing robustness manifest: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def tcrit(df: int) -> float:
    return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(df, 1.96)


def summary(values: list[float]) -> dict:
    n = len(values)
    mu = mean(values)
    sd = sample_std(values)
    half = 0.0 if n < 2 else tcrit(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mu, "std": sd, "ci95_low": mu - half, "ci95_high": mu + half}


def fmt(value: float) -> str:
    return f"{value:.3f}"


def phase3_key(row: dict) -> tuple[str, int, int, str]:
    cfg = row.get("config", {})
    dataset = row.get("dataset") if isinstance(row.get("dataset"), str) else cfg.get("dataset")
    shot = row.get("shot", row.get("shots", cfg.get("shot", cfg.get("shots"))))
    seed = row.get("seed", cfg.get("seed"))
    method = row.get("method") or row.get("adapter") or cfg.get("adapter")
    if method in {"orthoadapt", "ohsinglora", "OH-SingLoRA"}:
        method = "ohsinglora"
    elif method in {"lora", "CLIP-LoRA"}:
        method = "lora"
    return (dataset, int(shot), int(seed), method)


def phase3_accuracy(row: dict) -> float:
    for key in ["test_accuracy", "accuracy", "acc"]:
        if key in row and row[key] not in (None, ""):
            return float(row[key])
    metrics = row.get("metrics", {})
    for key in ["test_accuracy", "accuracy", "acc"]:
        if key in metrics and metrics[key] not in (None, ""):
            return float(metrics[key])
    raise KeyError("No accuracy field found in Phase 3 reference row")


def clean_consistency_audit(rows: list[dict], reference_path: Path, tolerance: float = 1.0) -> dict:
    if not reference_path.exists():
        return {"status": "no_reference", "reference": str(reference_path), "message": "Phase 3 reference manifest not found."}

    reference = {phase3_key(row): phase3_accuracy(row) for row in load_rows(reference_path)}
    comparisons = []
    missing = []
    for row in rows:
        if int(row["severity"]) != 0:
            continue
        key = (row["dataset"], int(row["shot"]), int(row["seed"]), row["method"])
        if key not in reference:
            missing.append(key)
            continue
        robust_acc = float(row["metrics"]["accuracy"])
        comparisons.append({"key": key, "diff": robust_acc - reference[key]})

    by_method = defaultdict(list)
    failures = []
    for item in comparisons:
        by_method[item["key"][3]].append(item["diff"])
        if abs(item["diff"]) > tolerance:
            failures.append(item)

    method_stats = {}
    for method, diffs in by_method.items():
        method_stats[method] = {
            "n": len(diffs),
            "mean_diff": mean(diffs),
            "min_diff": min(diffs),
            "max_diff": max(diffs),
            "failures": sum(abs(diff) > tolerance for diff in diffs),
        }

    return {
        "status": "pass" if not missing and not failures else "fail",
        "reference": reference_path.as_posix(),
        "tolerance": tolerance,
        "comparisons": len(comparisons),
        "missing_reference": len(missing),
        "failures": len(failures),
        "method_stats": method_stats,
        "failure_examples": failures[:10],
    }


def aggregate(rows: list[dict]) -> tuple[list[dict], dict]:
    expected_rows = len(DATASETS) * len(SHOTS) * len(SEEDS) * 2 * len(SEVERITIES)
    if len(rows) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} severity rows, found {len(rows)}")
    by_key = {}
    for row in rows:
        key = (row["dataset"], int(row["shot"]), int(row["seed"]), row["method"], int(row["severity"]))
        if key in by_key:
            raise SystemExit(f"Duplicate robustness key: {key}")
        by_key[key] = row

    pair_rows = []
    for dataset in DATASETS:
        for shot in SHOTS:
            for seed in SEEDS:
                clean_lora = float(by_key[(dataset, shot, seed, "lora", 0)]["metrics"]["accuracy"])
                clean_oh = float(by_key[(dataset, shot, seed, "ohsinglora", 0)]["metrics"]["accuracy"])
                for severity in SEVERITIES:
                    lora = float(by_key[(dataset, shot, seed, "lora", severity)]["metrics"]["accuracy"])
                    oh = float(by_key[(dataset, shot, seed, "ohsinglora", severity)]["metrics"]["accuracy"])
                    pair_rows.append(
                        {
                            "dataset": dataset,
                            "shot": shot,
                            "seed": seed,
                            "severity": severity,
                            "lora_accuracy": lora,
                            "ohsinglora_accuracy": oh,
                            "delta_oh_minus_lora": oh - lora,
                            "lora_drop": clean_lora - lora,
                            "ohsinglora_drop": clean_oh - oh,
                            "drop_delta_oh_minus_lora": (clean_oh - oh) - (clean_lora - lora),
                        }
                    )
    return pair_rows, {"manifest_rows": len(rows), "expected_rows": expected_rows}


def write_csv(pair_rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[(row["severity"], row["shot"])].append(row)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fields = ["severity", "shot", "paired_n", "mean_delta", "ci95_low", "ci95_high", "mean_lora_drop", "mean_oh_drop", "mean_drop_delta"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (severity, shot), rows in sorted(grouped.items()):
            delta = summary([row["delta_oh_minus_lora"] for row in rows])
            writer.writerow(
                {
                    "severity": severity,
                    "shot": shot,
                    "paired_n": delta["n"],
                    "mean_delta": f"{delta['mean']:.6f}",
                    "ci95_low": f"{delta['ci95_low']:.6f}",
                    "ci95_high": f"{delta['ci95_high']:.6f}",
                    "mean_lora_drop": f"{mean([row['lora_drop'] for row in rows]):.6f}",
                    "mean_oh_drop": f"{mean([row['ohsinglora_drop'] for row in rows]):.6f}",
                    "mean_drop_delta": f"{mean([row['drop_delta_oh_minus_lora'] for row in rows]):.6f}",
                }
            )


def write_report(pair_rows: list[dict], audit: dict, clean_audit: dict) -> None:
    by_sev = defaultdict(list)
    for row in pair_rows:
        by_sev[row["severity"]].append(row)
    severity_summaries = {}
    for severity in SEVERITIES:
        rows = by_sev[severity]
        severity_summaries[severity] = {
            "delta": summary([row["delta_oh_minus_lora"] for row in rows]),
            "lora_drop": mean([row["lora_drop"] for row in rows]),
            "oh_drop": mean([row["ohsinglora_drop"] for row in rows]),
            "drop_delta": mean([row["drop_delta_oh_minus_lora"] for row in rows]),
        }
    lines = [
        "# Phase 4 Robustness Report",
        "",
        f"Source: `{DEFAULT_MANIFEST.relative_to(ROOT).as_posix()}`.",
        "",
        f"- Severity rows: {audit['manifest_rows']} / {audit['expected_rows']}",
        "- Paired by dataset, shot, seed, and severity.",
        f"- Clean consistency gate: `{clean_audit['status']}` against `revision_materials/results/phase3_main_ramp100_results.jsonl`.",
        "",
        "| Severity | n pairs | Mean delta OH-LoRA | 95% CI | Mean LoRA drop | Mean OH drop | Drop delta OH-LoRA |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for severity in SEVERITIES:
        stats = severity_summaries[severity]
        delta = stats["delta"]
        lines.append(
            f"| {severity} | {delta['n']} | {fmt(delta['mean'])} | [{fmt(delta['ci95_low'])}, {fmt(delta['ci95_high'])}] | "
            f"{fmt(stats['lora_drop'])} | {fmt(stats['oh_drop'])} | "
            f"{fmt(stats['drop_delta'])} |"
        )
    clean_delta = severity_summaries[0]["delta"]
    severe_delta = severity_summaries[3]["delta"]
    severe_drop_delta = severity_summaries[3]["drop_delta"]
    lines += [
        "",
        "## Clean Consistency Audit",
        "",
    ]
    if clean_audit["status"] == "no_reference":
        lines.append(f"- {clean_audit['message']}")
    else:
        lines.append(
            f"- Compared severity-0 rows against Phase 3 test accuracy with tolerance {clean_audit['tolerance']} pp: "
            f"{clean_audit['comparisons']} comparisons, {clean_audit['failures']} failures, {clean_audit['missing_reference']} missing references."
        )
        lines += [
            "",
            "| Method | n | Mean clean-minus-Phase3 | Min | Max | Failures > tolerance |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for method in sorted(clean_audit["method_stats"]):
            stats = clean_audit["method_stats"][method]
            lines.append(
                f"| {method} | {stats['n']} | {fmt(stats['mean_diff'])} | {fmt(stats['min_diff'])} | "
                f"{fmt(stats['max_diff'])} | {stats['failures']} |"
            )
    lines += [
        "",
        "## Claim Gate",
        "",
    ]
    if clean_audit["status"] == "fail":
        lines += [
            "- Do not use this robustness manifest for scientific claims yet. Although coverage is structurally complete, the clean severity-0 check does not reproduce Phase 3 accuracy.",
            "- The current failure pattern indicates an evaluator/checkpoint-loading mismatch, especially for OH-SingLoRA. Fix the evaluator and rerun robustness before reporting any robustness numbers.",
            "- Reviewer-facing wording for now: paired robustness evaluation was audited but did not pass the clean-consistency gate, so robustness gains are removed/deferred.",
        ]
    else:
        lines += [
            f"- Do not keep the original robustness-improvement claim. At clean severity 0, OH-SingLoRA is lower than CLIP-LoRA by {fmt(clean_delta['mean'])} pp "
            f"(95% CI [{fmt(clean_delta['ci95_low'])}, {fmt(clean_delta['ci95_high'])}]).",
            f"- Under severe corruption, OH-SingLoRA remains lower by {fmt(severe_delta['mean'])} pp "
            f"(95% CI [{fmt(severe_delta['ci95_low'])}, {fmt(severe_delta['ci95_high'])}]).",
            f"- The severe corruption drop delta is {fmt(severe_drop_delta)} pp, meaning OH-SingLoRA loses fewer points relative to its own clean score, "
            "but this must not be reported as superior robust accuracy because the clean baseline is substantially lower.",
            "- Reviewer-facing wording: robustness is not supported by the paired Phase 4 evaluation; remove quantitative robustness-gain claims and keep only a limitation/future-work statement.",
        ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args(argv)
    rows = load_rows(Path(args.manifest))
    pair_rows, audit = aggregate(rows)
    clean_audit = clean_consistency_audit(rows, DEFAULT_REFERENCE)
    write_csv(pair_rows)
    write_report(pair_rows, audit, clean_audit)
    print(OUT_CSV.relative_to(ROOT).as_posix())
    print(OUT_REPORT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
