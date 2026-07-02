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


def write_report(pair_rows: list[dict], audit: dict) -> None:
    by_sev = defaultdict(list)
    for row in pair_rows:
        by_sev[row["severity"]].append(row)
    lines = [
        "# Phase 4 Robustness Report",
        "",
        f"Source: `{DEFAULT_MANIFEST.relative_to(ROOT).as_posix()}`.",
        "",
        f"- Severity rows: {audit['manifest_rows']} / {audit['expected_rows']}",
        "- Paired by dataset, shot, seed, and severity.",
        "",
        "| Severity | n pairs | Mean delta OH-LoRA | 95% CI | Mean LoRA drop | Mean OH drop | Drop delta OH-LoRA |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for severity in SEVERITIES:
        rows = by_sev[severity]
        delta = summary([row["delta_oh_minus_lora"] for row in rows])
        lines.append(
            f"| {severity} | {delta['n']} | {fmt(delta['mean'])} | [{fmt(delta['ci95_low'])}, {fmt(delta['ci95_high'])}] | "
            f"{fmt(mean([row['lora_drop'] for row in rows]))} | {fmt(mean([row['ohsinglora_drop'] for row in rows]))} | "
            f"{fmt(mean([row['drop_delta_oh_minus_lora'] for row in rows]))} |"
        )
    lines += [
        "",
        "## Claim Gate",
        "",
        "- Keep robustness claims only if paired deltas and drop deltas are favorable with uncertainty reported.",
        "- If severity-specific results are mixed, report robustness as mixed and avoid causal orthogonality wording.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args(argv)
    rows = load_rows(Path(args.manifest))
    pair_rows, audit = aggregate(rows)
    write_csv(pair_rows)
    write_report(pair_rows, audit)
    print(OUT_CSV.relative_to(ROOT).as_posix())
    print(OUT_REPORT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
