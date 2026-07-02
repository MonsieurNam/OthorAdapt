"""Generate Phase 4 diagnostics from existing validation manifests."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "revision_materials" / "results"
PLAN = ROOT / "revision_materials" / "plan"
VALIDATION = RESULTS / "validation_sweep_ramp100_results.jsonl"
W3_H1 = RESULTS / "w3_headcount_h1_ramp100_results.jsonl"
PHASE3 = RESULTS / "phase3_main_ramp100_results.jsonl"

OUT_CSV = RESULTS / "phase4_existing_diagnostics.csv"
OUT_MD = RESULTS / "phase4_existing_diagnostics.md"
OUT_INVENTORY = PLAN / "phase4_inventory.md"

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228, 20: 2.086, 30: 2.042}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def tcrit(df: int) -> float:
    if df <= 0:
        return 0.0
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    if df < 10:
        return T_CRIT_95[max(k for k in T_CRIT_95 if k < df)]
    if df < 20:
        return T_CRIT_95[10]
    if df < 30:
        return T_CRIT_95[20]
    return 1.96


def summary(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    mu = mean(values)
    sd = sample_std(values)
    half = 0.0 if n < 2 else tcrit(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mu, "std": sd, "ci95_low": mu - half, "ci95_high": mu + half, "ci95_half": half}


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def row_acc(row: dict) -> float:
    metrics = row.get("metrics") or {}
    value = metrics.get("selection_accuracy")
    if value is None:
        value = metrics.get("test_accuracy")
    if value is None:
        raise ValueError(f"missing accuracy for {row.get('config', {}).get('filename')}")
    return float(value)


def collect_existing() -> tuple[list[dict], dict]:
    rows = []
    sources = {
        "validation_sweep_ramp100": load_jsonl(VALIDATION),
        "w3_headcount_h1_ramp100": load_jsonl(W3_H1),
    }
    for source_name, source_rows in sources.items():
        for row in source_rows:
            cfg = row.get("config") or {}
            if cfg.get("adapter") != "ohsinglora":
                continue
            rows.append(
                {
                    "source": source_name,
                    "dataset": cfg.get("dataset"),
                    "shot": int(cfg.get("shots")),
                    "seed": int(cfg.get("seed")),
                    "H": int(cfg.get("num_heads")),
                    "r": int(cfg.get("r")),
                    "lambda_o": float(cfg.get("lambda_o")),
                    "ramp_up_steps": int(cfg.get("ramp_up_steps")),
                    "selection_split": cfg.get("selection_split"),
                    "report_test": bool(cfg.get("report_test")),
                    "accuracy": row_acc(row),
                    "parameter_count": int((row.get("metrics") or {}).get("trainable_parameters", 0)),
                    "checkpoint_sha256": (row.get("checkpoint") or {}).get("sha256", ""),
                }
            )
    details = {
        "source_counts": {name: len(value) for name, value in sources.items()},
        "phase3_rows": len(load_jsonl(PHASE3)),
    }
    return rows, details


def write_csv(groups: dict[tuple, dict]) -> None:
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_scope",
            "H",
            "r",
            "lambda_o",
            "shot",
            "n",
            "datasets",
            "seeds",
            "mean_accuracy",
            "std_accuracy",
            "ci95_low",
            "ci95_high",
            "parameter_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, item in sorted(groups.items(), key=lambda kv: (kv[0][4], kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
            H, r, lambda_o, shot, scope = key
            stats = item["stats"]
            writer.writerow(
                {
                    "source_scope": scope,
                    "H": H,
                    "r": r,
                    "lambda_o": lambda_o,
                    "shot": shot,
                    "n": stats["n"],
                    "datasets": " ".join(sorted(item["datasets"])),
                    "seeds": " ".join(str(seed) for seed in sorted(item["seeds"])),
                    "mean_accuracy": fmt(float(stats["mean"]), 6),
                    "std_accuracy": fmt(float(stats["std"]), 6),
                    "ci95_low": fmt(float(stats["ci95_low"]), 6),
                    "ci95_high": fmt(float(stats["ci95_high"]), 6),
                    "parameter_count": " ".join(str(v) for v in sorted(item["params"])),
                }
            )


def build_groups(rows: list[dict]) -> dict[tuple, dict]:
    grouped = defaultdict(list)
    for row in rows:
        scope = "validation_plus_h1" if row["shot"] == 4 and row["dataset"] in {"eurosat", "caltech101"} else "other"
        grouped[(row["H"], row["r"], row["lambda_o"], row["shot"], scope)].append(row)

    out = {}
    for key, group_rows in grouped.items():
        out[key] = {
            "stats": summary([row["accuracy"] for row in group_rows]),
            "datasets": {row["dataset"] for row in group_rows},
            "seeds": {row["seed"] for row in group_rows},
            "params": {row["parameter_count"] for row in group_rows},
        }
    return out


def write_md(rows: list[dict], groups: dict[tuple, dict]) -> None:
    lines = [
        "# Phase 4 Existing Diagnostics",
        "",
        "This report uses validation-only diagnostics already available before new Phase 4 runs. These values support configuration-sensitivity discussion only; they do not prove rank-fragmentation or robustness causality.",
        "",
        "## Coverage",
        "",
        f"- Validation sweep ramp100 rows used: {sum(1 for row in rows if row['source'] == 'validation_sweep_ramp100')}",
        f"- W3 H=1 ramp100 rows used: {sum(1 for row in rows if row['source'] == 'w3_headcount_h1_ramp100')}",
        "- All rows are validation split, 4-shot, EuroSAT/Caltech101, seeds 1/2/3.",
        "",
        "## H/r/lambda Sensitivity",
        "",
        "| H | r | lambda_o | n | Mean val acc | 95% CI | Params |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(groups.items(), key=lambda kv: (-float(kv[1]["stats"]["mean"]), kv[0])):
        H, r, lambda_o, shot, scope = key
        if scope != "validation_plus_h1" or shot != 4:
            continue
        stats = item["stats"]
        params = ",".join(str(v) for v in sorted(item["params"]))
        lines.append(
            f"| {H} | {r} | {lambda_o:g} | {stats['n']} | {fmt(float(stats['mean']))} | "
            f"[{fmt(float(stats['ci95_low']))}, {fmt(float(stats['ci95_high']))}] | {params} |"
        )
    lines += [
        "",
        "## Claim Gate",
        "",
        "- Keep: the selected ramp100 configuration is `H=2,r=8,lambda_o=0.03` under the frozen validation rule.",
        "- Keep: orthogonality/lambda effects are configuration-dependent in the validation subset.",
        "- Weaken/remove: any claim that higher head count universally helps or hurts.",
        "- Weaken/remove: rank-fragmentation explanations until checkpoint diagnostics support them.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inventory(details: dict) -> None:
    ckpt_count = sum(1 for _ in (ROOT / "revision_materials" / "checkpoints" / "phase3_main_ramp100").rglob("*.pt"))
    lines = [
        "# Phase 4 Inventory",
        "",
        "## Usable Evidence",
        "",
        f"- `{VALIDATION.relative_to(ROOT).as_posix()}`: {details['source_counts'].get('validation_sweep_ramp100', 0)} validation-only rows for H/r/lambda sensitivity.",
        f"- `{W3_H1.relative_to(ROOT).as_posix()}`: {details['source_counts'].get('w3_headcount_h1_ramp100', 0)} validation-only H=1 rows.",
        f"- `{PHASE3.relative_to(ROOT).as_posix()}`: {details['phase3_rows']} final Phase 3 rows for checkpoint pairing.",
        f"- Phase 3 ramp100 checkpoints on disk: {ckpt_count}.",
        "",
        "## Preliminary / Legacy Evidence",
        "",
        "- `data/result_scan_head`, `data/results_ablation_heads_lambda`, and `data/result_scan_loss` contain recovered logs but lack verified seed/split provenance; use only for context.",
        "- `img/figure_robustness.*` and `img/singular_value_spectrum-rank16.png` are legacy figures and must not be cited as regenerated evidence.",
        "",
        "## Missing Evidence Before Phase 4 Runs",
        "",
        "- Paired robustness manifest with deterministic corruption parameters.",
        "- Batch spectrum manifest from Phase 3 paired checkpoints.",
        "- ViT-L/14 pilot/subset manifest, or an explicit deferred report.",
    ]
    OUT_INVENTORY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, details = collect_existing()
    if not rows:
        raise SystemExit("No existing Phase 4 diagnostic rows found")
    bad = [row for row in rows if row["selection_split"] != "val" or row["report_test"]]
    if bad:
        raise SystemExit(f"Existing diagnostics contain non-validation rows: {bad[:3]}")
    groups = build_groups(rows)
    write_csv(groups)
    write_md(rows, groups)
    write_inventory(details)
    print(OUT_CSV.relative_to(ROOT).as_posix())
    print(OUT_MD.relative_to(ROOT).as_posix())
    print(OUT_INVENTORY.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
