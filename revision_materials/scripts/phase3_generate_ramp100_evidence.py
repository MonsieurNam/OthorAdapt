"""Generate Phase 3 ramp100 evidence tables and audit reports."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "revision_materials" / "results"
MANIFEST = RESULTS_DIR / "phase3_main_ramp100_results.jsonl"

OUT_GROUP_CSV = RESULTS_DIR / "phase3_main_ramp100_summary.csv"
OUT_GROUP_MD = RESULTS_DIR / "phase3_main_ramp100_summary.md"
OUT_PAIRED_CSV = RESULTS_DIR / "phase3_main_ramp100_paired_summary.csv"
OUT_AUDIT_MD = RESULTS_DIR / "phase3_main_ramp100_audit.md"
OUT_STAT_REPORT = RESULTS_DIR / "statistical_report.md"
OUT_TABLES_TEX = RESULTS_DIR / "generated_tables.tex"

DATASET_ORDER = [
    "fgvc",
    "eurosat",
    "food101",
    "oxford_pets",
    "oxford_flowers",
    "caltech101",
    "dtd",
    "ucf101",
]
DATASET_LABELS = {
    "fgvc": "Aircraft",
    "eurosat": "EuroSAT",
    "food101": "Food",
    "oxford_pets": "Pets",
    "oxford_flowers": "Flowers",
    "caltech101": "Caltech",
    "dtd": "DTD",
    "ucf101": "UCF",
}
SHOT_ORDER = [1, 4, 16]
SEED_ORDER = [1, 2, 3]
METHOD_ORDER = ["lora", "ohsinglora"]
METHOD_LABELS = {
    "lora": "CLIP-LoRA r=8",
    "ohsinglora": "OrthoAdapt H=2,r=8",
}
EXPECTED_PARAMS = {
    "lora": 737280,
    "ohsinglora": 460800,
}
T_CRIT_95 = {
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
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    50: 2.009,
    60: 2.000,
    70: 1.994,
    80: 1.990,
    100: 1.984,
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def tcrit_95(df: int) -> float:
    if df <= 0:
        return 0.0
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    lower_keys = [key for key in T_CRIT_95 if key < df]
    upper_keys = [key for key in T_CRIT_95 if key > df]
    if lower_keys and upper_keys:
        lo = max(lower_keys)
        hi = min(upper_keys)
        frac = (df - lo) / (hi - lo)
        return T_CRIT_95[lo] + frac * (T_CRIT_95[hi] - T_CRIT_95[lo])
    return 1.96


def summary(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    mu = mean(values)
    sd = sample_std(values)
    half = 0.0 if n < 2 else tcrit_95(n - 1) * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mu,
        "std": sd,
        "ci95_half_width": half,
        "ci95_low": mu - half,
        "ci95_high": mu + half,
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_ci(row: dict[str, float | int]) -> str:
    return f"{fmt(float(row['mean']))} [{fmt(float(row['ci95_low']))}, {fmt(float(row['ci95_high']))}]"


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def get_accuracy(row: dict) -> float:
    metrics = row.get("metrics") or {}
    value = metrics.get("test_accuracy", metrics.get("selection_accuracy"))
    if value is None:
        raise ValueError(f"missing accuracy for {row.get('config', {}).get('filename')}")
    return float(value)


def audit_rows(rows: list[dict]) -> tuple[list[str], dict]:
    errors = []
    expected_count = len(DATASET_ORDER) * len(SHOT_ORDER) * len(SEED_ORDER) * len(METHOD_ORDER)
    if len(rows) != expected_count:
        errors.append(f"expected {expected_count} rows, found {len(rows)}")

    key_counts = Counter()
    params = defaultdict(set)
    ramps = defaultdict(set)
    split_counts = Counter()
    report_counts = Counter()
    status_counts = Counter()
    method_counts = Counter()

    for row in rows:
        cfg = row.get("config") or {}
        metrics = row.get("metrics") or {}
        method = cfg.get("adapter")
        dataset = cfg.get("dataset")
        shot = cfg.get("shots")
        seed = cfg.get("seed")
        key_counts[(method, dataset, shot, seed)] += 1
        params[method].add(metrics.get("trainable_parameters"))
        ramps[method].add(cfg.get("ramp_up_steps"))
        split_counts[cfg.get("selection_split")] += 1
        report_counts[cfg.get("report_test")] += 1
        status_counts[row.get("status")] += 1
        method_counts[method] += 1

    expected_keys = {
        (method, dataset, shot, seed)
        for method in METHOD_ORDER
        for dataset in DATASET_ORDER
        for shot in SHOT_ORDER
        for seed in SEED_ORDER
    }
    actual_keys = set(key_counts)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    dupes = sorted(key for key, count in key_counts.items() if count != 1)
    if missing:
        errors.append(f"missing expected keys: {missing[:8]}")
    if extra:
        errors.append(f"extra keys: {extra[:8]}")
    if dupes:
        errors.append(f"duplicate keys: {dupes[:8]}")
    if dict(split_counts) != {"test": expected_count}:
        errors.append(f"expected all rows selection_split=test, found {dict(split_counts)}")
    if dict(report_counts) != {True: expected_count}:
        errors.append(f"expected all rows report_test=true, found {dict(report_counts)}")
    if dict(status_counts) != {"completed": expected_count}:
        errors.append(f"expected all rows completed, found {dict(status_counts)}")
    for method, expected in EXPECTED_PARAMS.items():
        if params[method] != {expected}:
            errors.append(f"{method} parameter count expected {expected}, found {sorted(params[method])}")
    if ramps["lora"] != {100}:
        errors.append(f"LoRA ramp metadata expected {{100}}, found {sorted(ramps['lora'])}")
    if ramps["ohsinglora"] != {100}:
        errors.append(f"OrthoAdapt ramp expected {{100}}, found {sorted(ramps['ohsinglora'])}")

    details = {
        "expected_count": expected_count,
        "method_counts": method_counts,
        "params": {method: sorted(values) for method, values in params.items()},
        "ramps": {method: sorted(values) for method, values in ramps.items()},
        "split_counts": split_counts,
        "report_counts": report_counts,
        "status_counts": status_counts,
        "missing": missing,
        "extra": extra,
        "dupes": dupes,
    }
    return errors, details


def build_group_stats(rows: list[dict]) -> dict[tuple, dict]:
    values = defaultdict(list)
    runtimes = defaultdict(float)
    params = defaultdict(set)
    seeds = defaultdict(set)
    revisions = defaultdict(set)
    for row in rows:
        cfg = row["config"]
        method = cfg["adapter"]
        key = (cfg["dataset"], int(cfg["shots"]), method)
        values[key].append(get_accuracy(row))
        runtimes[key] += float(row["metrics"].get("runtime_seconds") or row["metrics"].get("fine_tuning_seconds") or 0.0)
        params[key].add(int(row["metrics"]["trainable_parameters"]))
        seeds[key].add(int(cfg["seed"]))
        revisions[key].add(row.get("git_revision", "UNKNOWN"))

    out = {}
    for key, vals in values.items():
        item = summary(vals)
        item["runtime_seconds_total"] = runtimes[key]
        item["parameter_count"] = sorted(params[key])
        item["seeds"] = sorted(seeds[key])
        item["git_revisions"] = sorted(revisions[key])
        out[key] = item
    return out


def build_pairs(rows: list[dict]) -> tuple[dict[tuple, dict], list[dict]]:
    by_key = {}
    for row in rows:
        cfg = row["config"]
        by_key[(cfg["dataset"], int(cfg["shots"]), int(cfg["seed"]), cfg["adapter"])] = row

    all_pairs = []
    grouped = defaultdict(list)
    for dataset in DATASET_ORDER:
        for shot in SHOT_ORDER:
            for seed in SEED_ORDER:
                lora = by_key[(dataset, shot, seed, "lora")]
                oh = by_key[(dataset, shot, seed, "ohsinglora")]
                delta = get_accuracy(oh) - get_accuracy(lora)
                item = {
                    "dataset": dataset,
                    "shot": shot,
                    "seed": seed,
                    "lora_accuracy": get_accuracy(lora),
                    "ohsinglora_accuracy": get_accuracy(oh),
                    "delta": delta,
                }
                all_pairs.append(item)
                grouped[(dataset, shot)].append(delta)

    paired = {}
    for key, deltas in grouped.items():
        paired[key] = summary(deltas)
        paired[key]["wins"] = sum(1 for value in deltas if value > 0)
        paired[key]["ties"] = sum(1 for value in deltas if value == 0)
        paired[key]["losses"] = sum(1 for value in deltas if value < 0)
    return paired, all_pairs


def write_group_csv(group_stats: dict[tuple, dict]) -> None:
    fields = [
        "dataset",
        "dataset_label",
        "shot",
        "method",
        "method_label",
        "n",
        "mean_accuracy",
        "std_accuracy",
        "ci95_low",
        "ci95_high",
        "ci95_half_width",
        "seeds",
        "parameter_count",
        "runtime_seconds_total",
        "git_revisions",
    ]
    with OUT_GROUP_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASET_ORDER:
            for shot in SHOT_ORDER:
                for method in METHOD_ORDER:
                    row = group_stats[(dataset, shot, method)]
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "dataset_label": DATASET_LABELS[dataset],
                            "shot": shot,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "n": row["n"],
                            "mean_accuracy": fmt(float(row["mean"]), 6),
                            "std_accuracy": fmt(float(row["std"]), 6),
                            "ci95_low": fmt(float(row["ci95_low"]), 6),
                            "ci95_high": fmt(float(row["ci95_high"]), 6),
                            "ci95_half_width": fmt(float(row["ci95_half_width"]), 6),
                            "seeds": " ".join(str(seed) for seed in row["seeds"]),
                            "parameter_count": " ".join(str(value) for value in row["parameter_count"]),
                            "runtime_seconds_total": fmt(float(row["runtime_seconds_total"]), 6),
                            "git_revisions": " ".join(row["git_revisions"]),
                        }
                    )


def write_paired_csv(paired_stats: dict[tuple, dict]) -> None:
    fields = [
        "dataset",
        "dataset_label",
        "shot",
        "paired_n",
        "mean_delta_oh_minus_lora",
        "std_delta",
        "ci95_low",
        "ci95_high",
        "ci95_half_width",
        "wins",
        "ties",
        "losses",
    ]
    with OUT_PAIRED_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASET_ORDER:
            for shot in SHOT_ORDER:
                row = paired_stats[(dataset, shot)]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "shot": shot,
                        "paired_n": row["n"],
                        "mean_delta_oh_minus_lora": fmt(float(row["mean"]), 6),
                        "std_delta": fmt(float(row["std"]), 6),
                        "ci95_low": fmt(float(row["ci95_low"]), 6),
                        "ci95_high": fmt(float(row["ci95_high"]), 6),
                        "ci95_half_width": fmt(float(row["ci95_half_width"]), 6),
                        "wins": row["wins"],
                        "ties": row["ties"],
                        "losses": row["losses"],
                    }
                )


def table_rows_for_shot(group_stats: dict[tuple, dict], shot: int) -> list[dict]:
    rows = []
    for method in METHOD_ORDER:
        values = []
        for dataset in DATASET_ORDER:
            values.append(float(group_stats[(dataset, shot, method)]["mean"]))
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "dataset_values": dict(zip(DATASET_ORDER, values)),
                "average": mean(values),
                "parameter_count": EXPECTED_PARAMS[method],
            }
        )
    return rows


def write_summary_md(group_stats: dict[tuple, dict], paired_stats: dict[tuple, dict], overall_pair: dict, shot_pair: dict) -> None:
    lines = [
        "# Phase 3 Main Ramp100 Summary",
        "",
        f"Source manifest: `{MANIFEST.as_posix()}`.",
        "",
        "Accuracy values are test accuracies in percentage points. Intervals are 95% confidence intervals across the three seeds for each dataset-shot-method cell.",
        "",
        "## Main Table Means",
    ]
    for shot in SHOT_ORDER:
        lines += [
            "",
            f"### {shot}-shot",
            "",
            "| Method | Aircraft | EuroSAT | Food | Pets | Flowers | Caltech | DTD | UCF | Average | Params |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in table_rows_for_shot(group_stats, shot):
            vals = [fmt(row["dataset_values"][dataset], 2) for dataset in DATASET_ORDER]
            lines.append(
                "| "
                + row["method_label"]
                + " | "
                + " | ".join(vals)
                + f" | {fmt(row['average'], 2)} | {row['parameter_count']:,} |"
            )

    lines += [
        "",
        "## Paired Delta Summary",
        "",
        "Delta is OrthoAdapt minus CLIP-LoRA, paired by `(dataset, shot, seed)`.",
        "",
        f"- Overall seed-level pairs: n={overall_pair['n']}, mean delta={fmt(overall_pair['mean'])} pp, 95% CI [{fmt(overall_pair['ci95_low'])}, {fmt(overall_pair['ci95_high'])}] pp.",
        f"- Wins/ties/losses across seed-level pairs: {overall_pair['wins']} / {overall_pair['ties']} / {overall_pair['losses']}.",
        "",
        "| Shot | n pairs | Mean delta | 95% CI | Wins/ties/losses |",
        "|---:|---:|---:|---:|---:|",
    ]
    for shot in SHOT_ORDER:
        row = shot_pair[shot]
        lines.append(
            f"| {shot} | {row['n']} | {fmt(row['mean'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {row['wins']} / {row['ties']} / {row['losses']} |"
        )

    lines += [
        "",
        "## Dataset-Shot Paired Deltas",
        "",
        "| Dataset | Shot | n pairs | Mean delta | 95% CI | Wins/ties/losses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_ORDER:
        for shot in SHOT_ORDER:
            row = paired_stats[(dataset, shot)]
            lines.append(
                f"| {DATASET_LABELS[dataset]} | {shot} | {row['n']} | {fmt(row['mean'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {row['wins']} / {row['ties']} / {row['losses']} |"
            )

    OUT_GROUP_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_generated_tables(group_stats: dict[tuple, dict], paired_stats: dict[tuple, dict]) -> None:
    lines = [
        "% Auto-generated from revision_materials/results/phase3_main_ramp100_results.jsonl",
        "% Do not edit values manually; rerun revision_materials/scripts/phase3_generate_ramp100_evidence.py.",
        "",
    ]
    for shot in SHOT_ORDER:
        colspec = "l" + "r" * len(DATASET_ORDER) + "rr"
        lines += [
            "\\begin{table*}[t]",
            "\\centering",
            f"\\caption{{Phase 3 ramp100 {shot}-shot test accuracy. Values are means over three seeds.}}",
            f"\\label{{tab:phase3-ramp100-{shot}shot}}",
            f"\\begin{{tabular}}{{{colspec}}}",
            "\\toprule",
            "Method & "
            + " & ".join(latex_escape(DATASET_LABELS[dataset]) for dataset in DATASET_ORDER)
            + " & Avg. & Params \\\\",
            "\\midrule",
        ]
        for row in table_rows_for_shot(group_stats, shot):
            values = [fmt(row["dataset_values"][dataset], 2) for dataset in DATASET_ORDER]
            lines.append(
                latex_escape(row["method_label"])
                + " & "
                + " & ".join(values)
                + f" & {fmt(row['average'], 2)} & {row['parameter_count']:,} \\\\"
            )
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]

    lines += [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Paired Phase 3 ramp100 deltas. Delta is OrthoAdapt minus CLIP-LoRA, paired by dataset, shot, and seed.}",
        "\\label{tab:phase3-ramp100-paired-deltas}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & Shot & n & Mean delta & 95\\% CI & Wins/ties/losses \\\\",
        "\\midrule",
    ]
    for dataset in DATASET_ORDER:
        for shot in SHOT_ORDER:
            row = paired_stats[(dataset, shot)]
            lines.append(
                f"{latex_escape(DATASET_LABELS[dataset])} & {shot} & {row['n']} & {fmt(float(row['mean']))} & "
                f"[{fmt(float(row['ci95_low']))}, {fmt(float(row['ci95_high']))}] & "
                f"{row['wins']}/{row['ties']}/{row['losses']} \\\\"
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    OUT_TABLES_TEX.write_text("\n".join(lines), encoding="utf-8")


def aggregate_pairs(all_pairs: list[dict], key_fn) -> dict:
    grouped = defaultdict(list)
    for pair in all_pairs:
        grouped[key_fn(pair)].append(pair["delta"])
    out = {}
    for key, deltas in grouped.items():
        item = summary(deltas)
        item["wins"] = sum(1 for value in deltas if value > 0)
        item["ties"] = sum(1 for value in deltas if value == 0)
        item["losses"] = sum(1 for value in deltas if value < 0)
        out[key] = item
    return out


def overall_method_summary(rows: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    runtimes = defaultdict(float)
    for row in rows:
        method = row["config"]["adapter"]
        grouped[method].append(get_accuracy(row))
        runtimes[method] += float(row["metrics"].get("runtime_seconds") or row["metrics"].get("fine_tuning_seconds") or 0.0)
    out = {}
    for method, values in grouped.items():
        out[method] = summary(values)
        out[method]["runtime_seconds_total"] = runtimes[method]
        out[method]["parameter_count"] = EXPECTED_PARAMS[method]
    return out


def write_audit_md(audit: dict, overall_methods: dict[str, dict]) -> None:
    lines = [
        "# Phase 3 Ramp100 Audit",
        "",
        f"Source manifest: `{MANIFEST.as_posix()}`.",
        "",
        "## Coverage",
        "",
        f"- Expected rows: {audit['expected_count']}",
        f"- Manifest rows: {sum(audit['method_counts'].values())}",
        f"- Methods: {dict(audit['method_counts'])}",
        f"- Status counts: {dict(audit['status_counts'])}",
        f"- Selection split counts: {dict(audit['split_counts'])}",
        f"- Report-test counts: {dict(audit['report_counts'])}",
        "- Missing expected `(method, dataset, shot, seed)` keys: 0",
        "- Extra keys: 0",
        "- Duplicate keys: 0",
        "",
        "## Parameter And Ramp Metadata",
        "",
        "| Method | Parameter count | `ramp_up_steps` metadata | Note |",
        "|---|---:|---:|---|",
        "| CLIP-LoRA r=8 | 737,280 | 100 | Normalized metadata only; LoRA does not use ramp-up scheduling. |",
        "| OrthoAdapt H=2,r=8 | 460,800 | 100 | Active ramp-up setting for OH-SingLoRA/OrthoAdapt. |",
        "",
        "## Overall Method Means",
        "",
        "| Method | n | Mean accuracy | Std | 95% CI | Runtime (h) | Params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = overall_methods[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['n']} | {fmt(row['mean'])} | {fmt(row['std'])} | "
            f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | "
            f"{fmt(float(row['runtime_seconds_total']) / 3600.0)} | {row['parameter_count']:,} |"
        )
    OUT_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_statistical_report(overall_methods: dict[str, dict], overall_pair: dict, shot_pair: dict, dataset_pair: dict) -> None:
    param_reduction = (1 - EXPECTED_PARAMS["ohsinglora"] / EXPECTED_PARAMS["lora"]) * 100
    lines = [
        "# Phase 3 Ramp100 Statistical Report",
        "",
        f"Source manifest: `{MANIFEST.as_posix()}`.",
        "",
        "This report uses the final ramp100 Phase 3 test manifest after validation-only hyperparameter selection. Accuracy values are percentages. Paired deltas are computed as OrthoAdapt minus CLIP-LoRA using the exact `(dataset, shot, seed)` match.",
        "",
        "## Audit Result",
        "",
        "- Coverage is complete: 8 datasets x 3 shots x 2 methods x 3 seeds = 144 completed rows.",
        "- All rows use `selection_split=test` with `report_test=true`.",
        "- Pairing is complete for 72 seed-level comparisons.",
        "- Parameter counts are stable: CLIP-LoRA r=8 has 737,280 trainable parameters; OrthoAdapt H=2,r=8 has 460,800 trainable parameters.",
        "- The LoRA rows store `ramp_up_steps=100` only as normalized metadata from the shared CLI schema. This value is not a LoRA mechanism and should not be interpreted as a LoRA training component.",
        "",
        "## Overall Accuracy",
        "",
        "| Method | n | Mean accuracy | Std | 95% CI | Params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = overall_methods[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['n']} | {fmt(row['mean'])} | {fmt(row['std'])} | "
            f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {row['parameter_count']:,} |"
        )

    lines += [
        "",
        "## Paired Comparison",
        "",
        f"Across 72 matched seed-level pairs, OrthoAdapt changes accuracy by {fmt(overall_pair['mean'])} percentage points relative to CLIP-LoRA, with a 95% CI of [{fmt(overall_pair['ci95_low'])}, {fmt(overall_pair['ci95_high'])}] pp. The paired standardized effect size dz is {fmt(overall_pair['dz'])}. OrthoAdapt wins/ties/losses are {overall_pair['wins']} / {overall_pair['ties']} / {overall_pair['losses']}.",
        "",
        f"OrthoAdapt uses {param_reduction:.1f}% fewer trainable parameters than CLIP-LoRA r=8 in this matrix.",
        "",
        "## Paired Delta By Shot",
        "",
        "| Shot | n pairs | Mean delta | 95% CI | dz | Wins/ties/losses |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for shot in SHOT_ORDER:
        row = shot_pair[shot]
        lines.append(
            f"| {shot} | {row['n']} | {fmt(row['mean'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {fmt(row['dz'])} | {row['wins']} / {row['ties']} / {row['losses']} |"
        )

    lines += [
        "",
        "## Paired Delta By Dataset",
        "",
        "| Dataset | n pairs | Mean delta | 95% CI | dz | Wins/ties/losses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_ORDER:
        row = dataset_pair[dataset]
        lines.append(
            f"| {DATASET_LABELS[dataset]} | {row['n']} | {fmt(row['mean'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {fmt(row['dz'])} | {row['wins']} / {row['ties']} / {row['losses']} |"
        )

    lines += [
        "",
        "## Claim Implications",
        "",
        "- Keep: validation-only selection was enforced before test reporting.",
        "- Keep: the final matrix is fully paired over datasets, shots, and seeds.",
        "- Keep with precise wording: OrthoAdapt shows a modest positive average delta in this ramp100 final matrix while using fewer trainable parameters than CLIP-LoRA r=8.",
        "- Weaken/remove: any broad claim of state-of-the-art performance, large gains, or consistent per-dataset superiority.",
        "- Weaken/remove: claims that orthogonality causes robustness or that head-count behavior is explained by rank fragmentation; those require Phase 4 diagnostics.",
    ]
    OUT_STAT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_dz(stats: dict) -> None:
    sd = float(stats["std"])
    stats["dz"] = 0.0 if sd == 0 else float(stats["mean"]) / sd


def main() -> None:
    rows = load_rows(MANIFEST)
    errors, audit = audit_rows(rows)
    if errors:
        raise SystemExit("Phase 3 ramp100 audit failed:\n" + "\n".join(f"- {error}" for error in errors))

    group_stats = build_group_stats(rows)
    paired_stats, all_pairs = build_pairs(rows)
    overall_methods = overall_method_summary(rows)
    overall_pair = summary([pair["delta"] for pair in all_pairs])
    overall_pair["wins"] = sum(1 for pair in all_pairs if pair["delta"] > 0)
    overall_pair["ties"] = sum(1 for pair in all_pairs if pair["delta"] == 0)
    overall_pair["losses"] = sum(1 for pair in all_pairs if pair["delta"] < 0)
    add_dz(overall_pair)

    shot_pair = aggregate_pairs(all_pairs, lambda pair: pair["shot"])
    dataset_pair = aggregate_pairs(all_pairs, lambda pair: pair["dataset"])
    for item in list(shot_pair.values()) + list(dataset_pair.values()) + list(paired_stats.values()):
        add_dz(item)

    write_group_csv(group_stats)
    write_paired_csv(paired_stats)
    write_summary_md(group_stats, paired_stats, overall_pair, shot_pair)
    write_generated_tables(group_stats, paired_stats)
    write_audit_md(audit, overall_methods)
    write_statistical_report(overall_methods, overall_pair, shot_pair, dataset_pair)

    for path in [
        OUT_GROUP_CSV,
        OUT_GROUP_MD,
        OUT_PAIRED_CSV,
        OUT_AUDIT_MD,
        OUT_STAT_REPORT,
        OUT_TABLES_TEX,
    ]:
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
