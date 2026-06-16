#!/usr/bin/env python
"""Normalize Phase 0 recovered logs/CSVs and report missing Tier-A evidence.

This script is intentionally conservative: recovered logs and summaries are
evidence, but they are not marked as verified Tier-A results until seed/split
metadata and final rerun manifests exist.
"""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "revision_claude" / "plan"
RESULTS_DIR = ROOT / "revision_claude" / "results"
MANIFEST_PATH = PLAN_DIR / "phase0_raw_artifact_manifest.csv"
WORKBOOK_PATH = ROOT / "data" / "clip_fewshot_results.xlsx"

OUT_JSONL = RESULTS_DIR / "main_results_manifest.jsonl"
OUT_CSV = RESULTS_DIR / "main_results_manifest.csv"
OUT_MISSING_CSV = RESULTS_DIR / "missing_tier_a_matrix.csv"
OUT_MISSING_MD = RESULTS_DIR / "missing_tier_a_matrix.md"
OUT_XCHECK_CSV = RESULTS_DIR / "workbook_vs_logs_crosscheck.csv"
OUT_XCHECK_MD = RESULTS_DIR / "workbook_vs_logs_crosscheck.md"
OUT_SUMMARY_MD = RESULTS_DIR / "phase0_normalization_summary.md"

TIER_A_DATASETS = [
    "fgvc",
    "eurosat",
    "food101",
    "oxford_pets",
    "oxford_flowers",
    "caltech101",
    "dtd",
    "ucf101",
]
TIER_A_SHOTS = [1, 4, 16]
TIER_A_METHODS = ["lora", "singlora", "ohsinglora"]
WORKBOOK_DATASET_COLUMNS = {
    "aircraft": "fgvc",
    "eurosat": "eurosat",
    "food": "food101",
    "food101": "food101",
    "pets": "oxford_pets",
    "oxford_pets": "oxford_pets",
    "flowers": "oxford_flowers",
    "oxford_flowers": "oxford_flowers",
    "caltech": "caltech101",
    "caltech101": "caltech101",
    "dtd": "dtd",
    "ucf": "ucf101",
    "ucf101": "ucf101",
}


def read_manifest() -> List[Dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def as_float(value: Any) -> Optional[float]:
    text = clean(value).replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def as_int(value: Any) -> Optional[int]:
    text = clean(value).lower()
    if not text:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


def normalize_dataset(value: Any) -> str:
    text = clean(value).lower()
    aliases = {
        "aircraft": "fgvc",
        "fgvc_aircraft": "fgvc",
        "oxfordflowers": "oxford_flowers",
        "oxfordflower": "oxford_flowers",
        "flowers": "oxford_flowers",
        "oxfordpets": "oxford_pets",
        "pets": "oxford_pets",
    }
    return aliases.get(text, text)


def normalize_method(value: Any) -> str:
    text = clean(value).lower()
    aliases = {
        "clip-lora": "lora",
        "cliplora": "lora",
        "vitb16": "lora",
        "oh": "ohsinglora",
        "orthoadapt": "ohsinglora",
        "orthoadapt/ohsinglora": "ohsinglora",
    }
    return aliases.get(text, text)


def normalize_workbook_method(value: Any) -> str:
    text = clean(value).lower()
    if not text:
        return ""
    if "gmhsinglora" in text:
        return "gmhsinglora"
    if "ohsing" in text or "orthoadapt" in text:
        return "ohsinglora"
    if "singlora" in text:
        return "singlora"
    if "clip-lora" in text or text == "lora" or text.startswith("lora"):
        return "lora"
    return ""


def parse_rank_from_text(value: Any) -> Optional[int]:
    text = clean(value).lower()
    match = re.search(r"(?:^|[_\-\s])r(?:ank)?[_\-\s]?(\d+)", text)
    return int(match.group(1)) if match else None


def parse_heads_from_text(value: Any) -> Optional[int]:
    text = clean(value).lower()
    patterns = [
        r"(?:numheads|heads|head)[_\-\s]?(\d+)",
        r"(?:^|[_\-\s])h[_\-\s]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def parse_heads_from_path(value: Any) -> Optional[int]:
    """Parse explicit head-count tokens without matching dataset names.

    The raw Phase 0 artifact manifest may contain noisy `head_hint` values
    because a broad regex can match the `h101` suffix in `caltech101`.
    Normalization should only accept explicit tokens such as `heads4` or
    `_h2`, not arbitrary letters inside dataset names.
    """

    text = basename(clean(value))
    patterns = [
        r"(?:^|[_\-])heads?[_\-]?(\d+)(?:$|[_\-.])",
        r"(?:^|[_\-])h[_\-]?(\d+)(?:$|[_\-.])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def normalized_head_hint(row: Dict[str, str]) -> Optional[int]:
    parsed = parse_heads_from_path(row.get("path", ""))
    if parsed is not None:
        return parsed

    value = as_int(row.get("head_hint"))
    if value is None:
        return None

    # Known experiment head counts are small. Larger values here almost
    # always come from dataset names such as `caltech101`.
    return value if 1 <= value <= 16 else None


def basename(path: str) -> str:
    return Path(path.replace("\\", "/")).name.lower()


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_saved_checkpoint(line: str) -> str:
    text = clean(line)
    if not text:
        return ""
    match = re.search(r"saved to\s+(.+)$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"Saving .* weights to\s+(.+?)(?:\.\.\.)?$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def log_record(row: Dict[str, str]) -> Dict[str, Any]:
    method = normalize_method(row.get("method_hint"))
    dataset = normalize_dataset(row.get("dataset_hint"))
    return {
        "record_type": "log",
        "source_path": row.get("path", ""),
        "source_artifact_id": row.get("artifact_id", ""),
        "source_sha256": row.get("sha256", ""),
        "source_group": Path(row.get("path", "")).parts[1] if len(Path(row.get("path", "")).parts) > 1 else "",
        "method": method,
        "dataset": dataset,
        "shot": as_int(row.get("shot_hint")),
        "seed": row.get("seed_hint") or "unknown",
        "rank": as_int(row.get("rank_hint")),
        "heads": normalized_head_hint(row),
        "lambda_o": as_float(row.get("lambda_hint")),
        "loss": row.get("loss_hint", ""),
        "accuracy": as_float(row.get("log_final_accuracy")),
        "zero_shot_accuracy": as_float(row.get("log_zero_shot_accuracy")),
        "checkpoint_path_reported": parse_saved_checkpoint(row.get("log_saved_checkpoint_line", "")),
        "checkpoint_sha256": "",
        "log_line_count": as_int(row.get("log_line_count")),
        "status": "recovered_log_final_accuracy" if as_float(row.get("log_final_accuracy")) is not None else "recovered_log_no_final_accuracy",
        "verification_note": "seed/split not verified from recovered log",
    }


def csv_summary_records(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    log_by_base: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("extension") == ".log":
            log_by_base[basename(row.get("path", ""))].append(row)

    out: List[Dict[str, Any]] = []
    for artifact in rows:
        if artifact.get("extension") != ".csv":
            continue
        path = ROOT / artifact["path"]
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, src in enumerate(reader, start=2):
                log_file = clean(src.get("log_file"))
                matched = log_by_base.get(basename(log_file), [])
                log_match = matched[0] if matched else {}
                method = normalize_method(src.get("adapter") or artifact.get("method_hint"))
                dataset = normalize_dataset(src.get("dataset") or artifact.get("dataset_hint"))
                record = {
                    "record_type": "csv_summary",
                    "source_path": artifact.get("path", ""),
                    "source_artifact_id": artifact.get("artifact_id", ""),
                    "source_sha256": artifact.get("sha256", ""),
                    "source_group": Path(artifact.get("path", "")).parts[1] if len(Path(artifact.get("path", "")).parts) > 1 else "",
                    "csv_row": idx,
                    "method": method,
                    "dataset": dataset,
                    "shot": as_int(src.get("shots")),
                    "seed": "unknown",
                    "rank": as_int(src.get("rank")),
                    "heads": as_int(src.get("heads")),
                    "lambda_o": as_float(src.get("lambda_o")),
                    "loss": clean(src.get("loss")),
                    "accuracy": as_float(src.get("accuracy")),
                    "zero_shot_accuracy": "",
                    "log_file": log_file,
                    "matched_log_path": log_match.get("path", ""),
                    "matched_log_artifact_id": log_match.get("artifact_id", ""),
                    "matched_log_accuracy": as_float(log_match.get("log_final_accuracy")) if log_match else "",
                    "checkpoint_path_reported": clean(src.get("checkpoint_path")),
                    "checkpoint_sha256": "",
                    "params": clean(src.get("params")),
                    "time_seconds": as_float(src.get("time(s)")),
                    "status": "csv_summary_with_log_match" if log_match else "csv_summary_no_log_match",
                    "verification_note": "summary row; seed/split not verified",
                }
                out.append(record)
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, int], List[Dict[str, str]]]:
    out: Dict[Tuple[str, str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("extension") != ".pt":
            continue
        method = normalize_method(row.get("method_hint"))
        dataset = normalize_dataset(row.get("dataset_hint"))
        shot = as_int(row.get("shot_hint"))
        if method and dataset and shot:
            out[(method, dataset, shot)].append(row)
    return out


def is_strict_oh_tier_a(record: Dict[str, Any]) -> bool:
    if record.get("method") != "ohsinglora":
        return False
    return record.get("rank") == 2 and record.get("heads") == 2


def generate_missing_matrix(records: List[Dict[str, Any]], manifest_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    logs = [r for r in records if r.get("record_type") == "log" and r.get("accuracy") is not None]
    csvs = [r for r in records if r.get("record_type") == "csv_summary" and r.get("accuracy") is not None]
    ckpt_idx = checkpoint_index(manifest_rows)

    out: List[Dict[str, Any]] = []
    for dataset in TIER_A_DATASETS:
        for shot in TIER_A_SHOTS:
            for method in TIER_A_METHODS:
                relevant_logs = [r for r in logs if r.get("dataset") == dataset and r.get("shot") == shot and r.get("method") == method]
                relevant_csvs = [r for r in csvs if r.get("dataset") == dataset and r.get("shot") == shot and r.get("method") == method]
                strict_logs = [r for r in relevant_logs if is_strict_oh_tier_a(r)]
                strict_csvs = [r for r in relevant_csvs if is_strict_oh_tier_a(r)]
                ckpts = ckpt_idx.get((method, dataset, shot), [])
                ckpt_unique = len({r.get("sha256", "") for r in ckpts if r.get("sha256")})

                if method == "ohsinglora":
                    if strict_logs or strict_csvs:
                        status = "strict_r2_h2_evidence_available"
                    elif relevant_logs or relevant_csvs:
                        status = "evidence_available_config_unclear_or_nonstandard"
                    elif ckpts:
                        status = "checkpoint_available_no_log"
                    else:
                        status = "missing_recover_or_rerun"
                else:
                    if relevant_logs or relevant_csvs:
                        status = "log_or_csv_evidence_available"
                    elif ckpts:
                        status = "checkpoint_available_no_log"
                    else:
                        status = "missing_recover_or_rerun"

                out.append(
                    {
                        "dataset": dataset,
                        "shot": shot,
                        "method": method,
                        "log_count": len(relevant_logs),
                        "csv_summary_count": len(relevant_csvs),
                        "strict_oh_r2_h2_log_count": len(strict_logs),
                        "strict_oh_r2_h2_csv_count": len(strict_csvs),
                        "checkpoint_path_count": len(ckpts),
                        "checkpoint_unique_sha_count": ckpt_unique,
                        "seed_status": "seed_unknown_or_seed1_only; seed2_seed3_required",
                        "split_status": "split_hash_missing",
                        "status": status,
                    }
                )
    return out


def xlsx_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: List[str] = []
    for item in root.findall("x:si", ns):
        parts = [node.text or "" for node in item.findall(".//x:t", ns)]
        strings.append("".join(parts))
    return strings


def xlsx_sheet_targets(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    ns_main = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_by_id = {
        rel.attrib["Id"]: rel.attrib["Target"].lstrip("/")
        for rel in rels.findall("r:Relationship", ns_rel)
    }
    out: List[Tuple[str, str]] = []
    for sheet in wb.findall(".//x:sheet", ns_main):
        name = sheet.attrib.get("name", "")
        rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
        target = rel_by_id.get(rel_id, "")
        if target and not target.startswith("xl/"):
            target = "xl/" + target
        if name and target:
            out.append((name, target))
    return out


def xlsx_col_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def xlsx_cell_value(cell: ET.Element, shared: List[str]) -> Any:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        parts = [node.text or "" for node in cell.findall(".//x:t", ns)]
        return "".join(parts)
    value_node = cell.find("x:v", ns)
    if value_node is None or value_node.text is None:
        return ""
    text = value_node.text
    if cell_type == "s":
        idx = int(text)
        return shared[idx] if 0 <= idx < len(shared) else ""
    if cell_type == "str":
        return text
    number = as_float(text)
    if number is not None:
        if math.isclose(number, round(number)):
            return int(round(number))
        return number
    return text


def load_workbook_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with zipfile.ZipFile(WORKBOOK_PATH) as zf:
        shared = xlsx_shared_strings(zf)
        sheets = xlsx_sheet_targets(zf)
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for sheet, target in sheets:
            root = ET.fromstring(zf.read(target))
            materialized_rows: List[List[Any]] = []
            for row_node in root.findall(".//x:sheetData/x:row", ns):
                values: List[Any] = []
                for cell in row_node.findall("x:c", ns):
                    ref = cell.attrib.get("r", "")
                    col = xlsx_col_index(ref)
                    while len(values) <= col:
                        values.append("")
                    values[col] = xlsx_cell_value(cell, shared)
                materialized_rows.append(values)

            header: Optional[List[str]] = None
            for idx, values in enumerate(materialized_rows, start=1):
                cleaned = [clean(v) for v in values]
                if header is None:
                    if any(cleaned):
                        header = [c.lower().replace(" ", "_") for c in cleaned]
                    continue
                if not any(cleaned):
                    continue
                record = {header[i]: values[i] if i < len(values) else "" for i in range(len(header))}
                record["_sheet"] = sheet
                record["_row"] = idx
                rows.append(record)
    return rows


def workbook_crosscheck(manifest_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    log_rows = [r for r in manifest_rows if r.get("extension") == ".log"]
    logs_by_base: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    logs_by_signature: Dict[Tuple[str, str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in log_rows:
        logs_by_base[basename(row.get("path", ""))].append(row)
        method = normalize_method(row.get("method_hint"))
        dataset = normalize_dataset(row.get("dataset_hint"))
        shot = as_int(row.get("shot_hint"))
        if method and dataset and shot:
            logs_by_signature[(method, dataset, shot)].append(row)

    out: List[Dict[str, Any]] = []
    def append_match(
        *,
        sheet: str,
        workbook_row: int,
        dataset: str,
        shot: Optional[int],
        method: str,
        rank: Optional[int],
        heads: Optional[int],
        lambda_o: Optional[float],
        loss: str,
        accuracy: float,
        log_file: str,
        source_layout: str,
    ) -> None:
        matched_logs = logs_by_base.get(basename(log_file), []) if log_file else []
        match_mode = "log_file"

        if not matched_logs and method and dataset and shot:
            matched_logs = logs_by_signature.get((method, dataset, shot), [])
            match_mode = "method_dataset_shot"

        if matched_logs and (rank is not None or heads is not None):
            config_filtered = []
            for log in matched_logs:
                log_rank = as_int(log.get("rank_hint"))
                log_heads = normalized_head_hint(log)
                rank_ok = rank is None or log_rank == rank
                heads_ok = heads is None or log_heads == heads
                if rank_ok and heads_ok:
                    config_filtered.append(log)
            if config_filtered:
                matched_logs = config_filtered
                match_mode += "_config_filtered"

        best_log: Dict[str, str] = {}
        best_delta: Optional[float] = None
        for log in matched_logs:
            log_acc = as_float(log.get("log_final_accuracy"))
            if log_acc is None:
                continue
            delta = abs(log_acc - accuracy)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_log = log

        if not best_log:
            status = "no_matching_log"
        elif best_delta is not None and best_delta <= 0.01:
            status = "accuracy_match"
        else:
            status = "accuracy_mismatch"

        out.append(
            {
                "sheet": sheet,
                "workbook_row": workbook_row,
                "source_layout": source_layout,
                "dataset": dataset,
                "shot": shot,
                "method": method,
                "rank": rank,
                "heads": heads,
                "lambda_o": lambda_o,
                "loss": loss,
                "workbook_accuracy": accuracy,
                "workbook_log_file": log_file,
                "match_mode": match_mode if matched_logs else "",
                "candidate_log_count": len(matched_logs),
                "matched_log_path": best_log.get("path", ""),
                "matched_log_artifact_id": best_log.get("artifact_id", ""),
                "matched_log_accuracy": as_float(best_log.get("log_final_accuracy")) if best_log else "",
                "absolute_delta": best_delta if best_delta is not None else "",
                "status": status,
            }
        )

    for row in load_workbook_rows():
        keys = set(row)

        # Long layout: one row per run with explicit dataset/accuracy columns.
        if "accuracy" in keys and ("dataset" in keys or "log_file" in keys):
            accuracy = as_float(row.get("accuracy"))
            if accuracy is not None:
                raw_method = row.get("adapter") or row.get("method") or row.get("loss") or ""
                append_match(
                    sheet=row.get("_sheet"),
                    workbook_row=row.get("_row"),
                    dataset=normalize_dataset(row.get("dataset")),
                    shot=as_int(row.get("shots") or row.get("shot")),
                    method=normalize_method(raw_method) or normalize_workbook_method(raw_method),
                    rank=as_int(row.get("rank")),
                    heads=as_int(row.get("heads") or row.get("head")),
                    lambda_o=as_float(row.get("lambda_o")),
                    loss=clean(row.get("loss")),
                    accuracy=accuracy,
                    log_file=clean(row.get("log_file")),
                    source_layout="long",
                )

        # Wide layout: one row per method, dataset accuracies as columns.
        wide_method = normalize_workbook_method(row.get("method") or row.get("1_shots"))
        wide_shot = as_int(row.get("shots"))
        if wide_method and wide_shot in TIER_A_SHOTS:
            raw_method = row.get("method") or row.get("1_shots")
            rank = parse_rank_from_text(raw_method)
            heads = parse_heads_from_text(raw_method)
            for column, dataset in WORKBOOK_DATASET_COLUMNS.items():
                if column not in keys:
                    continue
                accuracy = as_float(row.get(column))
                if accuracy is None:
                    continue
                append_match(
                    sheet=row.get("_sheet"),
                    workbook_row=row.get("_row"),
                    dataset=dataset,
                    shot=wide_shot,
                    method=wide_method,
                    rank=rank,
                    heads=heads,
                    lambda_o=as_float(row.get("lambda_o")),
                    loss=clean(row.get("loss")),
                    accuracy=accuracy,
                    log_file="",
                    source_layout="wide",
                )
    return out


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(lines)


def write_missing_md(rows: List[Dict[str, Any]]) -> None:
    status_counts = Counter(row["status"] for row in rows)
    missing = [r for r in rows if r["status"] == "missing_recover_or_rerun"]
    unclear = [r for r in rows if r["status"] == "evidence_available_config_unclear_or_nonstandard"]
    lines = [
        "# Missing Tier-A Matrix",
        "",
        "Generated from `phase0_raw_artifact_manifest.csv`.",
        "",
        "## Status Counts",
        "",
        markdown_table(["status", "count"], [[k, v] for k, v in sorted(status_counts.items())]),
        "",
        "## Missing Or Rerun Required",
        "",
        markdown_table(
            ["dataset", "shot", "method", "status"],
            [[r["dataset"], r["shot"], r["method"], r["status"]] for r in missing],
        )
        if missing
        else "No dataset/shot/method cell is completely missing recovered log/CSV/checkpoint evidence.",
        "",
        "## OrthoAdapt Evidence With Unclear/Nonstandard Config",
        "",
        markdown_table(
            ["dataset", "shot", "log_count", "csv_count", "checkpoint_paths"],
            [[r["dataset"], r["shot"], r["log_count"], r["csv_summary_count"], r["checkpoint_path_count"]] for r in unclear],
        )
        if unclear
        else "No OrthoAdapt cells have only unclear/nonstandard config evidence.",
        "",
        "## Interpretation",
        "",
        "- `strict_r2_h2_evidence_available` means a recovered log or CSV row explicitly carries `rank=2` and `heads=2`.",
        "- `log_or_csv_evidence_available` means baseline evidence exists, but seed/split hashes remain missing.",
        "- All cells still require seed2/seed3 recovery or rerun before Tier-A mean/std claims are defensible.",
    ]
    OUT_MISSING_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_crosscheck_md(rows: List[Dict[str, Any]]) -> None:
    status_counts = Counter(row["status"] for row in rows)
    mismatches = [r for r in rows if r["status"] == "accuracy_mismatch"]
    no_match = [r for r in rows if r["status"] == "no_matching_log"]
    lines = [
        "# Workbook vs Logs Cross-Check",
        "",
        "Generated by matching workbook rows to recovered logs by `log_file` where possible, otherwise by method/dataset/shot.",
        "",
        "## Status Counts",
        "",
        markdown_table(["status", "count"], [[k, v] for k, v in sorted(status_counts.items())]),
        "",
        "## Accuracy Mismatches",
        "",
        "Rows from wide workbook sheets usually do not carry `log_file`, so these are nearest method/dataset/shot matches and should be treated as reconciliation targets rather than definitive contradictions.",
        "",
        markdown_table(
            ["sheet", "row", "dataset", "shot", "method", "workbook", "log", "delta", "log_path"],
            [
                [
                    r["sheet"],
                    r["workbook_row"],
                    r["dataset"],
                    r["shot"],
                    r["method"],
                    r["workbook_accuracy"],
                    r["matched_log_accuracy"],
                    r["absolute_delta"],
                    r["matched_log_path"],
                ]
                for r in mismatches[:50]
            ],
        )
        if mismatches
        else "No mismatches above tolerance were found among matched rows.",
        "",
        "## Rows Without Matching Log",
        "",
        markdown_table(
            ["sheet", "row", "dataset", "shot", "method", "accuracy", "log_file"],
            [
                [
                    r["sheet"],
                    r["workbook_row"],
                    r["dataset"],
                    r["shot"],
                    r["method"],
                    r["workbook_accuracy"],
                    r["workbook_log_file"],
                ]
                for r in no_match[:80]
            ],
        )
        if no_match
        else "Every workbook row with an accuracy found at least one candidate recovered log.",
    ]
    OUT_XCHECK_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(records: List[Dict[str, Any]], missing: List[Dict[str, Any]], xcheck: List[Dict[str, Any]], manifest_rows: List[Dict[str, str]]) -> None:
    manifest_counts = Counter(row.get("extension") for row in manifest_rows)
    log_records = [r for r in records if r.get("record_type") == "log"]
    csv_records = [r for r in records if r.get("record_type") == "csv_summary"]
    strict_cells = [r for r in missing if r["status"] == "strict_r2_h2_evidence_available"]
    missing_cells = [r for r in missing if r["status"] == "missing_recover_or_rerun"]
    lines = [
        "# Phase 0 Normalization Summary",
        "",
        "## Inputs",
        "",
        markdown_table(["artifact type", "count"], [[k or "(blank)", v] for k, v in sorted(manifest_counts.items())]),
        "",
        "## Generated Outputs",
        "",
        markdown_table(
            ["file", "purpose"],
            [
                [rel_path(OUT_JSONL), "Canonical log/CSV evidence manifest in JSONL"],
                [rel_path(OUT_CSV), "Same manifest in CSV"],
                [rel_path(OUT_MISSING_CSV), "Tier-A coverage matrix"],
                [rel_path(OUT_MISSING_MD), "Human-readable Tier-A coverage report"],
                [rel_path(OUT_XCHECK_CSV), "Workbook rows matched against recovered logs"],
                [rel_path(OUT_XCHECK_MD), "Human-readable workbook/log cross-check"],
                [rel_path(OUT_SUMMARY_MD), "This normalization summary"],
            ],
        ),
        "",
        "## Key Counts",
        "",
        markdown_table(
            ["metric", "count"],
            [
                ["canonical log records", len(log_records)],
                ["canonical CSV summary records", len(csv_records)],
                ["Tier-A cells with strict OrthoAdapt R2/H2 evidence", len(strict_cells)],
                ["Tier-A cells completely missing recovered evidence", len(missing_cells)],
                ["workbook cross-check rows", len(xcheck)],
                ["workbook rows with accuracy match", sum(1 for r in xcheck if r["status"] == "accuracy_match")],
                ["workbook rows with accuracy mismatch", sum(1 for r in xcheck if r["status"] == "accuracy_mismatch")],
                ["workbook rows without matching log", sum(1 for r in xcheck if r["status"] == "no_matching_log")],
            ],
        ),
        "",
        "## Gate Interpretation",
        "",
        "Recovered logs and CSVs materially improve provenance, but they do not by themselves satisfy the revised Tier-A protocol because seed identifiers, split hashes, validation-only selection evidence, and seed2/seed3 coverage remain unresolved.",
        "",
        "Recommended next step: inspect `missing_tier_a_matrix.md` and rerun/recover only the cells that remain missing or have unclear/nonstandard configuration evidence.",
    ]
    OUT_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_manifest()
    records = [log_record(row) for row in manifest_rows if row.get("extension") == ".log"]
    records.extend(csv_summary_records(manifest_rows))
    write_jsonl(OUT_JSONL, records)
    write_csv(OUT_CSV, records)

    missing = generate_missing_matrix(records, manifest_rows)
    write_csv(OUT_MISSING_CSV, missing)
    write_missing_md(missing)

    xcheck = workbook_crosscheck(manifest_rows)
    write_csv(OUT_XCHECK_CSV, xcheck)
    write_crosscheck_md(xcheck)

    write_summary_md(records, missing, xcheck, manifest_rows)

    print(f"Wrote {len(records)} canonical records")
    print(f"Wrote {len(missing)} Tier-A matrix rows")
    print(f"Wrote {len(xcheck)} workbook cross-check rows")
    print(f"Summary: {rel_path(OUT_SUMMARY_MD)}")


if __name__ == "__main__":
    main()
