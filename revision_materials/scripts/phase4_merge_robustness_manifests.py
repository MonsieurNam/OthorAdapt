#!/usr/bin/env python3
"""Merge Phase 4 robustness manifests safely (append-union, never overwrite).

Usage:
    python revision_materials/scripts/phase4_merge_robustness_manifests.py \
        --into revision_materials/results/phase4_robustness_manifest.jsonl \
        --from-file recovered_manifest_1.jsonl [--from-file more.jsonl ...] \
        [--dry-run]

Rules:
- Rows are keyed by (job_id, severity). First occurrence wins; later duplicates
  are dropped with a report line (values are compared; a conflicting duplicate
  aborts unless --prefer-existing/--prefer-incoming is given).
- Only rows with schema_version == "phase4.robustness.v1" are considered.
- A timestamped backup of --into is written next to it before any write.
- Prints a coverage report against the expected 576-row grid
  (8 datasets x {1,4,16} shots x seeds {1,2,3} x {lora,ohsinglora} x severities 0-3)
  and the list of jobs still missing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

SCHEMA = "phase4.robustness.v1"
DATASETS = ["fgvc", "eurosat", "food101", "oxford_pets", "oxford_flowers", "caltech101", "dtd", "ucf101"]
SHOTS = [1, 4, 16]
SEEDS = [1, 2, 3]
METHODS = ["lora", "ohsinglora"]
SEVERITIES = [0, 1, 2, 3]


def load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"warning: {path.name}:{line_no} malformed JSON, skipped")
            continue
        if row.get("schema_version") != SCHEMA:
            continue
        rows.append(row)
    return rows


def key_of(row: dict) -> tuple:
    return (str(row.get("job_id")), int(row.get("severity", -1)))


def accuracy_of(row: dict):
    metrics = row.get("metrics") or {}
    return metrics.get("accuracy")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--into", required=True, help="Canonical manifest to merge into (backed up first).")
    ap.add_argument("--from-file", action="append", default=[], help="Manifest(s) to merge from. Repeatable.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    ap.add_argument("--prefer-existing", action="store_true", help="On conflicting duplicate keys, keep the row already in --into.")
    ap.add_argument("--prefer-incoming", action="store_true", help="On conflicting duplicate keys, take the incoming row.")
    args = ap.parse_args()

    into = Path(args.into)
    base_rows = load_rows(into)
    merged: dict[tuple, dict] = {}
    dup_same = 0
    conflicts: list[tuple] = []

    for row in base_rows:
        k = key_of(row)
        if k in merged:
            dup_same += 1  # in-file duplicate; keep first
            continue
        merged[k] = row

    added = 0
    for src in args.from_file:
        src_path = Path(src)
        src_rows = load_rows(src_path)
        print(f"loaded {len(src_rows)} schema rows from {src_path}")
        for row in src_rows:
            k = key_of(row)
            if k not in merged:
                merged[k] = row
                added += 1
                continue
            # duplicate key: compare accuracy
            a_old, a_new = accuracy_of(merged[k]), accuracy_of(row)
            if a_old == a_new:
                dup_same += 1
            else:
                conflicts.append((k, a_old, a_new))
                if args.prefer_incoming:
                    merged[k] = row

    if conflicts and not (args.prefer_existing or args.prefer_incoming):
        for k, a_old, a_new in conflicts[:20]:
            print(f"CONFLICT {k}: existing accuracy={a_old} incoming accuracy={a_new}")
        raise SystemExit(
            f"aborting: {len(conflicts)} conflicting duplicate keys. "
            "Re-run with --prefer-existing or --prefer-incoming after inspecting."
        )

    # Coverage report
    completed_jobs: dict[str, set[int]] = defaultdict(set)
    for (job_id, sev), row in merged.items():
        if row.get("status") == "completed":
            completed_jobs[job_id].add(sev)
    full_jobs = {j for j, s in completed_jobs.items() if s >= set(SEVERITIES)}

    expected_jobs = []
    for ds in DATASETS:
        for shot in SHOTS:
            for seed in SEEDS:
                expected_jobs.append((f"{ds}_{shot}shot_seed{seed}_test_lora_r8", "lora"))
                expected_jobs.append((f"{ds}_{shot}shot_seed{seed}_test_ohsinglora_h2_r8_lo0p03_ramp100", "ohsinglora"))

    missing = [(j, m) for j, m in expected_jobs if j not in full_jobs]
    print("---- merge summary ----")
    print(f"rows merged total : {len(merged)} (target 576)")
    print(f"rows added        : {added}")
    print(f"identical dups    : {dup_same}")
    print(f"conflicts         : {len(conflicts)}")
    print(f"complete jobs     : {len(full_jobs)} / {len(expected_jobs)}")
    by_method = defaultdict(int)
    for j, m in missing:
        by_method[m] += 1
    print(f"missing jobs      : {len(missing)} (lora={by_method['lora']}, ohsinglora={by_method['ohsinglora']})")
    for j, m in sorted(missing):
        print(f"  MISSING {m:12s} {j}")

    if args.dry_run:
        print("dry-run: nothing written")
        return

    if into.exists():
        backup = into.with_suffix(f".backup_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        shutil.copy2(into, backup)
        print(f"backup written: {backup}")

    tmp = into.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for k in sorted(merged.keys()):
            f.write(json.dumps(merged[k], sort_keys=True) + "\n")
    tmp.replace(into)
    print(f"wrote {len(merged)} rows to {into}")


if __name__ == "__main__":
    main()
