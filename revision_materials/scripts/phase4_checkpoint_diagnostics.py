"""Compute Phase 4 head-overlap and orthogonality diagnostics from checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecr3_provenance import file_sha256

DEFAULT_MANIFEST = ROOT / "revision_materials" / "results" / "phase3_main_ramp100_results.jsonl"
OUT_JSONL = ROOT / "revision_materials" / "results" / "phase4_checkpoint_diagnostics.jsonl"
OUT_HEAD_MD = ROOT / "revision_materials" / "results" / "phase4_head_overlap_report.md"
OUT_ORTHO_MD = ROOT / "revision_materials" / "results" / "phase4_orthogonality_report.md"


def load_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for checkpoint diagnostics") from exc
    return torch


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def tensor_metrics(tensor) -> dict:
    torch = load_torch()
    A = tensor.detach().cpu().float()
    H = int(A.shape[0])
    head_norms = [float(torch.linalg.norm(A[i]).item()) for i in range(H)]
    raw_losses = []
    normalized_overlaps = []
    for i in range(H):
        for j in range(i + 1, H):
            product = A[i].T @ A[j]
            raw = float(torch.linalg.norm(product, ord="fro").pow(2).item())
            raw_losses.append(raw)
            qi = torch.linalg.qr(A[i], mode="reduced").Q
            qj = torch.linalg.qr(A[j], mode="reduced").Q
            overlap = torch.linalg.norm(qi.T @ qj, ord="fro").pow(2)
            normalized_overlaps.append(float((overlap / max(1, A.shape[-1])).item()))
    return {
        "head_norm_min": min(head_norms),
        "head_norm_mean": mean(head_norms),
        "head_norm_max": max(head_norms),
        "raw_ortho_sum": sum(raw_losses),
        "raw_ortho_mean": mean(raw_losses),
        "subspace_overlap_mean": mean(normalized_overlaps),
        "subspace_overlap_max": max(normalized_overlaps) if normalized_overlaps else 0.0,
        "num_head_pairs": len(raw_losses),
    }


def checkpoint_record(row: dict) -> dict:
    torch = load_torch()
    cfg = row["config"]
    checkpoint = row["checkpoint"]["path"]
    path = ROOT / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    weights = ckpt.get("weights", ckpt)
    lora_tensors = {key: value for key, value in weights.items() if key.endswith("lora_A_heads")}
    if not lora_tensors:
        raise ValueError(f"No lora_A_heads tensors found in {path}")

    tensor_rows = []
    for key, tensor in sorted(lora_tensors.items()):
        item = tensor_metrics(tensor)
        parts = key.split(".")
        item.update(
            {
                "key": key,
                "encoder": "vision" if key.startswith("visual.") else "text",
                "layer": int(parts[3] if key.startswith("visual.") else parts[2]),
                "param": parts[-2],
            }
        )
        tensor_rows.append(item)

    return {
        "schema_version": "phase4.checkpoint_diagnostics.v1",
        "dataset": cfg["dataset"],
        "shot": int(cfg["shots"]),
        "seed": int(cfg["seed"]),
        "method": cfg["adapter"],
        "num_heads": int(cfg["num_heads"]),
        "rank": int(cfg["r"]),
        "lambda_o": float(cfg["lambda_o"]),
        "ramp_up_steps": int(cfg["ramp_up_steps"]),
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": file_sha256(path),
        "manifest_checkpoint_sha256": row["checkpoint"]["sha256"],
        "tensor_count": len(tensor_rows),
        "head_norm_mean": mean([item["head_norm_mean"] for item in tensor_rows]),
        "head_norm_min": min(item["head_norm_min"] for item in tensor_rows),
        "head_norm_max": max(item["head_norm_max"] for item in tensor_rows),
        "raw_ortho_sum_total": sum(item["raw_ortho_sum"] for item in tensor_rows),
        "raw_ortho_mean_per_tensor": mean([item["raw_ortho_mean"] for item in tensor_rows]),
        "subspace_overlap_mean": mean([item["subspace_overlap_mean"] for item in tensor_rows]),
        "subspace_overlap_max": max(item["subspace_overlap_max"] for item in tensor_rows),
        "tensors": tensor_rows,
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def fmt(value: float) -> str:
    return f"{value:.6g}"


def grouped(records: list[dict], fields: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out = defaultdict(list)
    for record in records:
        out[tuple(record[field] for field in fields)].append(record)
    return out


def write_reports(records: list[dict]) -> None:
    overall_overlap = mean([record["subspace_overlap_mean"] for record in records])
    overall_raw = mean([record["raw_ortho_mean_per_tensor"] for record in records])
    lines = [
        "# Phase 4 Head Overlap Report",
        "",
        f"Source: `{OUT_JSONL.relative_to(ROOT).as_posix()}`.",
        "",
        f"- Completed checkpoints: {len(records)}",
        f"- Mean subspace overlap across checkpoints: {fmt(overall_overlap)}",
        f"- Mean raw pair orthogonality loss per tensor: {fmt(overall_raw)}",
        "",
        "## By Shot",
        "",
        "| Shot | n | Mean subspace overlap | Std | Mean head norm |",
        "|---:|---:|---:|---:|---:|",
    ]
    for (shot,), group_rows in sorted(grouped(records, ("shot",)).items()):
        overlaps = [row["subspace_overlap_mean"] for row in group_rows]
        norms = [row["head_norm_mean"] for row in group_rows]
        lines.append(f"| {shot} | {len(group_rows)} | {fmt(mean(overlaps))} | {fmt(sample_std(overlaps))} | {fmt(mean(norms))} |")
    lines += [
        "",
        "## Claim Gate",
        "",
        "- This report can support descriptive statements about learned head overlap.",
        "- It does not by itself prove that head overlap causes accuracy or robustness changes.",
    ]
    OUT_HEAD_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "# Phase 4 Orthogonality Report",
        "",
        f"Source: `{OUT_JSONL.relative_to(ROOT).as_posix()}`.",
        "",
        "The diagnostics below are computed directly from Phase 3 ramp100 OrthoAdapt checkpoints. They refresh the earlier orthogonality audit for the final selected configuration.",
        "",
        "| Dataset | Shot | n | Raw ortho mean/tensor | Subspace overlap mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for (dataset, shot), group_rows in sorted(grouped(records, ("dataset", "shot")).items()):
        lines.append(
            f"| {dataset} | {shot} | {len(group_rows)} | "
            f"{fmt(mean([row['raw_ortho_mean_per_tensor'] for row in group_rows]))} | "
            f"{fmt(mean([row['subspace_overlap_mean'] for row in group_rows]))} |"
        )
    lines += [
        "",
        "## Claim Gate",
        "",
        "- Keep: the final OrthoAdapt checkpoints contain measurable multi-head adapter tensors and can be audited for overlap.",
        "- Weaken: orthogonality should be described as configuration-dependent unless paired ablations show a consistent accuracy or robustness effect.",
    ]
    OUT_ORTHO_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out_jsonl", default=str(OUT_JSONL))
    args = parser.parse_args(argv)

    rows = load_manifest(Path(args.manifest))
    oh_rows = [row for row in rows if row.get("config", {}).get("adapter") == "ohsinglora"]
    if len(oh_rows) != 72:
        raise SystemExit(f"Expected 72 OrthoAdapt rows, found {len(oh_rows)}")
    records = [checkpoint_record(row) for row in oh_rows]
    bad_hashes = [r for r in records if r["checkpoint_sha256"] != r["manifest_checkpoint_sha256"]]
    if bad_hashes:
        raise SystemExit(f"Checkpoint hash mismatch: {bad_hashes[0]['checkpoint_path']}")
    out_path = Path(args.out_jsonl)
    write_jsonl(records, out_path)
    write_reports(records)
    print(out_path.relative_to(ROOT).as_posix())
    print(OUT_HEAD_MD.relative_to(ROOT).as_posix())
    print(OUT_ORTHO_MD.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
