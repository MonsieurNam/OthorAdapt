"""Run paired Phase 4 spectrum diagnostics from Phase 3 checkpoints."""

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

RESULTS = ROOT / "revision_materials" / "results"
DEFAULT_MANIFEST = RESULTS / "phase3_main_ramp100_results.jsonl"
OUT_JSONL = RESULTS / "phase4_spectrum_manifest.jsonl"
OUT_REPORT = RESULTS / "phase4_spectrum_report.md"
OUT_FIG_DIR = RESULTS / "figures" / "phase4_spectrum"
DATASET_ORDER = ["fgvc", "eurosat", "food101", "oxford_pets", "oxford_flowers", "caltech101", "dtd", "ucf101"]
PARAMS = ["q_proj", "v_proj"]


def load_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for spectrum diagnostics") from exc
    return torch


def load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("NumPy is required for spectrum diagnostics") from exc
    return np


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


def load_checkpoint(path: Path) -> dict:
    torch = load_torch()
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "metadata" not in ckpt:
        raise ValueError(f"Checkpoint metadata missing: {path}")
    return ckpt


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def lora_delta(ckpt: dict, vision_layer_idx: int, param: str):
    torch = load_torch()
    weights = ckpt.get("weights", ckpt)
    encoder = ckpt.get("metadata", {}).get("encoder")
    layer_idx = vision_layer_idx + 12 if encoder == "both" else vision_layer_idx
    layer_key = f"layer_{layer_idx}"
    if layer_key not in weights:
        raise ValueError(f"LoRA missing {layer_key}")
    item = weights[layer_key][param]
    return torch.matmul(item["w_lora_B"].float(), item["w_lora_A"].float())


def oh_delta(ckpt: dict, vision_layer_idx: int, param: str):
    torch = load_torch()
    weights = ckpt.get("weights", ckpt)
    key = f"visual.transformer.resblocks.{vision_layer_idx}.attn.{param}.lora_A_heads"
    if key not in weights:
        raise ValueError(f"OH checkpoint missing {key}")
    A = weights[key].detach().cpu().float()
    out_dim = A.shape[1]
    delta = torch.zeros(out_dim, out_dim)
    for head_idx in range(A.shape[0]):
        delta += A[head_idx] @ A[head_idx].T
    return delta, key


def spectrum(matrix, top_k: int) -> dict:
    torch = load_torch()
    s = torch.linalg.svdvals(matrix.float()).detach().cpu()
    if len(s) == 0:
        raise ValueError("empty singular spectrum")
    normalized = s / s[0].clamp_min(1e-12)
    energy = s.pow(2)
    stable_rank = float((energy.sum() / energy.max().clamp_min(1e-12)).item())
    probs = energy / energy.sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
    cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-12)
    rank90 = int((cumulative >= 0.90).nonzero()[0].item() + 1)
    return {
        "top_normalized_singular_values": [float(v) for v in normalized[:top_k]],
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "rank90_energy": rank90,
    }


def write_plot(record: dict, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    np = load_numpy()
    lora = np.asarray(record["lora"]["top_normalized_singular_values"])
    oh = np.asarray(record["ohsinglora"]["top_normalized_singular_values"])
    n = min(len(lora), len(oh))
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(n), lora[:n], "o--", label="CLIP-LoRA r=8", linewidth=1.5, markersize=3)
    plt.plot(range(n), oh[:n], "s-", label="OrthoAdapt H=2,r=8", linewidth=1.5, markersize=3)
    plt.yscale("log")
    plt.xlabel("Singular value index")
    plt.ylabel("Normalized singular value")
    plt.title(f"{record['dataset']} shot={record['shot']} seed={record['seed']} {record['param']}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_records(rows: list[dict], shot: int, seed: int, layer_idx: int, top_k: int) -> list[dict]:
    by_key = {}
    for row in rows:
        cfg = row.get("config") or {}
        by_key[(cfg.get("dataset"), int(cfg.get("shots")), int(cfg.get("seed")), cfg.get("adapter"))] = row

    records = []
    for dataset in DATASET_ORDER:
        lora_row = by_key[(dataset, shot, seed, "lora")]
        oh_row = by_key[(dataset, shot, seed, "ohsinglora")]
        lora_path = resolve_path(lora_row["checkpoint"]["path"])
        oh_path = resolve_path(oh_row["checkpoint"]["path"])
        lora_ckpt = load_checkpoint(lora_path)
        oh_ckpt = load_checkpoint(oh_path)
        for param in PARAMS:
            lora_spec = spectrum(lora_delta(lora_ckpt, layer_idx, param), top_k)
            oh_matrix, oh_key = oh_delta(oh_ckpt, layer_idx, param)
            oh_spec = spectrum(oh_matrix, top_k)
            record = {
                "schema_version": "phase4.spectrum.v1",
                "dataset": dataset,
                "shot": shot,
                "seed": seed,
                "encoder_scope": "vision",
                "vision_layer_idx": layer_idx,
                "param": param,
                "lora": {
                    "checkpoint_path": lora_row["checkpoint"]["path"],
                    "checkpoint_sha256": file_sha256(lora_path),
                    **lora_spec,
                },
                "ohsinglora": {
                    "checkpoint_path": oh_row["checkpoint"]["path"],
                    "checkpoint_sha256": file_sha256(oh_path),
                    "matrix_key": oh_key,
                    **oh_spec,
                },
            }
            fig = OUT_FIG_DIR / f"{dataset}_{shot}shot_seed{seed}_layer{layer_idx}_{param}.png"
            write_plot(record, fig)
            record["figure_path"] = fig.relative_to(ROOT).as_posix() if fig.exists() else ""
            records.append(record)
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def write_report(records: list[dict]) -> None:
    by_param = defaultdict(list)
    for record in records:
        by_param[record["param"]].append(record)
    lines = [
        "# Phase 4 Spectrum Report",
        "",
        f"Source: `{OUT_JSONL.relative_to(ROOT).as_posix()}`.",
        "",
        "Paired spectrum diagnostics use Phase 3 ramp100 checkpoints, vision layer 11, shot 4, seed 1, and parameters `q_proj` and `v_proj`. This is descriptive evidence only.",
        "",
        f"- Paired records: {len(records)}",
        f"- Figures directory: `{OUT_FIG_DIR.relative_to(ROOT).as_posix()}`",
        "",
        "| Param | n | LoRA stable rank | OrthoAdapt stable rank | LoRA rank90 | OrthoAdapt rank90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for param, group in sorted(by_param.items()):
        lines.append(
            f"| {param} | {len(group)} | "
            f"{mean([r['lora']['stable_rank'] for r in group]):.3f} | "
            f"{mean([r['ohsinglora']['stable_rank'] for r in group]):.3f} | "
            f"{mean([r['lora']['rank90_energy'] for r in group]):.3f} | "
            f"{mean([r['ohsinglora']['rank90_energy'] for r in group]):.3f} |"
        )
    lines += [
        "",
        "## Claim Gate",
        "",
        "- Keep only descriptive spectrum language tied to the selected layer/param subset.",
        "- Do not use this report alone as causal evidence for robustness or performance gains.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--layer_idx", type=int, default=11)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args(argv)
    rows = load_manifest(Path(args.manifest))
    records = build_records(rows, args.shot, args.seed, args.layer_idx, args.top_k)
    if len(records) != len(DATASET_ORDER) * len(PARAMS):
        raise SystemExit(f"Expected {len(DATASET_ORDER) * len(PARAMS)} spectrum records, found {len(records)}")
    write_jsonl(records, OUT_JSONL)
    write_report(records)
    print(OUT_JSONL.relative_to(ROOT).as_posix())
    print(OUT_REPORT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
