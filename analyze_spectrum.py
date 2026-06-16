import argparse
import json
import sys
from pathlib import Path

from ecr3_provenance import file_sha256


class SpectrumError(RuntimeError):
    pass


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SpectrumError("PyTorch is required to analyze checkpoints") from exc
    return torch


def _require_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SpectrumError("matplotlib is required to plot spectrum output") from exc
    return plt


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SpectrumError("NumPy is required to compute spectral values") from exc
    return np


def validate_spectrum_args(args):
    if not args.lora_path or not args.oh_path:
        raise SpectrumError("Spectrum analysis requires --lora_path and --oh_path; synthetic demo mode is disabled")

    for label, raw_path in [("LoRA", args.lora_path), ("OH-SingLoRA", args.oh_path)]:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            raise SpectrumError(f"Missing checkpoint for {label}: {path}")

    if args.top_k < 1:
        raise SpectrumError("--top_k must be >= 1")


def require_checkpoint_metadata(checkpoint):
    if not isinstance(checkpoint, dict):
        raise SpectrumError("Checkpoint must be a dict containing weights and metadata")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        raise SpectrumError("Checkpoint metadata is required before plotting spectrum figures")
    return metadata


def matrix_definition(adapter, layer_idx, param_name):
    if adapter == "lora":
        formula = "Delta W = B @ A for the selected LoRA projection"
    elif adapter == "ohsinglora":
        formula = "Delta proxy = sum_h A_h,out @ A_h,in.T for the selected OH-SingLoRA projection"
    else:
        raise SpectrumError(f"Unsupported adapter for matrix definition: {adapter}")
    return {
        "adapter": adapter,
        "layer_idx": layer_idx,
        "param": param_name,
        "formula": formula,
    }


def find_layer_weights_lora(weights_dict, layer_idx, param_name="q_proj"):
    layer_key = f"layer_{layer_idx}"

    if "weights" in weights_dict:
        weights_dict = weights_dict["weights"]

    if layer_key not in weights_dict:
        raise SpectrumError(f"[LoRA] Missing {layer_key}; available keys: {list(weights_dict.keys())[:5]}")

    if param_name not in weights_dict[layer_key]:
        raise SpectrumError(f"[LoRA] Missing parameter {param_name} in {layer_key}")

    try:
        w_a = weights_dict[layer_key][param_name]["w_lora_A"]
        w_b = weights_dict[layer_key][param_name]["w_lora_B"]
    except KeyError as exc:
        raise SpectrumError(f"[LoRA] Missing LoRA A/B tensor under {layer_key}/{param_name}") from exc

    return w_a, w_b


def find_layer_weights_oh_singlora(state_dict, layer_idx, param_name="q_proj"):
    if "weights" in state_dict:
        state_dict = state_dict["weights"]

    search_suffix = f"resblocks.{layer_idx}.attn.{param_name}.lora_A_heads"

    target_key = None
    for key in state_dict.keys():
        if key.endswith(search_suffix):
            target_key = key
            break

    if target_key is None:
        raise SpectrumError(f"[OH-SingLoRA] Missing key ending with '{search_suffix}'")

    return state_dict[target_key], target_key


def compute_delta_w_lora(w_a, w_b):
    torch = _require_torch()
    return torch.matmul(w_b, w_a)


def compute_delta_w_oh_singlora(lora_A_heads):
    torch = _require_torch()
    if len(lora_A_heads.shape) != 3:
        raise SpectrumError(f"OH-SingLoRA lora_A_heads must be rank-3, got shape {tuple(lora_A_heads.shape)}")

    num_heads, d_model, _ = lora_A_heads.shape
    lora_a_heads_out = lora_A_heads[:, :d_model, :]
    lora_a_heads_in = lora_A_heads[:, :d_model, :]
    delta_w_sum = torch.zeros(d_model, d_model, device=lora_A_heads.device)

    for idx in range(num_heads):
        head_update = torch.matmul(lora_a_heads_out[idx], lora_a_heads_in[idx].transpose(0, 1))
        delta_w_sum += head_update

    return delta_w_sum


def get_singular_values(matrix):
    np = _require_numpy()
    matrix_np = matrix.detach().cpu().float().numpy()
    _, singular_values, _ = np.linalg.svd(matrix_np, full_matrices=False)

    if len(singular_values) > 0 and singular_values[0] > 0:
        return singular_values / singular_values[0]
    return singular_values


def load_checkpoint(path):
    torch = _require_torch()
    return torch.load(path, map_location="cpu")


def analyze(args):
    validate_spectrum_args(args)

    lora_ckpt = load_checkpoint(args.lora_path)
    oh_ckpt = load_checkpoint(args.oh_path)
    lora_metadata = require_checkpoint_metadata(lora_ckpt)
    oh_metadata = require_checkpoint_metadata(oh_ckpt)

    w_a, w_b = find_layer_weights_lora(lora_ckpt, args.layer_idx, args.param)
    delta_lora = compute_delta_w_lora(w_a, w_b)
    s_lora = get_singular_values(delta_lora)

    lora_heads, oh_key = find_layer_weights_oh_singlora(oh_ckpt, args.layer_idx, args.param)
    delta_oh = compute_delta_w_oh_singlora(lora_heads)
    s_oh = get_singular_values(delta_oh)

    report = {
        "schema_version": "ecr1.spectrum.v1",
        "layer_idx": args.layer_idx,
        "param": args.param,
        "top_k": args.top_k,
        "lora": {
            "checkpoint_path": args.lora_path,
            "checkpoint_sha256": file_sha256(args.lora_path),
            "metadata": lora_metadata,
            "matrix_definition": matrix_definition("lora", args.layer_idx, args.param),
            "matrix_shape": list(delta_lora.shape),
            "normalized_singular_values": s_lora[: args.top_k].tolist(),
        },
        "ohsinglora": {
            "checkpoint_path": args.oh_path,
            "checkpoint_sha256": file_sha256(args.oh_path),
            "metadata": oh_metadata,
            "matrix_definition": matrix_definition("ohsinglora", args.layer_idx, args.param),
            "matrix_key": oh_key,
            "matrix_shape": list(delta_oh.shape),
            "normalized_singular_values": s_oh[: args.top_k].tolist(),
        },
    }
    return report


def plot_report(report, output_path):
    np = _require_numpy()
    plt = _require_pyplot()
    s_lora = np.asarray(report["lora"]["normalized_singular_values"])
    s_oh = np.asarray(report["ohsinglora"]["normalized_singular_values"])
    x_axis = np.arange(min(len(s_lora), len(s_oh), report["top_k"]))

    if len(x_axis) == 0:
        raise SpectrumError("No singular values available to plot")

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, s_lora[: len(x_axis)], "o--", color="#d62728", label="Standard LoRA", linewidth=2, markersize=5, alpha=0.8)
    plt.plot(x_axis, s_oh[: len(x_axis)], "s-", color="#1f77b4", label="OH-SingLoRA", linewidth=2, markersize=5)
    plt.title(f"Singular Value Spectrum Analysis\nLayer {report['layer_idx']} - {report['param']}", fontsize=14, fontweight="bold")
    plt.xlabel("Singular Value Index (Sorted)", fontsize=12)
    plt.ylabel("Normalized Singular Value (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_report(report, report_path):
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Fail-closed spectral analysis for CLIP adapters")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint (.pt)")
    parser.add_argument("--oh_path", type=str, required=True, help="Path to OH-SingLoRA checkpoint (.pt)")
    parser.add_argument("--layer_idx", type=int, default=11, help="Transformer layer index to analyze")
    parser.add_argument("--param", type=str, default="q_proj", choices=["q_proj", "v_proj"], help="Parameter to analyze")
    parser.add_argument("--top_k", type=int, default=50, help="Number of singular values to plot")
    parser.add_argument("--output", type=str, default="singular_value_spectrum.png", help="Output PNG path")
    parser.add_argument("--report_path", type=str, default="spectrum_report.json", help="Output JSON report path")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = analyze(args)
        plot_report(report, args.output)
        write_report(report, args.report_path)
    except SpectrumError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(f"[SUCCESS] Spectrum figure saved to: {args.output}")
    print(f"[SUCCESS] Spectrum report saved to: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
