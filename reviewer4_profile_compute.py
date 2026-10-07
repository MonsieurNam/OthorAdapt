"""Generate, measure, and summarize Reviewer #4 compute profiling on one RTX 3090."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from argparse import Namespace
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROFILE_METHODS = ("lora", "singlora", "static", "dynamic")
PROFILE_DATASETS = ("eurosat", "caltech101")
SCRIPT_VERSION = "reviewer4.profile.v1"
REQUIRED_PROFILE_METRICS = (
    "trainable_parameters",
    "training_forward_backward_flops_per_step",
    "estimated_training_flops_2000_steps",
    "inference_flops_per_image",
    "inference_flops_per_batch",
    "peak_gpu_allocated_bytes",
    "peak_gpu_reserved_bytes",
    "seconds_per_optimizer_step",
    "mean_complete_epoch_seconds_after_warmup",
    "median_complete_epoch_seconds_after_warmup",
    "end_to_end_training_seconds",
    "inference_median_latency_seconds",
    "inference_p95_latency_seconds",
    "inference_images_per_second",
    "text_classifier_construction_seconds",
    "gating_kernel_flops",
    "gating_kernel_seconds",
)


@dataclass(frozen=True)
class ProfileSpec:
    method: str
    dataset: str
    repetition: int
    shot: int = 4
    seed: int = 1

    @property
    def run_key(self):
        return f"{self.method}|{self.dataset}|4|1|rep{self.repetition}"


def build_training_profile_specs():
    specs = [
        ProfileSpec(method, dataset, repetition)
        for method in PROFILE_METHODS
        for dataset in PROFILE_DATASETS
        for repetition in (1, 2, 3)
    ]
    if len(specs) != 24 or len({spec.run_key for spec in specs}) != 24:
        raise ValueError("profiling matrix must contain 24 unique runs")
    return specs


def method_from_config(config):
    adapter = config.get("adapter")
    if adapter in {"lora", "singlora"}:
        return adapter
    if adapter == "ohsinglora":
        mode = config.get("routing_mode", "input_conditioned")
        if mode == "input_independent":
            return "static"
        if mode == "input_conditioned":
            return "dynamic"
    raise ValueError(f"unsupported profiling configuration: {config}")


def generate_training_profile_commands(data_root, split_reference):
    commands = []
    method_flags = {
        "lora": "--adapter lora --r 8 --alpha 1",
        "singlora": "--adapter singlora --r 8 --alpha 1 --ramp_up_steps 100",
        "static": (
            "--adapter ohsinglora --routing_mode input_independent --num_heads 2 "
            "--r 8 --alpha 1 --lambda_o 0.03 --ramp_up_steps 100 --ortho_reduction mean"
        ),
        "dynamic": (
            "--adapter ohsinglora --routing_mode input_conditioned --num_heads 2 "
            "--r 8 --alpha 1 --lambda_o 0.03 --ramp_up_steps 100 --ortho_reduction mean"
        ),
    }
    for spec in build_training_profile_specs():
        filename = f"profile_{spec.method}_{spec.dataset}_4shot_seed1_rep{spec.repetition}"
        log = f"revision_materials/reviewer4/raw/profile/logs/{filename}.log"
        command = (
            "${PYTHON} main.py "
            f"--root_path {data_root} --dataset {spec.dataset} --shots 4 --seed 1 "
            "--backbone ViT-B/16 --encoder both --position all --params q k v "
            "--n_iters 500 --batch_size 32 --loss_fn ce "
            f"{method_flags[spec.method]} "
            "--save_path revision_materials/reviewer4/raw/profile/checkpoints "
            f"--filename {filename} --profile_repetition {spec.repetition} "
            "--compute_profile_only "
            "--run_manifest revision_materials/reviewer4/raw/profile/compute_profile_train_manifest.jsonl "
            f"--split_reference_manifest {split_reference} "
            f"2>&1 | tee {log}"
        )
        commands.append(command)
    return commands


def generate_benchmark_commands(data_root):
    commands = []
    for spec in build_training_profile_specs():
        filename = f"profile_{spec.method}_{spec.dataset}_4shot_seed1_rep{spec.repetition}"
        log = f"revision_materials/reviewer4/raw/profile/logs/{filename}_microprofile.log"
        command = (
            "${PYTHON} reviewer4_profile_compute.py benchmark "
            f"--method {spec.method} --dataset {spec.dataset} --repetition {spec.repetition} "
            f"--data-root {data_root} "
            "--save-path revision_materials/reviewer4/raw/profile/checkpoints "
            f"--filename {filename} "
            "--split-reference revision_materials/reviewer4/raw/frozen_dynamic_reference.jsonl "
            "--output-jsonl revision_materials/reviewer4/raw/profile/compute_microprofile.jsonl "
            "--operator-dir revision_materials/reviewer4/raw/profile/operator_tables "
            f"2>&1 | tee {log}"
        )
        commands.append(command)
    return commands


def write_profile_script(path, commands):
    header = [
        "#!/usr/bin/env bash", "set -euo pipefail", ': "${PYTHON:=python3}"',
        ': "${DATA_ROOT:?Set DATA_ROOT=/path/to/eight-dataset-root}"',
        "mkdir -p revision_materials/reviewer4/raw/profile/logs", "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(header + commands) + "\n", encoding="utf-8")


def summarize_inference_timings(repetition_batch_seconds, batch_size):
    if len(repetition_batch_seconds) != 5:
        raise ValueError("inference profiling requires exactly five timing repetitions")
    lengths = {len(values) for values in repetition_batch_seconds}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("each timing repetition must have the same positive batch count")
    flattened = np.asarray([value for repetition in repetition_batch_seconds for value in repetition], dtype=float)
    if np.any(flattened <= 0):
        raise ValueError("latencies must be positive")
    median = float(np.median(flattened))
    return {
        "timing_repetitions": 5,
        "measured_batches_per_repetition": int(next(iter(lengths))),
        "median_batch_latency_seconds": median,
        "p95_batch_latency_seconds": float(np.percentile(flattened, 95)),
        "images_per_second": float(batch_size) / median,
    }


def gating_flops(batch_tokens, width, heads):
    if min(batch_tokens, width, heads) <= 0:
        raise ValueError("batch_tokens, width, and heads must be positive")
    return {
        "batch_tokens": int(batch_tokens), "width": int(width), "heads": int(heads),
        "macs": int(batch_tokens) * int(width) * int(heads),
        "mac_to_flop_multiplier": 2,
        "linear_flops": int(batch_tokens) * int(width) * int(heads) * 2,
        "softmax_flops": None,
        "note": "Only the dense routing projection is counted analytically; unsupported operators are not inferred.",
    }


def validate_profile_row(row):
    missing = [field for field in REQUIRED_PROFILE_METRICS if row.get(field) is None]
    if missing:
        raise ValueError(f"profile row is missing required metrics: {missing}")
    if int(row["trainable_parameters"]) <= 0:
        raise ValueError("trainable_parameters must be positive")


def measure_cuda_inference(callable_, batch_size=256, warmup_batches=50, measured_batches=200, repetitions=5):
    """Measure a zero-argument CUDA callable without touching model/random state."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference profiling")
    values = []
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(warmup_batches):
            callable_()
        torch.cuda.synchronize()
        for _ in range(repetitions):
            repetition = []
            for _ in range(measured_batches):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                callable_()
                end.record()
                torch.cuda.synchronize()
                repetition.append(float(start.elapsed_time(end)) / 1000.0)
            values.append(repetition)
    return {"raw_batch_seconds": values, **summarize_inference_timings(values, batch_size)}


def profile_cuda_callable_flops(callable_, operator_table_path):
    """Profile supported operator FLOPs once and preserve the raw operator table."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLOP profiling")
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=True) as profiler:
        callable_()
        torch.cuda.synchronize()
    rows = []
    total = 0
    for event in profiler.key_averages(group_by_input_shape=True):
        flops = int(event.flops or 0)
        total += flops
        rows.append({"operator": event.key, "input_shapes": str(event.input_shapes), "flops": flops})
    Path(operator_table_path).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return {"supported_operator_flops": total, "unsupported_operator_flops_inferred": False}


def _profile_args(method, dataset, repetition, data_root, save_path, filename):
    adapter = {"lora": "lora", "singlora": "singlora", "static": "ohsinglora", "dynamic": "ohsinglora"}[method]
    return Namespace(
        seed=1, root_path=data_root, dataset=dataset, shots=4, backbone="ViT-B/16",
        lr=2e-4, n_iters=500, batch_size=32, adapter=adapter, num_heads=2,
        routing_mode=("input_independent" if method == "static" else "input_conditioned"),
        position="all", encoder="both", params=["q", "k", "v"], r=8, alpha=1,
        lambda_o=(0.03 if adapter == "ohsinglora" else 0.0), ortho_reduction="mean",
        dropout_rate=0.25, ramp_up_steps=100, save_path=save_path, filename=filename,
        profile_repetition=repetition, loss_fn="ce", eval_only=True,
    )


def benchmark_checkpoint(
    method, dataset, repetition, data_root, save_path, filename, split_reference,
    output_jsonl, operator_dir,
):
    """Run the fixed cached-text inference and FLOP profile for one trained checkpoint."""
    import random
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import clip
    from datasets import build_dataset
    from datasets.utils import build_data_loader
    from ecr3_provenance import dataset_provenance
    from experiment_manifest import count_named_adapter_parameters
    from loralib.layers_OH_singlora import LinearOHsingLoRA, calculate_ortho_loss
    from loralib.utils import apply_adapter, apply_lora, load_adapter, load_lora, mark_only_lora_as_trainable
    from reviewer4_audit import load_jsonl
    from reviewer4_provenance import capture_environment, validate_split_reference, validate_target_environment
    from utils import clip_classifier

    if not torch.cuda.is_available():
        raise RuntimeError("RTX 3090 CUDA execution is required")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    environment = validate_target_environment(capture_environment())
    args = _profile_args(method, dataset, repetition, data_root, save_path, filename)
    model, preprocess = clip.load(args.backbone)
    model.eval().cuda()
    dataset_object = build_dataset(dataset, data_root, 4, preprocess)
    provenance = dataset_provenance(dataset_object)
    validate_split_reference(dataset, 4, 1, provenance, load_jsonl(split_reference))

    if args.adapter == "lora":
        layers = apply_lora(args, model)
        load_lora(args, layers)
    else:
        apply_adapter(args, model)
        load_adapter(args, model)
    mark_only_lora_as_trainable(model)
    trainable_parameters = count_named_adapter_parameters(model.named_parameters())

    test_loader = build_data_loader(
        data_source=dataset_object.test, batch_size=256, is_train=False,
        tfm=preprocess, shuffle=False, num_workers=8,
    )
    images, _ = next(iter(test_loader))
    images = images.cuda(non_blocking=True)

    torch.cuda.synchronize()
    classifier_start = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        text_features = clip_classifier(dataset_object.classnames, dataset_object.template, model)
    torch.cuda.synchronize()
    classifier_seconds = time.perf_counter() - classifier_start

    def inference_call():
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features @ text_features

    torch.cuda.reset_peak_memory_stats()
    inference_timing = measure_cuda_inference(inference_call, batch_size=images.shape[0])
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    operator_dir = Path(operator_dir)
    operator_dir.mkdir(parents=True, exist_ok=True)
    infer_flops = profile_cuda_callable_flops(
        inference_call, operator_dir / f"{method}_{dataset}_rep{repetition}_inference.json",
    )["supported_operator_flops"]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    train_loader = build_data_loader(
        data_source=dataset_object.train_x, batch_size=32, is_train=True,
        tfm=train_transform, shuffle=True, num_workers=8,
    )
    train_images, train_target = next(iter(train_loader))
    train_images, train_target = train_images.cuda(), train_target.cuda()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-4)
    model.train()

    def training_forward_backward():
        optimizer.zero_grad(set_to_none=True)
        template = dataset_object.template[0]
        tokens = clip.tokenize([template.format(name.replace("_", " ")) for name in dataset_object.classnames]).cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            class_embeddings = model.encode_text(tokens)
            train_text = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            train_image = model.encode_image(train_images)
            train_image = train_image / train_image.norm(dim=-1, keepdim=True)
            logits = 100 * train_image @ train_text.t()
            loss = F.cross_entropy(logits, train_target).float()
            if args.adapter == "ohsinglora":
                ortho = sum(
                    calculate_ortho_loss(module.lora_A_heads, reduction="mean")
                    for module in model.modules() if isinstance(module, LinearOHsingLoRA)
                )
                loss = loss + 0.03 * ortho
        loss.backward()
        return loss

    training_flops = profile_cuda_callable_flops(
        training_forward_backward,
        operator_dir / f"{method}_{dataset}_rep{repetition}_training.json",
    )["supported_operator_flops"]

    gate_shapes = []
    hooks = []
    gated_modules = [module for module in model.modules() if isinstance(module, LinearOHsingLoRA)]
    for module in gated_modules:
        hooks.append(module.register_forward_pre_hook(lambda current, inputs: gate_shapes.append((current, tuple(inputs[0].shape)))))
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        inference_call()
    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()
    shared_gate_inputs = {
        shape: torch.zeros(shape, device="cuda", dtype=next(module.parameters()).dtype)
        for module, shape in gate_shapes
    }

    if gate_shapes:
        def gating_call():
            last = None
            for module, shape in gate_shapes:
                last = module.routing_weights(shared_gate_inputs[shape])
            return last
        gating_timing = measure_cuda_inference(gating_call, batch_size=1)
        gating_seconds = gating_timing["median_batch_latency_seconds"]
        gating_total_flops = 0
        for module, shape in gate_shapes:
            tokens = math.prod(shape[:-1]) if module.routing_mode == "input_conditioned" else 1
            gating_total_flops += tokens * shape[-1] * module.num_heads * 2
    else:
        gating_seconds = 0.0
        gating_total_flops = 0

    row = {
        "method": method, "dataset": dataset, "shot": 4, "seed": 1, "repetition": repetition,
        "trainable_parameters": trainable_parameters,
        "training_forward_backward_flops_per_step": training_flops,
        "estimated_training_flops_2000_steps": training_flops * 2000,
        "inference_flops_per_batch": infer_flops,
        "inference_flops_per_image": infer_flops / int(images.shape[0]),
        "inference_median_latency_seconds": inference_timing["median_batch_latency_seconds"],
        "inference_p95_latency_seconds": inference_timing["p95_batch_latency_seconds"],
        "inference_images_per_second": inference_timing["images_per_second"],
        "text_classifier_construction_seconds": classifier_seconds,
        "gating_kernel_flops": gating_total_flops,
        "gating_kernel_seconds": gating_seconds,
        "microprofile_peak_gpu_allocated_bytes": peak_allocated,
        "microprofile_peak_gpu_reserved_bytes": peak_reserved,
        "gpu_name": environment["gpu_name"], "gpu_uuid": environment["gpu_uuid"],
        "software_stack_id": environment["software_stack_id"],
        "power_limit_watts": environment["power_limit_watts"],
        "split_identity": __import__("reviewer4_provenance").split_identity(provenance),
    }
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_jsonl).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return row


def summarize_profile_rows(rows):
    grouped = {}
    for row in rows:
        validate_profile_row(row)
        key = row["method"]
        grouped.setdefault(key, []).append(row)
    if set(grouped) != set(PROFILE_METHODS):
        raise ValueError("raw profile must contain lora, singlora, static, and dynamic")
    summary = []
    for method in PROFILE_METHODS:
        method_rows = grouped[method]
        if len(method_rows) != 6:
            raise ValueError(f"{method} must contain six rows (two datasets x three repetitions)")
        out = {"method": method, "profile_rows": len(method_rows)}
        for field in REQUIRED_PROFILE_METRICS:
            values = [float(row[field]) for row in method_rows]
            out[f"mean_{field}"] = statistics.fmean(values)
            out[f"median_{field}"] = statistics.median(values)
        summary.append(out)
    return summary


def dynamic_static_overhead(summary_rows):
    indexed = {row["method"]: row for row in summary_rows}
    if not {"static", "dynamic"}.issubset(indexed):
        raise ValueError("static and dynamic summary rows are required")
    metrics = (
        ("cached_text_inference_latency", "mean_inference_median_latency_seconds"),
        ("optimizer_step_time", "mean_seconds_per_optimizer_step"),
    )
    rows = []
    for label, field in metrics:
        static = float(indexed["static"][field])
        dynamic = float(indexed["dynamic"][field])
        difference = dynamic - static
        rows.append({
            "metric": label, "static_seconds": static, "dynamic_seconds": dynamic,
            "absolute_overhead_seconds": difference,
            "relative_overhead_percent": 100.0 * difference / static if static else None,
        })
    return rows


def flops_breakdown(summary_rows):
    return [
        {
            "method": row["method"],
            "training_forward_backward_flops_per_step": row["mean_training_forward_backward_flops_per_step"],
            "estimated_training_flops_2000_steps": row["mean_estimated_training_flops_2000_steps"],
            "inference_flops_per_image": row["mean_inference_flops_per_image"],
            "inference_flops_per_batch": row["mean_inference_flops_per_batch"],
            "gating_kernel_flops": row["mean_gating_kernel_flops"],
            "unsupported_operator_flops_inferred": False,
        }
        for row in summary_rows
    ]


def render_compute_table(summary_rows):
    indexed = {row["method"]: row for row in summary_rows}
    labels = {"lora": "CLIP-LoRA", "singlora": "SingLoRA", "static": "OrthoAdapt-static", "dynamic": "OrthoAdapt-dynamic"}
    lines = [
        "% Auto-generated by reviewer4_profile_compute.py; MAC=2 FLOPs; cached-text inference.",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Trainable params & Time/step (s) & Peak alloc. (GiB) & Images/s \\\\",
        "\\midrule",
    ]
    for method in PROFILE_METHODS:
        row = indexed[method]
        lines.append(
            f"{labels[method]} & {int(round(row['mean_trainable_parameters'])):,} & "
            f"{row['mean_seconds_per_optimizer_step']:.4f} & "
            f"{row['mean_peak_gpu_allocated_bytes'] / 2**30:.3f} & "
            f"{row['mean_inference_images_per_second']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_profile_evidence(training_records, micro_records):
    training_index = {}
    for record in training_records:
        config = record.get("config", {})
        key = (method_from_config(config), config.get("dataset"), int(config.get("profile_repetition", 0)))
        if key in training_index:
            raise ValueError(f"duplicate training profile row: {key}")
        training_index[key] = record
    micro_index = {}
    for record in micro_records:
        key = (record["method"], record["dataset"], int(record["repetition"]))
        if key in micro_index:
            raise ValueError(f"duplicate microprofile row: {key}")
        micro_index[key] = record
    expected = {(spec.method, spec.dataset, spec.repetition) for spec in build_training_profile_specs()}
    if set(training_index) != expected or set(micro_index) != expected:
        raise ValueError("training and microprofile evidence must each cover the same 24 frozen runs")
    rows = []
    for key in sorted(expected):
        training = training_index[key]
        metrics = training.get("metrics", {})
        micro = micro_index[key]
        environment = training.get("environment", {})
        if environment.get("gpu_uuid") != micro.get("gpu_uuid"):
            raise ValueError(f"training/microprofile GPU UUID mismatch for {key}")
        if environment.get("software_stack_id") != micro.get("software_stack_id"):
            raise ValueError(f"training/microprofile software stack mismatch for {key}")
        if str(environment.get("power_limit_watts")) != str(micro.get("power_limit_watts")):
            raise ValueError(f"training/microprofile power limit mismatch for {key}")
        row = {
            "method": key[0], "dataset": key[1], "repetition": key[2],
            "gpu_name": environment.get("gpu_name"), "gpu_uuid": environment.get("gpu_uuid"),
            "software_stack_id": environment.get("software_stack_id"),
            "trainable_parameters": metrics.get("trainable_parameters"),
            "training_forward_backward_flops_per_step": micro.get("training_forward_backward_flops_per_step"),
            "estimated_training_flops_2000_steps": micro.get("estimated_training_flops_2000_steps"),
            "inference_flops_per_image": micro.get("inference_flops_per_image"),
            "inference_flops_per_batch": micro.get("inference_flops_per_batch"),
            "peak_gpu_allocated_bytes": metrics.get("peak_gpu_allocated_bytes"),
            "peak_gpu_reserved_bytes": metrics.get("peak_gpu_reserved_bytes"),
            "seconds_per_optimizer_step": metrics.get("seconds_per_optimizer_step"),
            "mean_complete_epoch_seconds_after_warmup": metrics.get("mean_complete_epoch_seconds_after_warmup"),
            "median_complete_epoch_seconds_after_warmup": metrics.get("median_complete_epoch_seconds_after_warmup"),
            "end_to_end_training_seconds": metrics.get("fine_tuning_seconds"),
            "inference_median_latency_seconds": micro.get("inference_median_latency_seconds"),
            "inference_p95_latency_seconds": micro.get("inference_p95_latency_seconds"),
            "inference_images_per_second": micro.get("inference_images_per_second"),
            "text_classifier_construction_seconds": micro.get("text_classifier_construction_seconds"),
            "gating_kernel_flops": micro.get("gating_kernel_flops"),
            "gating_kernel_seconds": micro.get("gating_kernel_seconds"),
        }
        validate_profile_row(row)
        rows.append(row)
    return rows


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_sidecar(path, inputs):
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "UNKNOWN"
    artifact = Path(path)
    payload = {
        "artifact": str(artifact), "artifact_sha256": _file_sha256(artifact),
        "inputs": [{"path": str(Path(item)), "sha256": _file_sha256(item)} for item in inputs],
        "script_version": SCRIPT_VERSION, "git_revision": revision,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(str(artifact) + ".provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--out", required=True)
    generate.add_argument("--kind", choices=("training", "benchmark"), default="training")
    generate.add_argument("--data-root", default="${DATA_ROOT}")
    generate.add_argument(
        "--split-reference",
        default="revision_materials/reviewer4/raw/frozen_dynamic_reference.jsonl",
    )
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--raw-csv", required=True)
    summarize.add_argument("--summary-csv", required=True)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--method", choices=PROFILE_METHODS, required=True)
    benchmark.add_argument("--dataset", choices=PROFILE_DATASETS, required=True)
    benchmark.add_argument("--repetition", type=int, choices=(1, 2, 3), required=True)
    benchmark.add_argument("--data-root", required=True)
    benchmark.add_argument("--save-path", required=True)
    benchmark.add_argument("--filename", required=True)
    benchmark.add_argument("--split-reference", required=True)
    benchmark.add_argument("--output-jsonl", required=True)
    benchmark.add_argument("--operator-dir", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--training-manifest", required=True)
    merge.add_argument("--micro-jsonl", required=True)
    merge.add_argument("--raw-csv", required=True)
    merge.add_argument("--summary-csv", required=True)
    merge.add_argument("--overhead-csv")
    merge.add_argument("--flops-csv")
    merge.add_argument("--table-tex")
    args = parser.parse_args(argv)
    if args.command == "generate":
        commands = (
            generate_training_profile_commands(args.data_root, args.split_reference)
            if args.kind == "training"
            else generate_benchmark_commands(args.data_root)
        )
        write_profile_script(args.out, commands)
    elif args.command == "summarize":
        rows = _read_csv(args.raw_csv)
        _write_csv(args.summary_csv, summarize_profile_rows(rows))
    elif args.command == "benchmark":
        benchmark_checkpoint(
            args.method, args.dataset, args.repetition, args.data_root,
            args.save_path, args.filename, args.split_reference,
            args.output_jsonl, args.operator_dir,
        )
    else:
        rows = merge_profile_evidence(
            _load_jsonl(args.training_manifest), _load_jsonl(args.micro_jsonl),
        )
        summary_rows = summarize_profile_rows(rows)
        _write_csv(args.raw_csv, rows)
        _write_csv(args.summary_csv, summary_rows)
        if args.overhead_csv:
            _write_csv(args.overhead_csv, dynamic_static_overhead(summary_rows))
        if args.flops_csv:
            _write_csv(args.flops_csv, flops_breakdown(summary_rows))
        if args.table_tex:
            Path(args.table_tex).write_text(render_compute_table(summary_rows), encoding="utf-8")
        for artifact in filter(
            None,
            (args.raw_csv, args.summary_csv, args.overhead_csv, args.flops_csv, args.table_tex),
        ):
            _write_sidecar(artifact, [args.training_manifest, args.micro_jsonl])


if __name__ == "__main__":
    main()
