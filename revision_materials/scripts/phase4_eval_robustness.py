"""Deterministic Phase 4 robustness evaluation for one Phase 3 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecr3_provenance import file_sha256


SCHEMA_VERSION = "phase4.robustness.v1"
NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORM_STD = (0.26862954, 0.26130258, 0.27577711)


def severity_spec(severity: int, seed: int) -> dict:
    specs = {
        0: {"name": "clean", "noise_std": 0.0, "blur_kernel": 0, "blur_sigma": 0.0, "brightness": 1.0, "contrast": 1.0},
        1: {"name": "noise_light", "noise_std": 0.05, "blur_kernel": 0, "blur_sigma": 0.0, "brightness": 1.0, "contrast": 1.0},
        2: {"name": "noise_blur_medium", "noise_std": 0.10, "blur_kernel": 3, "blur_sigma": 1.0, "brightness": 1.0, "contrast": 1.0},
        3: {"name": "noise_blur_contrast_high", "noise_std": 0.15, "blur_kernel": 5, "blur_sigma": 2.0, "brightness": 1.25, "contrast": 1.35},
    }
    if severity not in specs:
        raise ValueError("severity must be one of 0,1,2,3")
    out = dict(specs[severity])
    out["seed"] = seed
    return out


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DeterministicNoise:
    def __init__(self, std: float, seed: int):
        self.std = float(std)
        self.generator = None
        self.seed = int(seed)

    def __call__(self, tensor):
        if self.std <= 0:
            return tensor
        import torch

        if self.generator is None:
            self.generator = torch.Generator(device=tensor.device)
            self.generator.manual_seed(self.seed)
        noise = torch.randn(tensor.size(), generator=self.generator, device=tensor.device, dtype=tensor.dtype)
        return tensor + noise * self.std


class FixedColor:
    def __init__(self, brightness: float, contrast: float):
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __call__(self, tensor):
        import torchvision.transforms.functional as F

        if self.brightness != 1.0:
            tensor = F.adjust_brightness(tensor, self.brightness)
        if self.contrast != 1.0:
            tensor = F.adjust_contrast(tensor, self.contrast)
        return tensor


def build_transform(n_px: int, severity: int, seed: int):
    import torchvision.transforms as T

    spec = severity_spec(severity, seed)
    transforms = [
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(n_px),
        T.ToTensor(),
    ]
    if spec["brightness"] != 1.0 or spec["contrast"] != 1.0:
        transforms.append(FixedColor(spec["brightness"], spec["contrast"]))
    if spec["blur_kernel"]:
        transforms.append(T.GaussianBlur(kernel_size=spec["blur_kernel"], sigma=spec["blur_sigma"]))
    if spec["noise_std"] > 0:
        transforms.append(DeterministicNoise(spec["noise_std"], seed))
    transforms.append(T.Normalize(NORM_MEAN, NORM_STD))
    return T.Compose(transforms), spec


def evaluate(model, loader, dataset):
    import clip
    import torch
    from utils import cls_acc

    model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace("_", " ")) for classname in dataset.classnames]
        texts = clip.tokenize(texts).cuda()
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.0
    total = 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.cuda(), target.cuda()
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ class_embeddings.t()
            acc += cls_acc(similarity, target) * len(similarity)
            total += len(similarity)
    return float(acc / total)


def load_weights_smart(model, checkpoint_path: str, adapter_type: str, args) -> None:
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cuda")
    weights = ckpt["weights"] if isinstance(ckpt, dict) and "weights" in ckpt else ckpt
    if adapter_type == "lora":
        from loralib.utils import INDEX_POSITIONS_TEXT, INDEX_POSITIONS_VISION

        layer_counter = 0
        if args.encoder in {"text", "both"}:
            for i, block in enumerate(model.transformer.resblocks):
                if i in INDEX_POSITIONS_TEXT[args.position]:
                    layer_key = f"layer_{layer_counter}"
                    if layer_key in weights:
                        for name, module in [("q_proj", block.attn.q_proj), ("k_proj", block.attn.k_proj), ("v_proj", block.attn.v_proj), ("proj", block.attn.proj)]:
                            if name in weights[layer_key]:
                                module.w_lora_A.data.copy_(weights[layer_key][name]["w_lora_A"])
                                module.w_lora_B.data.copy_(weights[layer_key][name]["w_lora_B"])
                    layer_counter += 1
        if args.encoder in {"vision", "both"}:
            for i, block in enumerate(model.visual.transformer.resblocks):
                if i in INDEX_POSITIONS_VISION[args.backbone][args.position]:
                    layer_key = f"layer_{layer_counter}"
                    if layer_key in weights:
                        for name, module in [("q_proj", block.attn.q_proj), ("k_proj", block.attn.k_proj), ("v_proj", block.attn.v_proj), ("proj", block.attn.proj)]:
                            if name in weights[layer_key]:
                                module.w_lora_A.data.copy_(weights[layer_key][name]["w_lora_A"])
                                module.w_lora_B.data.copy_(weights[layer_key][name]["w_lora_B"])
                    layer_counter += 1
    else:
        state_dict = {key.replace("module.", ""): value for key, value in weights.items()}
        model.load_state_dict(state_dict, strict=False)


def build_model(args):
    import clip
    from loralib.utils import apply_adapter, apply_lora

    model, _ = clip.load(args.backbone, device="cuda")

    class AdapterArgs:
        adapter = args.adapter
        encoder = args.encoder
        position = args.position
        params = args.params
        r = args.r
        alpha = args.alpha
        ramp_up_steps = args.ramp_up_steps
        num_heads = args.num_heads
        dropout_rate = args.dropout_rate
        backbone = args.backbone

    if args.adapter == "lora":
        apply_lora(AdapterArgs(), model)
    else:
        apply_adapter(AdapterArgs(), model)
    model.cuda()
    model.to(model.dtype)
    for module in model.modules():
        if "LayerNorm" in type(module).__name__:
            module.float()
    load_weights_smart(model, args.checkpoint, args.adapter, args)
    return model


def write_jsonl(path: str, records: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--adapter", required=True, choices=["lora", "ohsinglora"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--run_manifest", required=True)
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--lambda_o", type=float, default=0.0)
    parser.add_argument("--ramp_up_steps", type=int, default=100)
    parser.add_argument("--position", default="all")
    parser.add_argument("--encoder", default="both")
    parser.add_argument("--params", nargs="+", default=["q", "k", "v"])
    parser.add_argument("--dropout_rate", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args(argv)

    set_seed(args.seed)

    import torch
    from torch.utils.data import DataLoader
    from datasets import build_dataset
    from datasets.utils import DatasetWrapper

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Phase 4 robustness evaluation")

    start = time.time()
    model = build_model(args)
    dataset = build_dataset(args.dataset, args.root_path, args.shots, None)
    records = []
    for severity in [0, 1, 2, 3]:
        transform, spec = build_transform(model.visual.input_resolution, severity, args.seed * 1000 + severity)
        loader = DataLoader(
            DatasetWrapper(dataset.test, input_size=model.visual.input_resolution, transform=transform, is_train=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        severity_start = time.time()
        acc = evaluate(model, loader, dataset)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "job_id": args.filename,
                "dataset": args.dataset,
                "shot": args.shots,
                "seed": args.seed,
                "method": args.adapter,
                "backbone": args.backbone,
                "config": {
                    "r": args.r,
                    "alpha": args.alpha,
                    "num_heads": args.num_heads if args.adapter == "ohsinglora" else None,
                    "lambda_o": args.lambda_o if args.adapter == "ohsinglora" else 0.0,
                    "ramp_up_steps": args.ramp_up_steps if args.adapter == "ohsinglora" else 100,
                    "encoder": args.encoder,
                    "position": args.position,
                    "params": args.params,
                },
                "checkpoint": {
                    "path": args.checkpoint,
                    "sha256": file_sha256(args.checkpoint),
                },
                "severity": severity,
                "corruption": spec,
                "metrics": {
                    "accuracy": acc,
                    "runtime_seconds": time.time() - severity_start,
                },
            }
        )
    write_jsonl(args.run_manifest, records)
    print(json.dumps({"job_id": args.filename, "rows": len(records), "runtime_seconds": time.time() - start}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
