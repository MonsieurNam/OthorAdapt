# @title lora.py
# %%writefile /content/SingloRA_CLIP/lora.py


import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import clip

from losses.focal_loss import FocalLoss
from utils import *

from losses.center_loss import CenterLoss
from losses.arcface import ArcFaceLoss
from losses.cosface import CosFaceLoss
from losses.max_entropy import MaximumEntropyLoss

from ecr3_provenance import build_run_record, file_sha256, write_jsonl_record
from experiment_manifest import count_named_adapter_parameters
from loralib.utils import apply_lora, load_adapter, save_lora, load_lora, apply_adapter, mark_only_lora_as_trainable, get_lora_parameters, save_adapter
from loralib.layers_OH_singlora import LinearOHsingLoRA, calculate_ortho_loss


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def evaluate(args, clip_model, loader, dataset):
    """
    Đánh giá hiệu suất của mô hình trên một tập dữ liệu cụ thể.
    """
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)

    acc /= tot_samples
    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    """
    Hàm chính điều phối toàn bộ quy trình.
    """
    print("\nMoving model and data to GPU for evaluation...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        clip_model.cuda()

    selection_split = getattr(args, "selection_split", "val")
    report_test = getattr(args, "report_test", False)
    selection_loader = val_loader if selection_split == "val" else test_loader

    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)
    selection_features, selection_labels = pre_load_features(clip_model, selection_loader)

    selection_features = selection_features.cuda()
    selection_labels = selection_labels.cuda()

    clip_logits = logit_scale * selection_features @ textual_features
    zs_selection_acc = cls_acc(clip_logits, selection_labels)
    print(f"\n**** Zero-shot CLIP's {selection_split} accuracy: {zs_selection_acc:.2f}. ****\n")

    del selection_features, selection_labels
    torch.cuda.empty_cache()

    list_adapter_layers = []
    if args.adapter == 'lora':
        list_adapter_layers = apply_lora(args, clip_model)
    elif args.adapter in ['singlora', 'gmhsinglora', 'ohsinglora']:
        list_adapter_layers = apply_adapter(args, clip_model)
    adapter_parameter_count = count_named_adapter_parameters(clip_model.named_parameters())

    clip_model = clip_model.cuda()

    if args.eval_only:
        print(f"\nEvaluation-only mode for {args.adapter.upper()} adapter.")
        if args.adapter == 'lora':
            load_lora(args, list_adapter_layers)
        elif args.adapter in ['singlora', 'gmhsinglora', 'ohsinglora']:
            load_adapter(args, list_adapter_layers)

        acc_selection = evaluate(args, clip_model, selection_loader, dataset)
        acc_test = evaluate(args, clip_model, test_loader, dataset) if report_test else None
        print(f"**** {selection_split.capitalize()} accuracy: {acc_selection:.2f}. ****")
        if acc_test is not None:
            print(f"**** Test accuracy: {acc_test:.2f}. ****")
        print("")
        if args.run_manifest:
            metrics = {
                "selection_split": selection_split,
                "zero_shot_selection_accuracy": zs_selection_acc,
                "selection_accuracy": acc_selection,
                "test_accuracy": acc_test,
                "runtime_seconds": 0.0,
                "trainable_parameters": adapter_parameter_count,
            }
            record = build_run_record(
                args,
                getattr(args, "dataset_provenance", {}),
                metrics,
                status="eval_only",
            )
            write_jsonl_record(args.run_manifest, record)
        return

    mark_only_lora_as_trainable(clip_model)
    optimizer_params = get_lora_parameters(clip_model)
    trainable_parameters = count_named_adapter_parameters(clip_model.named_parameters())
    print(f"Number of trainable parameters: {trainable_parameters}")
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)

    print(f"\nUsing main loss function: {args.loss_fn.upper()}")
    classification_loss_fn = None
    center_loss_fn = None
    max_entropy_loss_fn = None
    optimizer_center = None

    if args.loss_fn == 'ce':
        classification_loss_fn = F.cross_entropy
    elif args.loss_fn == 'ce_ls':
        classification_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss_fn == 'arcface':
        classification_loss_fn = ArcFaceLoss(s=args.metric_s, m=args.metric_m).cuda()
    elif args.loss_fn == 'cosface':
        classification_loss_fn = CosFaceLoss(s=args.metric_s, m=args.metric_m).cuda()
    elif args.loss_fn == 'ce_center':
        classification_loss_fn = F.cross_entropy
        center_loss_fn = CenterLoss(num_classes=len(dataset.classnames), feat_dim=clip_model.visual.output_dim).cuda()
        optimizer_center = torch.optim.SGD(center_loss_fn.parameters(), lr=0.5)
    elif args.loss_fn == 'ce_maxent':
        classification_loss_fn = F.cross_entropy
        max_entropy_loss_fn = MaximumEntropyLoss().cuda()
    elif args.loss_fn == 'focal':
        print(f">> Using Focal Loss with gamma={args.gamma}")
        classification_loss_fn = FocalLoss(gamma=args.gamma).cuda()

    total_iters = args.n_iters * args.shots
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    print(f"\nStarting {args.adapter.upper()} fine-tuning with {args.loss_fn.upper()} loss for {total_iters} iterations...")
    start_time = time.time()

    scaler = torch.amp.GradScaler(device='cuda')
    count_iters = 0

    with tqdm(total=total_iters, initial=count_iters, desc="Training", unit="step") as progress_bar:
        while count_iters < total_iters:
            clip_model.train()
            loss_epoch, class_loss_epoch, ortho_loss_epoch, ortho_loss_raw_epoch, aux_loss_epoch = 0., 0., 0., 0., 0.
            acc_train = 0
            tot_samples = 0

            for i, (images, target) in enumerate(train_loader):
                images, target = images.cuda(), target.cuda()

                if args.encoder == 'text' or args.encoder == 'both':
                    template = dataset.template[0]
                    texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        texts = clip.tokenize(texts).cuda()
                        class_embeddings = clip_model.encode_text(texts)
                    text_features_train = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                if args.encoder == 'vision' or args.encoder == 'both':
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
                else:
                    with torch.no_grad():
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            image_features = clip_model.encode_image(images)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                cosine_similarity = image_features @ text_features_train.t()

                aux_loss = torch.tensor(0.0, device=images.device, dtype=torch.float32)

                if args.loss_fn in ['ce', 'ce_ls', 'ce_center', 'ce_maxent']:
                    class_loss = classification_loss_fn(logit_scale * cosine_similarity, target)
                else: # ArcFace, CosFace
                    class_loss = classification_loss_fn(cosine_similarity, target)

                if center_loss_fn is not None:
                    center_loss = center_loss_fn(image_features, target)
                    aux_loss += args.lambda_center * center_loss

                if max_entropy_loss_fn is not None:
                    max_entropy_loss = max_entropy_loss_fn(logit_scale * cosine_similarity)
                    aux_loss -= args.lambda_maxent * max_entropy_loss

                total_ortho_loss = torch.tensor(0.0, device=images.device, dtype=torch.float32)
                total_ortho_loss_raw = torch.tensor(0.0, device=images.device, dtype=torch.float32)
                if args.lambda_o > 0 and args.adapter in ['ohsinglora', 'gmhsinglora']:
                    for module in clip_model.modules():
                        if isinstance(module, LinearOHsingLoRA):
                            total_ortho_loss += calculate_ortho_loss(module.lora_A_heads, reduction=args.ortho_reduction)
                            total_ortho_loss_raw += calculate_ortho_loss(module.lora_A_heads, reduction='sum')

                loss = class_loss.float() + aux_loss + args.lambda_o * total_ortho_loss
                acc_train += cls_acc(logit_scale * cosine_similarity, target) * target.shape[0]
                loss_epoch += loss.item() * target.shape[0]
                class_loss_epoch += class_loss.item() * target.shape[0]
                ortho_loss_epoch += total_ortho_loss.item() * target.shape[0]
                ortho_loss_raw_epoch += total_ortho_loss_raw.item() * target.shape[0]
                aux_loss_epoch += aux_loss.item() * target.shape[0]
                tot_samples += target.shape[0]

                optimizer.zero_grad()
                if optimizer_center:
                    optimizer_center.zero_grad()

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=1.0)

                scaler.step(optimizer)
                if optimizer_center:
                    for param in center_loss_fn.parameters():
                        param.grad.data *= (1. / args.lambda_center)
                    scaler.step(optimizer_center)

                scaler.update()
                scheduler.step()

                count_iters += 1
                elapsed = time.time() - start_time
                steps_per_second = count_iters / elapsed if elapsed > 0 else 0.0
                remaining_steps = max(total_iters - count_iters, 0)
                eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0.0
                progress_bar.set_postfix({
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                    "elapsed": _format_duration(elapsed),
                    "eta": _format_duration(eta_seconds),
                })
                progress_bar.update(1)
                if count_iters >= total_iters:
                    break

            if count_iters < total_iters and tot_samples > 0:
                acc_train /= tot_samples
                loss_epoch /= tot_samples
                class_loss_epoch /= tot_samples
                ortho_loss_epoch /= tot_samples
                ortho_loss_raw_epoch /= tot_samples
                aux_loss_epoch /= tot_samples

                current_lr = scheduler.get_last_lr()[0]
                progress_bar.write(f'Iter {count_iters}/{total_iters} | LR: {current_lr:.6f}, '
                    f'Acc: {acc_train:.4f}, Total Loss: {loss_epoch:.4f} '
                    f'(Class: {class_loss_epoch:.4f}, Ortho-{args.ortho_reduction}: {ortho_loss_epoch:.4f}, '
                    f'Ortho-raw-sum: {ortho_loss_raw_epoch:.4f}, Aux: {aux_loss_epoch:.4f})')

    end_time = time.time()
    total_finetuning_time = end_time - start_time
    print(f"\nFine-tuning finished in {total_finetuning_time:.2f} seconds.")

    acc_selection = evaluate(args, clip_model, selection_loader, dataset)
    acc_test = evaluate(args, clip_model, test_loader, dataset) if report_test else None
    print(f"**** Final {selection_split} accuracy with {args.adapter.upper()}: {acc_selection:.2f}. ****")
    if acc_test is not None:
        print(f"**** Final test accuracy with {args.adapter.upper()}: {acc_test:.2f}. ****")
    print("")

    checkpoint_path = ""
    if args.save_path is not None:
        print(f"Saving {args.adapter.upper()} weights to {args.save_path}...")
        if args.adapter == 'lora':
            checkpoint_path = save_lora(args, list_adapter_layers)
        elif args.adapter in ['singlora', 'gmhsinglora', 'ohsinglora']:
            checkpoint_path = save_adapter(args, clip_model)

    if args.run_manifest:
        metrics = {
            "selection_split": selection_split,
            "zero_shot_selection_accuracy": zs_selection_acc,
            "selection_accuracy": acc_selection,
            "test_accuracy": acc_test,
            "fine_tuning_seconds": total_finetuning_time,
            "runtime_seconds": total_finetuning_time,
            "trainable_parameters": trainable_parameters,
            "train_total_iterations": total_iters,
            "ortho_reduction": getattr(args, "ortho_reduction", "sum"),
        }
        record = build_run_record(
            args,
            getattr(args, "dataset_provenance", {}),
            metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=file_sha256(checkpoint_path),
            status="completed",
        )
        write_jsonl_record(args.run_manifest, record)

    return acc_test if acc_test is not None else acc_selection
