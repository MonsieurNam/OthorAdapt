"""Matched production-path CLIP timings; no monkeypatches or optimizer updates."""
from contextlib import contextmanager, nullcontext
import gc
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from optimization_evidence import ARMS, DATASETS, tensor_sha, write_csv, write_json
from optimization_experiment import (backend_counts, build_data, checked_checkpoint,
    configure_backend, find_reference, frozen_records, load_model, runtime_args, seed_all, normalize_features)


def audit_backend_counts(counts, arm):
    if ARMS[arm][1] == 'sdpa':
        if counts.get('sdpa', 0) == 0 or counts.get('manual', 0) or counts.get('fallback', 0):
            raise ValueError('SDPA path silently used manual fallback')
        if counts.get('mask_validations', 0):
            raise ValueError('mask was revalidated during steady-state timing')


def reset_counts(model):
    from loralib.layers_singlora import PlainMultiheadAttentionAdapter
    for module in model.modules():
        if isinstance(module, PlainMultiheadAttentionAdapter):
            for key in module.execution_counts:
                module.execution_counts[key] = 0


@contextmanager
def full_precision(model):
    dtypes = {n: p.dtype for n, p in model.named_parameters()}
    model.float()
    try:
        yield
    finally:
        model.zero_grad(set_to_none=True)
        for n, p in model.named_parameters():
            p.data = p.data.to(dtypes[n])


def objective(model, images, tokens, labels, cached=None, amp=True):
    from loralib.layers_OH_singlora import LinearOHsingLoRA, calculate_ortho_loss
    model.zero_grad(set_to_none=True)
    if cached is None:
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
            text = model.encode_text(tokens)
        text = normalize_features(text)
    else:
        text = cached
    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
        image = model.encode_image(images)
    image = normalize_features(image)
    cosine_similarity = image @ text.t()
    logits = 100 * cosine_similarity
    if model.training:
        loss = F.cross_entropy(logits, labels).float()
        # Retain both bookkeeping computations present in the training source.
        matrices = [m.lora_A_heads for m in model.modules() if isinstance(m, LinearOHsingLoRA)]
        ortho = sum(calculate_ortho_loss(a, 'mean') for a in matrices)
        raw = sum(calculate_ortho_loss(a, 'sum') for a in matrices)
        loss = loss + .03 * ortho
        loss.backward()
        return logits, loss, raw
    return logits, None, None


def enforce_numerical_policy(rows, strict=True):
    if not rows or not all(r['finite'] for r in rows):
        raise RuntimeError('nonfinite full-CLIP numerical smoke; always a hard failure')
    if strict and not all(r['passed'] for r in rows):
        raise RuntimeError('full-CLIP numerical smoke failed')


def numerical_smoke(model, images, tokens, labels, out, strict=True):
    ramp = {n: b.clone() for n, b in model.named_buffers() if n.endswith('training_step')}
    rows = []
    for amp in (False, True):
        context = nullcontext() if amp else full_precision(model)
        with context:
            for scope in ('training', 'inference'):
                training = scope == 'training'
                model.train(training)
                reference_logits = reference_grads = None
                for arm in ARMS:
                    with torch.no_grad():
                        for name, buf in model.named_buffers():
                            if name in ramp:
                                buf.copy_(ramp[name])
                    configure_backend(model, arm)
                    with nullcontext() if training else torch.no_grad():
                        cached = None
                        if not training:
                            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
                                encoded = model.encode_text(tokens)
                            cached = normalize_features(encoded)
                        logits, loss, _ = objective(model, images[:32] if training else images,
                            tokens, labels[:32] if training else labels, cached=cached, amp=amp)
                    grads = {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters()
                             if p.requires_grad and p.grad is not None}
                    logits = logits.detach().float().cpu()
                    if reference_logits is None:
                        reference_logits, reference_grads = logits, grads
                    atol, rtol = (.02, .02) if amp else (1e-5, 1e-4)
                    output_passed = torch.allclose(reference_logits, logits, atol=atol, rtol=rtol)
                    passed = bool(output_passed and set(reference_grads) == set(grads) and
                        all(torch.allclose(reference_grads[n], value, atol=atol, rtol=rtol) for n, value in grads.items()))
                    finite = bool(torch.isfinite(logits).all() and all(torch.isfinite(g).all() for g in grads.values()))
                    rows.append(dict(precision='fp16' if amp else 'fp32', scope=scope, arm=arm,
                        batch=32 if training else 256, text_classes=len(tokens),
                        max_abs_logit_error=float((reference_logits-logits).abs().max()),
                        loss=float(loss.detach()) if loss is not None else None,
                        passed=passed, finite=finite, output_passed=bool(output_passed),
                        gradient_tensor_failures=sum(not torch.allclose(reference_grads[n],g,atol=atol,rtol=rtol)
                                                    for n,g in grads.items()),
                        tolerance_policy='strict' if strict else 'report_only_hard_fail_nonfinite',
                        atol=atol, rtol=rtol))
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, b in model.named_buffers():
            if n in ramp:
                b.copy_(ramp[n])
    write_json(Path(out) / 'full_clip_numerical_smoke.json', rows)
    enforce_numerical_policy(rows, strict)
    return rows


def measure(call, model):
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    reset_counts(model)
    torch.cuda.reset_peak_memory_stats()
    wall, cuda = [], []
    for _ in range(100):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter()-t)*1000)
        cuda.append(start.elapsed_time(end))
    return dict(median_ms=float(np.median(wall)), p95_ms=float(np.percentile(wall, 95)),
        cuda_median_ms=float(np.median(cuda)), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), raw_wall_ms=wall, raw_cuda_ms=cuda)


def audit_sdpa_operators(names, arm):
    fused = [name for name in names if any(key in name.lower() for key in (
        'flash_attention', 'efficient_attention', 'cudnn_attention'))]
    math_fallback = [name for name in names if '_scaled_dot_product_attention_math' in name]
    if ARMS[arm][1] == 'sdpa' and (not fused or math_fallback):
        raise ValueError('SDPA fused kernel missing or math fallback detected')
    return dict(fused_attention_operators=fused, math_fallback_operators=math_fallback)


def trace_call(call, out, arm):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        call()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(Path(out) / 'trace.json'))
    rows = []
    for e in prof.key_averages():
        rows.append(dict(operator=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
                         self_device_us=getattr(e, 'self_device_time_total', 0)))
    write_csv(Path(out) / 'operator_table.csv', rows)
    write_json(Path(out) / 'kernel_audit.json', audit_sdpa_operators([r['operator'] for r in rows], arm))


def profile_suite(opts, out, smoke_only=False, strict_numerical=True):
    import clip
    rows = []
    for dataset_name in DATASETS:
        for routing in ('dynamic', 'static'):
            cell = dict(dataset=dataset_name, shots=4, seed=1, routing=routing)
            reference = find_reference(frozen_records(opts.evidence_root, routing), cell)
            checkpoint = checked_checkpoint(opts.evidence_root, reference)
            args = runtime_args(cell, 'original', opts.data_root, out)
            seed_all(1)
            base, preprocess = clip.load(args.backbone, device='cpu')
            dataset, loader, _ = build_data(args, preprocess, reference)
            del base
            model, _ = load_model(args, checkpoint)
            images_cpu, labels_cpu = next(iter(loader))
            if len(images_cpu) != 256:
                raise ValueError('full inference batch required')
            tokens_cpu = clip.tokenize([dataset.template[0].format(n.replace('_', ' ')) for n in dataset.classnames])
            images, labels, tokens = images_cpu.cuda(), labels_cpu.cuda(), tokens_cpu.cuda()
            identity = dict(image_sha256=tensor_sha(images_cpu), token_sha256=tensor_sha(tokens_cpu),
                input_kind='fixed real test images for compute; accuracy evaluated separately')
            case = Path(out) / f'{dataset_name}_{routing}'
            case.mkdir()
            write_json(case / 'inputs.json', identity)
            numerical_smoke(model, images, tokens, labels, case, strict=strict_numerical)
            if smoke_only:
                del model, images, labels, tokens
                gc.collect(); torch.cuda.empty_cache()
                continue
            for scope in ('training', 'inference'):
                training = scope == 'training'
                batch = 32 if training else 256
                x, y = images[:batch], labels[:batch]
                for rep in range(1, 6):
                    for arm in (tuple(ARMS) if rep % 2 else tuple(reversed(ARMS))):
                        configure_backend(model, arm)
                        model.eval()
                        # no_grad, not inference_mode: CLIP's fixed causal mask keeps a version counter.
                        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                            encoded = model.encode_text(tokens) if not training else None
                        cached = normalize_features(encoded).detach() if encoded is not None else None
                        model.train(training)
                        def call():
                            with nullcontext() if training else torch.inference_mode():
                                return objective(model, x, tokens, y, cached=cached)
                        stats = measure(call, model)
                        counts = backend_counts(model)
                        audit_backend_counts(counts, arm)
                        raw = dict(dataset=dataset_name, routing=routing, scope=scope, arm=arm,
                            repetition=rep, batch=batch, images_per_second=batch/(stats['median_ms']/1000),
                            backend_counts=counts, **identity, **stats)
                        write_json(case / f'{scope}_{arm}_rep{rep}.json', raw)
                        row = {k: v for k, v in raw.items() if k not in ('raw_wall_ms', 'raw_cuda_ms', 'backend_counts')}
                        row.update({f'backend_{k}': v for k, v in counts.items()})
                        rows.append(row)
                        print('PROFILE', dataset_name, routing, scope, arm, rep, stats['median_ms'], flush=True)
                        if rep == 1:
                            trace_dir = case / f'trace_{scope}_{arm}'
                            trace_dir.mkdir()
                            trace_call(call, trace_dir, arm)
                model.zero_grad(set_to_none=True)
            del model, images, labels, tokens, cached
            gc.collect(); torch.cuda.empty_cache()
    if not smoke_only:
        write_csv(Path(out) / 'timings.csv', rows)
