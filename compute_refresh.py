"""Matched eight-workload compute benchmark. Never performs an optimizer update."""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from optimization_evidence import sha256, state_sha, tensor_sha, write_csv, write_json

VERSION = 'orthadapt.compute_refresh.v1.1'
GPU_UUID = 'GPU-a3e47412-65b6-9425-91a2-6d1d6d897dad'
DATASETS = ('eurosat', 'caltech101', 'dtd', 'fgvc', 'food101', 'oxford_flowers', 'oxford_pets', 'ucf101')
METHODS = ('lora', 'singlora', 'original', 'optimized')
PHASES = ('forward', 'backward', 'forward_backward', 'inference')
PARAMS = dict(lora=737280, singlora=368640, original=460800, optimized=460800)


def timing_settings(profile='full', smoke=False):
    profiles = {'full': dict(repetitions=5, warmup=20, calls=100),
                'economy': dict(repetitions=3, warmup=10, calls=30)}
    if profile not in profiles:
        raise ValueError('unknown timing profile')
    return dict(repetitions=1, warmup=2, calls=3) if smoke else dict(profiles[profile])


def cells(profile='full'):
    repetitions = timing_settings(profile)['repetitions']
    return [(d, m, p, r) for d in DATASETS for m in METHODS for p in PHASES for r in range(1, repetitions + 1)]


def audit_coverage(actual, profile='full'):
    actual = list(actual)
    if len(actual) != len(set(actual)) or set(actual) != set(cells(profile)):
        raise ValueError('duplicate/missing/unexpected compute cells')


def timing_summary(wall, cuda, batch, expected=100):
    a, b = np.asarray(wall), np.asarray(cuda)
    if a.shape != (expected,) or b.shape != (expected,) or batch <= 0:
        raise ValueError('timing coverage mismatch')
    if not (np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all()):
        raise ValueError('nonpositive/nonfinite timing')
    return dict(median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)),
        cuda_median_ms=float(np.median(b)), cuda_p95_ms=float(np.percentile(b, 95)),
        images_per_second=float(batch * 1000 / np.median(a)))


def parameter_digest(model):
    return state_sha({n: p for n, p in model.named_parameters() if p.requires_grad})


def assert_unchanged(model, before):
    if parameter_digest(model) != before:
        raise ValueError('trainable parameters changed without authorized training')


def convert_legacy(payload, expected, record, args):
    """Explicit user-authorized migration, never part of a silent loader fallback."""
    from adapter_checkpoint_v2 import adapter_metadata, validate_adapter_state_dict
    if record.get('status') != 'completed' or record['metrics'].get('train_total_iterations') != 2000:
        raise ValueError('completed 2000-step training evidence required')
    config = record['config']
    metadata = payload['metadata']
    if 'schema_version' in metadata:
        raise ValueError('migration is restricted to pre-schema legacy checkpoints')
    for key in ('adapter', 'r', 'alpha', 'params', 'position', 'encoder', 'backbone'):
        if metadata.get(key) != config.get(key):
            raise ValueError('contradictory top-level legacy metadata: ' + key)
    for key in ('num_heads', 'ramp_up_steps'):
        if key in metadata and metadata[key] != config.get(key):
            raise ValueError('contradictory optional legacy metadata: ' + key)
    for key, approved in (('routing_mode', 'input_conditioned'), ('routing_context_id', 'not_applicable')):
        if key in metadata and metadata[key] != approved:
            raise ValueError('unexpected legacy routing declaration: ' + key)
    embedded = payload['metadata'].get('ecr3', {}).get('config', {})
    for k in ('adapter', 'r', 'num_heads', 'alpha', 'ramp_up_steps', 'params', 'position', 'encoder',
              'backbone', 'dataset', 'shots', 'seed', 'lambda_o', 'n_iters'):
        if k not in config or embedded.get(k) != config[k]:
            raise ValueError('legacy metadata/manifest mismatch: ' + k)
    for k in ('adapter', 'r', 'num_heads', 'alpha', 'ramp_up_steps', 'params', 'position', 'encoder', 'backbone'):
        if config[k] != getattr(args, k):
            raise ValueError('legacy target mismatch: ' + k)
    if config['ramp_up_steps'] != 100 or config['shots'] != 4 or config['seed'] != 1 or config['lambda_o'] != .03:
        raise ValueError('not the approved saturated legacy protocol')
    weights = payload['weights']
    if any(k.endswith(('training_step', 'routing_context')) for k in weights):
        raise ValueError('partial/already-present buffer state cannot be inferred')
    added = sorted(k for k in expected if k.endswith(('training_step', 'routing_context')))
    if set(weights) != set(expected) - set(added):
        raise ValueError('legacy trainable key coverage mismatch')
    new_weights = {k: v.detach().cpu().clone() for k, v in weights.items()}
    for k in added:
        new_weights[k] = (torch.full_like(expected[k].cpu(), 100) if k.endswith('training_step')
                          else expected[k].detach().cpu().clone())
    validate_adapter_state_dict(expected, new_weights)
    if any(not torch.equal(weights[k].cpu(), new_weights[k]) for k in weights):
        raise ValueError('migration altered a historical weight')
    audit = dict(source_sha256=record['checkpoint']['sha256'], added_keys=added,
        trainable_tensors_bitwise_unchanged=True, ramp_state_inferred=True, inferred_training_step=100,
        inference='Saturated ramp from completed 2000-step manifest; exact historical counter not recovered',
        authorization='User approved separate audited conversion on 2026-10-18; no retraining')
    metadata = dict(copy.deepcopy(payload['metadata']), **adapter_metadata(args), conversion_audit=audit)
    return dict(weights=new_weights, metadata=metadata), audit


def validate_lora_weights(weights, layers):
    if set(weights) != {f'layer_{i}' for i in range(len(layers))}:
        raise ValueError('LoRA layer coverage mismatch')
    for i, layer in enumerate(layers):
        projections = weights[f'layer_{i}']
        if set(projections) != {'q_proj', 'k_proj', 'v_proj'}:
            raise ValueError('LoRA projection coverage mismatch')
        for name, tensors in projections.items():
            if set(tensors) != {'w_lora_A', 'w_lora_B'}:
                raise ValueError('LoRA tensor coverage mismatch')
            for key, value in tensors.items():
                if value.shape != getattr(getattr(layer, name), key).shape or not torch.isfinite(value).all():
                    raise ValueError('LoRA shape/nonfinite mismatch')


def args_for(dataset, method, out):
    from optimization_experiment import runtime_args
    args = runtime_args(dict(dataset=dataset, shots=4, seed=1, routing='dynamic'),
                        'combined' if method == 'optimized' else 'original', '/root/DATA', out)
    args.adapter = method if method in ('lora', 'singlora') else 'ohsinglora'
    args.lambda_o = .03 if args.adapter == 'ohsinglora' else 0.
    return args


def load_model(args, checkpoint, method):
    import clip
    from adapter_checkpoint_v2 import (adapter_state_dict, LEGACY_CHECKPOINT_SCHEMA_VERSION,
        validate_adapter_metadata, validate_adapter_state_dict, validate_saturated_ramp_state)
    from loralib.utils import apply_adapter, apply_lora, mark_only_lora_as_trainable
    model, preprocess = clip.load('ViT-B/16', device='cuda')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if args.adapter == 'lora':
        layers = apply_lora(args, model)
        for k in ('r', 'alpha', 'encoder', 'params', 'position'):
            if payload['metadata'].get(k) != getattr(args, k): raise ValueError('LoRA metadata mismatch: ' + k)
        validate_lora_weights(payload['weights'], layers)
        with torch.no_grad():
            for i, layer in enumerate(layers):
                for proj, tensors in payload['weights'][f'layer_{i}'].items():
                    for k, value in tensors.items():
                        target = getattr(getattr(layer, proj), k)
                        target.copy_(value)
                        if not torch.equal(target.cpu(), value.to(target.dtype)): raise ValueError('LoRA copy mismatch')
    else:
        apply_adapter(args, model)
        validate_adapter_metadata(payload['metadata'], args)
        expected = adapter_state_dict(model)
        if payload['metadata']['schema_version'] == LEGACY_CHECKPOINT_SCHEMA_VERSION:
            expected = {k: v for k, v in expected.items() if not k.endswith('routing_context')}
        validate_adapter_state_dict(expected, payload['weights'])
        validate_saturated_ramp_state(payload['weights'], 100)
        if not all(torch.isfinite(v).all() for v in payload['weights'].values()): raise ValueError('nonfinite checkpoint')
        result = model.load_state_dict(payload['weights'], strict=False)
        if result.unexpected_keys: raise ValueError('unexpected checkpoint keys')
        actual = model.state_dict()
        if any(not torch.equal(actual[k].cpu(), v.to(actual[k].dtype)) for k, v in payload['weights'].items()):
            raise ValueError('checkpoint copy mismatch')
    mark_only_lora_as_trainable(model)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != PARAMS[method]:
        raise ValueError('trainable parameter count mismatch')
    return model.cuda(), preprocess


def validate_gpu_mapping(physical_uuids, visible_count, current_device, mapping):
    if physical_uuids != [GPU_UUID] or visible_count != 1 or current_device != 0:
        raise ValueError('exactly one pinned physical and visible CUDA GPU required')
    if mapping not in (None, '0', GPU_UUID):
        raise ValueError('unverified CUDA_VISIBLE_DEVICES remapping')


def environment():
    from reviewer4_provenance import capture_environment
    env = capture_environment()
    if env['gpu_uuid'] != GPU_UUID or abs(float(env['power_limit_watts']) - 250) > .01:
        raise ValueError('GPU UUID/power mismatch')
    if env['pytorch'] != '2.5.1+cu124' or env['cuda_runtime'] != '12.4' or env['cudnn'] != 90100:
        raise ValueError('software stack mismatch')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    import os
    physical = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True)
    validate_gpu_mapping([p.strip() for p in physical.splitlines() if p.strip()],
        torch.cuda.device_count(), torch.cuda.current_device(), os.environ.get('CUDA_VISIBLE_DEVICES'))
    env['actual_cuda_device'] = 0
    env['physical_uuid_mapping_verified'] = True
    if any(int(p.strip()) != os.getpid() for p in processes.splitlines() if p.strip().isdigit()):
        raise ValueError('competing GPU process')
    return env


def inputs(dataset, reference, preprocess, out):
    import clip
    from PIL import Image
    from datasets import build_dataset
    from ecr3_provenance import dataset_provenance
    from reviewer4_provenance import validate_split_reference
    from optimization_experiment import seed_all
    seed_all(1)
    ds = build_dataset(dataset, '/root/DATA', 4, preprocess)
    prov = dataset_provenance(ds)
    identity = validate_split_reference(dataset, 4, 1, prov, [reference])
    train, test = ds.train_x[:32], ds.test[:256]
    if len(train) != 32 or len(test) != 256: raise ValueError('insufficient full fixed batch')
    def batch(records):
        tensors = []
        for x in records:
            with Image.open(x.impath) as img: tensors.append(preprocess(img.convert('RGB')))
        return torch.stack(tensors), torch.tensor([x.label for x in records], dtype=torch.long)
    x, y = batch(train); xx, yy = batch(test)
    tokens = clip.tokenize([ds.template[0].format(n.replace('_', ' ')) for n in ds.classnames])
    audit = dict(dataset=dataset, split_identity=identity['split_identity'],
        train_ids=[x.impath for x in train], test_ids=[x.impath for x in test],
        file_sha256={x.impath: sha256(x.impath) for x in train + test},
        train_image_sha256=tensor_sha(x), test_image_sha256=tensor_sha(xx), token_sha256=tensor_sha(tokens),
        train_label_sha256=tensor_sha(y), test_label_sha256=tensor_sha(yy), classes=len(tokens),
        source='first frozen train32/test256 records; deterministic CLIP preprocessing; not accuracy evaluation')
    write_json(Path(out) / f'inputs_{dataset}.json', audit)
    return x.cuda(), y.cuda(), xx.cuda(), yy.cuda(), tokens.cuda(), audit


def forward_objective(model, images, labels, tokens):
    from loralib.layers_OH_singlora import LinearOHsingLoRA, calculate_ortho_loss
    from optimization_experiment import normalize_features
    with torch.amp.autocast('cuda', dtype=torch.float16): text = model.encode_text(tokens)
    text = normalize_features(text)
    with torch.amp.autocast('cuda', dtype=torch.float16): image = model.encode_image(images)
    logits = 100 * (normalize_features(image) @ text.t())
    loss = F.cross_entropy(logits, labels).float()
    ortho = torch.zeros((), device=images.device, dtype=torch.float32)
    raw = torch.zeros_like(ortho)
    for module in model.modules():
        if isinstance(module, LinearOHsingLoRA):
            ortho = ortho + calculate_ortho_loss(module.lora_A_heads, 'mean')
            raw = raw + calculate_ortho_loss(module.lora_A_heads, 'sum')
    return logits, loss + .03 * ortho, raw


def check_finite(model, result, gradients):
    values = [x for x in result if x is not None]
    if gradients:
        params = [p for p in model.parameters() if p.requires_grad]
        if any(p.grad is None for p in params): raise ValueError('missing trainable gradient')
        values += [p.grad for p in params]
    if not all(torch.isfinite(x).all() for x in values): raise ValueError('nonfinite output/loss/gradient')


def measure_phase(model, phase, data, warmup, calls, trace_out=None):
    from optimization_experiment import normalize_features, backend_counts
    from optimization_profile import reset_counts, audit_backend_counts, audit_sdpa_operators
    x, y, xx, yy, tokens, _ = data
    training = phase != 'inference'
    model.train(training)
    cached = None
    classifier_ms = None
    if not training:
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16): text = model.encode_text(tokens)
        cached = normalize_features(text)
        torch.cuda.synchronize(); classifier_ms = (time.perf_counter() - start) * 1000
    def forward(): return forward_objective(model, x, y, tokens)
    def prepare():
        model.zero_grad(set_to_none=True)
        return forward() if phase == 'backward' else None
    def call(prepared):
        if phase == 'inference':
            with torch.inference_mode():
                with torch.amp.autocast('cuda', dtype=torch.float16): image = model.encode_image(xx)
                return (normalize_features(image) @ cached.t(),)
        result = prepared if phase == 'backward' else forward()
        if phase in ('backward', 'forward_backward'): result[1].backward()
        return result
    for _ in range(warmup):
        prepared = prepare(); result = call(prepared); del prepared, result
    torch.cuda.synchronize(); model.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache(); reset_counts(model)
    torch.cuda.reset_peak_memory_stats()
    wall, cuda, allocated, reserved = [], [], [], []
    for _ in range(calls):
        torch.cuda.reset_peak_memory_stats()
        prepared = prepare()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter(); start.record(); result = call(prepared); end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t) * 1000); cuda.append(start.elapsed_time(end))
        # Snapshot compute peaks BEFORE allocating finite-check temporaries.
        allocated.append(torch.cuda.max_memory_allocated()); reserved.append(torch.cuda.max_memory_reserved())
        # Deliberately outside timing; reject any nonfinite call, not merely the last one.
        check_finite(model, result, phase in ('backward', 'forward_backward'))
        del prepared, result
    metrics = dict(timing_summary(wall, cuda, 256 if not training else 32, calls),
        raw_wall_ms=wall, raw_cuda_ms=cuda, peak_allocated_bytes=max(allocated),
        peak_reserved_bytes=max(reserved), raw_peak_allocated_bytes=allocated, raw_peak_reserved_bytes=reserved,
        memory_scope='per-call peak snapshot before validation; reserved includes existing warmed allocator cache',
        backend_counts=backend_counts(model),
        classifier_construction_ms=classifier_ms, backward_memory_includes_prepared_graph=(phase == 'backward'))
    optimized = any(getattr(m, 'attention_backend', '') == 'sdpa' for m in model.modules())
    if optimized: audit_backend_counts(metrics['backend_counts'], 'combined')
    prepared = prepare(); result = call(prepared)
    check_finite(model, result, phase in ('backward', 'forward_backward'))
    del prepared, result
    if trace_out is not None:
        trace_out.mkdir(parents=True)
        prepared = prepare(); torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                     record_shapes=True, with_flops=True) as prof:
            result = call(prepared); torch.cuda.synchronize()
        events = [dict(operator=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
            self_device_us=getattr(e, 'self_device_time_total', 0), supported_flops=int(e.flops or 0))
                  for e in prof.key_averages()]
        write_csv(trace_out / 'operators.csv', events)
        if phase in ('inference', 'forward_backward'): prof.export_chrome_trace(str(trace_out / 'trace.json'))
        # Backward-only trace has no forward SDPA operator by design: require fused backward kernel.
        kernel = audit_sdpa_operators([e['operator'] for e in events], 'combined' if optimized else 'original')
        write_json(trace_out / 'audit.json', dict(kernel, supported_operator_flops=sum(e['supported_flops'] for e in events),
            total_flops=False, unsupported_operator_flops_inferred=False))
        del prepared, result
    model.zero_grad(set_to_none=True)
    return metrics


def run(opts):
    import clip
    from optimization_experiment import seed_all
    out = Path(opts.out); out.mkdir(parents=True, exist_ok=False)
    spec = json.loads(Path(opts.inventory).read_text())
    profile = opts.timing_profile
    settings = timing_settings(profile, opts.smoke)
    before = environment(); write_json(out / 'environment_before.json', before)
    write_json(out / 'protocol.json', dict(version=VERSION, smoke=opts.smoke, datasets=DATASETS,
        methods=METHODS, phases=PHASES, timing_profile=profile, **settings, optimizer_updates=0, training_runs=0,
        original_numerical_gate_passed=False, previous_all_arm_accuracy_gate_passed=False,
        accuracy_evaluations=0, protocol_markdown_sha256=sha256(opts.protocol),
        git_revision=opts.git_revision,
        source_inventory_sha256=sha256(opts.source_inventory), checkpoint_inventory_sha256=sha256(opts.inventory)))
    source_inventory = json.loads(Path(opts.source_inventory).read_text())
    source_root = Path(__file__).parent
    def source_check():
        for row in source_inventory:
            if sha256(source_root / row['path']) != row['sha256']: raise ValueError('source changed: ' + row['path'])
    source_check()
    for entry in spec:
        if sha256(entry['path']) != entry['sha256']: raise ValueError('checkpoint SHA mismatch')
    rows = []
    datasets = DATASETS[:1] if opts.smoke else DATASETS
    for d in datasets:
        reference = next(e['record'] for e in spec if e['dataset'] == d and e['method'] == 'original')
        _, preprocess = clip.load('ViT-B/16', device='cpu')
        data = inputs(d, reference, preprocess, out)
        for rep in range(1, settings['repetitions'] + 1):
            current = environment()
            if current['software_stack_id'] != before['software_stack_id']: raise ValueError('software identity changed')
            write_json(out / f'environment_{d}_rep{rep}.json', current)
            order = METHODS[(rep - 1) % 4:] + METHODS[:(rep - 1) % 4]
            for method in order:
                seed_all(1)
                entry = next(e for e in spec if e['dataset'] == d and e['method'] == method)
                model, _ = load_model(args_for(d, method, out), entry['path'], method)
                digest = parameter_digest(model)
                for phase in PHASES:
                    directory = out / 'cells' / d / f'{method}_{phase}_rep{rep}'
                    directory.mkdir(parents=True)
                    metrics = measure_phase(model, phase, data, settings['warmup'], settings['calls'],
                        trace_out=directory / 'profile' if rep == 1 else None)
                    assert_unchanged(model, digest)
                    row = dict(dataset=d, method=method, phase=phase, repetition=rep, checkpoint_sha256=entry['sha256'],
                        timing_profile=profile, warmup_calls=settings['warmup'], measured_calls=settings['calls'],
                        trainable_parameters=PARAMS[method], input_audit_sha256=sha256(out / f'inputs_{d}.json'),
                        gpu_uuid=before['gpu_uuid'], power_watts=250, software_stack_id=before['software_stack_id'], **metrics)
                    write_json(directory / 'measurement.json', row)
                    with (out / 'timings.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(row, allow_nan=False) + '\n'); f.flush()
                    rows.append(row)
                    print('CELL_COMPLETE', d, method, phase, rep, round(row['median_ms'], 3), flush=True)
                del model; gc.collect(); torch.cuda.empty_cache()
        del data; gc.collect(); torch.cuda.empty_cache()
    if not opts.smoke: audit_coverage(((r['dataset'], r['method'], r['phase'], r['repetition']) for r in rows), profile)
    source_check()
    for entry in spec:
        if sha256(entry['path']) != entry['sha256']: raise ValueError('checkpoint changed during compute')
    write_json(out / 'environment_after.json', environment())
    write_csv(out / 'timing_summary.csv', [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows])
    write_json(out / 'completed.json', dict(passed=True, rows=len(rows), smoke=opts.smoke, optimizer_updates=0,
        timing_profile=profile,
        full_experiment=False, accuracy_preservation_proven_for_eight_datasets=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'inventory', 'protocol', 'source-inventory'): parser.add_argument('--' + name, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--timing-profile', choices=('full', 'economy'), default='full')
    parser.add_argument('--git-revision', default=None, help='Verified staged source commit; null for old portable snapshots')
    opts = parser.parse_args()
    try: run(opts)
    except Exception as exc:
        traceback.print_exc()
        if Path(opts.out).is_dir():
            write_json(Path(opts.out) / 'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
