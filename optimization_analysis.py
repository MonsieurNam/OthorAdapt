"""Independent coverage/gate aggregation for the bounded optimization evidence."""
import csv
import json
from pathlib import Path
import statistics
import math

import numpy as np

from optimization_evidence import (ARMS, DATASETS, GPU_UUID, bounded_cells, paired_speed_gate,
    sha256, validate_unique_coverage, write_csv, write_json, compare_predictions, state_sha)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_prediction_group(directory, arms):
    directory = Path(directory)
    result = []
    reference = identity = None
    all_disagreements = set()
    disagreeing_arms = []
    for arm in arms:
        case = directory / arm
        with np.load(case / 'predictions.npz', allow_pickle=False) as package:
            data = {k: package[k] for k in ('logits', 'labels', 'sample_ids', 'prediction', 'margin')}
        inputs = _json(case / 'inputs.json')
        reported = _json(case / 'comparison.json')
        if reference is None:
            reference, identity = data, inputs
        if not np.array_equal(data['labels'], reference['labels']) or not np.array_equal(data['sample_ids'], reference['sample_ids']):
            raise ValueError('raw prediction sample/label identity mismatch')
        if len(set(data['sample_ids'])) != len(data['labels']):
            raise ValueError('duplicate raw test samples')
        if inputs['image_batch_sha256'] != identity['image_batch_sha256'] or inputs['token_sha256'] != identity['token_sha256']:
            raise ValueError('raw eval image/token hashes mismatch')
        if not np.array_equal(data['prediction'], data['logits'].argmax(1)):
            raise ValueError('saved predictions do not match logits')
        actual = compare_predictions(reference['logits'], data['logits'], data['labels'])
        for key in ('passed', 'accuracy_delta_pp', 'agreement_percent'):
            if isinstance(actual[key], bool):
                equal = actual[key] == reported[key]
            else:
                equal = abs(actual[key]-float(reported[key])) < 1e-8
            if not equal:
                raise ValueError('CSV/JSON accuracy verdict contradicts raw logits')
        if ARMS[arm][1] == 'sdpa':
            counts = inputs['backend_counts']
            if not counts['sdpa'] or counts['manual'] or counts['fallback']:
                raise ValueError('raw accuracy path used manual SDPA fallback')
        disagreements = np.flatnonzero(reference['prediction'] != data['prediction'])
        if len(disagreements):
            all_disagreements.update(disagreements.tolist())
            disagreeing_arms.append(arm)
        result.append(dict(arm=arm, **actual))
    if all_disagreements:
        selected = sorted(all_disagreements)
        fp32_reference = fp32_inputs_reference = None
        diagnostic = _json(directory / 'fp32_disagreement_comparison.json')
        for arm in ('original', *disagreeing_arms):
            case = directory / f'fp32_{arm}'
            with np.load(case / 'predictions.npz', allow_pickle=False) as p:
                data = {k:p[k] for k in ('logits','labels','sample_ids','prediction')}
            inputs = _json(case / 'inputs.json')
            if (not np.array_equal(data['sample_ids'], reference['sample_ids'][selected])
                    or not np.array_equal(data['labels'], reference['labels'][selected])
                    or not np.isfinite(data['logits']).all()
                    or len(set(data['sample_ids'])) != len(selected)
                    or not np.array_equal(data['prediction'], data['logits'].argmax(1))
                    or inputs['precision'] != 'FP32'
                    or inputs['token_sha256'] != identity['token_sha256']):
                raise ValueError('FP32 disagreement sample coverage/identity/precision mismatch')
            if fp32_reference is None:
                fp32_reference, fp32_inputs_reference = data, inputs
            elif inputs['image_batch_sha256'] != fp32_inputs_reference['image_batch_sha256']:
                raise ValueError('FP32 paired image hashes mismatch')
            if arm != 'original':
                comparison = compare_predictions(fp32_reference['logits'],data['logits'],data['labels'])
                if comparison != diagnostic[arm]:
                    raise ValueError('FP32 reported comparison contradicts raw logits')
    return result


def audit_raw_evidence(out, old, new, timings, numerical_phases=('smoke','profile'), strict_numerical=True):
    import torch
    from adapter_checkpoint_v2 import validate_saturated_ramp_state
    from optimization_experiment import checked_checkpoint, find_reference, frozen_records, read_rows, runtime_args
    from optimization_profile import audit_sdpa_operators
    from loralib.utils import adapter_checkpoint_path
    from adapter_checkpoint_v2 import validate_adapter_metadata
    out = Path(out)
    protocol = _json(out / 'protocol.json')
    root = Path(protocol['evidence_root'])
    for cell in bounded_cells():
        checkpoint = checked_checkpoint(root, find_reference(frozen_records(root, cell['routing']), cell))
        rows = [r for r in old if all(r[k] == cell[k] for k in cell)]
        if len(rows) != 4 or any(r['checkpoint_sha256'] != sha256(checkpoint) for r in rows):
            raise ValueError('old checkpoint hash or row consistency failure')
        group = out / 'old-eval' / f"{cell['dataset']}_{cell['routing']}_{cell['shots']}shot"
        recomputed = audit_prediction_group(group, tuple(ARMS))
        for actual in recomputed:
            row = next(r for r in rows if r['arm'] == actual['arm'])
            if abs(float(row['accuracy_delta_pp'])-actual['accuracy_delta_pp']) > 1e-8 or (row['passed'] == 'True') != actual['passed']:
                raise ValueError('aggregate old accuracy CSV contradicts raw evidence')
    for dataset in DATASETS:
        initial_reference = batch_reference = None
        accuracies = []
        for arm in ('original', 'combined'):
            case = out / 'train' / f'{dataset}_{arm}'
            initial = _json(case / 'audit' / 'initial_state.json')
            steps = read_rows(case / 'audit' / 'steps.jsonl')
            if len(steps) != 2000 or [s['step'] for s in steps] != list(range(2000)):
                raise ValueError('raw optimizer-step coverage failure')
            batches = [(r['image_sha256'], r['label_sha256']) for r in steps]
            if initial_reference is None:
                initial_reference, batch_reference = initial, batches
            elif initial != initial_reference or batches != batch_reference:
                raise ValueError('raw paired initialization/augmentation mismatch')
            if initial['trainable_parameters'] != 460800:
                raise ValueError('raw training parameter count mismatch')
            audit = _json(case / 'training_audit.json')
            args = runtime_args(dict(dataset=dataset, shots=4, seed=1, routing='dynamic'), arm, protocol['data_root'], case)
            checkpoint = Path(adapter_checkpoint_path(args))
            if sha256(checkpoint) != audit['checkpoint_sha256']:
                raise ValueError('new checkpoint SHA mismatch')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            validate_adapter_metadata(payload['metadata'], args)
            validate_saturated_ramp_state(payload['weights'], 100)
            if any(not bool(torch.isfinite(t).all()) for t in payload['weights'].values()):
                raise ValueError('nonfinite saved checkpoint')
            if {int(t.item()) for n, t in payload['weights'].items() if n.endswith('training_step')} != {2000}:
                raise ValueError('saved ramp counters do not equal 2000 forwards')
            rows = [r for r in new if r['dataset'] == dataset and r['training_arm'] == arm]
            if len(rows) != 2 or any(r['checkpoint_sha256'] != sha256(checkpoint) for r in rows):
                raise ValueError('new checkpoint evaluation hash inconsistency')
            actual = audit_prediction_group(case / 'final_eval', ('original', 'combined'))
            for a in actual:
                row = next(r for r in rows if r['arm'] == a['arm'])
                if abs(float(row['accuracy_delta_pp'])-a['accuracy_delta_pp']) > 1e-8 or (row['passed'] == 'True') != a['passed']:
                    raise ValueError('new accuracy CSV contradicts raw logits')
            accuracies.append(actual[0]['reference_accuracy'])
        screen = _json(out / 'train' / f'{dataset}_training_screen.json')
        delta = accuracies[1]-accuracies[0]
        if abs(screen['accuracy_delta_pp']-delta) > 1e-8 or screen['passed'] != (abs(delta) <= .5):
            raise ValueError('training screen contradicts raw predictions')
    for dataset in DATASETS:
        for routing in ('dynamic', 'static'):
            for phase in numerical_phases:
                case = out / phase / f'{dataset}_{routing}'
                smoke = _json(case / 'full_clip_numerical_smoke.json')
                validate_unique_coverage(smoke, [(p,s,a) for p in ('fp32','fp16') for s in ('training','inference') for a in ARMS],
                    ('precision','scope','arm'))
                if not all((r['passed'] if strict_numerical else r.get('finite',False))
                           and r['batch'] == (32 if r['scope']=='training' else 256) for r in smoke):
                    raise ValueError('numerical smoke rejected or wrong batch')
            case = out / 'profile' / f'{dataset}_{routing}'
            identity = _json(case / 'inputs.json')
            for scope in ('training', 'inference'):
                for arm in ARMS:
                    trace = case / f'trace_{scope}_{arm}'
                    kernel = _json(trace / 'kernel_audit.json')
                    trace_data = _json(trace / 'trace.json')
                    events = trace_data['traceEvents']
                    if not any(e.get('cat') == 'kernel' for e in events):
                        raise ValueError('profiler trace contains no CUDA kernels')
                    operators = _read_csv(trace / 'operator_table.csv')
                    actual = audit_sdpa_operators([r['operator'] for r in operators], arm)
                    if actual != kernel:
                        raise ValueError('kernel audit contradicts operator table')
                    for rep in range(1,6):
                        raw = _json(case / f'{scope}_{arm}_rep{rep}.json')
                        row = next(r for r in timings if (r['dataset'],r['routing'],r['scope'],r['arm'],r['repetition']) ==
                                   (dataset,routing,scope,arm,rep))
                        if raw['image_sha256'] != identity['image_sha256'] or raw['token_sha256'] != identity['token_sha256']:
                            raise ValueError('profile input identity mismatch')
                        if len(raw['raw_wall_ms']) != 100 or len(raw['raw_cuda_ms']) != 100:
                            raise ValueError('profile call coverage failure')
                        if any(not math.isfinite(x) or x <= 0 for x in raw['raw_wall_ms'] + raw['raw_cuda_ms']):
                            raise ValueError('invalid raw timings')
                        if abs(float(row['median_ms'])-float(np.median(raw['raw_wall_ms']))) > 1e-8:
                            raise ValueError('profile CSV median contradicts raw timing')
                        counts=raw['backend_counts']
                        if ARMS[arm][1]=='sdpa' and (not counts['sdpa'] or counts['fallback'] or counts['manual'] or counts['mask_validations']):
                            raise ValueError('SDPA timing fallback/revalidation')
    return dict(raw_prediction_groups=12, training_traces=4, profile_traces=32,
                numerical_checks=64*len(numerical_phases), checked_old_checkpoint_hashes=8, checked_new_checkpoint_hashes=4)


def summarize_profile(rows):
    grouped = {}
    for row in rows:
        key = (row['dataset'], row['routing'], row['scope'])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, group in grouped.items():
        arms = sorted({r['arm'] for r in group})
        validate_unique_coverage(group, [(a, r) for a in arms for r in range(1, 6)], ('arm', 'repetition'))
        reference = sorted([r for r in group if r['arm'] == 'original'], key=lambda r: r['repetition'])
        if len(reference) != 5:
            raise ValueError('missing reference repetitions')
        for arm in arms:
            candidates = sorted([r for r in group if r['arm'] == arm], key=lambda r: r['repetition'])
            gate = paired_speed_gate([float(r['median_ms']) for r in reference],
                                     [float(r['median_ms']) for r in candidates])
            result.append(dict(dataset=key[0], routing=key[1], scope=key[2], arm=arm,
                median_ms=statistics.median(float(r['median_ms']) for r in candidates),
                p95_ms=statistics.median(float(r['p95_ms']) for r in candidates),
                peak_allocated_GiB=statistics.median(float(r['peak_allocated_bytes']) for r in candidates) / 2**30,
                peak_reserved_GiB=statistics.median(float(r['peak_reserved_bytes']) for r in candidates) / 2**30,
                median_latency_reduction_percent=gate['median_reduction_percent'],
                faster_repetitions=gate['faster_repetitions'], speed_gate_passed=gate['passed']))
    return result


def _read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def analyze(out):
    out = Path(out)
    old = _read_csv(out / 'old-eval' / 'accuracy_comparison.csv')
    new = _read_csv(out / 'train' / 'new_checkpoint_accuracy.csv')
    timings = _read_csv(out / 'profile' / 'timings.csv')
    for row in old + new:
        row['shots'], row['seed'] = int(row['shots']), int(row['seed'])
    expected_old = [(c['dataset'], c['routing'], c['shots'], c['seed'], arm)
                    for c in bounded_cells() for arm in ARMS]
    validate_unique_coverage(old, expected_old, ('dataset', 'routing', 'shots', 'seed', 'arm'))
    validate_unique_coverage(new, [(d, 'dynamic', 4, 1, ta, a) for d in DATASETS
        for ta in ('original', 'combined') for a in ('original', 'combined')],
        ('dataset', 'routing', 'shots', 'seed', 'training_arm', 'arm'))
    for row in timings:
        row['repetition'] = int(row['repetition'])
    validate_unique_coverage(timings, [(d, r, s, a, rep) for d in DATASETS
        for r in ('dynamic', 'static') for s in ('training', 'inference') for a in ARMS
        for rep in range(1, 6)], ('dataset', 'routing', 'scope', 'arm', 'repetition'))
    summary = summarize_profile(timings)
    raw_audit = audit_raw_evidence(out, old, new, timings)
    before = json.loads((out / 'environment_before.json').read_text())
    after = json.loads((out / 'environment_after.json').read_text())
    if any(before[k] != after[k] for k in ('gpu_uuid', 'software_stack_id', 'power_limit_watts')):
        raise ValueError('environment changed during experiment')
    screens = [json.loads((out / 'train' / f'{d}_training_screen.json').read_text()) for d in DATASETS]
    checkpoint_passed = all(r['passed'] == 'True' for r in old + new)
    speed_passed = all(r['speed_gate_passed'] for r in summary if r['routing'] == 'dynamic' and r['arm'] == 'combined')
    gates = dict(old_test_passes=len(old), new_test_passes=len(new), timing_rows=len(timings),
        raw_evidence_audit=raw_audit,
        checkpoint_accuracy_passed=checkpoint_passed, training_screen_passed=all(r['passed'] for r in screens),
        dynamic_speed_passed=speed_passed,
        limited_acceptance=checkpoint_passed and speed_passed and all(r['passed'] for r in screens),
        defaults_remain_legacy=True,
        limitation='Two datasets, one training seed. No equivalence/generalization claim over all eight datasets.')
    analysis = out / 'analysis'
    analysis.mkdir()
    write_csv(analysis / 'runtime_summary.csv', summary)
    write_csv(analysis / 'accuracy_summary.csv', old + new)
    write_json(analysis / 'coverage_and_gates.json', gates)
    report = ['# Kiểm chứng giới hạn factorized + SDPA', '',
        '## Kết luận', '', f'- Gate checkpoint accuracy: {checkpoint_passed}.',
        f'- Gate training screening: {gates["training_screen_passed"]}.',
        f'- Gate tốc độ dynamic: {speed_passed}.', '',
        'Đây là kiểm chứng hai dataset và một seed training, không phải bằng chứng tương đương trên toàn bộ benchmark.', '',
        '## Tốc độ dynamic — so với code cũ trong cùng đợt 250 W', '',
        '| Dataset | Phép đo | Bản | Median (ms) | Giảm latency (%) |',
        '|---|---|---|---:|---:|']
    for row in summary:
        if row['routing'] == 'dynamic':
            report.append(f"| {row['dataset']} | {row['scope']} | {row['arm']} | {row['median_ms']:.3f} | {row['median_latency_reduction_percent']:.2f} |")
    report += ['', '## Giới hạn diễn giải', '',
        '- Các speedup riêng không cộng được; combined là đối chứng riêng, không thay ramp/cast/merge.',
        '- Forward/backward không bao gồm optimizer; end-to-end training được báo riêng trong manifests.',
        '- Ngưỡng training 0.5 pp chỉ là screening, không phải kiểm định tương đương.',
        '- Accuracy chính thức, checkpoint nguồn và manuscript chưa bị thay thế. Backend mặc định vẫn là dense/manual.',
        '- Không so latency tuyệt đối 250 W với batch 320 W trước đây.']
    (analysis / 'REPORT_vi.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    paths = sorted(p for p in out.rglob('*') if p.is_file())
    write_json(analysis / 'artifact_inventory.json', [{'path': str(p.relative_to(out)), 'sha256': sha256(p),
                'bytes': p.stat().st_size} for p in paths])
    # Failure to meet a speed objective is a negative result, not invalid raw data.
    return gates
