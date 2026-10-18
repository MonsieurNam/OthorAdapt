"""Bounded real-data equivalence experiment; never edits frozen evidence."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from optimization_evidence import (ARMS, DATASETS, GPU_UUID, VERSION, bounded_cells,
    compare_predictions, sha256, tensor_sha, write_csv, write_json)


def runtime_args(cell, arm, data_root, out):
    compute, attention = ARMS[arm]
    return SimpleNamespace(
        adapter='ohsinglora', backbone='ViT-B/16', encoder='both', position='all',
        params=['q', 'k', 'v'], r=8, num_heads=2, alpha=1, lambda_o=.03,
        ramp_up_steps=100, ortho_reduction='mean', lr=2e-4, n_iters=500,
        batch_size=32, loss_fn='ce', dropout_rate=.25, label_smoothing=.1,
        gamma=2., metric_s=30., metric_m=.5, lambda_center=.003, lambda_maxent=.1,
        root_path=str(data_root), dataset=cell['dataset'], shots=cell['shots'], seed=cell['seed'],
        routing_mode='input_conditioned' if cell['routing'] == 'dynamic' else 'input_independent',
        adapter_compute=compute, attention_backend=attention, eval_only=False,
        selection_split='val', report_test=False, sweep_mode=False, compute_profile_only=False,
        profile_repetition=0, split_reference_manifest=None,
        save_path=str(Path(out) / 'checkpoints'), filename=f"limited_{cell['dataset']}_{arm}",
        run_manifest=str(Path(out) / 'train_manifest.jsonl'))


def find_reference(rows, cell):
    matches = [r for r in rows if all(r['config'].get(k) == cell[k] for k in ('dataset', 'shots', 'seed'))]
    if len(matches) != 1:
        raise ValueError(f'expected one frozen reference for {cell}, found {len(matches)}')
    return matches[0]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_features(features):
    """Legacy normalization outside encoder autocast, including its activation dtype."""
    return features / features.norm(dim=-1, keepdim=True)


def prediction_and_margin(logits):
    top_two = np.partition(logits, -2, axis=1)[:, -2:]
    return logits.argmax(1), top_two[:, 1] - top_two[:, 0]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def frozen_records(root, routing):
    filename = 'dynamic_new_train_manifest.jsonl' if routing == 'dynamic' else 'static_train_manifest.jsonl'
    return read_rows(Path(root) / 'reviewer4' / 'raw' / filename)


def checked_checkpoint(root, record):
    root = Path(root).resolve()
    relative = Path(record['checkpoint']['path'])
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('checkpoint path escaped frozen evidence')
    # FINAL portable archive flattens revision_materials/, but keeps manifest paths.
    # This is a declared layout mapping, never a recursive filename search.
    if not path.is_file() and relative.parts[:2] == ('revision_materials', 'reviewer4'):
        path = (root / Path(*relative.parts[1:])).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('checkpoint path escaped frozen evidence or is missing')
    if sha256(path).lower() != record['checkpoint']['sha256'].lower():
        raise ValueError('checkpoint SHA-256 mismatch')
    return path


def build_data(args, preprocess, reference):
    from datasets import build_dataset
    from datasets.utils import build_data_loader
    from ecr3_provenance import dataset_provenance
    from reviewer4_provenance import validate_split_reference
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    provenance = dataset_provenance(dataset)
    validate_split_reference(args.dataset, args.shots, args.seed, provenance, [reference])
    loader = build_data_loader(data_source=dataset.test, batch_size=256, tfm=preprocess,
                               is_train=False, shuffle=False, num_workers=8)
    return dataset, loader, provenance


def load_model(args, checkpoint=None):
    import clip
    from loralib.utils import apply_adapter, load_adapter, mark_only_lora_as_trainable
    model, preprocess = clip.load(args.backbone, device='cuda')
    apply_adapter(args, model)
    if checkpoint is not None:
        # Existing strict loader defines the directory layout; do not bypass it.
        checkpoint = Path(checkpoint)
        load_args = SimpleNamespace(**vars(args))
        load_args.save_path = str(checkpoint.parents[5])
        load_args.filename = checkpoint.name
        actual = Path(load_adapter(load_args, model)).resolve()
        if actual != checkpoint.resolve():
            raise ValueError('checkpoint loader resolved a different file')
    mark_only_lora_as_trainable(model)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 460800:
        raise ValueError('trainable parameter mismatch')
    model.cuda().eval()
    return model, preprocess


def configure_backend(model, arm):
    from loralib.layers_OH_singlora import LinearOHsingLoRA
    from loralib.layers_singlora import PlainMultiheadAttentionAdapter
    compute, attention = ARMS[arm]
    for module in model.modules():
        if isinstance(module, LinearOHsingLoRA):
            module.adapter_compute = compute
        if isinstance(module, PlainMultiheadAttentionAdapter):
            module.attention_backend = attention


def backend_counts(model):
    from loralib.layers_singlora import PlainMultiheadAttentionAdapter
    total = {}
    for module in model.modules():
        if isinstance(module, PlainMultiheadAttentionAdapter):
            for key, value in module.execution_counts.items():
                total[key] = total.get(key, 0) + value
    return total


def predict(model, dataset, loader, out, fp32_indices=None):
    import clip
    model.eval()
    token_cpu = clip.tokenize([dataset.template[0].format(n.replace('_', ' ')) for n in dataset.classnames])
    tokens = token_cpu.cuda()
    logits, labels, input_hashes = [], [], []
    selected = None if fp32_indices is None else set(fp32_indices)
    offset = 0
    with torch.inference_mode():
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=selected is None):
            text = model.encode_text(tokens)
        text = normalize_features(text)
        text_hash = tensor_sha(text)
        for images, y in loader:
            indices = list(range(offset, offset + len(y)))
            offset += len(y)
            if selected is not None:
                keep = [i for i, idx in enumerate(indices) if idx in selected]
                if not keep:
                    continue
                images, y = images[keep], y[keep]
            input_hashes.append(tensor_sha(images))
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=selected is None):
                image = model.encode_image(images.cuda())
            image = normalize_features(image)
            scores = image @ text.t()
            logits.append(scores.float().cpu().numpy())
            labels.append(y.numpy())
    a, y = np.concatenate(logits), np.concatenate(labels)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite real-data logits')
    prediction, margin = prediction_and_margin(a)
    paths = [str(item.impath) for item in dataset.test]
    if selected is not None:
        paths = [path for i, path in enumerate(paths) if i in selected]
    if len(paths) != len(y) or len(set(paths)) != len(paths):
        raise ValueError('test sample identity/coverage mismatch')
    np.savez_compressed(Path(out) / 'predictions.npz', logits=a, labels=y,
                        prediction=prediction, margin=margin, sample_ids=np.asarray(paths))
    write_json(Path(out) / 'inputs.json', dict(image_batch_sha256=input_hashes,
               token_sha256=tensor_sha(token_cpu), text_features_sha256=text_hash,
               classifier='own text encoder, template[0], normalized unscaled cosine logits',
               precision='FP16 autocast' if selected is None else 'FP32', backend_counts=backend_counts(model)))
    return a, y, input_hashes


def fp32_disagreements(args, checkpoint, dataset, loader, indices, out, arms):
    results = {}
    for arm in ('original', *arms):
        model, _ = load_model(args, checkpoint)
        model.float()
        configure_backend(model, arm)
        directory = Path(out) / f'fp32_{arm}'
        directory.mkdir()
        logits, labels, _ = predict(model, dataset, loader, directory, indices)
        results[arm] = logits
        del model
        gc.collect(); torch.cuda.empty_cache()
    write_json(Path(out) / 'fp32_disagreement_comparison.json',
               {arm: compare_predictions(results['original'], results[arm], labels) for arm in arms})


def evaluate_checkpoint(cell, checkpoint, reference, args, out, arms=tuple(ARMS)):
    import clip
    seed_all(cell['seed'])
    base, preprocess = clip.load('ViT-B/16', device='cpu')
    dataset, loader, provenance = build_data(args, preprocess, reference)
    del base
    rows, reference_logits, reference_labels, reference_hashes = [], None, None, None
    all_disagreements = set()
    disagreeing_arms = []
    for arm in arms:
        directory = Path(out) / arm
        directory.mkdir(parents=True)
        model, _ = load_model(args, checkpoint)
        configure_backend(model, arm)
        logits, y, input_hashes = predict(model, dataset, loader, directory)
        if reference_logits is None:
            reference_logits, reference_labels, reference_hashes = logits, y, input_hashes
        if not np.array_equal(y, reference_labels) or input_hashes != reference_hashes:
            raise ValueError('paired eval input identities mismatch')
        comparison = compare_predictions(reference_logits, logits, y)
        disagreements = np.flatnonzero(reference_logits.argmax(1) != logits.argmax(1)).tolist()
        all_disagreements.update(disagreements)
        if disagreements:
            disagreeing_arms.append(arm)
        row = dict(cell, arm=arm, checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint), **comparison)
        rows.append(row)
        write_json(directory / 'comparison.json', dict(row, disagreement_indices=disagreements,
                   split_provenance=provenance))
        print('EVAL', cell, arm, comparison, flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
    if all_disagreements:
        fp32_disagreements(args, checkpoint, dataset, loader, sorted(all_disagreements), out, disagreeing_arms)
    return rows


def old_checkpoint_suite(opts, out):
    rows = []
    for cell in bounded_cells():
        reference = find_reference(frozen_records(opts.evidence_root, cell['routing']), cell)
        checkpoint = checked_checkpoint(opts.evidence_root, reference)
        args = runtime_args(cell, 'original', opts.data_root, out)
        rows.extend(evaluate_checkpoint(cell, checkpoint, reference, args,
                    Path(out) / f"{cell['dataset']}_{cell['routing']}_{cell['shots']}shot"))
    write_csv(Path(out) / 'accuracy_comparison.csv', rows)
    if not all(row['passed'] for row in rows):
        raise RuntimeError('old-checkpoint accuracy gate failed; do not launch training')


def training_suite(opts, out):
    import clip
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from datasets.utils import DatasetWrapper, build_data_loader
    from ecr3_provenance import args_to_config, git_revision
    from lora import run_lora
    from loralib.utils import adapter_checkpoint_path
    from optimization_training import TrainingObserver
    from reviewer4_provenance import capture_environment, split_identity

    eval_rows = []
    for dataset_name in DATASETS:
        cell = dict(dataset=dataset_name, shots=4, seed=1, routing='dynamic')
        reference = find_reference(frozen_records(opts.evidence_root, 'dynamic'), cell)
        paired_initial = paired_batches = None
        paired_accuracy = []
        for arm in ('original', 'combined'):
            directory = Path(out) / f'{dataset_name}_{arm}'
            directory.mkdir(parents=True)
            args = runtime_args(cell, arm, opts.data_root, directory)
            seed_all(1)
            model, preprocess = clip.load('ViT-B/16', device='cuda')
            dataset, test_loader, provenance = build_data(args, preprocess, reference)
            val_loader = build_data_loader(data_source=dataset.val, batch_size=256, tfm=preprocess,
                                           is_train=False, shuffle=False, num_workers=8)
            transform = T.Compose([T.RandomResizedCrop(224, scale=(.08, 1.), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(.5), T.ToTensor(),
                T.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
            # Independent worker/sampler RNG isolates data from backend CUDA implementation details.
            generator = torch.Generator().manual_seed(20261016)
            train_loader = DataLoader(DatasetWrapper(dataset.train_x, input_size=224, transform=transform,
                is_train=True), batch_size=32, shuffle=True, num_workers=8, pin_memory=True, generator=generator)
            args.dataset_provenance = provenance
            args.runtime_environment = capture_environment()
            args.git_revision = git_revision()
            args.run_command = sys.argv
            args.checkpoint_extra_metadata = dict(environment=args.runtime_environment,
                git_revision=args.git_revision, split_identity=split_identity(provenance),
                execution_backend=dict(adapter_compute=args.adapter_compute, attention_backend=args.attention_backend),
                optimization_version=VERSION)
            observer = TrainingObserver(directory / 'audit', paired_initial, paired_batches)
            args._optimization_observer = observer
            try:
                run_lora(args, model, 100, dataset, train_loader, val_loader, test_loader)
            finally:
                observer.close()
            checkpoint = Path(adapter_checkpoint_path(args))
            initial = json.loads((directory / 'audit' / 'initial_state.json').read_text())
            batches = read_rows(directory / 'audit' / 'steps.jsonl')
            if len(batches) != 2000 or [r['step'] for r in batches] != list(range(2000)):
                raise ValueError('training did not produce exactly 2000 audited steps')
            identity = [(r['image_sha256'], r['label_sha256']) for r in batches]
            if paired_initial is None:
                paired_initial, paired_batches = initial, identity
            elif initial != paired_initial or identity != paired_batches:
                raise ValueError('paired training initialization/augmentation mismatch')
            write_json(directory / 'training_audit.json', dict(initial=initial, checkpoint_sha256=sha256(checkpoint),
                batch_sequence_matched=True, optimizer_steps=2000,
                amp_skipped_steps=sum(r['amp_skipped_step'] for r in batches),
                config=args_to_config(args)))
            del model, train_loader, val_loader, test_loader
            gc.collect(); torch.cuda.empty_cache()
            rows = evaluate_checkpoint(cell, checkpoint, reference, args, directory / 'final_eval',
                                       arms=('original', 'combined'))
            eval_rows.extend(dict(row, training_arm=arm) for row in rows)
            paired_accuracy.append(rows[0]['reference_accuracy'])
            if not all(row['passed'] for row in rows):
                write_csv(Path(out) / 'new_checkpoint_accuracy.csv', eval_rows)
                raise RuntimeError('new-checkpoint backend accuracy gate failed')
        delta = paired_accuracy[1] - paired_accuracy[0]
        write_json(Path(out) / f'{dataset_name}_training_screen.json', dict(accuracy_delta_pp=delta,
            passed=abs(delta) <= .5, limitation='single seed; screening threshold is NOT statistical equivalence'))
        if abs(delta) > .5:
            write_csv(Path(out) / 'new_checkpoint_accuracy.csv', eval_rows)
            raise RuntimeError('training accuracy drift exceeds 0.5 pp; stop and investigate')
    write_csv(Path(out) / 'new_checkpoint_accuracy.csv', eval_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke', 'old-eval', 'train', 'profile', 'all'))
    parser.add_argument('--evidence-root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', required=True)
    opts = parser.parse_args(argv)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=False)
    from reviewer4_provenance import capture_environment, validate_target_environment
    environment = capture_environment()
    validate_target_environment(environment)
    if environment['gpu_uuid'] != GPU_UUID or float(environment['power_limit_watts']) != 250:
        raise RuntimeError('physical GPU/power mismatch')
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True).stdout.splitlines()
    if any(line.strip().isdigit() and int(line.strip()) != os.getpid() for line in processes):
        raise RuntimeError('competing GPU process; refusing to run')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = dict(version=VERSION, command=sys.argv, cells=bounded_cells(), arms=ARMS,
        evidence_root=str(Path(opts.evidence_root).resolve()), data_root=str(Path(opts.data_root).resolve()),
        training_runs=4, training_steps=2000, old_test_passes=32, new_test_passes=8,
        numeric_tolerances=dict(fp32=[1e-5, 1e-4], fp16=[.02, .02]),
        accuracy_gate=dict(absolute_delta_pp=.05, agreement_percent=99.9),
        training_screen_pp=.5, profile=dict(warmup=20, calls=100, repetitions=5),
        speed_gate=dict(primary='full CLIP forward/backward and cached-text inference wall median',
                        reduction_percent=10, faster_repetitions=4),
        no_full_matrix=True, no_manuscript_changes=True)
    write_json(out / 'protocol.json', protocol)
    source_root = Path(__file__).resolve().parent
    source_files = sorted(list(source_root.glob('*.py')) +
        [p for directory in ('clip', 'loralib', 'datasets', 'losses', 'tests')
         for p in (source_root / directory).rglob('*.py')])
    write_json(out / 'source_inventory.json', dict(version=VERSION,
        base_git_revision='ce709fa5b32b7392db532a3776ab7ecbda8500bc',
        portable_git_note='Portable deployment may have no .git; source SHA-256 defines execution content.',
        files=[dict(path=str(p.relative_to(source_root)), sha256=sha256(p)) for p in source_files]))
    write_json(out / 'environment_before.json', environment)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (out / 'pip_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    stages = ('smoke', 'old-eval', 'train', 'profile') if opts.stage == 'all' else (opts.stage,)
    for stage in stages:
        write_json(out / f'{stage}_started.json', dict(timestamp=time.time()))
        directory = out / stage
        directory.mkdir()
        print('STAGE START', stage, flush=True)
        if stage == 'smoke':
            from optimization_profile import profile_suite
            profile_suite(opts, directory, smoke_only=True)
        elif stage == 'old-eval':
            old_checkpoint_suite(opts, directory)
        elif stage == 'train':
            training_suite(opts, directory)
        else:
            from optimization_profile import profile_suite
            profile_suite(opts, directory)
        write_json(out / f'{stage}_completed.json', dict(timestamp=time.time()))
    write_json(out / 'environment_after.json', capture_environment())
    if opts.stage == 'all':
        from optimization_analysis import analyze
        analyze(out)
        write_json(out / 'completed.json', dict(exit_code=0, version=VERSION, timestamp=time.time()))
    else:
        write_json(out / 'partial_stage_completed.json', dict(stage=opts.stage, full_experiment_completed=False))


if __name__ == '__main__':
    main()
