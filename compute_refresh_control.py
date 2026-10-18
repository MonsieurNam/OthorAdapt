"""Fail-closed preflight, smoke/main controller and evidence packaging (no training)."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

from optimization_evidence import sha256, write_json


def prepare(root):
    import clip
    import torch
    from adapter_checkpoint_v2 import adapter_state_dict, validate_adapter_metadata, validate_adapter_state_dict
    from compute_refresh import (DATASETS, args_for, convert_legacy, load_model, environment)
    from ecr3_provenance import dataset_provenance
    from reviewer4_provenance import validate_split_reference
    from datasets import build_dataset
    from loralib.utils import apply_adapter
    from optimization_experiment import seed_all
    env = environment(); write_json(root / 'environment_preflight.json', env)
    originals = json.loads((root / 'original_inventory.json').read_text())
    if len(originals) != 24 or len({(e['dataset'], e['method']) for e in originals}) != 24:
        raise ValueError('24 unique original checkpoint entries required')
    # Only a CPU template is needed to establish expected state shapes/context buffers.
    template, preprocess = clip.load('ViT-B/16', device='cpu')
    apply_adapter(args_for('eurosat', 'original', root), template)
    expected = adapter_state_dict(template)
    inventory, gates = [], []
    for d in DATASETS:
        entries = [e for e in originals if e['dataset'] == d]
        if {e['method'] for e in entries} != {'lora', 'singlora', 'original'}:
            raise ValueError('per-dataset checkpoint coverage mismatch')
        seed_all(1); ds = build_dataset(d, '/root/DATA', 4, preprocess)
        provenance = dataset_provenance(ds)
        missing = [x.impath for split in (ds.train_x, ds.val, ds.test) for x in split if not Path(x.impath).is_file()]
        if missing: raise ValueError('missing image: ' + missing[0])
        for e in entries:
            p = root / e['relative_path']
            if sha256(p) != e['sha256']: raise ValueError('original checkpoint hash mismatch')
            validate_split_reference(d, 4, 1, provenance, [e['record']])
            payload = torch.load(p, map_location='cpu', weights_only=False)
            embedded = payload['metadata'].get('ecr3', {}).get('config', {})
            for key in ('adapter', 'dataset', 'shots', 'seed', 'r', 'alpha', 'params', 'encoder', 'position', 'backbone'):
                if embedded.get(key) != e['record']['config'].get(key):
                    raise ValueError('checkpoint embedded provenance mismatch: ' + key)
            item = dict(e, path=str(p))
            if e['method'] == 'original':
                new, audit = convert_legacy(payload, expected, e['record'], args_for(d, 'original', root))
                if len(audit['added_keys']) != 144 or len(payload['weights']) != 144:
                    raise ValueError('unexpected real CLIP migration key count')
                target = root / 'derived_checkpoints' / (d + '_ramp_inferred_v3.pt')
                target.parent.mkdir(exist_ok=True)
                if target.exists(): raise ValueError('derived checkpoint already exists')
                torch.save(new, target)
                validate_adapter_metadata(new['metadata'], args_for(d, 'original', root))
                validate_adapter_state_dict(expected, new['weights'])
                write_json(target.with_suffix('.audit.json'), dict(audit, derived_sha256=sha256(target)))
                item.update(path=str(target), sha256=sha256(target), original_sha256=e['sha256'])
                inventory.extend([item, dict(item, method='optimized')])
            else: inventory.append(item)
        gates.append(dict(dataset=d, split_passed=True, missing_images=0))
    del template
    write_json(root / 'checkpoint_inventory.json', inventory)
    write_json(root / 'preflight_passed.json', dict(passed=True, entries=len(inventory), checks=gates))
    # Load all checkpoints on GPU before launching the matrix; validation is strict for every method.
    import gc
    for e in inventory:
        model, _ = load_model(args_for(e['dataset'], e['method'], root), e['path'], e['method'])
        del model; gc.collect(); torch.cuda.empty_cache()
    write_json(root / 'checkpoint_load_gate.json', dict(passed=True, strict_loads=32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--timing-profile', choices=('full', 'economy'), default='full')
    parser.add_argument('--git-revision', default=None)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if args.prepare_only:
        prepare(root)
        return 0
    code, stage = 1, 'tests'
    try:
        with (root / 'tests.log').open('xb') as log:
            subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(root / 'resource_detector.py'), '--output', str(root / 'resources.json')], check=True)
        with (root / 'pip_freeze.txt').open('x', encoding='utf-8') as f:
            f.write(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        stage = 'prepare'
        # Release the preflight CUDA context before measurement subprocesses start.
        # The controller itself must not appear as a competing GPU process.
        subprocess.run([sys.executable, '-u', __file__, '--run-root', str(root), '--prepare-only'], check=True)
        for smoke in (True, False):
            stage = 'smoke' if smoke else 'main'
            print('STAGE', stage, datetime.now(timezone.utc).isoformat(), flush=True)
            cmd = [sys.executable, '-u', str(Path(__file__).with_name('compute_refresh.py')),
                '--out', str(root / ('smoke_output' if smoke else 'output')),
                '--inventory', str(root / 'checkpoint_inventory.json'),
                '--protocol', str(root / 'IMPLEMENTATION_PLAN.md'),
                '--source-inventory', str(root / 'source_inventory.json'),
                '--timing-profile', args.timing_profile]
            if smoke: cmd.append('--smoke')
            if args.git_revision: cmd.extend(['--git-revision', args.git_revision])
            result = subprocess.run(cmd, check=False)
            if result.returncode: raise RuntimeError(f'{stage} exited {result.returncode}; no automatic retry')
            done = json.loads((root / ('smoke_output' if smoke else 'output') / 'completed.json').read_text())
            from compute_refresh import cells
            if (not done['passed'] or done['timing_profile'] != args.timing_profile
                    or done['rows'] != (16 if smoke else len(cells(args.timing_profile)))):
                raise ValueError('completion coverage mismatch')
        code = 0
    except Exception as exc:
        traceback.print_exc()
        write_json(root / 'controller_failure.json', dict(stage=stage, message=str(exc), type=type(exc).__name__))
    finally:
        write_json(root / 'finish.json', dict(exit_code=code, stage=stage,
            timing_profile=args.timing_profile,
            full_experiment=False, compute_refresh_completed=(code == 0),
            training_runs=0, optimizer_updates=0, original_numerical_gate_passed=False,
            previous_all_arm_accuracy_gate_passed=False, completed_utc=datetime.now(timezone.utc).isoformat()))
        package = root / 'compute_refresh_evidence.zip'
        with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
            for p in sorted(root.rglob('*')):
                if p.is_file() and p != package and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
                    z.write(p, p.relative_to(root).as_posix())
        write_json(root / 'archive_ready.json', dict(exit_code=code, stage=stage,
            compute_refresh_completed=(code == 0), full_experiment=False,
            timing_profile=args.timing_profile,
            archive=package.name, sha256=sha256(package), bytes=package.stat().st_size))
    return code


if __name__ == '__main__': sys.exit(main())
