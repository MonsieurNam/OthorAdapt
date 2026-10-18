"""Explicit post-preflight diagnostic amendment; never relabels original gate failures."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from optimization_evidence import (ARMS,DATASETS,GPU_UUID,bounded_cells,compare_predictions,
    sha256,validate_unique_coverage,write_csv,write_json)

AMENDMENT_VERSION='orthadapt.optimization.amendment.v1'
PRIOR_ARCHIVE_SHA='c25e9a932a4126a2d7754b8da3c64ed7b98b0652fa7007006d0a5195645b2cff'


def frozen_amendment():
    return dict(version=AMENDMENT_VERSION,classification='post-preflight exploratory diagnostic amendment',
        user_authorization='tiếp tục theo plan đấy đi',original_numerical_gate_passed=False,
        original_failure_archive_sha256=PRIOR_ARCHIVE_SHA,
        stage_order=['old-eval','profile','train'],cells=bounded_cells(),arms=ARMS,
        old_test_passes=32,new_test_passes=8,training_runs=4,training_steps=2000,
        accuracy_gate=dict(absolute_delta_pp=.05,agreement_percent=99.9),
        training_screen_pp=.5,numeric_tolerances=dict(fp32=[1e-5,1e-4],fp16=[.02,.02]),
        numeric_tolerance_policy='report_only_hard_fail_nonfinite',
        profile=dict(warmup=20,calls=100,repetitions=5),
        speed_gate=dict(primary='full CLIP forward/backward and cached-text inference wall median',
                        reduction_percent=10,faster_repetitions=4),
        automatic_production_promotion=False,defaults_remain_legacy=True,
        no_full_matrix=True,no_manuscript_changes=True,
        interpretation='New exploratory evidence cannot retroactively pass the original numerical gate.')


def enforce_speed_gate(summary):
    chosen=[r for r in summary if r['routing']=='dynamic' and r['arm']=='combined']
    validate_unique_coverage(chosen,[(d,s) for d in DATASETS for s in ('training','inference')],
                            ('dataset','scope'))
    if not all(r['speed_gate_passed'] for r in chosen):
        raise RuntimeError('dynamic speed target failed; training must not start')


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_old_stage(opts,out):
    from optimization_analysis import audit_prediction_group
    from optimization_experiment import checked_checkpoint,find_reference,frozen_records
    rows=read_csv(Path(out)/'old-eval/accuracy_comparison.csv')
    for r in rows:r['shots'],r['seed']=int(r['shots']),int(r['seed'])
    validate_unique_coverage(rows,[(c['dataset'],c['routing'],c['shots'],c['seed'],a)
        for c in bounded_cells() for a in ARMS],('dataset','routing','shots','seed','arm'))
    results=[]
    for cell in bounded_cells():
        reference=find_reference(frozen_records(opts.evidence_root,cell['routing']),cell)
        checkpoint=checked_checkpoint(opts.evidence_root,reference)
        group=Path(out)/'old-eval'/f"{cell['dataset']}_{cell['routing']}_{cell['shots']}shot"
        actual=audit_prediction_group(group,tuple(ARMS))
        for a in actual:
            row=next(r for r in rows if all(r[k]==cell[k] for k in cell) and r['arm']==a['arm'])
            if (row['checkpoint_sha256']!=sha256(checkpoint)
                or abs(float(row['accuracy_delta_pp'])-a['accuracy_delta_pp'])>1e-8
                or (row['passed']=='True')!=a['passed']):
                raise ValueError('old-eval aggregate/hash contradicts raw evidence')
            results.append(dict(cell,**a))
    write_json(Path(out)/'old_accuracy_raw_audit.json',dict(artifact_audit_passed=True,
        accuracy_gate_passed=all(r['passed'] for r in results),rows=results))
    return results


def audit_profile_stage(out):
    from optimization_analysis import summarize_profile
    from optimization_profile import audit_backend_counts,audit_sdpa_operators
    directory=Path(out)/'profile'
    rows=read_csv(directory/'timings.csv')
    for r in rows:r['repetition']=int(r['repetition'])
    validate_unique_coverage(rows,[(d,r,s,a,k) for d in DATASETS for r in ('dynamic','static')
        for s in ('training','inference') for a in ARMS for k in range(1,6)],
        ('dataset','routing','scope','arm','repetition'))
    for d in DATASETS:
        for routing in ('dynamic','static'):
            case=directory/f'{d}_{routing}'
            identity=read_json(case/'inputs.json')
            smoke=read_json(case/'full_clip_numerical_smoke.json')
            validate_unique_coverage(smoke,[(p,s,a) for p in ('fp32','fp16') for s in ('training','inference')
                for a in ARMS],('precision','scope','arm'))
            if not all(r['finite'] and r['batch']==(32 if r['scope']=='training' else 256)
                       and r['tolerance_policy']=='report_only_hard_fail_nonfinite' for r in smoke):
                raise ValueError('nonfinite/wrong-shape numerical diagnostic')
            for scope in ('training','inference'):
                for arm in ARMS:
                    trace=case/f'trace_{scope}_{arm}'
                    events=read_json(trace/'trace.json')['traceEvents']
                    if not any(e.get('cat')=='kernel' for e in events):
                        raise ValueError('missing CUDA kernel trace')
                    operators=read_csv(trace/'operator_table.csv')
                    audit=audit_sdpa_operators([r['operator'] for r in operators],arm)
                    if audit!=read_json(trace/'kernel_audit.json'):
                        raise ValueError('kernel audit mismatch')
                    for k in range(1,6):
                        raw=read_json(case/f'{scope}_{arm}_rep{k}.json')
                        row=next(r for r in rows if (r['dataset'],r['routing'],r['scope'],r['arm'],r['repetition'])==
                            (d,routing,scope,arm,k))
                        if (raw['image_sha256']!=identity['image_sha256']
                            or raw['token_sha256']!=identity['token_sha256']
                            or len(raw['raw_wall_ms'])!=100 or len(raw['raw_cuda_ms'])!=100):
                            raise ValueError('profile input identity/call coverage mismatch')
                        values=np.asarray(raw['raw_wall_ms']+raw['raw_cuda_ms'])
                        if not np.isfinite(values).all() or (values<=0).any():
                            raise ValueError('invalid raw timings')
                        if abs(float(row['median_ms'])-float(np.median(raw['raw_wall_ms'])))>1e-8:
                            raise ValueError('profile median mismatch')
                        audit_backend_counts(raw['backend_counts'],arm)
    summary=summarize_profile(rows)
    write_csv(Path(out)/'runtime_summary.csv',summary)
    return summary


def capture_and_validate_environment(reference=None):
    from reviewer4_provenance import capture_environment,validate_target_environment
    environment=capture_environment()
    validate_target_environment(environment)
    if environment['gpu_uuid']!=GPU_UUID or float(environment['power_limit_watts'])!=250:
        raise RuntimeError('physical GPU/power mismatch')
    pids=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                         check=True,capture_output=True,text=True).stdout.splitlines()
    if any(p.strip().isdigit() and int(p)!=os.getpid() for p in pids):
        raise RuntimeError('another GPU compute process is running')
    if reference and any(environment[k]!=reference[k] for k in ('gpu_uuid','software_stack_id','power_limit_watts')):
        raise RuntimeError('GPU/software/power identity changed')
    return environment


def validate_prior_failure(prior,source):
    prior,source=Path(prior),Path(source)
    marker=read_json(prior/'archive_ready.json')
    if marker['exit_code']!=1 or marker['full_experiment'] or marker['sha256']!=PRIOR_ARCHIVE_SHA:
        raise ValueError('prior numerical failure receipt mismatch')
    if sha256(prior/marker['archive'])!=PRIOR_ARCHIVE_SHA:
        raise ValueError('prior failure archive SHA mismatch')
    core=['loralib/layers_OH_singlora.py','loralib/layers_singlora.py','loralib/utils.py',
          'execution_backends.py','run_utils.py','clip/model.py','adapter_checkpoint_v2.py','lora.py']
    hashes=[]
    with zipfile.ZipFile(prior/marker['archive']) as archive:
        numeric=json.loads(archive.read('output/smoke/eurosat_dynamic/full_clip_numerical_smoke.json'))
        if len(numeric)!=16 or sum(not r['passed'] for r in numeric)!=11:
            raise ValueError('prior archived failure content mismatch')
        for name in core:
            archived_sha=hashlib.sha256(archive.read('source/'+name)).hexdigest()
            if sha256(source/name)!=archived_sha:
                raise ValueError('research execution changed since archived failed preflight: '+name)
            hashes.append(dict(path=name,sha256=archived_sha))
    return dict(original_failure_preserved=True,original_numerical_gate_passed=False,core_source=hashes)


def write_progress_report(out,error=None):
    out=Path(out)
    stages={name:(out/f'{name}_completed.json').is_file() for name in ('old-eval','profile','train')}
    lines=['# Nhánh chẩn đoán factorized + SDPA — protocol bổ sung','',
        'Gate numerical của lần chạy gốc vẫn FAIL; đây là nhánh chẩn đoán bổ sung được người dùng chấp thuận.',
        'Không đổi tolerance cũ, không tự bật production và không sửa evidence/manuscript chính thức.','',
        f"- Eval checkpoint cũ hoàn tất: {stages['old-eval']}.",
        f"- Profiling hoàn tất: {stages['profile']}.",
        f"- Bốn training runs hoàn tất: {stages['train']}."]
    if error:lines+=['',f'Dừng: {type(error).__name__}: {error}']
    csv_path=out/'old-eval/accuracy_comparison.csv'
    if csv_path.exists():
        rows=read_csv(csv_path)
        failed=[r for r in rows if r['passed']!='True']
        lines+=['',f'Accuracy gate: {len(rows)-len(failed)}/{len(rows)} hàng đạt.',
            'Ngưỡng giữ nguyên: |delta| ≤0,05 pp và prediction agreement ≥99,9% trên từng checkpoint/arm.','']
        if failed:
            lines+=['| Dataset | Routing/shot | Bản | Delta (pp) | Agreement (%) |',
                    '|---|---|---|---:|---:|']
            for r in failed:lines.append(f"| {r['dataset']} | {r['routing']}/{r['shots']} | {r['arm']} | {float(r['accuracy_delta_pp']):+.6f} | {float(r['agreement_percent']):.6f} |")
    if (out/'runtime_summary.csv').exists():
        lines+=['','Bảng tốc độ: runtime_summary.csv. Training chỉ được phép bắt đầu sau khi đủ bốn gate tốc độ dynamic đạt.']
    lines+=['','Kết quả hai dataset/một seed không chứng minh tương đương trên toàn bộ tám dataset.']
    with (out/'REPORT_vi.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--prior-run-root',default='/root/orthadapt_optimization_20261016_v3')
    opts=parser.parse_args(argv)
    out=Path(opts.out);out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parent
    protocol=frozen_amendment()
    protocol.update(created_utc=datetime.now(timezone.utc).isoformat(),evidence_root=opts.evidence_root,
                    data_root=opts.data_root,prior_run_root=opts.prior_run_root,command=sys.argv)
    write_json(out/'protocol.json',protocol)
    write_json(out/'original_failure_audit.json',validate_prior_failure(opts.prior_run_root,source))
    write_json(out/'source_inventory.json',dict(files=[dict(path=str(p.relative_to(source)),sha256=sha256(p))
        for p in sorted(source.rglob('*.py')) if '__pycache__' not in p.parts]))
    env=capture_and_validate_environment();write_json(out/'environment_before.json',env)
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (out/'pip_freeze.txt').write_text(freeze.stdout,encoding='utf-8')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from optimization_experiment import old_checkpoint_suite,training_suite
    from optimization_profile import profile_suite
    try:
        for stage in protocol['stage_order']:
            write_json(out/f'{stage}_started.json',dict(timestamp=time.time()))
            write_json(out/f'{stage}_environment.json',capture_and_validate_environment(env))
            directory=out/stage;directory.mkdir()
            print('AMENDED STAGE START',stage,flush=True)
            if stage=='old-eval':
                try:old_checkpoint_suite(opts,directory)
                finally:
                    if (directory/'accuracy_comparison.csv').is_file():audit_old_stage(opts,out)
            elif stage=='profile':
                profile_suite(opts,directory,strict_numerical=False)
                summary=audit_profile_stage(out)
                enforce_speed_gate(summary)
            else:
                training_suite(opts,directory)
            write_json(out/f'{stage}_completed.json',dict(timestamp=time.time()))
        from optimization_analysis import audit_raw_evidence
        old=read_csv(out/'old-eval/accuracy_comparison.csv');new=read_csv(out/'train/new_checkpoint_accuracy.csv')
        for row in old+new:row['shots'],row['seed']=int(row['shots']),int(row['seed'])
        timings=read_csv(out/'profile/timings.csv')
        for row in timings:row['repetition']=int(row['repetition'])
        audit=audit_raw_evidence(out,old,new,timings,numerical_phases=('profile',),strict_numerical=False)
        write_json(out/'final_raw_audit.json',dict(audit_passed=True,original_numerical_gate_passed=False,**audit))
        write_json(out/'environment_after.json',capture_and_validate_environment(env))
        write_progress_report(out)
        write_json(out/'amended_completed.json',dict(amended_bounded_workflow_completed=True,
            original_numerical_gate_passed=False,automatic_production_promotion=False))
    except Exception as error:
        write_json(out/'stopped.json',dict(type=type(error).__name__,message=str(error),timestamp=time.time(),
            original_numerical_gate_passed=False,full_experiment_completed=False))
        write_progress_report(out,error)
        raise


if __name__=='__main__':main()
