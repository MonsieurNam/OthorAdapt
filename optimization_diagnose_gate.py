"""Explain a failed locked numerical gate. No optimizer, training or tolerance change."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch


def tensor_error(reference, candidate, atol, rtol):
    a, b = reference.detach().float().cpu(), candidate.detach().float().cpu()
    if a.shape != b.shape:
        raise ValueError('diagnostic tensor shape mismatch')
    difference = (a-b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    # torch.allclose(input=reference, other=candidate) uses candidate in rtol.
    threshold = atol + rtol * b.abs()
    violations = int((difference > threshold).sum())
    return dict(shape=list(a.shape), finite=finite,
        passed=bool(torch.allclose(a,b,atol=atol,rtol=rtol)),
        violations=violations, elements=a.numel(), max_abs_error=float(difference.max()),
        rms_error=float(difference.square().mean().sqrt()),
        max_threshold_ratio=float((difference/threshold).max()), atol=atol, rtol=rtol)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--evidence-root',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--out',required=True)
    opts=parser.parse_args()
    source=Path(opts.source_root).resolve()
    sys.path.insert(0,str(source))
    import clip
    from optimization_experiment import (runtime_args,seed_all,build_data,load_model,configure_backend,
        checked_checkpoint,find_reference,frozen_records,normalize_features)
    from optimization_profile import full_precision,objective
    from optimization_evidence import ARMS,sha256,write_json,write_csv
    from reviewer4_provenance import capture_environment
    out=Path(opts.out);out.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    environment=capture_environment()
    if environment['gpu_uuid']!='GPU-6a3242b5-13dc-58ea-be01-e96440e63771' or float(environment['power_limit_watts'])!=250:
        raise ValueError('diagnostic GPU/power mismatch')
    cell=dict(dataset='eurosat',shots=4,seed=1,routing='dynamic')
    reference=find_reference(frozen_records(opts.evidence_root,'dynamic'),cell)
    checkpoint=checked_checkpoint(opts.evidence_root,reference)
    args=runtime_args(cell,'original',opts.data_root,out)
    seed_all(1)
    base,preprocess=clip.load('ViT-B/16',device='cpu')
    dataset,loader,_=build_data(args,preprocess,reference)
    del base
    model,_=load_model(args,checkpoint)
    images,labels=next(iter(loader));images=images.cuda();labels=labels.cuda()
    tokens=clip.tokenize([dataset.template[0].format(n.replace('_',' ')) for n in dataset.classnames]).cuda()
    ramp={n:b.clone() for n,b in model.named_buffers() if n.endswith('training_step')}
    write_json(out/'provenance.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        script_sha256=sha256(__file__),source_root=str(source),environment=environment,
        optimizer_updates=0,accuracy_evaluations=0,thresholds_unchanged=True,
        note='Diagnostic of original failed gate only; not acceptance or a rerun of accuracy experiments.'))
    rows=[]
    for amp in (False,True):
        context=full_precision(model) if not amp else __import__('contextlib').nullcontext()
        with context:
            for scope in ('training','inference'):
                training=scope=='training';model.train(training)
                baseline=None
                for arm in ('original','original_repeat','factorized','sdpa','combined'):
                    configure_backend(model,'original' if arm=='original_repeat' else arm)
                    with torch.no_grad():
                        for n,b in model.named_buffers():
                            if n in ramp:b.copy_(ramp[n])
                    with __import__('contextlib').nullcontext() if training else torch.no_grad():
                        cached=None
                        if not training:
                            with torch.amp.autocast('cuda',dtype=torch.float16,enabled=amp):
                                cached=model.encode_text(tokens)
                            cached=normalize_features(cached)
                        logits,loss,_=objective(model,images[:32] if training else images,tokens,
                            labels[:32] if training else labels,cached=cached,amp=amp)
                    values=dict(logits=logits.detach().float().cpu(),
                        gradients={n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters()
                                   if p.requires_grad and p.grad is not None})
                    if baseline is None:baseline=values
                    atol,rtol=(.02,.02) if amp else (1e-5,1e-4)
                    output=tensor_error(baseline['logits'],values['logits'],atol,rtol)
                    gradients={n:tensor_error(baseline['gradients'][n],v,atol,rtol)
                               for n,v in values['gradients'].items()}
                    failed={n:v for n,v in gradients.items() if not v['passed']}
                    record=dict(precision='fp16' if amp else 'fp32',scope=scope,arm=arm,
                        batch=32 if training else 256,output=output,gradient_failures=failed,
                        gradients_checked=len(gradients),failed_gradient_tensors=len(failed),
                        loss=float(loss.detach()) if loss is not None else None)
                    write_json(out/f"{record['precision']}_{scope}_{arm}.json",record)
                    torch.save(values,out/f"{record['precision']}_{scope}_{arm}.pt")
                    rows.append(dict(precision=record['precision'],scope=scope,arm=arm,
                        output_passed=output['passed'],output_violations=output['violations'],
                        output_elements=output['elements'],max_abs_logit_error=output['max_abs_error'],
                        max_threshold_ratio=output['max_threshold_ratio'],
                        failed_gradient_tensors=len(failed),gradients_checked=len(gradients)))
                    print(rows[-1],flush=True)
    write_csv(out/'gate_diagnostic_summary.csv',rows)
    write_json(out/'completed.json',dict(diagnostic_only=True,optimizer_updates=0,
        original_gate_still_failed=True,thresholds_unchanged=True))


if __name__=='__main__':main()
