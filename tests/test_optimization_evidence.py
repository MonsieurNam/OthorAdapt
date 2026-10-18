import json
from pathlib import Path

import numpy as np
import pytest

from optimization_evidence import (
    ARMS, bounded_cells, compare_predictions, paired_speed_gate,
    validate_unique_coverage, write_json, sha256,
)


def test_locked_bounded_matrix():
    cells = bounded_cells()
    assert len(cells) == 8
    assert sum(c['routing'] == 'dynamic' for c in cells) == 6
    assert sum(c['routing'] == 'static' for c in cells) == 2
    assert len(ARMS) == 4
    assert {c['dataset'] for c in cells} == {'eurosat', 'caltech101'}
    assert {c['seed'] for c in cells} == {1}


def test_accuracy_gate_uses_percentage_points_and_no_cancellation():
    y = np.zeros(10000, dtype=np.int64)
    ref = np.zeros((10000, 2), dtype=np.float32)
    ref[:, 0] = 1
    candidate = ref.copy()
    candidate[:5] = [0, 1]
    result = compare_predictions(ref, candidate, y)
    assert result['passed']
    assert result['accuracy_delta_pp'] == pytest.approx(-0.05)
    candidate[5] = [0, 1]
    assert not compare_predictions(ref, candidate, y)['passed']


def test_prediction_identity_gate_catches_cancelling_accuracy():
    y = np.arange(1000) % 2
    ref = np.tile([1., 0.], (1000, 1))
    other = ref.copy()
    other[:2] = [0., 1.]
    result = compare_predictions(ref, other, y)
    assert result['accuracy_delta_pp'] == 0
    assert not result['passed']


def test_nonfinite_and_shape_errors_hard_fail():
    with pytest.raises(ValueError):
        compare_predictions(np.array([[np.nan, 0.]]), np.ones((1, 2)), np.array([0]))
    with pytest.raises(ValueError):
        compare_predictions(np.ones((2, 2)), np.ones((1, 2)), np.array([0]))


def test_coverage_rejects_duplicate_and_missing():
    expected = [('a', 1), ('b', 1)]
    validate_unique_coverage([{'x': 'a', 'r': 1}, {'x': 'b', 'r': 1}], expected, ('x', 'r'))
    with pytest.raises(ValueError):
        validate_unique_coverage([{'x': 'a', 'r': 1}] * 2, expected, ('x', 'r'))


def test_speed_gate_requires_four_faster_repeats_and_ten_percent():
    assert paired_speed_gate([1] * 5, [.8] * 4 + [1.1])['passed']
    assert not paired_speed_gate([1] * 5, [.95] * 5)['passed']
    assert not paired_speed_gate([1] * 5, [.8] * 3 + [1.1] * 2)['passed']


def test_json_writer_never_overwrites(tmp_path):
    path = tmp_path / 'artifact.json'
    write_json(path, {'x': 1})
    assert len(sha256(path)) == 64
    assert json.loads(path.read_text()) == {'x': 1}
    with pytest.raises(FileExistsError):
        write_json(path, {'x': 2})


def test_training_observer_does_not_mutate_rng_gradient_or_buffers(tmp_path):
    import torch
    import optimization_training
    model = torch.nn.Linear(3, 2)
    model.weight.grad = torch.ones_like(model.weight)
    params = list(model.parameters())
    observer = optimization_training.TrainingObserver(tmp_path)
    before = torch.get_rng_state().clone()
    gradient = model.weight.grad.clone()
    observer.on_start(model)
    observer.on_batch(0, torch.ones(2, 3), torch.tensor([0, 1]))
    observer.on_backward(0, torch.tensor(1.), params, 65536.)
    observer.on_step(0, 65536., 65536.)
    observer.close()
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(model.weight.grad, gradient)
    row = json.loads((tmp_path / 'steps.jsonl').read_text())
    assert row['amp_scale'] == 65536
    assert not row['amp_skipped_step']


def test_training_instrumentation_keeps_legacy_amp_order():
    import ast
    source = (Path(__file__).parents[1] / 'lora.py').read_text(encoding='utf-8')
    assert '_optimization_observer' in source
    assert source.index('scaler.scale(loss).backward()') < source.index('clip_grad_norm_')
    assert source.index('clip_grad_norm_') < source.index('scaler.step(optimizer)')
    assert 'unscale_(' not in source


def test_runtime_args_do_not_confuse_test_selection_or_execution_metadata():
    from optimization_experiment import runtime_args
    cell = dict(dataset='eurosat', shots=4, seed=1, routing='dynamic')
    args = runtime_args(cell, 'combined', '/data', '/new')
    assert args.n_iters * args.shots == 2000
    assert args.adapter_compute == 'factorized' and args.attention_backend == 'sdpa'
    assert args.selection_split == 'val' and not args.report_test
    assert args.r == 8 and args.num_heads == 2 and args.ramp_up_steps == 100


def test_reference_selection_requires_unique_matching_checkpoint():
    from optimization_experiment import find_reference
    cell = dict(dataset='eurosat', shots=4, seed=1, routing='dynamic')
    row = dict(config=dict(dataset='eurosat', shots=4, seed=1), checkpoint=dict(path='x', sha256='a'))
    assert find_reference([row], cell) is row
    with pytest.raises(ValueError):
        find_reference([row, row], cell)


def test_profile_summary_checks_exact_repetition_coverage():
    from optimization_analysis import summarize_profile
    rows = [dict(dataset='eurosat', routing='dynamic', scope='training', arm=arm,
                 repetition=rep, median_ms=value, p95_ms=value,
                 peak_allocated_bytes=1024, peak_reserved_bytes=2048)
            for arm, value in [('original', 100.), ('combined', 80.)]
            for rep in range(1, 6)]
    result = summarize_profile(rows)
    assert next(r for r in result if r['arm'] == 'combined')['speed_gate_passed']
    with pytest.raises(ValueError):
        summarize_profile(rows[:-1])


def test_observer_records_amp_overflow_without_changing_it(tmp_path):
    import torch
    from optimization_training import TrainingObserver
    p = torch.nn.Parameter(torch.ones(2))
    p.grad = torch.full_like(p, float('inf'))
    observer = TrainingObserver(tmp_path)
    observer.on_batch(0, torch.ones(1, 2), torch.tensor([0]))
    observer.on_backward(0, torch.tensor(1.), [p], 65536.)
    observer.on_step(0, 65536., 32768.)
    observer.close()
    row = json.loads((tmp_path / 'steps.jsonl').read_text())
    assert row['amp_skipped_step'] and row['unscaled_gradient_norm'] is None
    assert torch.isinf(p.grad).all()


def test_profile_backend_gate_rejects_silent_manual_fallback():
    from optimization_profile import audit_backend_counts
    audit_backend_counts(dict(sdpa=24, manual=0, fallback=0, mask_validations=0), 'combined')
    with pytest.raises(ValueError):
        audit_backend_counts(dict(sdpa=12, manual=12, fallback=12, mask_validations=0), 'sdpa')


def test_profile_builds_frozen_split_before_adapter_rng_consumption():
    source = (Path(__file__).parents[1] / 'optimization_profile.py').read_text()
    body = source[source.index('def profile_suite('):]
    assert body.index('build_data(') < body.index('load_model(')


def test_production_autocast_boundary_helpers_match_direct_normalization():
    import torch
    from optimization_experiment import normalize_features
    x = torch.tensor([[1., 2., 3.]], dtype=torch.float16)
    expected = x / x.norm(dim=-1, keepdim=True)
    assert torch.equal(normalize_features(x), expected)
    assert normalize_features(x).dtype == x.dtype


def test_sdpa_kernel_audit_rejects_math_fallback():
    from optimization_profile import audit_sdpa_operators
    audit_sdpa_operators(['aten::_scaled_dot_product_flash_attention', 'flash_fwd_kernel'], 'combined')
    with pytest.raises(ValueError):
        audit_sdpa_operators(['aten::_scaled_dot_product_attention_math', 'aten::bmm'], 'sdpa')


def test_numerical_smoke_covers_actual_training_and_inference_shapes():
    source = (Path(__file__).parents[1] / 'optimization_profile.py').read_text()
    body = source[source.index('def numerical_smoke('):source.index('def measure(')]
    assert 'images[:32]' in body and "'inference'" in body
    assert 'images[:2]' not in body


def test_raw_prediction_audit_recomputes_instead_of_trusting_summary(tmp_path):
    from optimization_analysis import audit_prediction_group
    reference = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    labels = np.array([0, 1])
    for arm in ('original', 'combined'):
        directory = tmp_path / arm
        directory.mkdir()
        logits = reference if arm == 'original' else reference[:, ::-1]
        np.savez_compressed(directory / 'predictions.npz', logits=logits, labels=labels,
                            sample_ids=np.array(['a', 'b']), prediction=logits.argmax(1), margin=np.array([1., 1.]))
        write_json(directory / 'inputs.json', dict(image_batch_sha256=['same'], token_sha256='same',
                   backend_counts=dict(sdpa=1, manual=0, fallback=0), text_features_sha256='own'))
        write_json(directory / 'comparison.json', dict(passed=True, accuracy_delta_pp=0, agreement_percent=100))
    with pytest.raises(ValueError):
        audit_prediction_group(tmp_path, ('original', 'combined'))


def test_tied_top_scores_use_argmax_not_unstable_argsort():
    from optimization_experiment import prediction_and_margin
    logits = np.zeros((1, 101), dtype=np.float32)
    logits[0, 2:4] = 1.
    prediction, margin = prediction_and_margin(logits)
    assert prediction.tolist() == [2]
    assert margin.tolist() == [0.]


def test_frozen_archive_flattened_revision_materials_layout(tmp_path):
    from optimization_experiment import checked_checkpoint
    path = tmp_path / 'reviewer4' / 'raw' / 'checkpoint.pt'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'known checkpoint')
    record = dict(checkpoint=dict(path='revision_materials/reviewer4/raw/checkpoint.pt', sha256=sha256(path)))
    assert checked_checkpoint(tmp_path, record) == path.resolve()
    record['checkpoint']['sha256'] = 'wrong'
    with pytest.raises(ValueError):
        checked_checkpoint(tmp_path, record)
    record['checkpoint']['path'] = '../outside.pt'
    with pytest.raises(ValueError):
        checked_checkpoint(tmp_path, record)


def test_gate_diagnostic_separates_relative_and_absolute_errors():
    import torch
    from optimization_diagnose_gate import tensor_error
    a = torch.tensor([0., 100.])
    b = torch.tensor([.001, 100.001])
    report = tensor_error(a, b, 1e-5, 1e-4)
    assert not report['passed'] and report['violations'] == 1
    assert report['max_abs_error'] == pytest.approx(.001, abs=1e-5)
    assert report['finite']


def test_amended_protocol_keeps_original_failure_and_thresholds():
    from optimization_amended import frozen_amendment
    protocol = frozen_amendment()
    assert protocol['original_numerical_gate_passed'] is False
    assert protocol['stage_order'] == ['old-eval', 'profile', 'train']
    assert protocol['accuracy_gate'] == {'absolute_delta_pp': .05, 'agreement_percent': 99.9}
    assert protocol['numeric_tolerances'] == {'fp32': [1e-5, 1e-4], 'fp16': [.02, .02]}
    assert protocol['numeric_tolerance_policy'] == 'report_only_hard_fail_nonfinite'
    assert protocol['automatic_production_promotion'] is False


def test_amended_speed_gate_cannot_pass_empty_or_partial_coverage():
    from optimization_amended import enforce_speed_gate
    with pytest.raises(ValueError):
        enforce_speed_gate([])
    rows = [dict(dataset=d,routing='dynamic',scope=s,arm='combined',speed_gate_passed=True)
            for d in ('eurosat','caltech101') for s in ('training','inference')]
    enforce_speed_gate(rows)
    rows[-1]['speed_gate_passed'] = False
    with pytest.raises(RuntimeError):
        enforce_speed_gate(rows)


def test_amended_job_does_not_label_original_experiment_complete():
    from optimization_job import experiment_command
    from types import SimpleNamespace
    opts = SimpleNamespace(stage='all',amended_protocol=True,evidence_root='/frozen',data_root='/data')
    command = experiment_command(opts, Path('/new/output'))
    assert 'optimization_amended.py' in command
    assert 'optimization_experiment.py' not in command


def test_report_only_numerical_policy_still_rejects_nonfinite():
    from optimization_profile import enforce_numerical_policy
    enforce_numerical_policy([dict(passed=False,finite=True)], strict=False)
    with pytest.raises(RuntimeError):
        enforce_numerical_policy([dict(passed=False,finite=True)], strict=True)
    with pytest.raises(RuntimeError):
        enforce_numerical_policy([dict(passed=True,finite=False)], strict=False)


def test_amended_prior_source_and_verdict_come_from_verified_archive(tmp_path,monkeypatch):
    import zipfile
    import optimization_amended as amended
    prior,source=tmp_path/'prior',tmp_path/'source'
    prior.mkdir();source.mkdir()
    names=['loralib/layers_OH_singlora.py','loralib/layers_singlora.py','loralib/utils.py',
           'execution_backends.py','run_utils.py','clip/model.py','adapter_checkpoint_v2.py','lora.py']
    archive=prior/'evidence.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('output/smoke/eurosat_dynamic/full_clip_numerical_smoke.json',
                   json.dumps([{'passed':i<5} for i in range(16)]))
        for name in names:
            p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('archived content')
            z.writestr('source/'+name,p.read_bytes())
    write_json(prior/'archive_ready.json',dict(exit_code=1,full_experiment=False,sha256=sha256(archive),archive=archive.name))
    monkeypatch.setattr(amended,'PRIOR_ARCHIVE_SHA',sha256(archive))
    result=amended.validate_prior_failure(prior,source)
    assert result['original_failure_preserved']
    (source/'lora.py').write_text('modified after archived failure')
    with pytest.raises(ValueError):amended.validate_prior_failure(prior,source)
