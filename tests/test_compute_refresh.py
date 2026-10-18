import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def api():
    import compute_refresh
    return compute_refresh


def test_frozen_matrix_has_640_unique_cells():
    m = api()
    cells = m.cells()
    assert len(cells) == len(set(cells)) == 640
    assert set(x[2] for x in cells) == {'forward', 'backward', 'forward_backward', 'inference'}
    m.audit_coverage(cells)
    with pytest.raises(ValueError):
        m.audit_coverage(cells[:-1])
    with pytest.raises(ValueError):
        m.audit_coverage(cells + cells[:1])


def test_economy_profile_is_explicit_and_has_384_unique_cells():
    m = api()
    assert m.timing_settings('economy') == dict(repetitions=3, warmup=10, calls=30)
    assert m.timing_settings('full') == dict(repetitions=5, warmup=20, calls=100)
    assert m.timing_settings('economy', smoke=True) == dict(repetitions=1, warmup=2, calls=3)
    expected = m.cells('economy')
    assert len(expected) == len(set(expected)) == 384
    m.audit_coverage(expected, 'economy')
    with pytest.raises(ValueError): m.audit_coverage(expected)
    with pytest.raises(ValueError): m.audit_coverage(expected[:-1], 'economy')
    with pytest.raises(ValueError): m.audit_coverage(expected + expected[:1], 'economy')
    with pytest.raises(ValueError): m.timing_settings('unknown')


def test_economy_raw_count_is_30_not_silently_mixed_with_100():
    m = api()
    assert m.timing_summary([2.] * 30, [1.] * 30, 32, expected=30)['median_ms'] == 2.
    with pytest.raises(ValueError): m.timing_summary([2.] * 100, [1.] * 100, 32, expected=30)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 0, -1])
def test_bad_timings_rejected(bad):
    with pytest.raises(ValueError):
        api().timing_summary([bad] * 100, [1.] * 100, 32)


def test_raw_timings_recompute_median_p95_throughput():
    out = api().timing_summary(list(range(1, 101)), list(range(2, 102)), 32)
    assert out['median_ms'] == 50.5
    assert out['p95_ms'] == pytest.approx(np.percentile(range(1, 101), 95))
    assert out['images_per_second'] == pytest.approx(32000 / 50.5)
    with pytest.raises(ValueError):
        api().timing_summary([1], [1], 32)


def legacy():
    args = SimpleNamespace(adapter='ohsinglora', r=8, num_heads=2, alpha=1,
        ramp_up_steps=100, params=['q', 'k', 'v'], position='all', encoder='both',
        backbone='ViT-B/16', routing_mode='input_conditioned')
    config = dict(vars(args), dataset='eurosat', shots=4, seed=1, n_iters=500, lambda_o=.03)
    weights = {'q.lora_A_heads': torch.ones(2, 4, 4),
               'q.gating_network.0.weight': torch.ones(2, 4)}
    payload = dict(weights=weights, metadata=dict(vars(args), ecr3=dict(config=config)))
    expected = dict(weights, **{'q.training_step': torch.tensor(0),
                                'q.routing_context': torch.ones(4) / 2})
    record = dict(config=config, status='completed', metrics={'train_total_iterations': 2000},
                  checkpoint={'sha256': 'a' * 64})
    return payload, expected, record, args


def test_legacy_conversion_adds_only_audited_buffers_preserves_tensors():
    payload, expected, record, args = legacy()
    old = copy.deepcopy(payload)
    new, audit = api().convert_legacy(payload, expected, record, args)
    assert new['metadata']['schema_version'] == 'orthadapt.adapter_checkpoint.v3'
    assert set(new['weights']) == set(expected)
    assert new['weights']['q.training_step'].item() == 100
    assert audit['ramp_state_inferred'] is True
    assert sorted(audit['added_keys']) == ['q.routing_context', 'q.training_step']
    for k, v in payload['weights'].items():
        assert torch.equal(v, new['weights'][k]) and torch.equal(v, old['weights'][k])
    assert 'schema_version' not in payload['metadata']


@pytest.mark.parametrize('problem', ['partial', 'missing_weight', 'extra', 'shape', 'incomplete', 'metadata', 'top_metadata'])
def test_conversion_fails_closed(problem):
    payload, expected, record, args = legacy()
    if problem == 'partial': payload['weights']['q.training_step'] = torch.tensor(50)
    if problem == 'missing_weight': del payload['weights']['q.lora_A_heads']
    if problem == 'extra': payload['weights']['surprise'] = torch.tensor(1)
    if problem == 'shape': payload['weights']['q.lora_A_heads'] = torch.ones(2, 4, 3)
    if problem == 'incomplete': record['metrics']['train_total_iterations'] = 99
    if problem == 'metadata': payload['metadata']['ecr3']['config']['r'] = 4
    if problem == 'top_metadata': payload['metadata']['r'] = 4
    with pytest.raises(ValueError): api().convert_legacy(payload, expected, record, args)


def test_strict_lora_nested_coverage_and_shapes():
    layer = SimpleNamespace(**{p: SimpleNamespace(w_lora_A=torch.zeros(8, 4),
        w_lora_B=torch.zeros(4, 8)) for p in ('q_proj', 'k_proj', 'v_proj')})
    weights = {'layer_0': {p: {'w_lora_A': torch.ones(8, 4), 'w_lora_B': torch.ones(4, 8)}
                          for p in ('q_proj', 'k_proj', 'v_proj')}}
    api().validate_lora_weights(weights, [layer])
    del weights['layer_0']['q_proj']['w_lora_B']
    with pytest.raises(ValueError): api().validate_lora_weights(weights, [layer])


def test_parameters_unchanged_check_detects_update():
    model = torch.nn.Linear(2, 2)
    before = api().parameter_digest(model)
    api().assert_unchanged(model, before)
    with torch.no_grad(): model.weight.add_(1)
    with pytest.raises(ValueError): api().assert_unchanged(model, before)


def test_physical_visible_gpu_mapping_is_explicit():
    m = api()
    m.validate_gpu_mapping([m.GPU_UUID], 1, 0, None)
    m.validate_gpu_mapping([m.GPU_UUID], 1, 0, m.GPU_UUID)
    for ids, count, current, mapping in [([m.GPU_UUID, 'other'], 1, 0, None),
        ([m.GPU_UUID], 2, 0, None), ([m.GPU_UUID], 1, 1, None), ([m.GPU_UUID], 1, 0, '1')]:
        with pytest.raises(ValueError): m.validate_gpu_mapping(ids, count, current, mapping)
