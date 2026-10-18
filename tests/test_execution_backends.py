"""Execution backends must preserve the frozen dense/manual mathematical model."""
import copy
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from argparse import Namespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from adapter_checkpoint_v2 import CHECKPOINT_SCHEMA_VERSION, LEGACY_CHECKPOINT_SCHEMA_VERSION

from loralib.layers_OH_singlora import LinearOHsingLoRA
from loralib.layers_singlora import LinearSingLoRA, PlainMultiheadAttentionAdapter
from loralib.utils import apply_adapter, load_adapter, save_adapter

BASE = "ce709fa5b32b7392db532a3776ab7ecbda8500bc"
ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def frozen():
    # Prefer the immutable reviewer import; git's immutable base is portable to CI.
    folder = Path(os.environ.get("ORTHO_FROZEN_SOURCE", r"D:\RESEARCH\CLIP-LoRA_group\OthorAdapt\revision_materials\reviewer4_import_20261009T093638Z"))
    package = "_frozen_backend_reference"
    sys.modules[package] = types.ModuleType(package)
    modules = []
    for stem in ("layers_singlora", "layers_OH_singlora"):
        source_file = folder / "loralib" / (stem + ".py")
        if "ORTHO_FROZEN_SOURCE" in os.environ:
            assert source_file.is_file(), f"Explicit frozen reference is missing: {source_file}"
        if source_file.exists():
            source = source_file.read_text(encoding="utf-8")
        else:
            source = subprocess.check_output(["git", "show", f"{BASE}:loralib/{stem}.py"], cwd=ROOT).decode("utf-8")
        name = package + "." + stem
        module = types.ModuleType(name)
        sys.modules[name] = module
        exec(compile(source, str(source_file), "exec"), module.__dict__)
        modules.append(module)
    return modules[1].LinearOHsingLoRA, modules[0].PlainMultiheadAttentionAdapter


def linear(cls=LinearOHsingLoRA, width=16, mode="input_conditioned", compute="dense"):
    layer = cls(nn.Linear(width, width), r=8, num_heads=2, routing_mode=mode,
                adapter_compute=compute, ramp_up_steps=100)
    with torch.no_grad():
        layer.lora_A_heads.normal_(std=0.06)
        layer.gating_network[0].weight.normal_(std=0.05)
    return layer


def wrapper(cls=PlainMultiheadAttentionAdapter, linear_cls=LinearOHsingLoRA,
            width=16, mode="input_conditioned", compute="dense", backend="manual",
            batch_first=False, dropout=0.0):
    return cls(nn.MultiheadAttention(width, 2, dropout=dropout, batch_first=batch_first),
               linear_cls, r=8, num_heads=2, routing_mode=mode,
               adapter_compute=compute, attention_backend=backend, ramp_up_steps=100)


@pytest.mark.parametrize("width", [512, 768])
@pytest.mark.parametrize("mode", ["input_conditioned", "input_independent"])
@pytest.mark.parametrize("step", [0, 1, 99, 100, 101])
def test_factorized_output_gradients_and_ramp_match_frozen(frozen, width, mode, step):
    torch.manual_seed(34)
    reference = linear(frozen[0], width, mode)
    current = linear(width=width, mode=mode, compute="factorized")
    current.load_state_dict(reference.state_dict())
    assert current.adapter_compute == "factorized"
    for module in (reference, current):
        module.training_step.fill_(step)
        module.train()
    x = torch.randn(2, 2, width, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = reference(x), current(y)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(y.grad, x.grad, atol=1e-5, rtol=1e-4)
    for (name, p), (_, q) in zip(reference.named_parameters(), current.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(q.grad, p.grad, atol=1e-5, rtol=1e-4, msg=name)
    assert reference.training_step.item() == current.training_step.item() == step + 1


def test_dense_default_is_bit_exact_with_frozen(frozen):
    reference = linear(frozen[0])
    current = linear()
    current.load_state_dict(reference.state_dict())
    x = torch.randn(3, 2, 16)
    torch.testing.assert_close(current(x), reference(x), rtol=0, atol=0)
    assert current.adapter_compute == "dense"
    assert set(current.state_dict()) == set(reference.state_dict())


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mode", ["input_conditioned", "input_independent"])
def test_sdpa_combined_matches_frozen_and_steps_once(frozen, batch_first, causal, mode):
    reference = wrapper(frozen[1], frozen[0], mode=mode, batch_first=batch_first)
    current = wrapper(mode=mode, compute="factorized", backend="sdpa", batch_first=batch_first)
    current.load_state_dict(reference.state_dict())
    x = torch.randn((2, 4, 16) if batch_first else (4, 2, 16), requires_grad=True)
    y = x.detach().clone().requires_grad_()
    mask = torch.full((4, 4), float("-inf")).triu_(1) if causal else None
    expected = reference(x, x, x, need_weights=False, attn_mask=mask)[0]
    actual, weights = current(y, y, y, need_weights=False, attn_mask=mask)
    assert weights is None
    assert current.execution_counts["sdpa"] == 1
    assert current.execution_counts["manual"] == 0
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(y.grad, x.grad, atol=1e-5, rtol=1e-4)
    for name in ("q_proj", "k_proj", "v_proj"):
        assert getattr(current, name).training_step.item() == 1
        for (pn, p), (_, q) in zip(getattr(reference, name).named_parameters(), getattr(current, name).named_parameters()):
            if p.requires_grad:
                torch.testing.assert_close(q.grad, p.grad, atol=1e-5, rtol=1e-4, msg=pn)


@pytest.mark.parametrize("case", ["weights", "padding", "boolean", "3d", "noncausal", "fullmasked", "dropout"])
def test_sdpa_fallback_preserves_manual_including_nan_rows(case):
    current = wrapper(backend="sdpa", dropout=0.1 if case == "dropout" else 0)
    reference = copy.deepcopy(current)
    reference.attention_backend = "manual"
    kwargs = {"need_weights": case == "weights", "average_attn_weights": False}
    if case == "padding":
        kwargs["key_padding_mask"] = torch.tensor([[False, False, True, True], [False] * 4])
    elif case == "boolean":
        kwargs["attn_mask"] = torch.ones(4, 4, dtype=torch.bool).triu(1)
    elif case == "3d":
        kwargs["attn_mask"] = torch.zeros(1, 4, 4)
    elif case == "noncausal":
        kwargs["attn_mask"] = torch.zeros(4, 4)
    elif case == "fullmasked":
        kwargs["attn_mask"] = torch.full((4, 4), float("-inf"))
    x = torch.randn(4, 2, 16)
    torch.manual_seed(59)
    expected, expected_weights = reference(x, x, x, **kwargs)
    torch.manual_seed(59)
    actual, actual_weights = current(x, x, x, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    if expected_weights is not None:
        torch.testing.assert_close(actual_weights, expected_weights, atol=0, rtol=0, equal_nan=True)
    assert current.execution_counts["fallback"] == current.execution_counts["manual"] == 1
    assert current.execution_counts["sdpa"] == 0
    assert current.q_proj.training_step.item() == 1
    if case != "fullmasked":
        assert torch.isfinite(actual).all()


def test_mask_validation_cache_tracks_identity_and_inplace_version():
    current = wrapper(backend="sdpa")
    x = torch.randn(4, 2, 16)
    mask = torch.full((4, 4), float("-inf")).triu_(1)
    for _ in range(3):
        current(x, x, x, need_weights=False, attn_mask=mask)
    assert current.execution_counts["mask_validations"] == 1
    assert current.execution_counts["sdpa"] == 3
    mask[0, 0] = float("-inf")
    current(x, x, x, need_weights=False, attn_mask=mask)
    assert current.execution_counts["mask_validations"] == 2
    assert current.execution_counts["fallback"] == 1
    mask = torch.full((4, 4), float("-inf")).triu_(1)
    current(x, x, x, need_weights=False, attn_mask=mask)
    assert current.execution_counts["mask_validations"] == 3
    assert current.execution_counts["sdpa"] == 4


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
@pytest.mark.parametrize("mode", ["input_conditioned", "input_independent"])
@pytest.mark.parametrize("width", [512, 768])
def test_default_adamw_gradscaler_parameter_updates_match_frozen(frozen, device, mode, width):
    torch.manual_seed(97)
    reference = wrapper(frozen[1], frozen[0], width=width, mode=mode).to(device)
    current = wrapper(width=width, mode=mode).to(device)
    current.load_state_dict(reference.state_dict())
    before = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}
    # Nonzero gate gradients and a saturated ramp exercise both optimizer groups.
    for module in (reference, current):
        module.out_proj.requires_grad_(False)
        for proj in (module.q_proj, module.k_proj, module.v_proj):
            proj.training_step.fill_(99)
    x = torch.randn(3, 2, width, device=device)
    outputs = []
    for module in (reference, current):
        params = [p for p in module.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-2, betas=(0.9, 0.999))
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            output = module(x, x, x, need_weights=False)[0]
            assert output.dtype == (torch.float16 if device == "cuda" else torch.float32)
            loss = output.float().square().mean()
        scaler.scale(loss).backward()
        # Preserve the legacy production order: clipping operates on scaled
        # gradients. GradScaler unscales internally only when step is called.
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        outputs.append(output)
    torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
    for (name, p), (_, q) in zip(reference.named_parameters(), current.named_parameters()):
        if p.requires_grad:
            expected_delta = p.detach() - before[name]
            actual_delta = q.detach() - before[name]
            assert torch.count_nonzero(expected_delta).item() > 0, f"Frozen parameter did not update: {name}"
            # Exact deltas reject both skipped and sign-reversed updates, even
            # when the update is much smaller than a forward-output tolerance.
            torch.testing.assert_close(actual_delta, expected_delta, atol=0, rtol=0, msg=name)
            torch.testing.assert_close(q.grad, p.grad, atol=0, rtol=0, msg=name)


def load_cli():
    stub = types.ModuleType("lora")
    stub.run_lora = object()
    spec = importlib.util.spec_from_file_location("backend_run_utils", ROOT / "run_utils.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"lora": stub}):
        spec.loader.exec_module(module)
    return module


def test_cli_defaults_and_optimized_configuration():
    cli = load_cli()
    with patch.object(sys, "argv", ["main.py"]):
        args = cli.get_arguments()
    assert args.adapter_compute == "dense"
    assert args.attention_backend == "manual"
    with patch.object(sys, "argv", ["main.py", "--adapter", "ohsinglora", "--adapter_compute", "factorized", "--attention_backend", "sdpa"]):
        args = cli.get_arguments()
    assert args.adapter_compute == "factorized"
    assert args.attention_backend == "sdpa"


@pytest.mark.parametrize("argv", [
    ["--adapter", "lora", "--adapter_compute", "factorized"],
    ["--adapter", "singlora", "--attention_backend", "sdpa"],
    ["--adapter", "gmhsinglora", "--adapter_compute", "factorized"],
    ["--adapter", "ohsinglora", "--adapter_compute", "factorized", "--params", "q", "o"],
    ["--adapter", "ohsinglora", "--attention_backend", "sdpa", "--params", "mlp"],
])
def test_unsupported_cli_configurations_are_rejected(argv):
    with patch.object(sys, "argv", ["main.py", *argv]), patch.object(sys, "stderr", io.StringIO()) as stderr:
        with pytest.raises(SystemExit):
            load_cli().get_arguments()
    assert "unrecognized arguments" not in stderr.getvalue()
    assert "ohsinglora" in stderr.getvalue() or "q/k/v" in stderr.getvalue()


def test_constructor_and_apply_adapter_configuration_guards():
    with pytest.raises(ValueError, match="square"):
        LinearOHsingLoRA(nn.Linear(8, 12), r=8, adapter_compute="factorized")
    with pytest.raises(ValueError, match="adapter_compute"):
        linear(compute="unknown")
    with pytest.raises(ValueError, match="attention_backend"):
        wrapper(backend="unknown")
    with pytest.raises(ValueError, match="ohsinglora|LinearOHsingLoRA"):
        wrapper(linear_cls=LinearSingLoRA, backend="sdpa")
    for adapter in ("singlora", "gmhsinglora"):
        with pytest.raises(ValueError, match="ohsinglora"):
            apply_adapter(Namespace(adapter=adapter, adapter_compute="factorized"), nn.Module())


@pytest.mark.parametrize("schema", [CHECKPOINT_SCHEMA_VERSION, LEGACY_CHECKPOINT_SCHEMA_VERSION])
def test_v3_and_v2_backend_crossload_preserves_keys_and_output(schema):
    with tempfile.TemporaryDirectory() as folder:
        args = Namespace(save_path=folder, adapter="ohsinglora", backbone="ViT-B/16", dataset="test", shots=4,
                         seed=1, filename="backend", r=8, alpha=1, num_heads=2, ramp_up_steps=100,
                         params=["q", "k", "v"], position="all", encoder="both", lambda_o=0., ortho_reduction="mean",
                         routing_mode="input_conditioned", adapter_compute="dense", attention_backend="manual")
        dense = wrapper()
        fast = wrapper(compute="factorized", backend="sdpa")
        fast.load_state_dict(dense.state_dict())
        for proj in (dense.q_proj, dense.k_proj, dense.v_proj):
            proj.training_step.fill_(100)
        path = save_adapter(args, dense)
        payload = torch.load(path, weights_only=False)
        assert payload["metadata"]["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert "adapter_compute" not in payload["metadata"]
        if schema == LEGACY_CHECKPOINT_SCHEMA_VERSION:
            payload["metadata"]["schema_version"] = schema
            payload["metadata"].pop("routing_mode")
            payload["metadata"].pop("routing_context_id")
            payload["weights"] = {name: value for name, value in payload["weights"].items() if not name.endswith("routing_context")}
            torch.save(payload, path)
        args.adapter_compute, args.attention_backend = "factorized", "sdpa"
        load_adapter(args, fast)
        assert set(dense.state_dict()) == set(fast.state_dict())
        dense.eval()
        fast.eval()
        x = torch.randn(4, 2, 16)
        torch.testing.assert_close(fast(x, x, x, need_weights=False)[0], dense(x, x, x, need_weights=False)[0], atol=1e-5, rtol=1e-4)


def test_optimized_adapter_budget_remains_460800():
    for mode in ("input_conditioned", "input_independent"):
        count = 0
        for width in (512, 768):
            layer = linear(width=width, mode=mode, compute="factorized")
            count += 12 * 3 * sum(p.numel() for p in layer.parameters() if p.requires_grad)
        assert count == 460800


@pytest.mark.parametrize("mode", ["input_conditioned", "input_independent"])
@pytest.mark.parametrize("step,ramp", [(0, 100), (0, 0), (100, 100)])
def test_eval_and_disabled_ramp_match_frozen_without_advancing(frozen, mode, step, ramp):
    dense = linear(frozen[0], mode=mode)
    fast = linear(mode=mode, compute="factorized")
    fast.load_state_dict(dense.state_dict())
    for module in (dense, fast):
        module.eval()
        module.ramp_up_steps = ramp
        module.training_step.fill_(step)
    x = torch.randn(2, 16)
    torch.testing.assert_close(fast(x), dense(x), atol=1e-5, rtol=1e-4)
    assert fast.training_step.item() == step


def test_apply_adapter_propagates_both_backends_without_forwarding_attention_to_linear():
    class StrictOH(LinearOHsingLoRA):
        def __init__(self, existing_linear, **kwargs):
            assert "attention_backend" not in kwargs
            super().__init__(existing_linear, **kwargs)
    direct = wrapper(linear_cls=StrictOH, compute="factorized", backend="sdpa")
    assert direct.q_proj.adapter_compute == "factorized"
    model = nn.Module()
    model.transformer = nn.Module()
    block = nn.Module()
    block.attn = nn.MultiheadAttention(16, 2)
    model.transformer.resblocks = nn.ModuleList([block])
    args = Namespace(adapter="ohsinglora", encoder="text", position="all", params=["q", "k", "v"],
                     r=8, alpha=1, ramp_up_steps=100, num_heads=2, routing_mode="input_independent",
                     adapter_compute="factorized", attention_backend="sdpa")
    apply_adapter(args, model)
    attention = block.attn
    assert attention.attention_backend == "sdpa"
    assert attention.q_proj.adapter_compute == "factorized"
    assert attention.q_proj.routing_mode == "input_independent"


def test_inference_tensor_mask_is_revalidated_after_inplace_change():
    attention = wrapper(backend="sdpa")
    with torch.inference_mode():
        x = torch.randn(4, 2, 16)
        mask = torch.full((4, 4), float("-inf")).triu_(1)
        attention(x, x, x, need_weights=False, attn_mask=mask)
        mask[0, 0] = float("-inf")
        result = attention(x, x, x, need_weights=False, attn_mask=mask)[0]
    assert attention.execution_counts["sdpa"] == 1
    assert attention.execution_counts["fallback"] == 1
    assert attention.execution_counts["mask_validations"] == 2
    assert torch.isnan(result[0]).all()


@pytest.mark.parametrize("width", [512, 768])
@pytest.mark.parametrize("mode", ["input_conditioned", "input_independent"])
@pytest.mark.parametrize("step", [0, 1, 99, 100, 101])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
def test_explicit_half_factorization_preserves_dtype_and_gradients(frozen, width, mode, step, device):
    dense = linear(frozen[0], width=width, mode=mode).to(device=device, dtype=torch.float16)
    fast = linear(width=width, mode=mode, compute="factorized").to(device=device, dtype=torch.float16)
    fast.load_state_dict(dense.state_dict())
    for module in (dense, fast):
        module.training_step.fill_(step)
    x = torch.randn(2, width, device=device, dtype=torch.float16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = dense(x), fast(y)
    assert dense.training_step.item() == fast.training_step.item() == step + 1
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(y.grad, x.grad, atol=0.02, rtol=0.02)
    for (name, p), (_, q) in zip(dense.named_parameters(), fast.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(q.grad, p.grad, atol=0.02, rtol=0.02, msg=name)


def test_backend_selection_consumes_no_extra_rng_or_state(frozen):
    torch.manual_seed(71)
    reference = wrapper(frozen[1], frozen[0])
    reference_rng = torch.get_rng_state()
    torch.manual_seed(71)
    fast = wrapper(compute="factorized", backend="sdpa")
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert set(fast.state_dict()) == set(reference.state_dict())
    for name, value in fast.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)


def test_default_attention_output_and_gradients_are_bit_exact(frozen):
    reference = wrapper(frozen[1], frozen[0])
    current = wrapper()
    current.load_state_dict(reference.state_dict())
    x = torch.randn(4, 2, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    mask = torch.full((4, 4), float("-inf")).triu_(1)
    expected = reference(x, x, x, attn_mask=mask)[0]
    actual = current(y, y, y, attn_mask=mask)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(y.grad, x.grad, atol=0, rtol=0)
    for (name, p), (_, q) in zip(reference.named_parameters(), current.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(q.grad, p.grad, atol=0, rtol=0, msg=name)
