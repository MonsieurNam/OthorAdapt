import unittest
from argparse import Namespace

import torch
import torch.nn as nn

from loralib.layers_OH_singlora import LinearOHsingLoRA
from loralib.utils import apply_adapter


def make_layer(in_features=8, out_features=8, r=4, routing_mode="input_conditioned"):
    base = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.eye(out_features, in_features))
    return LinearOHsingLoRA(
        base,
        r=r,
        lora_alpha=1,
        ramp_up_steps=100,
        num_heads=2,
        routing_mode=routing_mode,
    )


class Reviewer4RoutingTest(unittest.TestCase):
    def test_dynamic_forward_matches_legacy_formula(self):
        torch.manual_seed(5)
        layer = make_layer(routing_mode="input_conditioned")
        layer.eval()
        with torch.no_grad():
            layer.lora_A_heads.copy_(torch.randn_like(layer.lora_A_heads))
            layer.gating_network[0].weight.copy_(
                torch.randn_like(layer.gating_network[0].weight)
            )
            layer.training_step.fill_(100)

        inputs = torch.randn(3, 4, 8)
        original = torch.nn.functional.linear(inputs, layer.weight, layer.bias)
        gates = layer.gating_network(inputs.detach())
        heads_out = layer.lora_A_heads[:, : layer.out_features, :]
        heads_in = layer.lora_A_heads[:, : layer.in_features, :]
        delta = heads_out @ heads_in.transpose(1, 2)
        legacy_adjustment = torch.einsum("...i,hio,...h->...o", inputs, delta, gates)
        expected = original + legacy_adjustment * layer.u() * layer.scaling

        torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=1e-6)

    def test_input_independent_routing_is_invariant_to_input(self):
        torch.manual_seed(7)
        layer = make_layer(routing_mode="input_independent")
        with torch.no_grad():
            layer.gating_network[0].weight.copy_(
                torch.randn_like(layer.gating_network[0].weight)
            )

        first = layer.routing_weights(torch.randn(2, 3, 8))
        second = layer.routing_weights(torch.randn(5, 1, 8))

        torch.testing.assert_close(first[0, 0], second[0, 0])
        torch.testing.assert_close(first, first[0, 0].expand_as(first))
        torch.testing.assert_close(second, second[0, 0].expand_as(second))

    def test_input_conditioned_routing_can_vary_with_input(self):
        layer = make_layer(routing_mode="input_conditioned")
        with torch.no_grad():
            layer.gating_network[0].weight.zero_()
            layer.gating_network[0].weight[0, 0] = 1.0
            layer.gating_network[0].weight[1, 0] = -1.0

        inputs = torch.zeros(2, 8)
        inputs[0, 0] = 1.0
        inputs[1, 0] = -1.0
        gates = layer.routing_weights(inputs)

        self.assertFalse(torch.allclose(gates[0], gates[1]))

    def test_input_independent_and_conditioned_have_same_parameter_count(self):
        dynamic = make_layer(routing_mode="input_conditioned")
        static = make_layer(routing_mode="input_independent")

        dynamic_count = sum(p.numel() for p in dynamic.parameters() if p.requires_grad)
        static_count = sum(p.numel() for p in static.parameters() if p.requires_grad)

        self.assertEqual(dynamic_count, static_count)

    def test_vit_b16_qkv_adapter_budget_is_460800_for_both_modes(self):
        def budget(mode):
            modules = []
            for width in [512] * 12 + [768] * 12:
                for _ in range(3):
                    modules.append(
                        make_layer(
                            in_features=width,
                            out_features=width,
                            r=8,
                            routing_mode=mode,
                        )
                    )
            return sum(
                parameter.numel()
                for module in modules
                for parameter in module.parameters()
                if parameter.requires_grad
            )

        self.assertEqual(budget("input_conditioned"), 460_800)
        self.assertEqual(budget("input_independent"), 460_800)

    def test_unknown_routing_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "routing_mode"):
            make_layer(routing_mode="uniformish")

    def test_apply_adapter_propagates_input_independent_mode(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(8, 2)
                self.mlp = nn.Sequential()

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.resblocks = nn.ModuleList([Block()])

        class Clip(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = Encoder()

        args = Namespace(
            adapter="ohsinglora",
            encoder="text",
            position="bottom",
            backbone="ViT-B/16",
            params=["q", "k", "v"],
            r=4,
            alpha=1,
            ramp_up_steps=100,
            num_heads=2,
            routing_mode="input_independent",
        )
        model = Clip()

        apply_adapter(args, model)

        self.assertEqual(
            model.transformer.resblocks[0].attn.q_proj.routing_mode,
            "input_independent",
        )

    def test_apply_adapter_rejects_static_routing_for_singlora(self):
        args = Namespace(adapter="singlora", routing_mode="input_independent")
        with self.assertRaisesRegex(ValueError, "input_independent"):
            apply_adapter(args, nn.Module())


if __name__ == "__main__":
    unittest.main()
