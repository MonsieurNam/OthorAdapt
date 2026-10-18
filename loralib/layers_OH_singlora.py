# @title loralib/layers_OH_singlora.py
# %%writefile /content/SingloRA_CLIP/loralib/layers_OH_singlora.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import lớp cha từ file gốc
from .layers_singlora import SingLoRALayer

class LinearOHsingLoRA(nn.Linear, SingLoRALayer):
    """
    Implements Orthogonal-Head SingLoRA (OH-SingLoRA).

    Đây là bước đầu tiên trong việc xây dựng DO-SingLoRA. Giai đoạn này tập trung
    vào việc triển khai loss điều chuẩn trực giao, trong khi vẫn giữ lại
    kiến trúc gating của GMH-SingLoRA.
    """
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 2,
        lora_alpha: int = 1,
        ramp_up_steps: int = 100,
        num_heads: int = 2,
        routing_mode: str = "input_conditioned",
        adapter_compute: str = "dense",
        **kwargs
    ):
        if adapter_compute not in {"dense", "factorized"}:
            raise ValueError("adapter_compute must be 'dense' or 'factorized'")
        if adapter_compute == "factorized" and existing_linear.in_features != existing_linear.out_features:
            raise ValueError("adapter_compute=factorized supports square q/k/v projections only")
        self.adapter_compute = adapter_compute
        # Khởi tạo các lớp cha y hệt như LinearGMHSingLoRA
        nn.Linear.__init__(
            self,
            in_features=existing_linear.in_features,
            out_features=existing_linear.out_features,
            bias=existing_linear.bias is not None
        )
        SingLoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, ramp_up_steps=ramp_up_steps)

        if routing_mode not in {"input_conditioned", "input_independent"}:
            raise ValueError(
                "routing_mode must be either 'input_conditioned' or "
                "'input_independent'"
            )

        self.num_heads = num_heads
        self.routing_mode = routing_mode
        self.routing_context_id = (
            "normalized_ones_v1"
            if routing_mode == "input_independent"
            else "not_applicable"
        )
        self.register_buffer(
            "routing_context",
            torch.full(
                (self.in_features,),
                1.0 / math.sqrt(self.in_features),
                dtype=torch.float32,
            ),
        )
        if self.r > 0:
            assert r % num_heads == 0, f"Rank 'r' ({r}) must be divisible by 'num_heads' ({num_heads})."
            self.rank_per_head = r // num_heads

        self.register_buffer('training_step', torch.tensor(0, dtype=torch.long))
        self.load_state_dict(existing_linear.state_dict(), strict=False)

        if self.r > 0:
            # Đóng băng trọng số gốc
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

            self.lora_A_heads = nn.Parameter(
                torch.zeros(
                    self.num_heads,
                    max(self.in_features, self.out_features),
                    self.rank_per_head
                )
            )
            nn.init.normal_(self.lora_A_heads, std=0.01)

            self.gating_network = nn.Sequential(
                nn.Linear(self.in_features, self.num_heads, bias=False),
                nn.Softmax(dim=-1)
            )
            nn.init.zeros_(self.gating_network[0].weight)

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return token-wise or broadcast input-independent routing weights."""
        gate_weight = self.gating_network[0].weight
        if self.routing_mode == "input_conditioned":
            gate_input = x.detach().to(device=gate_weight.device, dtype=gate_weight.dtype)
            return self.gating_network(gate_input)

        context = self.routing_context.to(device=gate_weight.device, dtype=gate_weight.dtype)
        fixed_weights = self.gating_network(context)
        target_shape = (*x.shape[:-1], self.num_heads)
        return fixed_weights.expand(target_shape)

    def forward(self, x: torch.Tensor, **kwargs):
        dtype = x.dtype
        weight = self.weight.to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None

        if self.r == 0:
            return F.linear(x, weight, bias)

        original_output = F.linear(x, weight, bias)

        if self.training:
            self.step()

        gating_weights = self.routing_weights(x)
        gating_weights = gating_weights.to(dtype)

        lora_A_heads = self.lora_A_heads.to(dtype)
        lora_A_heads_out = lora_A_heads[:, :self.out_features, :]
        lora_A_heads_in = lora_A_heads[:, :self.in_features, :]

        if self.adapter_compute == "factorized":
            low = torch.einsum('...i,hik->...hk', x, lora_A_heads_in)
            outputs = torch.einsum('...hk,hok->...ho', low, lora_A_heads_out)
            # Explicit reduction dtype prevents CUDA autocast promoting to FP32.
            lora_adjustment = (
                outputs * gating_weights.to(outputs.dtype).unsqueeze(-1)
            ).sum(dim=-2, dtype=outputs.dtype)
        else:
            all_delta_W = lora_A_heads_out @ lora_A_heads_in.transpose(1, 2)

            lora_adjustment = torch.einsum('...i,hio,...h->...o', x, all_delta_W, gating_weights)

        final_adjustment = lora_adjustment * self.u() * self.scaling

        return original_output + final_adjustment


def calculate_ortho_loss(lora_A_heads: torch.Tensor, reduction: str = 'sum') -> torch.Tensor:
    """
    Tính toán loss điều chuẩn trực giao cho các head của adapter.
    Loss này sẽ nhỏ khi các không gian con được sinh bởi các ma trận A_i là trực giao,
    và lớn khi chúng trùng lặp.

    Loss = sum_{i < j} || A_i^T @ A_j ||^2_F
    Use reduction='mean' to divide by the number of head pairs.

    Args:
        lora_A_heads (torch.Tensor): Tensor chứa các ma trận A của các head.
                                     Shape: (num_heads, D, rank_per_head)

    Returns:
        torch.Tensor: Một scalar tensor chứa giá trị loss.
    """
    if reduction not in {'sum', 'mean'}:
        raise ValueError("reduction must be either 'sum' or 'mean'")

    if lora_A_heads.shape[0] <= 1:
        return torch.tensor(0.0, device=lora_A_heads.device, dtype=lora_A_heads.dtype)

    num_heads = lora_A_heads.shape[0]
    num_pairs = num_heads * (num_heads - 1) // 2

    ortho_loss = torch.tensor(0.0, device=lora_A_heads.device, dtype=lora_A_heads.dtype)

    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            A_i = lora_A_heads[i] # Shape: (D, rank)
            A_j = lora_A_heads[j] # Shape: (D, rank)
            product = A_i.T @ A_j
            loss_pair = torch.norm(product, p='fro')**2
            ortho_loss += loss_pair

    if reduction == 'mean':
        ortho_loss = ortho_loss / num_pairs

    return ortho_loss
