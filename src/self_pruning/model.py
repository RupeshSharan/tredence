from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PrunableLinear(nn.Module):
    """A linear layer whose weights are modulated by learnable sigmoid gates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_init_std = gate_init_std

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        # Keep sigmoid(gate_scores) near 0.5 at the start so pruning gradients
        # can flow before the sparsity regularizer ramps up.
        nn.init.normal_(self.gate_scores, mean=0.0, std=self.gate_init_std)

    def gate_values(self) -> Tensor:
        return torch.sigmoid(self.gate_scores)

    def masked_weight(self) -> Tensor:
        return self.weight * self.gate_values()

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.masked_weight(), self.bias)

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        return (self.gate_values() < threshold).float().mean().item()


class SelfPruningNet(nn.Module):
    """A small MLP baseline for CIFAR-10 using custom prunable linear layers."""

    def __init__(
        self,
        input_dim: int = 32 * 32 * 3,
        hidden_dim_1: int = 512,
        hidden_dim_2: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.fc1 = PrunableLinear(input_dim, hidden_dim_1)
        self.fc2 = PrunableLinear(hidden_dim_1, hidden_dim_2)
        self.fc3 = PrunableLinear(hidden_dim_2, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
