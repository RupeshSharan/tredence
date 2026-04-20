from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor, nn

from .model import PrunableLinear


def iter_prunable_layers(model: nn.Module) -> Iterator[tuple[str, PrunableLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, PrunableLinear):
            yield name, module


def collect_gate_values(model: nn.Module) -> Tensor:
    gates = [module.gate_values().detach().reshape(-1).cpu() for _, module in iter_prunable_layers(model)]
    if not gates:
        return torch.empty(0)
    return torch.cat(gates)


def sparsity_regularizer(model: nn.Module) -> Tensor:
    penalties = [module.gate_values().sum() for _, module in iter_prunable_layers(model)]
    if penalties:
        return torch.stack(penalties).sum()
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    return torch.zeros((), device=device)


def layerwise_sparsity(model: nn.Module, threshold: float = 1e-2) -> dict[str, float]:
    return {name: module.get_sparsity(threshold=threshold) for name, module in iter_prunable_layers(model)}


def overall_sparsity(model: nn.Module, threshold: float = 1e-2) -> float:
    gates = collect_gate_values(model)
    if gates.numel() == 0:
        return 0.0
    return (gates < threshold).float().mean().item()


def gate_statistics(model: nn.Module, threshold: float = 1e-2) -> dict[str, object]:
    gate_values = collect_gate_values(model)
    layer_sparsity = layerwise_sparsity(model, threshold=threshold)
    if gate_values.numel() == 0:
        return {
            "dead_gates": 0,
            "total_gates": 0,
            "overall_sparsity": 0.0,
            "gate_mean": 0.0,
            "gate_std": 0.0,
            "layer_sparsity": layer_sparsity,
        }

    dead_gates = int((gate_values < threshold).sum().item())
    total_gates = int(gate_values.numel())
    return {
        "dead_gates": dead_gates,
        "total_gates": total_gates,
        "overall_sparsity": dead_gates / total_gates,
        "gate_mean": float(gate_values.mean().item()),
        "gate_std": float(gate_values.std(unbiased=False).item()),
        "layer_sparsity": layer_sparsity,
    }

