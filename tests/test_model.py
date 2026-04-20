from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_pruning.engine import ExperimentConfig, get_lambda
from self_pruning.metrics import sparsity_regularizer
from self_pruning.model import PrunableLinear, SelfPruningNet


class ModelTests(unittest.TestCase):
    def test_gate_scores_are_registered_parameters(self) -> None:
        layer = PrunableLinear(4, 3)
        self.assertIsInstance(layer.gate_scores, nn.Parameter)
        self.assertIn("gate_scores", dict(layer.named_parameters()))

    def test_forward_shape_matches_cifar10_classifier(self) -> None:
        model = SelfPruningNet()
        dummy = torch.randn(4, 3, 32, 32)
        output = model(dummy)
        self.assertEqual(output.shape, (4, 10))

    def test_model_uses_no_standard_linear_layers(self) -> None:
        model = SelfPruningNet()
        linear_layers = [module for module in model.modules() if type(module) is nn.Linear]
        self.assertEqual(linear_layers, [])

    def test_full_sparsity_when_gates_are_forced_closed(self) -> None:
        layer = PrunableLinear(5, 2)
        with torch.no_grad():
            layer.gate_scores.fill_(-100.0)
        self.assertAlmostEqual(layer.get_sparsity(), 1.0, places=5)

    def test_lambda_schedule_warmup_and_ramp(self) -> None:
        self.assertEqual(get_lambda(epoch=0, target_lambda=1e-3, warmup=2, rampup=3), 0.0)
        self.assertAlmostEqual(get_lambda(epoch=2, target_lambda=1e-3, warmup=2, rampup=3), 1e-3 / 3)
        self.assertEqual(get_lambda(epoch=6, target_lambda=1e-3, warmup=2, rampup=3), 1e-3)

    def test_sparsity_regularizer_is_positive(self) -> None:
        model = SelfPruningNet()
        penalty = sparsity_regularizer(model)
        self.assertGreater(penalty.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

