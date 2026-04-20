"""Self-pruning network utilities for the Tredence AI Engineering case study."""

from .engine import ExperimentConfig, get_lambda, run_experiment, run_sweep
from .metrics import gate_statistics, overall_sparsity, sparsity_regularizer
from .model import PrunableLinear, SelfPruningNet

__all__ = [
    "ExperimentConfig",
    "PrunableLinear",
    "SelfPruningNet",
    "gate_statistics",
    "get_lambda",
    "overall_sparsity",
    "run_experiment",
    "run_sweep",
    "sparsity_regularizer",
]

