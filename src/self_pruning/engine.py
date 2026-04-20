from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .data import build_dataloaders
from .metrics import collect_gate_values, gate_statistics, sparsity_regularizer
from .model import SelfPruningNet
from .plots import plot_gate_histogram, plot_layerwise_sparsity, plot_training_curves
from .reporting import build_results_rows, write_csv, write_json, write_markdown_summary


@dataclass
class ExperimentConfig:
    output_dir: Path
    data_dir: Path = Path("data")
    dataset: str = "cifar10"
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    target_lambda: float = 1e-3
    warmup_epochs: int = 5
    ramp_epochs: int = 10
    threshold: float = 1e-2
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    hidden_dim_1: int = 512
    hidden_dim_2: int = 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_lambda(epoch: int, target_lambda: float, warmup: int, rampup: int) -> float:
    if epoch < warmup:
        return 0.0
    if rampup <= 0:
        return target_lambda
    if epoch < warmup + rampup:
        progress = (epoch - warmup + 1) / rampup
        return target_lambda * progress
    return target_lambda


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    epoch: int,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.train()
    active_lambda = get_lambda(epoch, config.target_lambda, config.warmup_epochs, config.ramp_epochs)

    total_examples = 0
    total_loss = 0.0
    total_task_loss = 0.0
    total_sparsity_loss = 0.0
    total_correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        task_loss = F.cross_entropy(logits, labels)
        sparse_loss = sparsity_regularizer(model)
        loss = task_loss + active_lambda * sparse_loss
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        predictions = logits.argmax(dim=1)
        total_examples += batch_size
        total_correct += (predictions == labels).sum().item()
        total_loss += loss.item() * batch_size
        total_task_loss += task_loss.item() * batch_size
        total_sparsity_loss += sparse_loss.item() * batch_size

    return {
        "train_accuracy": total_correct / max(1, total_examples),
        "train_loss": total_loss / max(1, total_examples),
        "train_task_loss": total_task_loss / max(1, total_examples),
        "train_sparsity_loss": total_sparsity_loss / max(1, total_examples),
        "active_lambda": active_lambda,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return {
        "test_loss": total_loss / max(1, total_examples),
        "test_accuracy": total_correct / max(1, total_examples),
    }


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_name=config.dataset,
    )

    model = SelfPruningNet(
        hidden_dim_1=config.hidden_dim_1,
        hidden_dim_2=config.hidden_dim_2,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: list[dict[str, float]] = []
    best_test_accuracy = 0.0
    best_epoch = 0

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, config)
        test_metrics = evaluate(model, test_loader, device)
        gate_summary = gate_statistics(model, threshold=config.threshold)
        epoch_time = time.perf_counter() - epoch_start

        history_row = {
            "epoch": epoch + 1,
            **train_metrics,
            **test_metrics,
            "overall_sparsity": gate_summary["overall_sparsity"],
            "gate_mean": gate_summary["gate_mean"],
            "gate_std": gate_summary["gate_std"],
            "epoch_seconds": epoch_time,
        }
        history.append(history_row)

        if test_metrics["test_accuracy"] >= best_test_accuracy:
            best_test_accuracy = test_metrics["test_accuracy"]
            best_epoch = epoch + 1
            # Accuracy and sparsity can drift apart late in training, so keep
            # the best-performing checkpoint as well as the final model.
            torch.save(model.state_dict(), config.output_dir / "best_model.pt")

        print(
            f"[lambda={config.target_lambda:g}] "
            f"epoch {epoch + 1:02d}/{config.epochs} "
            f"train_acc={train_metrics['train_accuracy'] * 100:6.2f}% "
            f"test_acc={test_metrics['test_accuracy'] * 100:6.2f}% "
            f"sparsity={gate_summary['overall_sparsity'] * 100:6.2f}% "
            f"active_lambda={train_metrics['active_lambda']:.6f}"
        )

    final_gate_summary = gate_statistics(model, threshold=config.threshold)
    gate_values = collect_gate_values(model)

    write_csv(config.output_dir / "history.csv", history)
    write_json(config.output_dir / "history.json", history)
    torch.save(model.state_dict(), config.output_dir / "final_model.pt")

    plot_training_curves(
        history,
        config.output_dir / "training_curves.png",
        title=f"Training curves (lambda={config.target_lambda:g})",
    )
    plot_gate_histogram(
        gate_values,
        config.output_dir / "gate_histogram.png",
        title=f"Gate value distribution (lambda={config.target_lambda:g})",
    )

    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir), "data_dir": str(config.data_dir)},
        "target_lambda": config.target_lambda,
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_accuracy,
        "final_test_accuracy": history[-1]["test_accuracy"],
        "final_test_loss": history[-1]["test_loss"],
        "overall_sparsity": final_gate_summary["overall_sparsity"],
        "gate_mean": final_gate_summary["gate_mean"],
        "gate_std": final_gate_summary["gate_std"],
        "dead_gates": final_gate_summary["dead_gates"],
        "total_gates": final_gate_summary["total_gates"],
        "layer_sparsity": final_gate_summary["layer_sparsity"],
        "device": str(device),
    }
    write_json(config.output_dir / "summary.json", summary)
    return summary


def run_sweep(base_config: ExperimentConfig, lambda_values: list[float]) -> list[dict[str, object]]:
    sweep_dir = base_config.output_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for lambda_value in lambda_values:
        run_name = f"lambda_{str(lambda_value).replace('.', 'p').replace('-', 'm')}"
        config = ExperimentConfig(
            **{**asdict(base_config), "target_lambda": lambda_value, "output_dir": sweep_dir / run_name}
        )
        summaries.append(run_experiment(config))

    write_json(sweep_dir / "summary.json", summaries)
    write_csv(sweep_dir / "summary.csv", build_results_rows(summaries))
    write_markdown_summary(sweep_dir / "summary.md", summaries)
    plot_layerwise_sparsity(summaries, sweep_dir / "layerwise_sparsity.png")
    return summaries
