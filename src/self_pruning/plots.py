from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_gate_histogram(gate_values, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(gate_values.tolist(), bins=40, color="#1f77b4", edgecolor="white")
    plt.xlabel("Gate value after sigmoid")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_training_curves(history: list[dict[str, float]], output_path: Path, title: str) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    test_accuracy = [row["test_accuracy"] * 100 for row in history]
    sparsity = [row["overall_sparsity"] * 100 for row in history]
    lambda_values = [row["active_lambda"] for row in history]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(epochs, test_accuracy, marker="o", color="#2ca02c", label="Test accuracy")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(title)
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.plot(epochs, sparsity, marker="o", color="#d62728", label="Overall sparsity")
    ax2.plot(epochs, lambda_values, marker="s", linestyle="--", color="#1f77b4", label="Active lambda")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Value")
    ax2.grid(alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_layerwise_sparsity(
    experiment_summaries: list[dict[str, object]],
    output_path: Path,
    title: str = "Layer-wise sparsity by lambda",
) -> None:
    if not experiment_summaries:
        return

    layer_names = list(experiment_summaries[0]["layer_sparsity"].keys())
    lambdas = [summary["target_lambda"] for summary in experiment_summaries]
    bar_width = 0.8 / max(1, len(experiment_summaries))
    positions = list(range(len(layer_names)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for index, summary in enumerate(experiment_summaries):
        values = [summary["layer_sparsity"][name] * 100 for name in layer_names]
        shifted = [pos + index * bar_width for pos in positions]
        plt.bar(shifted, values, width=bar_width, label=f"lambda={lambdas[index]:g}")

    tick_positions = [pos + bar_width * (len(experiment_summaries) - 1) / 2 for pos in positions]
    plt.xticks(tick_positions, layer_names)
    plt.ylabel("Sparsity (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

