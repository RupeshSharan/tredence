from __future__ import annotations

import csv
import json
from pathlib import Path


def write_json(output_path: Path, payload: dict | list) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_results_rows(experiment_summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary in experiment_summaries:
        row = {
            "lambda": summary["target_lambda"],
            "best_test_accuracy": round(summary["best_test_accuracy"] * 100, 2),
            "final_test_accuracy": round(summary["final_test_accuracy"] * 100, 2),
            "overall_sparsity": round(summary["overall_sparsity"] * 100, 2),
            "gate_mean": round(summary["gate_mean"], 4),
            "gate_std": round(summary["gate_std"], 4),
        }
        for layer_name, layer_value in summary["layer_sparsity"].items():
            row[f"{layer_name}_sparsity"] = round(layer_value * 100, 2)
        rows.append(row)
    return rows


def write_markdown_summary(output_path: Path, experiment_summaries: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not experiment_summaries:
        output_path.write_text("# Experiment Summary\n\nNo runs completed.\n", encoding="utf-8")
        return

    rows = build_results_rows(experiment_summaries)
    headers = list(rows[0].keys())
    divider = ["---"] * len(headers)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")

    best_accuracy_run = max(experiment_summaries, key=lambda item: item["best_test_accuracy"])
    sparsest_run = max(experiment_summaries, key=lambda item: item["overall_sparsity"])
    layer_names = list(experiment_summaries[0]["layer_sparsity"].keys())
    average_layer_sparsity = {
        layer_name: sum(summary["layer_sparsity"][layer_name] for summary in experiment_summaries) / len(experiment_summaries)
        for layer_name in layer_names
    }
    most_pruned_layer = max(average_layer_sparsity, key=average_layer_sparsity.get)
    layer_spread = max(average_layer_sparsity.values()) - min(average_layer_sparsity.values())
    accuracy_spread = max(summary["best_test_accuracy"] for summary in experiment_summaries) - min(
        summary["best_test_accuracy"] for summary in experiment_summaries
    )
    sparsity_spread = max(summary["overall_sparsity"] for summary in experiment_summaries) - min(
        summary["overall_sparsity"] for summary in experiment_summaries
    )

    if layer_spread < 1e-6:
        layer_finding = (
            "The current sweep did not separate the layers much in terms of sparsity, so there is no strong "
            "layer-wise claim to make yet. That is normal for a smoke test or for very short runs, and it is a "
            "signal to rely on the full CIFAR-10 experiments before drawing conclusions."
        )
    else:
        layer_finding = (
            f"Across the sweep, `{most_pruned_layer}` was the most aggressively pruned layer on average. This is a "
            "useful sanity check because earlier fully connected layers often contain more redundant connections than "
            "the final classifier head, which still needs enough capacity to separate the ten CIFAR-10 classes."
        )

    if accuracy_spread < 1e-6 and sparsity_spread < 1e-6:
        tradeoff_finding = (
            "This particular sweep did not produce a meaningful accuracy-versus-sparsity separation yet. That usually "
            "means the run was too short, the dataset was synthetic, or pruning pressure had not ramped up long enough "
            "to create divergence between lambda values."
        )
    else:
        tradeoff_finding = (
            f"The best accuracy in this sweep came from `lambda={best_accuracy_run['target_lambda']:g}`, while the "
            f"sparsest model came from `lambda={sparsest_run['target_lambda']:g}`. That gap captures the central "
            "trade-off of the assignment: stronger pruning is possible, but it usually costs predictive performance."
        )

    lines = [
        "# Self-Pruning Network Report",
        "",
        "## What and why",
        "I implemented a custom MLP for CIFAR-10 where every weight is multiplied by a learnable sigmoid gate. During training, cross-entropy still drives classification performance, while an L1 penalty on the gates pushes unimportant connections toward zero so the model can prune itself.",
        "",
        "## Why L1 creates sparsity",
        "L1 keeps applying pressure even when a gate value is already small, so the optimizer still has an incentive to move that gate closer to zero. L2 would shrink the gradient near zero, which usually leaves many tiny-but-nonzero connections instead of producing clear sparsity.",
        "",
        "## Results table",
        *table_lines,
        "",
        "## Layer-wise finding",
        layer_finding,
        "",
        "## Lambda annealing",
        "I used a warmup-plus-ramp schedule so the model could first learn useful features before sparsity pressure became strong. In practice, this prevents the network from collapsing too early, which is especially important when the regularizer is applied to every gate in a wide first layer.",
        "",
        "## Best trade-off",
        tradeoff_finding,
        "",
        "## Honest limitation",
        "This baseline uses a plain MLP on flattened CIFAR-10 images, so the absolute accuracy ceiling is lower than a convolutional model. That is acceptable for the assignment because the focus is on gate learning and pruning behavior, but it is still a real limitation worth stating.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
