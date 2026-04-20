from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_pruning.engine import ExperimentConfig, run_experiment, run_sweep
from self_pruning.model import SelfPruningNet

DEFAULT_SWEEP = [0.0, 1e-4, 1e-3, 1e-2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-pruning MLP on CIFAR-10.")
    parser.add_argument("--lambda", dest="single_lambda", type=float, default=None, help="Target lambda for a single run.")
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=None,
        help="Multiple lambda values for a sweep.",
    )
    parser.add_argument(
        "--default-sweep",
        action="store_true",
        help="Run the four recommended lambdas: 0.0, 1e-4, 1e-3, 1e-2.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--ramp-epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", choices=["cifar10", "fake"], default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--hidden-dim-1", type=int, default=512)
    parser.add_argument("--hidden-dim-2", type=int, default=256)
    parser.add_argument(
        "--shape-check",
        action="store_true",
        help="Run a dummy forward pass before training.",
    )
    return parser.parse_args()


def run_shape_check() -> None:
    model = SelfPruningNet()
    dummy = __import__("torch").randn(4, 3, 32, 32)
    output = model(dummy)
    assert output.shape == (4, 10), f"Expected output shape (4, 10), got {tuple(output.shape)}"
    print("Shape check passed: model(dummy) -> (4, 10)")


def main() -> None:
    args = parse_args()
    if args.shape_check:
        run_shape_check()
        return

    lambda_values = []
    if args.default_sweep:
        lambda_values = DEFAULT_SWEEP
    elif args.lambda_values:
        lambda_values = args.lambda_values
    elif args.single_lambda is not None:
        lambda_values = [args.single_lambda]
    else:
        lambda_values = [1e-3]

    base_config = ExperimentConfig(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        target_lambda=lambda_values[0],
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        threshold=args.threshold,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
    )

    if len(lambda_values) == 1:
        summary = run_experiment(base_config)
        print(
            "Final summary:",
            {
                "lambda": summary["target_lambda"],
                "best_test_accuracy": round(summary["best_test_accuracy"] * 100, 2),
                "overall_sparsity": round(summary["overall_sparsity"] * 100, 2),
                "output_dir": str(base_config.output_dir),
            },
        )
        return

    summaries = run_sweep(base_config, lambda_values)
    print(
        "Sweep complete:",
        [
            {
                "lambda": summary["target_lambda"],
                "best_test_accuracy": round(summary["best_test_accuracy"] * 100, 2),
                "overall_sparsity": round(summary["overall_sparsity"] * 100, 2),
            }
            for summary in summaries
        ],
    )


if __name__ == "__main__":
    main()
