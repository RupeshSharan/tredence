from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 ResNet on GPU only.")
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "resnet")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic cuDNN settings for reproducibility instead of maximum throughput.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=15,
        help="Stop if test accuracy does not improve for this many epochs after the minimum stop epoch.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.001,
        help="Minimum absolute test-accuracy improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=30,
        help="Do not allow early stopping before this many epochs have completed.",
    )
    parser.add_argument("--amp", dest="amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision.")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. This script is intentionally GPU-only. "
            "Install a CUDA-enabled PyTorch build and run on a machine with an NVIDIA GPU."
        )
    return torch.device("cuda")


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    train_set = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **common_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **common_kwargs)
    return train_loader, test_loader


def build_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        model = resnet18(weights=None)
    elif model_name == "resnet34":
        model = resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # CIFAR-10 images are small, so the ImageNet stem is unnecessarily aggressive.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return {
        "test_loss": total_loss / total_examples,
        "test_accuracy": total_correct / total_examples,
    }


def write_history_csv(output_path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_curves(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_acc = [row["train_accuracy"] * 100 for row in history]
    test_acc = [row["test_accuracy"] * 100 for row in history]
    train_loss = [row["train_loss"] for row in history]
    test_loss = [row["test_loss"] for row in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(epochs, train_acc, label="Train accuracy", color="#1f77b4", marker="o")
    ax1.plot(epochs, test_acc, label="Test accuracy", color="#2ca02c", marker="o")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.plot(epochs, train_loss, label="Train loss", color="#d62728", marker="o")
    ax2.plot(epochs, test_loss, label="Test loss", color="#9467bd", marker="o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = require_cuda()
    set_seed(args.seed, deterministic=args.deterministic)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = bool(args.amp)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history: list[dict[str, float]] = []
    best_test_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    print(
        f"Training {args.model} on CUDA: {torch.cuda.get_device_name(0)} | "
        f"epochs={args.epochs} batch_size={args.batch_size} amp={use_amp} "
        f"workers={args.num_workers} deterministic={args.deterministic} "
        f"early_stop(patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}, "
        f"min_epochs={args.early_stop_min_epochs})"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        test_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": train_metrics["train_loss"],
            "train_accuracy": train_metrics["train_accuracy"],
            "test_loss": test_metrics["test_loss"],
            "test_accuracy": test_metrics["test_accuracy"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        if test_metrics["test_accuracy"] > best_test_accuracy + args.early_stop_min_delta:
            best_test_accuracy = test_metrics["test_accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"lr={lr:.5f} "
            f"train_acc={train_metrics['train_accuracy'] * 100:6.2f}% "
            f"test_acc={test_metrics['test_accuracy'] * 100:6.2f}% "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"test_loss={test_metrics['test_loss']:.4f} "
            f"time={epoch_seconds:6.1f}s"
        )

        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and epochs_without_improvement >= args.early_stop_patience
        ):
            stopped_early = True
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best test accuracy was {best_test_accuracy * 100:.2f}% at epoch {best_epoch}."
            )
            break

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    write_history_csv(output_dir / "history.csv", history)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "model": args.model,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "label_smoothing": args.label_smoothing,
        "amp": use_amp,
        "deterministic": args.deterministic,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stop_min_epochs": args.early_stop_min_epochs,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_accuracy,
        "final_test_accuracy": history[-1]["test_accuracy"],
        "final_train_accuracy": history[-1]["train_accuracy"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_curves(history, output_dir / "training_curves.png")

    print(
        "Done:",
        {
            "best_epoch": best_epoch,
            "best_test_accuracy": round(best_test_accuracy * 100, 2),
            "final_test_accuracy": round(history[-1]["test_accuracy"] * 100, 2),
            "output_dir": str(output_dir),
        },
    )


if __name__ == "__main__":
    main()
