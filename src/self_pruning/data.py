from __future__ import annotations

from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 0,
    dataset_name: str = "cifar10",
) -> tuple[DataLoader, DataLoader]:
    transform = build_transforms()
    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "cifar10":
        try:
            train_set = datasets.CIFAR10(
                root=str(data_dir),
                train=True,
                download=True,
                transform=transform,
            )
            test_set = datasets.CIFAR10(
                root=str(data_dir),
                train=False,
                download=True,
                transform=transform,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to load CIFAR-10. If the dataset is not already present locally, "
                "download access is required. For a local smoke test, run with "
                "`--dataset fake` instead."
            ) from exc
    elif dataset_name == "fake":
        train_set = datasets.FakeData(
            size=2048,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )
        test_set = datasets.FakeData(
            size=512,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader
