from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def cifar10_stats():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    return mean, std


def build_transforms(config: dict, train: bool):
    aug = config.get("augmentation", {})

    transform_list = []

    if train and aug.get("random_crop", False):
        transform_list.append(transforms.RandomCrop(32, padding=4))

    if train and aug.get("random_horizontal_flip", False):
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())

    if aug.get("normalize", "cifar10") == "cifar10":
        mean, std = cifar10_stats()
        transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)


def maybe_subset(dataset, fraction, seed: int):
    if fraction is None or float(fraction) >= 1.0:
        return dataset

    fraction = float(fraction)
    if fraction <= 0:
        raise ValueError("train_fraction must be larger than 0.")

    n = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()

    return Subset(dataset, indices)


def build_dataloaders(config: dict):
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    root = dataset_cfg.get("root", "./data")
    download = bool(dataset_cfg.get("download", True))
    batch_size = int(training_cfg.get("batch_size", 128))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    seed = int(config.get("seed", 42))

    val_ratio = float(dataset_cfg.get("val_ratio", 0.1))
    train_fraction = dataset_cfg.get("train_fraction", None)

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    train_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=build_transforms(config, train=True),
    )

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=build_transforms(config, train=False),
    )

    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        train_full,
        [train_size, val_size],
        generator=generator,
    )

    train_set = maybe_subset(train_set, train_fraction, seed)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader