"""Loader for CIFAR-10 on frozen ResNet50 features.

HDO never touches the ResNet50 itself — the paper fine-tunes only the
last (classification) layer on top of frozen pretrained features, and so
do we. The expensive ResNet50 forward pass happens exactly ONCE, offline,
via scripts/extract_cifar10_features.py; everything this loader returns
is just a small (N, 2048) feature tensor + labels, cheap enough that
zeroth-order's extra forward passes during the actual HDO simulation are
affordable.

Expected local files (produced by scripts/extract_cifar10_features.py):
  data/cifar10_features/train_features.pt
  data/cifar10_features/train_labels.pt
  data/cifar10_features/val_features.pt
  data/cifar10_features/val_labels.pt
"""

import os
from typing import Tuple

import torch


def load_cifar10_features(
    data_dir: str = "data/cifar10_features",
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    paths = {
        "train_X": os.path.join(data_dir, "train_features.pt"),
        "train_y": os.path.join(data_dir, "train_labels.pt"),
        "val_X": os.path.join(data_dir, "val_features.pt"),
        "val_y": os.path.join(data_dir, "val_labels.pt"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing precomputed CIFAR-10 feature files: "
            + ", ".join(missing)
            + ". Run `python scripts/extract_cifar10_features.py` first "
              "(see README.md's 'Dataset: CIFAR-10' section)."
        )

    X_train = torch.load(paths["train_X"])
    y_train = torch.load(paths["train_y"]).long()
    X_val = torch.load(paths["val_X"])
    y_val = torch.load(paths["val_y"]).long()
    return (X_train, y_train), (X_val, y_val)
