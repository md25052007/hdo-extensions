"""Loader for the Year Prediction MSD dataset (UCI / LibSVM), matching the
paper's own Appendix B.3 setup: standardized 90-dim audio features,
min-max normalized year labels, trained with a linear model + MSE loss.

Expected local file (download instructions are in the project README):
  data/yearmsd/YearPredictionMSD.txt

Format: one row per song, comma-separated, first column = release year
(label), remaining 90 columns = audio features.
"""

import os
from typing import Tuple

import torch


def load_year_msd(
    data_dir: str = "data/yearmsd",
    filename: str = "YearPredictionMSD.txt",
    n_train: int = 2 ** 17,
    n_val: int = 10000,
    seed: int = 0,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download YearPredictionMSD.txt (see README.md's "
            f"'Dataset: Year-MSD' section) and place it at exactly that path."
        )

    years = []
    features = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts or parts[0] == "":
                continue
            years.append(float(parts[0]))
            features.append([float(v) for v in parts[1:91]])

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(years, dtype=torch.float32)

    if n_train + n_val > len(X):
        raise ValueError(
            f"Requested n_train+n_val={n_train + n_val} exceeds dataset size {len(X)}."
        )

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=gen)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Standardize inputs using TRAIN statistics only (avoid val leakage).
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Min-max normalize the year label using TRAIN statistics only.
    y_min, y_max = y_train.min(), y_train.max()
    y_train = (y_train - y_min) / (y_max - y_min + 1e-8)
    y_val = (y_val - y_min) / (y_max - y_min + 1e-8)

    return (X_train, y_train), (X_val, y_val)
