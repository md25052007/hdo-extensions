"""One-time preprocessing: download CIFAR-10, run it through a pretrained
ResNet50, save the 2048-dim pooled features + labels to disk.

Run this ONCE before using the cifar10 task:
    python scripts/extract_cifar10_features.py

It downloads CIFAR-10 automatically via torchvision into data/cifar10_raw/
(standard torchvision cache behavior), then writes:
    data/cifar10_features/train_features.pt
    data/cifar10_features/train_labels.pt
    data/cifar10_features/val_features.pt
    data/cifar10_features/val_labels.pt

Note on fidelity vs. the paper: the paper trains its OWN ResNet50 first
(SGD, 150 epochs) and then freezes it. We use a standard ImageNet-pretrained
ResNet50 instead — a disclosed, reasonable substitution for a course
project, not the paper's exact feature extractor. Say so in the report.
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

RAW_DIR = "data/cifar10_raw"
OUT_DIR = "data/cifar10_features"


def build_feature_extractor() -> nn.Module:
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    resnet = torchvision.models.resnet50(weights=weights)
    resnet.fc = nn.Identity()  # drop the classification head -> 2048-dim pooled features
    resnet.eval()
    return resnet, weights.transforms()


@torch.no_grad()
def extract(loader: DataLoader, model: nn.Module, device: str):
    feats, labels = [], []
    total = len(loader.dataset)
    done = 0
    for images, targets in loader:
        images = images.to(device)
        out = model(images)
        feats.append(out.cpu())
        labels.append(targets)
        done += images.size(0)
        print(f"    {done}/{total} images", end="\r", flush=True)
    print()
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocess = build_feature_extractor()
    model = model.to(device)

    train_set = torchvision.datasets.CIFAR10(root=RAW_DIR, train=True, download=True, transform=preprocess)
    val_set = torchvision.datasets.CIFAR10(root=RAW_DIR, train=False, download=True, transform=preprocess)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2)

    print("Extracting train features...")
    train_X, train_y = extract(train_loader, model, device)
    print(f"  train_X: {train_X.shape}, train_y: {train_y.shape}")

    print("Extracting val features...")
    val_X, val_y = extract(val_loader, model, device)
    print(f"  val_X: {val_X.shape}, val_y: {val_y.shape}")

    torch.save(train_X, os.path.join(OUT_DIR, "train_features.pt"))
    torch.save(train_y, os.path.join(OUT_DIR, "train_labels.pt"))
    torch.save(val_X, os.path.join(OUT_DIR, "val_features.pt"))
    torch.save(val_y, os.path.join(OUT_DIR, "val_labels.pt"))
    print(f"Saved precomputed features to {OUT_DIR}/")


if __name__ == "__main__":
    main()
