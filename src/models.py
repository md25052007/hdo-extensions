"""Generic linear model used by every task in this project.

The HDO paper's convex experiments (Year-MSD regression, and the ResNet50
frozen-feature fine-tuning on CIFAR-10/Flowers) all reduce, at the point
where HDO actually operates, to optimizing a single linear layer on top of
a fixed feature vector:

  - Year-MSD: raw standardized audio features (d=90)  -> 1 regression output
  - CIFAR-10: frozen ResNet50 pooled features (d=2048) -> 10 class logits

So one model class covers both tasks; only input_dim/output_dim and the
loss/eval functions used around it (see src/tasks.py) differ per task.
"""

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """A single linear layer: y = W x + b."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
