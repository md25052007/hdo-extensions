"""Task specifications: bundles together the loss function, the evaluation
metric, and the model shape for each dataset this project runs on.

Every other module (node.py, estimators.py, simulate.py) is written against
this TaskSpec interface and never hardcodes "cross entropy" or "10 classes"
anywhere — that's what makes the same simulation/novelty code reusable
across Year-MSD (regression) and CIFAR-10 (classification) without
duplicating the HDO logic.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TaskSpec:
    name: str
    input_dim: int
    output_dim: int
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    eval_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[float, float]]
    # metric_name is whatever the second number out of eval_fn means
    # ("accuracy %" for classification, "R^2" for regression).
    metric_name: str


def _classification_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


def _classification_eval(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    loss = F.cross_entropy(logits, y).item()
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == y).float().mean().item() * 100.0
    return loss, acc


def _regression_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.view(-1), y.view(-1))


def _regression_eval(pred: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    pred = pred.view(-1)
    y = y.view(-1)
    mse = F.mse_loss(pred, y).item()
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
    r2 = (1.0 - ss_res / ss_tot).item()
    return mse, r2


def year_msd_task(input_dim: int = 90) -> TaskSpec:
    """Year-MSD: standardized 90-dim audio features -> release-year regression."""
    return TaskSpec(
        name="year-msd",
        input_dim=input_dim,
        output_dim=1,
        loss_fn=_regression_loss,
        eval_fn=_regression_eval,
        metric_name="R^2",
    )


def cifar10_task(input_dim: int = 2048, num_classes: int = 10) -> TaskSpec:
    """CIFAR-10 on frozen ResNet50 features -> 10-way classification."""
    return TaskSpec(
        name="cifar10",
        input_dim=input_dim,
        output_dim=num_classes,
        loss_fn=_classification_loss,
        eval_fn=_classification_eval,
        metric_name="accuracy(%)",
    )
