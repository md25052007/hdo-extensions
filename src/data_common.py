"""Dataset-agnostic population-sharding helper, shared by every task."""

from typing import List, Optional, Tuple

import torch


def split_population(
    X: torch.Tensor,
    y: torch.Tensor,
    n_nodes: int,
    generator: Optional[torch.Generator] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Split a dataset into n_nodes roughly equal, non-overlapping shards.

    Per the HDO paper, each population (ZO / FO) receives an independent
    partition of the entire training dataset — call this once per
    population, not once globally, so the FO and ZO groups each see their
    own full copy of the data distributed within that group.
    """
    if n_nodes <= 0:
        return []

    n_samples = len(X)
    indices = torch.randperm(n_samples, generator=generator)
    chunk_size = (n_samples + n_nodes - 1) // n_nodes

    shards = []
    for i in range(n_nodes):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n_samples)
        shard_indices = indices[start_idx:end_idx]
        shards.append((X[shard_indices], y[shard_indices]))
    return shards
