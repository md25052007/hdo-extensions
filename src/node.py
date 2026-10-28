"""Node class representing a single decentralized agent (FO or ZO).

This is a generalized version of the original MNIST-only Node: it takes a
TaskSpec (so the same class works for Year-MSD regression and CIFAR-10
classification) and carries the extra per-node state the four novelty
extensions need (a running noise/variance estimate, an elite buffer of
historical best states, and a "dual-capable" flag for the adaptive mixing
idea). None of that extra state does anything unless the corresponding
novelty flag is turned on in NoveltyConfig (see src/novelty/config.py) —
with everything off, a Node behaves exactly like the original baseline.
"""

import heapq
import itertools
from typing import Optional, Tuple

import torch

from src.models import LinearModel
from src.tasks import TaskSpec


class Node:
    """
    Simulated decentralized node carrying local model parameters X_i,
    a local data shard, and an assigned node type ('FO' or 'ZO').
    """

    def __init__(
        self,
        node_type: str,
        data_X: torch.Tensor,
        data_y: torch.Tensor,
        task: TaskSpec,
        batch_size: int = 32,
        initial_model: Optional[LinearModel] = None,
        dual_capable: bool = False,
        elite_buffer_size: int = 0,
    ):
        assert node_type in ("FO", "ZO"), f"Invalid node_type: {node_type}. Must be 'FO' or 'ZO'."
        self.node_type = node_type
        self.data_X = data_X
        self.data_y = data_y
        self.task = task
        self.batch_size = batch_size

        self.model = LinearModel(task.input_dim, task.output_dim)
        if initial_model is not None:
            self.model.load_state_dict(initial_model.state_dict())

        # --- shared novelty bookkeeping (inert unless a novelty flag uses it) ---
        # Running noise/variance proxy for this node's gradient estimator,
        # updated every step via an EMA of ||g_t - g_{t-1}||^2.
        # Used by Idea 1 (variance-weighted merge) and Idea 2 (adaptive ZO budget).
        self.var_ema: float = 1.0  # start at a neutral, non-zero value
        self._prev_grad: Optional[torch.Tensor] = None

        # Idea 2: adaptive ZO query budget (only meaningful for ZO nodes).
        self.current_q: int = 1

        # Idea 3: elite buffer of (loss, tie_breaker, flat_params) tuples, a min-heap
        # keyed on loss so the *worst* entry is always at the top (heapq default),
        # which is exactly what you want for an O(log K) "replace worst if better" push.
        self.elite_buffer_size = elite_buffer_size
        self._elite_buffer: list = []
        self._tie_breaker = itertools.count()

        # Idea 4: whether this node computes BOTH an FO and a ZO gradient each
        # step and blends them (only applies to ZO-type nodes in our population;
        # giving an FO node a "dual" ZO estimate too would be free in principle,
        # but the paper's own model assumes FO nodes don't incur ZO's cost, so we
        # only ever mark ZO nodes as dual-capable).
        self.dual_capable = dual_capable and node_type == "ZO"
        self.last_alpha: Optional[float] = None
        self.last_cosine: Optional[float] = None

    def sample_batch(self, generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random mini-batch with replacement from the local data shard."""
        n_samples = len(self.data_X)
        if n_samples == 0:
            raise ValueError("Local data shard is empty.")
        indices = torch.randint(0, n_samples, (self.batch_size,), generator=generator)
        return self.data_X[indices], self.data_y[indices]

    # ------------------------------------------------------------------
    # Idea 1 / 2 support: track a cheap running variance proxy per node.
    # ------------------------------------------------------------------
    def update_variance_signal(self, grad: torch.Tensor, beta: float = 0.9) -> None:
        """EMA of the squared change between consecutive gradient estimates.

        This is a dataset- and estimator-agnostic noise proxy: a node whose
        gradient estimate is jumping around a lot step-to-step (true of ZO,
        especially with few random directions, and of any node on a noisy
        minibatch) gets a high var_ema; a node with smooth, consistent
        updates gets a low one. It needs no extra forward/backward passes
        beyond what the estimator already computes.
        """
        if self._prev_grad is not None and self._prev_grad.shape == grad.shape:
            delta_sq = (grad - self._prev_grad).pow(2).mean().item()
            self.var_ema = beta * self.var_ema + (1 - beta) * delta_sq
        self._prev_grad = grad.detach().clone()

    # ------------------------------------------------------------------
    # Idea 3 support: a bounded top-K buffer of this node's best states.
    # ------------------------------------------------------------------
    def maybe_push_elite(self, loss_value: float, flat_params: torch.Tensor) -> None:
        if self.elite_buffer_size <= 0:
            return
        entry = (-loss_value, next(self._tie_breaker), flat_params.detach().clone())
        # We store -loss so that heapq (a min-heap) keeps the *best* (lowest
        # true loss / highest -loss) entries, and pops the worst first when
        # the buffer overflows.
        if len(self._elite_buffer) < self.elite_buffer_size:
            heapq.heappush(self._elite_buffer, entry)
        else:
            heapq.heappushpop(self._elite_buffer, entry)

    def best_elite(self) -> Optional[torch.Tensor]:
        if not self._elite_buffer:
            return None
        best = max(self._elite_buffer, key=lambda e: e[0])  # highest -loss = lowest loss
        return best[2]

    def merge_elite_from(self, other: "Node") -> None:
        """Gossip step for Idea 3: pull the peer's buffered states into ours
        too, keeping only the combined top-K by loss."""
        if self.elite_buffer_size <= 0:
            return
        for entry in other._elite_buffer:
            neg_loss, _, params = entry
            self.maybe_push_elite(-neg_loss, params)
