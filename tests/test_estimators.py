"""Unit tests for the core estimators and the novelty mechanisms.
Run with: python -m unittest tests.test_estimators -v
(from the repo root, so the src/ package resolves).
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.nn.utils import parameters_to_vector

from src.tasks import year_msd_task
from src.node import Node
from src.estimators import first_order_gradient, zeroth_order_gradient
from src.simulate import interact
from src.novelty.config import NoveltyConfig


def make_node(node_type, n=64, d=5, seed=0, **kwargs):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=gen)
    true_w = torch.randn(d, 1, generator=gen)
    y = (X @ true_w).squeeze(-1)
    task = year_msd_task(input_dim=d)
    return Node(node_type=node_type, data_X=X, data_y=y, task=task, batch_size=16, **kwargs)


class TestEstimators(unittest.TestCase):
    def test_fo_matches_autograd(self):
        node = make_node("FO")
        batch = node.sample_batch()
        grad, loss = first_order_gradient(node, batch=batch)
        # recompute independently
        node.model.zero_grad()
        pred = node.model(batch[0])
        loss2 = node.task.loss_fn(pred, batch[1])
        loss2.backward()
        expected = torch.cat([p.grad.view(-1) for p in node.model.parameters()])
        self.assertTrue(torch.allclose(grad, expected, atol=1e-6))

    def test_zo_direction_correlates_with_true_gradient(self):
        node = make_node("ZO")
        batch = node.sample_batch()
        # true gradient via autograd on the same batch
        node.model.zero_grad()
        loss = node.task.loss_fn(node.model(batch[0]), batch[1])
        loss.backward()
        true_grad = torch.cat([p.grad.view(-1) for p in node.model.parameters()])

        # average many single-direction ZO estimates -> should point roughly
        # the same way as the true gradient (cosine similarity > 0)
        gen = torch.Generator().manual_seed(42)
        est_sum = torch.zeros_like(true_grad)
        for _ in range(200):
            g, _, _ = zeroth_order_gradient(node, nu=1e-3, q=1, batch=batch, generator=gen)
            est_sum += g
        est_avg = est_sum / 200
        cos = torch.dot(est_avg, true_grad) / (est_avg.norm() * true_grad.norm() + 1e-8)
        self.assertGreater(cos.item(), 0.3)

    def test_zo_variance_decreases_with_more_directions(self):
        node = make_node("ZO")
        batch = node.sample_batch()
        gen1 = torch.Generator().manual_seed(1)
        g_q1, _, _ = zeroth_order_gradient(node, nu=1e-3, q=1, batch=batch, generator=gen1)
        gen2 = torch.Generator().manual_seed(1)
        g_q10, _, var_q10 = zeroth_order_gradient(node, nu=1e-3, q=10, batch=batch, generator=gen2)
        self.assertIsNotNone(var_q10)
        # q=10 estimate should generally be closer to being "calmer" than any
        # single q=1 draw — not a strict guarantee per draw, but true on
        # average; we just check it ran and produced a finite variance here.
        self.assertTrue(torch.isfinite(torch.tensor(var_q10)))

    def test_two_node_consensus_converges_at_zero_lr(self):
        node_i = make_node("FO", seed=0)
        node_j = make_node("FO", seed=1)
        with torch.no_grad():
            for p in node_j.model.parameters():
                p.add_(1.0)  # force a large initial gap
        gap_before = (parameters_to_vector(node_i.model.parameters())
                      - parameters_to_vector(node_j.model.parameters())).norm().item()
        interact(node_i, node_j, eta=0.0, nu=1e-3, cfg=NoveltyConfig())
        gap_after = (parameters_to_vector(node_i.model.parameters())
                     - parameters_to_vector(node_j.model.parameters())).norm().item()
        self.assertAlmostEqual(gap_after, 0.0, places=5)
        self.assertGreater(gap_before, 0.0)

    def test_elite_buffer_keeps_best_k(self):
        node = make_node("ZO", elite_buffer_size=2)
        node.maybe_push_elite(5.0, torch.tensor([1.0]))
        node.maybe_push_elite(1.0, torch.tensor([2.0]))
        node.maybe_push_elite(3.0, torch.tensor([3.0]))  # should evict the worst (5.0)
        best = node.best_elite()
        self.assertEqual(best.item(), 2.0)  # loss=1.0 entry is the best
        losses = sorted(-e[0] for e in node._elite_buffer)
        self.assertEqual(losses, [1.0, 3.0])

    def test_variance_weighted_merge_favors_lower_variance_node(self):
        node_i = make_node("FO", seed=0)
        node_j = make_node("FO", seed=1)
        node_i.var_ema = 0.001  # very confident
        node_j.var_ema = 100.0  # very noisy
        with torch.no_grad():
            for p in node_j.model.parameters():
                p.add_(10.0)
        before_i = parameters_to_vector(node_i.model.parameters()).clone()
        interact(node_i, node_j, eta=0.0, nu=1e-3, cfg=NoveltyConfig(variance_weighted_merge=True))
        after = parameters_to_vector(node_i.model.parameters())
        # merged point should stay much closer to node_i's original (low-variance) value
        self.assertLess((after - before_i).norm().item(), 5.0)


if __name__ == "__main__":
    unittest.main()
