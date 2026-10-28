"""Algorithm 1 implementation and round-based decentralized simulation loop,
generalized across tasks and wired up to all four novelty extensions via a
single NoveltyConfig. With a default (all-False) NoveltyConfig, interact()
is exactly the paper's three-line pairwise step:

    X_i <- X_i - eta * G_i(X_i)
    X_j <- X_j - eta * G_j(X_j)
    avg <- (X_i + X_j) / 2 ; X_i, X_j <- avg
"""

import random
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.models import LinearModel
from src.node import Node
from src.tasks import TaskSpec
from src.estimators import compute_node_gradient
from src.novelty.config import NoveltyConfig


def _clip_grad_norm(grad: torch.Tensor, max_norm: Optional[float]) -> torch.Tensor:
    if max_norm is None:
        return grad
    norm = grad.norm()
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad


def interact(
    node_i: Node,
    node_j: Node,
    eta: float,
    nu: float,
    cfg: NoveltyConfig,
    generator: Optional[torch.Generator] = None,
    max_grad_norm: Optional[float] = 50.0,
) -> None:
    res_i = compute_node_gradient(node_i, nu, cfg, generator=generator)
    res_j = compute_node_gradient(node_j, nu, cfg, generator=generator)

    # Safety net, not one of the four novelty ideas: the Nesterov-Spokoiny
    # ZO estimator has variance that scales with dimension, and on an
    # unbounded loss (e.g. MSE regression) a single bad random direction
    # can produce a huge gradient estimate that, combined with too large an
    # eta, diverges in a handful of rounds (we hit this for real on
    # Year-MSD at eta=0.05 — loss went to 1e17 within ~100 rounds). Clipping
    # the gradient NORM (not the estimator itself) bounds the damage from a
    # single bad draw without changing what the estimator is estimating.
    # It rarely binds once eta is set sensibly for the task; it's here so a
    # bad hyperparameter choice fails as "converges slowly" instead of
    # silently overflowing to nan/inf.
    grad_i = _clip_grad_norm(res_i.grad, max_grad_norm)
    grad_j = _clip_grad_norm(res_j.grad, max_grad_norm)

    x_i = parameters_to_vector(node_i.model.parameters()) - eta * grad_i
    x_j = parameters_to_vector(node_j.model.parameters()) - eta * grad_j

    # ---- Idea 3: push each node's pre-step state into its elite buffer,
    # then gossip-exchange buffers between the interacting pair. ----
    if cfg.elite_buffer:
        node_i.maybe_push_elite(res_i.current_loss, parameters_to_vector(node_i.model.parameters()))
        node_j.maybe_push_elite(res_j.current_loss, parameters_to_vector(node_j.model.parameters()))
        i_buf_snapshot = list(node_i._elite_buffer)
        j_buf_snapshot = list(node_j._elite_buffer)
        for neg_loss, _, params in j_buf_snapshot:
            node_i.maybe_push_elite(-neg_loss, params)
        for neg_loss, _, params in i_buf_snapshot:
            node_j.maybe_push_elite(-neg_loss, params)

    # ---- Idea 1: variance-aware weighted merge instead of a flat average.
    # Weight each side by the inverse of its tracked noise proxy, clipped so
    # neither side ever gets pushed to ~0 (which would stop information
    # flowing from the noisier node entirely). ----
    if cfg.variance_weighted_merge:
        eps = 1e-8
        w_i = 1.0 / (node_i.var_ema + eps)
        w_j = 1.0 / (node_j.var_ema + eps)
        ratio = w_i / w_j
        clip = cfg.merge_weight_clip
        if ratio > clip:
            w_i, w_j = clip, 1.0
        elif ratio < 1.0 / clip:
            w_i, w_j = 1.0, clip
        avg_params = (w_i * x_i + w_j * x_j) / (w_i + w_j)
    else:
        avg_params = (x_i + x_j) / 2.0

    vector_to_parameters(avg_params, node_i.model.parameters())
    vector_to_parameters(avg_params, node_j.model.parameters())


def update_adaptive_budgets(nodes: List[Node], cfg: NoveltyConfig) -> None:
    """Idea 2: after a round, compare each ZO node's noise proxy (var_ema)
    against the ZO population's OWN running average (not a fixed absolute
    constant, which wouldn't transfer across tasks/scales) and raise/lower
    its query budget q accordingly.
    """
    zo_nodes = [n for n in nodes if n.node_type == "ZO"]
    if not zo_nodes:
        return
    pop_mean_var = sum(n.var_ema for n in zo_nodes) / len(zo_nodes)
    if pop_mean_var <= 0:
        return
    for n in zo_nodes:
        if n.var_ema > cfg.budget_increase_threshold * pop_mean_var:
            n.current_q = min(cfg.max_q, n.current_q + 1)
        else:
            # NOTE: this used to be `int(round(n.current_q * cfg.budget_decay))`
            # (budget_decay=0.98). For every integer q in [1, max_q=10], a 2%
            # multiplicative shrink rounds right back to q itself (e.g.
            # round(10*0.98)=10, round(5*0.98)=5) -- the decay was mathematically
            # inert at this scale, so q could only ever ratchet UP and got stuck
            # at max_q permanently once any node crossed the threshold once
            # (confirmed: avg_q==10.00 on every seed, both tasks). A flat
            # per-round decrement actually moves at integer granularity.
            n.current_q = max(cfg.base_q, n.current_q - 1)


def run_round(
    nodes: List[Node],
    eta: float,
    nu: float,
    cfg: NoveltyConfig,
    generator: Optional[torch.Generator] = None,
    max_grad_norm: Optional[float] = 50.0,
) -> None:
    """One round: shuffle all n nodes, form disjoint pairs, interact() each pair."""
    shuffled = list(nodes)
    random.shuffle(shuffled)
    n_pairs = len(shuffled) // 2
    for k in range(n_pairs):
        interact(shuffled[2 * k], shuffled[2 * k + 1], eta, nu, cfg, generator=generator, max_grad_norm=max_grad_norm)

    if cfg.adaptive_zo_budget:
        update_adaptive_budgets(nodes, cfg)


def evaluate(model: LinearModel, task: TaskSpec, X_val: torch.Tensor, y_val: torch.Tensor):
    model.eval()
    with torch.no_grad():
        pred = model(X_val)
        return task.eval_fn(pred, y_val)


def compute_metrics(nodes: List[Node], task: TaskSpec, X_val: torch.Tensor, y_val: torch.Tensor):
    """Returns (val_loss, val_metric, gamma_t, extra_diagnostics)."""
    param_vectors = [parameters_to_vector(n.model.parameters()) for n in nodes]
    mu_t = torch.stack(param_vectors, dim=0).mean(dim=0)
    gamma_t = torch.stack([torch.norm(p - mu_t) ** 2 for p in param_vectors]).mean().item()

    eval_model = LinearModel(task.input_dim, task.output_dim)
    vector_to_parameters(mu_t, eval_model.parameters())
    val_loss, val_metric = evaluate(eval_model, task, X_val, y_val)

    zo_nodes = [n for n in nodes if n.node_type == "ZO"]
    avg_q = sum(n.current_q for n in zo_nodes) / len(zo_nodes) if zo_nodes else 0.0
    mean_var = sum(n.var_ema for n in nodes) / len(nodes)
    alphas = [n.last_alpha for n in nodes if n.last_alpha is not None]
    mean_alpha = sum(alphas) / len(alphas) if alphas else None

    extra = {"avg_q": avg_q, "mean_var_ema": mean_var, "mean_alpha": mean_alpha}
    return val_loss, val_metric, gamma_t, extra


def simulate(
    nodes: List[Node],
    task: TaskSpec,
    total_rounds: int,
    eta: float,
    nu: float,
    eval_every: int,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: Optional[NoveltyConfig] = None,
    generator: Optional[torch.Generator] = None,
    verbose: bool = True,
    log_tag: str = "",
    max_grad_norm: Optional[float] = 50.0,
) -> Dict:
    cfg = cfg or NoveltyConfig()

    history = {"round": [], "val_loss": [], "val_metric": [], "gamma": [],
               "avg_q": [], "mean_var_ema": [], "mean_alpha": []}

    def record(t):
        val_loss, val_metric, gamma_t, extra = compute_metrics(nodes, task, X_val, y_val)
        history["round"].append(t)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)
        history["gamma"].append(gamma_t)
        history["avg_q"].append(extra["avg_q"])
        history["mean_var_ema"].append(extra["mean_var_ema"])
        history["mean_alpha"].append(extra["mean_alpha"])
        return val_loss, val_metric, gamma_t

    val_loss, val_metric, gamma_t = record(0)
    if verbose:
        print(f"[{log_tag}] Round 0/{total_rounds} | Val Loss: {val_loss:.4f} | "
              f"Val {task.metric_name}: {val_metric:.4f} | Gamma_t: {gamma_t:.6f}")

    for t in range(1, total_rounds + 1):
        run_round(nodes, eta, nu, cfg, generator=generator, max_grad_norm=max_grad_norm)
        if t % eval_every == 0 or t == total_rounds:
            val_loss, val_metric, gamma_t = record(t)
            if verbose and (t % (eval_every * 5) == 0 or t == total_rounds):
                print(f"[{log_tag}] Round {t}/{total_rounds} | Val Loss: {val_loss:.4f} | "
                      f"Val {task.metric_name}: {val_metric:.4f} | Gamma_t: {gamma_t:.6f}")

    return history
