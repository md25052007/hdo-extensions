"""First-order (FO) and zeroth-order (ZO) gradient estimators for HDO,
generalized across tasks (via TaskSpec.loss_fn) and extended with the
hooks Ideas 2, 3 and 4 need. Idea 1 (variance-weighted merge) lives in
simulate.py since it only touches the averaging step, not estimation.

With every NoveltyConfig flag off, `compute_node_gradient` reduces exactly
to: FO nodes use a plain backprop gradient, ZO nodes use the single-draw
biased Nesterov-Spokoiny estimator — i.e. the original baseline.
"""

from typing import NamedTuple, Optional, Tuple

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.node import Node
from src.novelty.config import NoveltyConfig


class GradResult(NamedTuple):
    grad: torch.Tensor
    current_loss: float          # loss at the node's actual pre-step params, for Idea 3's buffer
    sample_variance: Optional[float]  # only set when q > 1 gave us an exact empirical variance
    alpha: Optional[float] = None
    cosine: Optional[float] = None


def _eval_loss(node: Node, params: torch.Tensor, batch) -> float:
    batch_X, batch_y = batch
    with torch.no_grad():
        vector_to_parameters(params, node.model.parameters())
        return node.task.loss_fn(node.model(batch_X), batch_y).item()


def first_order_gradient(node: Node, batch=None) -> Tuple[torch.Tensor, float]:
    """G_i(x) = grad F_i(x, xi) via backprop. Returns (grad, loss_value)."""
    if batch is None:
        batch_X, batch_y = node.sample_batch()
    else:
        batch_X, batch_y = batch

    node.model.zero_grad()
    pred = node.model(batch_X)
    loss = node.task.loss_fn(pred, batch_y)
    loss_value = loss.item()
    loss.backward()

    grad_list = []
    for param in node.model.parameters():
        grad_list.append(param.grad.view(-1) if param.grad is not None else torch.zeros_like(param).view(-1))
    return torch.cat(grad_list), loss_value


def _finite_difference_estimate(
    node: Node,
    base_params: torch.Tensor,
    nu: float,
    q: int,
    batch,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, float, Optional[float]]:
    """Average the Nesterov-Spokoiny estimator over q random directions,
    all evaluated at `base_params` on the same batch.

    Returns (grad_estimate, loss_at_base_params, sample_variance_or_None).
    sample_variance is the variance of the q scalar finite-difference
    quotients, which is an exact empirical noise measurement whenever q > 1
    (free byproduct of averaging, no extra passes beyond the q already
    being spent).
    """
    batch_X, batch_y = batch
    d = base_params.numel()

    loss_base = _eval_loss(node, base_params, batch)

    scalars = []
    directions = []
    for _ in range(q):
        u = torch.randn(d, dtype=base_params.dtype, device=base_params.device, generator=generator)
        with torch.no_grad():
            vector_to_parameters(base_params + nu * u, node.model.parameters())
            loss_plus = node.task.loss_fn(node.model(batch_X), batch_y).item()
        scalar = (loss_plus - loss_base) / nu
        scalars.append(scalar)
        directions.append(u)

    # Restore parameters to the unperturbed point — critical, otherwise the
    # last perturbation silently leaks into the model's real state.
    with torch.no_grad():
        vector_to_parameters(base_params, node.model.parameters())

    grad_estimate = torch.zeros(d, dtype=base_params.dtype, device=base_params.device)
    for s, u in zip(scalars, directions):
        grad_estimate += s * u
    grad_estimate /= q

    variance = float(torch.tensor(scalars).var(unbiased=True).item()) if q > 1 else None
    return grad_estimate, loss_base, variance


def zeroth_order_gradient(
    node: Node,
    nu: float = 1e-4,
    q: int = 1,
    batch=None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, float, Optional[float]]:
    """Biased Nesterov-Spokoiny estimator, optionally averaged over q
    random directions (q=1 reproduces the original baseline estimator
    exactly). Returns (grad_estimate, loss_value, variance_or_None).
    """
    if batch is None:
        batch = node.sample_batch()
    base_params = parameters_to_vector(node.model.parameters())
    return _finite_difference_estimate(node, base_params, nu, q, batch, generator)


def compute_node_gradient(
    node: Node,
    nu: float,
    cfg: NoveltyConfig,
    generator: Optional[torch.Generator] = None,
) -> GradResult:
    """The single entry point simulate.py calls per node per interaction.

    Dispatches to plain FO, plain ZO, or one of the novelty-modified paths.
    Always updates node.var_ema so Ideas 1/2 have a signal to read on the
    *next* step (this step's merge/budget decision reads the value left
    over from the previous step — a lagging proxy, same as any EMA-based
    adaptive controller).
    """
    batch = node.sample_batch()
    current_params = parameters_to_vector(node.model.parameters())

    if node.node_type == "FO":
        grad, loss_value = first_order_gradient(node, batch=batch)
        node.update_variance_signal(grad)
        return GradResult(grad=grad, current_loss=loss_value, sample_variance=None)

    # node_type == "ZO"
    q = node.current_q if cfg.adaptive_zo_budget else cfg.base_q

    use_anchor = (
        cfg.elite_buffer
        and node.elite_buffer_size > 0
        and torch.rand(1, generator=generator).item() < cfg.elite_anchor_prob
    )
    anchor = node.best_elite() if use_anchor else None
    base_params = anchor if anchor is not None else current_params

    zo_grad, loss_at_base, sample_var = _finite_difference_estimate(node, base_params, nu, q, batch, generator)

    # current_loss (for the elite buffer push) must reflect the node's
    # *actual* current parameters, not a possibly-stale anchor.
    current_loss = loss_at_base if anchor is None else _eval_loss(node, current_params, batch)

    # Make sure the model's live parameters are back at the node's real
    # current point before we return — the step below is always taken
    # from here, never from a stale anchor, even if we just estimated the
    # gradient there.
    with torch.no_grad():
        vector_to_parameters(current_params, node.model.parameters())

    alpha = cosine = None
    if cfg.adaptive_mixing and node.dual_capable:
        fo_grad, _ = first_order_gradient(node, batch=batch)
        eps = 1e-8
        cosine = (torch.dot(fo_grad, zo_grad) / (fo_grad.norm() * zo_grad.norm() + eps)).item()
        target_alpha = min(max((cosine + 1.0) / 2.0, cfg.mixing_clip_low), cfg.mixing_clip_high)
        prev_alpha = node.last_alpha if node.last_alpha is not None else 0.5
        alpha = cfg.mixing_ema_beta * prev_alpha + (1 - cfg.mixing_ema_beta) * target_alpha
        node.last_alpha, node.last_cosine = alpha, cosine
        grad = alpha * fo_grad + (1 - alpha) * zo_grad
    else:
        grad = zo_grad

    node.update_variance_signal(grad)
    if sample_var is not None:
        node.var_ema = 0.5 * node.var_ema + 0.5 * sample_var

    # NOTE: the actual q-adjustment (Idea 2) happens once per round in
    # simulate.run_round, not here — it needs to compare each node's
    # var_ema against the *population's* average, which this function
    # (looking at one node in isolation) doesn't have access to. See
    # `update_adaptive_budgets` in simulate.py.

    return GradResult(grad=grad, current_loss=current_loss, sample_variance=sample_var, alpha=alpha, cosine=cosine)
