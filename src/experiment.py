"""Population construction + the ablation-suite runner shared by both
Year-MSD and CIFAR-10. Generalized from the original run_all_configs,
now also wiring up the four novelty extensions and multi-seed aggregation.
"""

import random
from typing import Dict, List, Optional

import torch

from src.models import LinearModel
from src.data_common import split_population
from src.node import Node
from src.tasks import TaskSpec
from src.simulate import simulate
from src.novelty.config import NoveltyConfig


def build_population(
    n0: int,
    n1: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    task: TaskSpec,
    batch_size: int,
    shared_init_model: LinearModel,
    cfg: NoveltyConfig,
    seed: int = 0,
) -> List[Node]:
    """Construct n = n0 + n1 nodes with shared initial parameters and
    independent per-population data splits (each population gets its own
    full copy of the training data, per the paper's setup)."""
    gen = torch.Generator().manual_seed(seed)

    zo_shards = split_population(X_train, y_train, n0, generator=gen) if n0 > 0 else []
    fo_shards = split_population(X_train, y_train, n1, generator=gen) if n1 > 0 else []

    n_dual = int(round(cfg.dual_capable_fraction * n0)) if cfg.adaptive_mixing else 0
    elite_size = cfg.elite_buffer_size if cfg.elite_buffer else 0

    nodes: List[Node] = []
    for idx, (shard_X, shard_y) in enumerate(zo_shards):
        nodes.append(Node(
            node_type="ZO",
            data_X=shard_X,
            data_y=shard_y,
            task=task,
            batch_size=batch_size,
            initial_model=shared_init_model,
            dual_capable=(idx < n_dual),
            elite_buffer_size=elite_size,
        ))
    for shard_X, shard_y in fo_shards:
        nodes.append(Node(
            node_type="FO",
            data_X=shard_X,
            data_y=shard_y,
            task=task,
            batch_size=batch_size,
            initial_model=shared_init_model,
            elite_buffer_size=elite_size,
        ))
    return nodes


def _run_one(
    name: str,
    n0: int,
    n1: int,
    X_train, y_train, X_val, y_val,
    task: TaskSpec,
    total_rounds: int,
    eta: float,
    nu: float,
    batch_size: int,
    eval_every: int,
    cfg: NoveltyConfig,
    seed: int,
    max_grad_norm: float = 50.0,
) -> Dict:
    torch.manual_seed(seed)
    shared_init_model = LinearModel(task.input_dim, task.output_dim)

    random.seed(seed)
    torch.manual_seed(seed)
    nodes = build_population(n0, n1, X_train, y_train, task, batch_size, shared_init_model, cfg, seed=seed)

    gen = torch.Generator().manual_seed(seed)
    history = simulate(
        nodes=nodes, task=task, total_rounds=total_rounds, eta=eta, nu=nu,
        eval_every=eval_every, X_val=X_val, y_val=y_val, cfg=cfg, generator=gen,
        verbose=True, log_tag=f"{name}/seed{seed}", max_grad_norm=max_grad_norm,
    )
    return history


def run_mono_and_hybrid_baseline(
    X_train, y_train, X_val, y_val, task: TaskSpec,
    n_total: int, hybrid_zo_frac: float,
    total_rounds: int, eta: float, nu: float, batch_size: int, eval_every: int,
    seed: int = 0,
    max_grad_norm: float = 50.0,
) -> Dict[str, Dict]:
    """Plain baseline (NoveltyConfig() = all off): all-FO, all-ZO, hybrid.
    This is the reference curve every novelty ablation gets compared against.
    """
    n0_hybrid = int(round(hybrid_zo_frac * n_total))
    n1_hybrid = n_total - n0_hybrid
    configs = {"all-FO": (0, n_total), "all-ZO": (n_total, 0), "hybrid": (n0_hybrid, n1_hybrid)}

    cfg = NoveltyConfig()
    results = {}
    for name, (n0, n1) in configs.items():
        print(f"\n=== baseline/{name} (n0={n0} ZO, n1={n1} FO) ===")
        results[name] = _run_one(name, n0, n1, X_train, y_train, X_val, y_val, task,
                                  total_rounds, eta, nu, batch_size, eval_every, cfg, seed,
                                  max_grad_norm=max_grad_norm)
    return results


def run_ablation_suite(
    X_train, y_train, X_val, y_val, task: TaskSpec,
    n_total: int, hybrid_zo_frac: float,
    total_rounds: int, eta: float, nu: float, batch_size: int, eval_every: int,
    seeds: List[int],
    ideas: Optional[Dict[str, NoveltyConfig]] = None,
    max_grad_norm: float = 50.0,
) -> Dict[str, List[Dict]]:
    """Run baseline + each of the four novelty ideas on the SAME hybrid
    population composition, across multiple seeds, so results can be
    reported as mean +/- std rather than a single noisy run.

    Returns {config_name: [history_seed0, history_seed1, ...]}.
    """
    n0 = int(round(hybrid_zo_frac * n_total))
    n1 = n_total - n0

    if ideas is None:
        ideas = {
            "baseline": NoveltyConfig(),
            "idea1_variance_weighted_merge": NoveltyConfig(variance_weighted_merge=True),
            "idea2_adaptive_zo_budget": NoveltyConfig(adaptive_zo_budget=True),
            "idea3_elite_buffer": NoveltyConfig(elite_buffer=True),
            "idea4_adaptive_mixing": NoveltyConfig(adaptive_mixing=True),
        }

    all_results: Dict[str, List[Dict]] = {name: [] for name in ideas}
    for name, cfg in ideas.items():
        for seed in seeds:
            print(f"\n=== {name} | hybrid n0={n0} n1={n1} | seed={seed} ===")
            history = _run_one(name, n0, n1, X_train, y_train, X_val, y_val, task,
                                total_rounds, eta, nu, batch_size, eval_every, cfg, seed,
                                max_grad_norm=max_grad_norm)
            all_results[name].append(history)

    return all_results
