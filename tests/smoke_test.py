"""Quick synthetic smoke test: exercises baseline + all 4 novelty ablations
on a tiny synthetic regression task, with tiny n/rounds, purely to catch
bugs fast before running anything on real data. Not a unit test suite —
just a fast end-to-end sanity check.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.tasks import year_msd_task
from src.experiment import run_mono_and_hybrid_baseline, run_ablation_suite
from src.aggregate import aggregate_suite


def make_synthetic(n_samples=2000, d=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, d, generator=gen)
    true_w = torch.randn(d, 1, generator=gen)
    y = (X @ true_w).squeeze(-1) + 0.1 * torch.randn(n_samples, generator=gen)
    y = (y - y.min()) / (y.max() - y.min())
    return X, y


def main():
    task = year_msd_task(input_dim=10)
    X, y = make_synthetic()
    X_train, y_train = X[:1500], y[:1500]
    X_val, y_val = X[1500:], y[1500:]

    print("### Smoke test: baseline mono/hybrid ###")
    results = run_mono_and_hybrid_baseline(
        X_train, y_train, X_val, y_val, task,
        n_total=6, hybrid_zo_frac=0.67,
        total_rounds=20, eta=0.01, nu=1e-3, batch_size=16, eval_every=5, seed=0,
    )
    for name, h in results.items():
        assert len(h["val_loss"]) > 0
        assert all(torch.isfinite(torch.tensor(h["val_loss"])))
        print(f"  {name}: final val_loss={h['val_loss'][-1]:.4f} gamma={h['gamma'][-1]:.6f}")

    print("\n### Smoke test: full 4-idea ablation suite ###")
    all_results = run_ablation_suite(
        X_train, y_train, X_val, y_val, task,
        n_total=6, hybrid_zo_frac=0.67,
        total_rounds=20, eta=0.01, nu=1e-3, batch_size=16, eval_every=5,
        seeds=[0, 1],
    )
    for name, histories in all_results.items():
        for h in histories:
            assert all(torch.isfinite(torch.tensor(h["val_loss"]))), f"{name} produced non-finite loss"

    aggregated = aggregate_suite(all_results)
    for name, agg in aggregated.items():
        print(f"  {name}: mean final val_loss={agg['val_loss_mean'][-1]:.4f} +/- {agg['val_loss_std'][-1]:.4f}, "
              f"avg_q={agg['avg_q_mean'][-1]:.2f}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
