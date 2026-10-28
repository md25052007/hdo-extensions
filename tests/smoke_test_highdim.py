"""Smoke test at CIFAR-10-like scale (d=2048, 10-class classification) to
catch shape bugs that only show up at higher dimension, and sanity check
runtime is reasonable before running this for real."""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.tasks import cifar10_task
from src.experiment import run_ablation_suite
from src.aggregate import aggregate_suite


def make_synthetic(n_samples=600, d=2048, num_classes=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, d, generator=gen)
    true_w = torch.randn(d, num_classes, generator=gen)
    logits = X @ true_w
    y = torch.argmax(logits, dim=-1)
    return X, y


def main():
    task = cifar10_task(input_dim=2048, num_classes=10)
    X, y = make_synthetic()
    X_train, y_train = X[:500], y[:500]
    X_val, y_val = X[500:], y[500:]

    t0 = time.time()
    all_results = run_ablation_suite(
        X_train, y_train, X_val, y_val, task,
        n_total=6, hybrid_zo_frac=0.67,
        total_rounds=10, eta=0.01, nu=1e-3, batch_size=16, eval_every=5,
        seeds=[0],
    )
    elapsed = time.time() - t0
    for name, histories in all_results.items():
        for h in histories:
            assert all(torch.isfinite(torch.tensor(h["val_loss"]))), f"{name} produced non-finite loss"
    aggregated = aggregate_suite(all_results)
    for name, agg in aggregated.items():
        print(f"  {name}: final val_loss={agg['val_loss_mean'][-1]:.4f} final_acc={agg['val_metric_mean'][-1]:.2f}")
    print(f"\nElapsed for 5 configs x 1 seed x 10 rounds x n=6, d=2048: {elapsed:.1f}s")
    print("HIGH-DIM SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
