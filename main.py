"""
HDO Extensions — Main Entry Point

Usage:
    # Plain baseline (all-FO / all-ZO / hybrid), no novelty code active:
    python main.py --task yearmsd --mode baseline

    # Full four-idea ablation suite on the hybrid population, multi-seed:
    python main.py --task yearmsd --mode ablation --seeds 0 1 2

    # Same, on CIFAR-10 (run scripts/extract_cifar10_features.py first):
    python main.py --task cifar10 --mode ablation --seeds 0 1 2
"""

import os
import json
import argparse

import torch

from src.tasks import year_msd_task, cifar10_task
from src.data_yearmsd import load_year_msd
from src.data_cifar10 import load_cifar10_features
from src.experiment import run_mono_and_hybrid_baseline, run_ablation_suite
from src.aggregate import aggregate_suite
from src.plot_utils import plot_baseline_comparison, plot_ablation_metric


def main():
    parser = argparse.ArgumentParser(description="HDO Extensions")
    parser.add_argument("--task", choices=["yearmsd", "cifar10"], required=True)
    parser.add_argument("--mode", choices=["baseline", "ablation"], required=True)
    parser.add_argument("--n_total", type=int, default=20)
    parser.add_argument("--hybrid_zo_frac", type=float, default=0.8)
    parser.add_argument("--total_rounds", type=int, default=2000)
    # NOTE on eta: 0.05 was the original MNIST-tuned value and is UNSAFE for
    # Year-MSD (an unbounded MSE regression loss) — the ZO estimator's
    # variance combined with eta=0.05 diverges to ~1e17 within ~100 rounds.
    # 0.01 is empirically stable and converges well for all-FO/all-ZO/hybrid
    # on this task; see README's "Known hyperparameter gotcha" section.
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--nu", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=50.0,
                         help="Clip each node's gradient estimate to this L2 norm before the local step. "
                              "A safety net against ZO's high-variance estimator diverging, not a novelty idea.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="single seed, used in --mode baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds used in --mode ablation")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or f"results/{args.task}/{args.mode}"
    os.makedirs(output_dir, exist_ok=True)

    if args.task == "yearmsd":
        task = year_msd_task()
        (X_train, y_train), (X_val, y_val) = load_year_msd()
    else:
        task = cifar10_task()
        (X_train, y_train), (X_val, y_val) = load_cifar10_features()

    print(f"Task: {task.name} | input_dim={task.input_dim} | train={len(X_train)} | val={len(X_val)}")

    if args.mode == "baseline":
        results = run_mono_and_hybrid_baseline(
            X_train, y_train, X_val, y_val, task,
            n_total=args.n_total, hybrid_zo_frac=args.hybrid_zo_frac,
            total_rounds=args.total_rounds, eta=args.eta, nu=args.nu,
            batch_size=args.batch_size, eval_every=args.eval_every, seed=args.seed,
            max_grad_norm=args.max_grad_norm,
        )
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(results, f, indent=2)

        plot_baseline_comparison(results, "val_loss", "Validation Loss", f"{task.name}: Loss vs Rounds",
                                  os.path.join(output_dir, "loss_comparison.png"))
        plot_baseline_comparison(results, "val_metric", f"Validation {task.metric_name}", f"{task.name}: {task.metric_name} vs Rounds",
                                  os.path.join(output_dir, "metric_comparison.png"))
        plot_baseline_comparison(results, "gamma", "Consensus Potential (Gamma_t)", f"{task.name}: Consensus vs Rounds",
                                  os.path.join(output_dir, "consensus_potential.png"), yscale="log")

        print("\n=== Baseline summary (final round) ===")
        for name, h in results.items():
            print(f"  {name:<8}: val_loss={h['val_loss'][-1]:.4f}  val_{task.metric_name}={h['val_metric'][-1]:.4f}  gamma={h['gamma'][-1]:.6f}")

    else:  # ablation
        all_results = run_ablation_suite(
            X_train, y_train, X_val, y_val, task,
            n_total=args.n_total, hybrid_zo_frac=args.hybrid_zo_frac,
            total_rounds=args.total_rounds, eta=args.eta, nu=args.nu,
            batch_size=args.batch_size, eval_every=args.eval_every, seeds=args.seeds,
            max_grad_norm=args.max_grad_norm,
        )
        with open(os.path.join(output_dir, "raw_history_all_seeds.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        aggregated = aggregate_suite(all_results)
        with open(os.path.join(output_dir, "aggregated.json"), "w") as f:
            json.dump(aggregated, f, indent=2)

        plot_ablation_metric(aggregated, "val_loss", "Validation Loss", f"{task.name}: Ablations — Loss vs Rounds (mean +/- std)",
                              os.path.join(output_dir, "ablation_loss.png"))
        plot_ablation_metric(aggregated, "val_metric", f"Validation {task.metric_name}", f"{task.name}: Ablations — {task.metric_name} vs Rounds",
                              os.path.join(output_dir, "ablation_metric.png"))
        plot_ablation_metric(aggregated, "gamma", "Consensus Potential (Gamma_t)", f"{task.name}: Ablations — Consensus vs Rounds",
                              os.path.join(output_dir, "ablation_consensus.png"), yscale="log")
        plot_ablation_metric(aggregated, "avg_q", "Average ZO query budget (q)", f"{task.name}: Ablations — Adaptive Budget over time",
                              os.path.join(output_dir, "ablation_avg_q.png"))

        print(f"\n=== Ablation summary (final round, mean over seeds {args.seeds}) ===")
        for name, agg in aggregated.items():
            print(f"  {name:<32}: val_loss={agg['val_loss_mean'][-1]:.4f} (+/-{agg['val_loss_std'][-1]:.4f})  "
                  f"val_{task.metric_name}={agg['val_metric_mean'][-1]:.4f} (+/-{agg['val_metric_std'][-1]:.4f})  "
                  f"avg_q={agg['avg_q_mean'][-1]:.2f}")


if __name__ == "__main__":
    main()
