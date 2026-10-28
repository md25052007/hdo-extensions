"""Plotting utilities: mean +/- std curves across seeds, for the ablation
suite, plus the plain baseline (mono-type vs. hybrid) plots."""

import os
from typing import Dict

import matplotlib.pyplot as plt


def _style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["lines.linewidth"] = 2.0


def plot_ablation_metric(
    aggregated: Dict[str, Dict],
    metric_key: str,
    ylabel: str,
    title: str,
    save_path: str,
    yscale: str = "linear",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _style()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)

    for name, agg in aggregated.items():
        rounds = agg["round"]
        mean = agg[f"{metric_key}_mean"]
        std = agg[f"{metric_key}_std"]
        ax.plot(rounds, mean, label=name)
        lower = [m - s for m, s in zip(mean, std)]
        upper = [m + s for m, s in zip(mean, std)]
        ax.fill_between(rounds, lower, upper, alpha=0.15)

    ax.set_xlabel("Interaction Rounds (t)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale(yscale)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_baseline_comparison(results_dict: Dict, metric: str, ylabel: str, title: str, save_path: str, yscale: str = "linear") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _style()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    colors = {"all-FO": "#1a6b3d", "hybrid": "#c07a12", "all-ZO": "#a63a3a"}
    styles = {"all-FO": "-", "hybrid": "--", "all-ZO": "-."}
    for name, history in results_dict.items():
        ax.plot(history["round"], history[metric], label=name, color=colors.get(name), linestyle=styles.get(name, "-"))
    ax.set_xlabel("Interaction Rounds (t)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale(yscale)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")
