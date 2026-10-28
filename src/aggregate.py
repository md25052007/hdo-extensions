"""Aggregate multiple seeds' histories into mean +/- std curves."""

from typing import Dict, List

import torch


def aggregate_histories(histories: List[Dict]) -> Dict:
    """All histories must share the same 'round' checkpoints (true as long
    as the same total_rounds/eval_every were used across seeds)."""
    rounds = histories[0]["round"]
    out = {"round": rounds}
    for key in ("val_loss", "val_metric", "gamma", "avg_q", "mean_var_ema"):
        stacked = torch.tensor([h[key] for h in histories], dtype=torch.float32)  # [n_seeds, n_points]
        out[f"{key}_mean"] = stacked.mean(dim=0).tolist()
        out[f"{key}_std"] = stacked.std(dim=0).tolist() if stacked.shape[0] > 1 else [0.0] * stacked.shape[1]
    return out


def aggregate_suite(all_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    return {name: aggregate_histories(hs) for name, hs in all_results.items()}
