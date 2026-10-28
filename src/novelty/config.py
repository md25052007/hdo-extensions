"""Single config object controlling all four novelty extensions.

Every flag defaults to False/0, which means: with a default NoveltyConfig,
the simulation reduces exactly to baseline HDO (Algorithm 1 from the paper).
Each idea is turned on independently so you can ablate one at a time.
"""

from dataclasses import dataclass


@dataclass
class NoveltyConfig:
    # ---- Idea 1: variance-aware weighted averaging ----
    variance_weighted_merge: bool = False
    merge_weight_clip: float = 20.0  # max allowed ratio w_i/w_j, keeps merges from degenerating

    # ---- Idea 2: adaptive zeroth-order query budget ----
    adaptive_zo_budget: bool = False
    base_q: int = 1
    max_q: int = 10
    budget_increase_threshold: float = 1.5  # raise q if var_ema > threshold * running baseline
    budget_decay: float = 0.98  # q relaxes back toward base_q when things are calm

    # ---- Idea 3: elite buffer + gossip ----
    elite_buffer: bool = False
    elite_buffer_size: int = 3  # K
    elite_anchor_prob: float = 0.2  # p

    # ---- Idea 4: variance/cosine-gated adaptive FO-ZO mixing ----
    adaptive_mixing: bool = False
    dual_capable_fraction: float = 0.25  # fraction of ZO nodes that also compute an FO-style grad
    mixing_ema_beta: float = 0.9
    mixing_clip_low: float = 0.1
    mixing_clip_high: float = 0.9

    @property
    def is_baseline(self) -> bool:
        return not (
            self.variance_weighted_merge
            or self.adaptive_zo_budget
            or self.elite_buffer
            or self.adaptive_mixing
        )

    def label(self) -> str:
        if self.is_baseline:
            return "baseline"
        parts = []
        if self.variance_weighted_merge:
            parts.append("variance_weighted_merge")
        if self.adaptive_zo_budget:
            parts.append("adaptive_zo_budget")
        if self.elite_buffer:
            parts.append("elite_buffer")
        if self.adaptive_mixing:
            parts.append("adaptive_mixing")
        return "+".join(parts)
