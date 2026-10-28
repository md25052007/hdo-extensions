# Progress log

One entry per commit, newest first. Each entry says what the commit
contains and why, not just what the diff touched — this is the quick
way to answer "what's actually been done so far" without re-reading
every file.

---

## 2026-10-28 — Generalize HDO sim across tasks, add 4 novelty extensions, fix eta divergence on Year-MSD

**What's in this commit:**
- Rewrote the original MNIST-only simulation (`src/`) into a task-agnostic
  version via `src/tasks.py` (`TaskSpec`), so the same Algorithm-1 code runs
  on Year-MSD (regression) and CIFAR-10 (classification via frozen
  ResNet50 features) without duplicating logic.
- Implemented all four proposed novelty ideas behind independent flags in
  `src/novelty/config.py` — everything off reduces exactly to baseline HDO:
  1. Variance-aware weighted merge (`variance_weighted_merge`)
  2. Adaptive zeroth-order query budget (`adaptive_zo_budget`)
  3. Elite buffer + gossip (`elite_buffer`)
  4. Cosine-similarity-gated adaptive FO/ZO mixing (`adaptive_mixing`)
- `main.py` supports `--mode baseline` (plain all-FO/all-ZO/hybrid) and
  `--mode ablation` (baseline + all 4 ideas, multi-seed, mean±std).
- 6 unit tests (`tests/test_estimators.py`) + 2 end-to-end smoke tests
  covering both task scales — all passing.
- **Bug found and fixed while running the real Year-MSD download**: the
  original MNIST-tuned `eta=0.05` diverges on Year-MSD (unbounded MSE
  loss) — `all-ZO`/`hybrid` loss went to ~1e17 within ~100 rounds. Fixed
  by lowering the default to `eta=0.01` (verified stable over the full
  2000-round run) and adding a `--max_grad_norm` safety-net clip (default
  50.0) as defense-in-depth for the ablation sweeps coming next.

**Verified on real data:** `python main.py --task yearmsd --mode baseline`
completes cleanly on the real downloaded `YearPredictionMSD.txt`
(131072 train / 10000 val). `all-FO` converges smoothly to R²≈0.236;
`all-ZO`/`hybrid` drop fast then oscillate in a noisy band without
diverging — consistent with the paper's own theory (the ZO
variance/bias terms in Theorem 7 form a floor that doesn't vanish with
more rounds, unlike the FO-only case).

**Not done yet:** the 4-idea ablation suite hasn't been run on real data
yet (next command: `python main.py --task yearmsd --mode ablation --seeds 0 1 2`);
CIFAR-10 feature extraction hasn't been run; Phase 0 (authors' repo
baseline) hasn't been started.
