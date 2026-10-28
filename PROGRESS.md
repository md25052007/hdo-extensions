# Progress log

One entry per commit, newest first. Each entry says what the commit
contains and why, not just what the diff touched — this is the quick
way to answer "what's actually been done so far" without re-reading
every file.

---

## 2026-10-28 — Run full 4-idea ablation suite on real Year-MSD data; fix repo root nesting

**What's in this commit:**
- Ran `python main.py --task yearmsd --mode ablation --seeds 0 1 2` on the
  real downloaded `YearPredictionMSD.txt` (131072 train / 10000 val),
  hybrid population (n0=16 ZO / n1=4 FO), 2000 rounds, 3 seeds. Results
  (mean +/- std across seeds, final round):

  | config                          | val_loss          | val_R^2            | avg_q |
  |----------------------------------|-------------------|---------------------|-------|
  | baseline                         | 0.0222 (+/-0.0040) | -0.4324 (+/-0.2580) | 1.00  |
  | idea1_variance_weighted_merge    | 0.0120 (+/-0.0001) |  0.2228 (+/-0.0037) | 1.00  |
  | idea2_adaptive_zo_budget         | 0.0119 (+/-0.0000) |  0.2304 (+/-0.0012) | 10.00 |
  | idea3_elite_buffer                | 0.0203 (+/-0.0011) | -0.3082 (+/-0.0720) | 1.00  |
  | idea4_adaptive_mixing             | 0.0169 (+/-0.0014) | -0.0942 (+/-0.0926) | 1.00  |

- **Fixed a repo-structuring bug**: the previous commit (`b5d2f69`) was
  made without running `git init` inside `hdo-extensions/` itself, so it
  silently used a pre-existing `.git` at the home directory one level up —
  every file landed on GitHub nested under `Desktop/hdo-extensions/...`
  instead of at repo root. Fixed by running `git init` inside the project
  folder (this creates its own, nearer `.git` that Git prefers, without
  touching the home-directory one) and re-doing add/commit/branch/remote,
  then `git push --force` to overwrite the single wrongly-rooted commit
  already on GitHub (safe here: one throwaway root-commit, no
  collaborators). New root-commit: `e0a1366`.

**What this shows:** Idea 1 (variance-weighted merge) and Idea 2 (adaptive
ZO budget) both clearly and substantially beat baseline, and come close to
matching the all-FO-only population's own quality (R^2≈0.236 from the
earlier baseline run) — on a *hybrid* ZO/FO population. Idea 3 (elite
buffer) is statistically indistinguishable from baseline (both strongly
negative R^2, high variance across seeds). Idea 4 (adaptive mixing) lands
in between, better than baseline but well short of ideas 1/2. This matches
the ranking predicted from theory before any code was run. Note: idea 2's
`avg_q` is pinned at `max_q=10.00` for the whole run — it never settles
below the cap, so a follow-up sweep with a higher `max_q` could be worth
trying as future work, to see if it's a genuine equilibrium or just
hitting the ceiling we set.

**Not done yet:** CIFAR-10 feature extraction hasn't been run; Phase 0
(authors' repo baseline) hasn't been started; no combination of ideas has
been tried (out of scope per current plan — stopping at the 4 individual
ablations).

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
