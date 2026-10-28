# Progress log

One entry per commit, newest first. Each entry says what the commit
contains and why, not just what the diff touched — this is the quick
way to answer "what's actually been done so far" without re-reading
every file.

---

## 2026-10-29 — Fix adaptive-ZO-budget ratchet bug; rerun both ablation suites

**What's in this commit:**
- **Real bug found and fixed in `src/simulate.py`'s
  `update_adaptive_budgets()` (idea 2, adaptive ZO budget).** The decay
  branch was `n.current_q = max(cfg.base_q, int(round(n.current_q *
  cfg.budget_decay)))` with `budget_decay=0.98`. For every integer q in
  `[1, max_q=10]`, `round(q * 0.98) == q` — a 2% multiplicative shrink is
  mathematically inert at this integer scale, so q could only ever
  ratchet *up* (via the increase branch) and got stuck at `max_q`
  permanently the first time any node crossed
  `budget_increase_threshold`. This exactly explained a pattern already
  observed and flagged as merely "curious" in the previous two full
  ablation runs: `avg_q==10.00` on every seed, on both Year-MSD and
  CIFAR-10. Fixed by switching to a flat per-round decrement
  (`n.current_q - 1`), which actually moves at integer granularity.
- Reran both ablation suites (`python main.py --task yearmsd --mode
  ablation --seeds 0 1 2` and `--task cifar10 --mode ablation --seeds 0
  1 2`) with the fix in place. Only `idea2_adaptive_zo_budget` changes —
  baseline/idea1/idea3/idea4 are bit-for-bit consistent with the pre-fix
  runs, confirming the fix is isolated to its intended code path.

  | task | idea2 val metric (before → after) | avg_q (before → after) |
  |---|---|---|
  | Year-MSD (val R²) | 0.2304 → **−0.1022** | 10.00 → **1.42** |
  | CIFAR-10 (val accuracy %) | 87.01 → **85.03** | 10.00 → **1.04** |

**What this shows:** idea 2 only ever looked like a strong second-place
idea because of the bug. With a genuinely adaptive controller, it barely
raises its budget above the floor on either task (avg_q settles at 1.42
and 1.04, not 10) and its performance collapses to roughly baseline —
on CIFAR-10, essentially indistinguishable from (very slightly below)
baseline's 85.24%. Idea 1 (variance-weighted merge) remains the clear
best performer on both tasks by a wide margin; this doesn't change.
Documented as a negative result / limitation (the increase threshold is
apparently too conservative to fire under real conditions) rather than
silently dropped — updated `README.md`'s Results tables and
cross-task-reading paragraph accordingly, and regenerated
`results/yearmsd/ablation/` and `results/cifar10/ablation/` (JSON +
plots) from the new runs.

**Not done yet:** Phase 0 (authors' repo CIFAR-10 baseline) — setup
instructions are fully resolved and ready in
`phase0_author_repo/SETUP_INSTRUCTIONS.md`, not yet run.

---

## 2026-10-29 — Run full 4-idea ablation suite on real CIFAR-10 (ResNet50 features); add live progress to feature extraction

**What's in this commit:**
- Ran `scripts/extract_cifar10_features.py` for real (GPU-accelerated:
  `torch`/`torchvision` reinstalled with CUDA 13.0 wheels to match the
  local RTX 3050 — the CPU-only default build was going to take an
  open-ended amount of time for 60k ImageNet-ResNet50 forward passes).
  Produced `data/cifar10_features/{train,val}_{features,labels}.pt`
  (train 50000x2048, val 10000x2048).
- Added a live `done/total images` progress line to
  `extract_cifar10_features.py`'s `extract()` loop — previously it printed
  nothing until an entire split finished, which made a long-running
  extraction indistinguishable from a hang.
- Ran `python main.py --task cifar10 --mode ablation --seeds 0 1 2`
  (hybrid n0=16 ZO / n1=4 FO, 2000 rounds, 3 seeds). Results (mean +/- std,
  final round):

  | config                          | val_loss           | val_accuracy(%)    | avg_q |
  |----------------------------------|--------------------|---------------------|-------|
  | baseline                         | 0.5117 (+/-0.0023) | 85.2433 (+/-0.2023) | 1.00  |
  | idea1_variance_weighted_merge    | 0.3976 (+/-0.0011) | 87.5167 (+/-0.1447) | 1.00  |
  | idea2_adaptive_zo_budget         | 0.4082 (+/-0.0007) | 87.0100 (+/-0.0361) | 10.00 |
  | idea3_elite_buffer                | 0.5092 (+/-0.0058) | 85.3433 (+/-0.1387) | 1.00  |
  | idea4_adaptive_mixing             | 0.4831 (+/-0.0020) | 85.7767 (+/-0.1845) | 1.00  |

**What this shows:** The CIFAR-10 ranking exactly matches the Year-MSD
ranking from the previous commit — ideas 1 and 2 clearly and
reproducibly beat baseline (+2.3pp / +1.8pp accuracy, very low
cross-seed variance), idea 3 is statistically indistinguishable from
baseline, idea 4 is a smaller but real improvement in between. This is
now a cross-task-consistent result (regression + classification, d=90 +
d=2048), not a single-task artifact. Idea 2's `avg_q` again saturates at
`max_q=10.00` across all seeds, same as on Year-MSD — a second
confirmation that it's hitting the cap rather than settling naturally; a
`max_q` sweep would be a clean follow-up if time allows. No divergence
issues on this task (as expected: CIFAR-10's cross-entropy loss is
bounded, unlike Year-MSD's MSE, so the eta=0.05 failure mode from before
never applied here) — ran at the default `eta=0.01` with no incidents.

**Not done yet:** Phase 0 (authors' repo baseline) hasn't been started;
no combination of ideas has been tried (out of scope per current plan).
Both task tracks (Year-MSD, CIFAR-10) now have complete baseline +
4-idea ablation results — the core experimental work for the report is
essentially done pending Phase 0.

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
