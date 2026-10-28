# hdo-extensions

Course project for Optimization Techniques (NITK Surathkal), extending
**"Hybrid Decentralized Optimization: First- and Zeroth-Order Optimizers
Can Be Jointly Leveraged for Faster Convergence"** (Talaei, Nadiradze,
Alistarh — AAAI 2025, [arXiv:2210.07703](https://arxiv.org/abs/2210.07703)).

Team: D Maithreya (241AI013), Maharshi Kundu (241AI022), Siddhanth R Naik
(241AI040), Sai (241AI0xx) — Guide: Dr. Bhawana Rudra.

## What's in here

- `src/` — a from-scratch simulation of HDO's Algorithm 1, generalized
  across tasks (`src/tasks.py`), with four optional novelty extensions
  that are all off by default (`src/novelty/config.py`). With every flag
  off, this reduces exactly to the paper's baseline algorithm.
- `phase0_author_repo/` — instructions for running the paper's own repo
  (`github.com/ShayanTalaei/HDO`) directly, as an independent baseline.
  That repo is cloned elsewhere, not into this one.
- `scripts/extract_cifar10_features.py` — one-time ResNet50 feature
  extraction for the CIFAR-10 task.
- `tests/` — fast synthetic smoke tests (no real data needed) that
  exercise every novelty path end-to-end.

## The four novelty ideas (`src/novelty/config.py`)

1. **`variance_weighted_merge`** — replace the flat pairwise average
   `(X_i+X_j)/2` with an inverse-variance-weighted merge, using a running
   per-node noise proxy (`Node.var_ema`).
2. **`adaptive_zo_budget`** — vary the number of random directions `q` a
   ZO node averages over, per-node, based on that same noise signal
   relative to the ZO population's own average.
3. **`elite_buffer`** — each node keeps a top-K heap of its best historical
   states; with probability `p` a ZO step anchors its perturbation there
   instead of the current iterate; buffers are gossip-merged on interaction.
4. **`adaptive_mixing`** — a subset of ZO nodes also compute an FO-style
   gradient and blend the two via a cosine-similarity-gated, EMA-smoothed
   weight `alpha_t`.

## Datasets — sources and exact save locations

Download these yourself; the loaders expect them at exactly these paths
relative to the repo root (both `data/` paths are gitignored — raw data
never gets committed).

### Year-MSD (regression task)

- **Source:** UCI Machine Learning Repository — "YearPredictionMSD"
  https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
  Direct zip: https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip
  (~201 MB zipped, ~428 MB extracted)
- **Save to:** unzip it so you end up with exactly this file:
  ```
  hdo-extensions/data/yearmsd/YearPredictionMSD.txt
  ```
- Each line: `year, feature_1, ..., feature_90` (comma-separated). The
  loader (`src/data_yearmsd.py`) standardizes features and min-max
  normalizes the year label itself, using only the train split's
  statistics.

### CIFAR-10 (classification task, via frozen ResNet50 features)

- **Source:** standard CIFAR-10, downloaded automatically by
  `torchvision.datasets.CIFAR10` — you don't need to manually fetch this
  one, just run the extraction script below and it downloads to
  `data/cifar10_raw/` on its own.
- **Steps:**
  ```
  pip install -r requirements.txt
  python scripts/extract_cifar10_features.py
  ```
  This downloads CIFAR-10, runs every image through an ImageNet-pretrained
  ResNet50 (classification head removed), and saves the resulting 2048-dim
  features to:
  ```
  hdo-extensions/data/cifar10_features/train_features.pt
  hdo-extensions/data/cifar10_features/train_labels.pt
  hdo-extensions/data/cifar10_features/val_features.pt
  hdo-extensions/data/cifar10_features/val_labels.pt
  ```
  Run this once; the actual HDO simulation afterward only ever touches
  these small `.pt` files, never the raw images or the ResNet50 again —
  that's what keeps zeroth-order's extra forward passes affordable.
- **Disclosed deviation from the paper:** the paper trains its own
  ResNet50 first (SGD, 150 epochs) and then freezes it; we use an
  off-the-shelf ImageNet-pretrained ResNet50 instead. Reasonable for a
  course project, but say so explicitly in the report — don't present it
  as using the paper's exact feature extractor.

## Known hyperparameter gotcha (found while testing on the real Year-MSD data)

The original MNIST-tuned default was `eta=0.05`. On Year-MSD specifically
— an **unbounded MSE regression loss**, unlike MNIST's bounded
cross-entropy — that value diverges: the ZO estimator's variance combined
with that large a step size pushed the loss to ~1e17 within ~100 rounds
for the `all-ZO` and `hybrid` configs (verified on the real downloaded
file, not just synthetic data). `main.py`'s default is now `eta=0.01`,
which is empirically stable and converges cleanly for all three
population types. There's also a `--max_grad_norm` safety net (default
50.0) that clips any single node's gradient estimate before it's applied,
as defense-in-depth against the same failure mode if you sweep `eta`
higher in your own experiments — it rarely binds at sane learning rates,
it's just there so a bad hyperparameter choice degrades gracefully
instead of silently overflowing to `inf`. If you deliberately want the
old unsafe default back, pass `--eta 0.05` explicitly.

## Running it

```bash
pip install -r requirements.txt

# Plain baseline (all-FO / all-ZO / hybrid), no novelty code active:
python main.py --task yearmsd --mode baseline

# Full four-idea ablation suite on the hybrid population, multi-seed:
python main.py --task yearmsd --mode ablation --seeds 0 1 2

# Same on CIFAR-10 (run the feature extraction script first):
python main.py --task cifar10 --mode ablation --seeds 0 1 2
```

Results (plots + JSON histories) land in `results/<task>/<mode>/`.

Before trusting a CIFAR-10 run, run the synthetic smoke tests
(`python tests/smoke_test.py` and `python tests/smoke_test_highdim.py`) —
they catch bugs in the shared novelty code in seconds instead of minutes.

## Project status

See the project's working notes (kept outside this repo) for the current
phase status. Phase 0 (author's-repo baseline) is independent of Phases 1/2
and can run in parallel; Phases 1 (Year-MSD) and 2 (CIFAR-10) share the
same `src/` code, so validate on Year-MSD before trusting CIFAR-10 runs of
the same novelty code.
