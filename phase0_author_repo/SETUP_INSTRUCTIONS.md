# Phase 0 — Running the authors' own repo as the baseline

Goal: generate baseline numbers by running the paper's own code
(`github.com/ShayanTalaei/HDO`) directly, rather than reproducing it
ourselves. This track is independent of everything else in this repo —
it doesn't share any code with `src/`, and nothing else here depends on it.

This is **not** committed into `hdo-extensions` itself — clone it
somewhere else on your machine (e.g. a sibling folder), since it's a
separate, unmodified third-party codebase, not part of our submission.

## Resolved findings (checked directly against the source, 2026-10-29)

These were open questions the first version of this doc left for whoever
started Phase 0. They're now answered — no need to re-investigate.

- **Model is ResNet-18, not ResNet-50 — confirmed a real paper/code
  mismatch, not just ambiguous wording.** The paper's Appendix B.3 text
  says ResNet50. The actual code
  (`model_bases/resnet_base.py:ResNetModel.__init__`) does
  `models.resnet18(weights=ResNet18_Weights.DEFAULT)` — there is no
  ResNet50 anywhere in this repo. Document this explicitly as a
  discrepancy between the paper and its own released code if you cite
  Phase 0 numbers next to the paper's reported figures.

- **No Year-MSD / LibSVM loader exists in this repo at all.** Checked
  `datasets/datasets.py` directly: the only supported `--dataset` values
  are `bracket`, `mnist`, `cifar10`, `cifar100` (see the `DATASETS` list
  and the `elif` chain in `get_dataset`). The paper describes a Year-MSD
  linear-regression experiment in Appendix B.3, but that code was never
  published in this repo. **Phase 0 cannot produce a Year-MSD baseline as
  originally planned — recommended fallback below.**

- **MPI and W&B are hard requirements, not optional.** `main.py` does
  `from mpi4py import MPI` unconditionally at the top (no try/except), so
  it must be launched via `mpiexec -np <N> python main.py ...`, not plain
  `python main.py`. Rank 0 also calls `wandb.login()` /
  `wandb.init(...)` unconditionally — `WANDB_MODE=offline` as an
  environment variable avoids needing real network access to W&B's
  servers, but you still need `wandb` installed and importable.

- **CIFAR-10 in this repo is trained end-to-end during the HDO
  simulation itself** (on-the-fly, via `--dataset cifar10 --model
  resnet`), not via the paper's described two-stage "train ResNet,
  extract frozen features, then train a linear probe" process. Pass
  `--freeze_model` to freeze the ResNet18 backbone and train only the
  final `fc` layer — that's the closest match to what the paper and our
  own `src/` code do (a linear model on top of fixed features). Without
  `--freeze_model`, this fine-tunes the whole ResNet18, which is a
  different experiment.

- **Exact estimator flag for a fair comparison:** the README's own
  example command uses `--z_grad zeroth_order_forward-mode_AD_sim`,
  which is the paper's *unbiased* forward-mode estimator — not what our
  `src/estimators.py` implements. Our code (and the paper's "Biased
  Zeroth-order" estimator, Appendix B.2) is the one-sided finite
  difference `(F(X+νu)−F(X))/ν · u`, averaged over directions — that's
  `--z_grad zeroth_order_rge` in this repo (confirmed by reading
  `optimizers.py`: `zeroth_order_rge` does exactly this one-sided
  average; `zeroth_order_cge` is an unrelated full coordinate-wise
  estimator, and `zeroth_order_forward-mode_AD_sim` is the unbiased
  JVP-based one). Use `rge` to compare against our results.

- **Paper's exact hyperparameters** (Appendix B.3, for when you want a
  config matching a published figure rather than just "a reasonable
  config"):
  - Year-MSD: linear model (90→1), MSELoss, batch size 128, lr 0.001
    (unless a figure states otherwise), 2^17/10000 train/val split,
    inputs standardized + outputs min-max normalized — **this is exactly
    what `src/data_yearmsd.py` already does**, so our own `all-FO` curve
    from `python main.py --task yearmsd --mode baseline` is a faithful
    stand-in for "the paper's own Year-MSD protocol," even though it's
    our code, not theirs (see fallback below).
  - CIFAR-10 (ResNet fine-tuning): CrossEntropyLoss, batch size 128,
    lr 0.001 (unless stated otherwise). The repo's own CLI defaults
    differ (`--lr0` defaults to 0.01 for ZO, `--lr1` to 0.001 for FO) —
    pass `--lr0 0.001` explicitly to match the paper.
  - Number of random directions (`--rv` / our `q`): the paper sweeps
    5/10/15 in its own ablation figures (Fig. 1) and uses up to 50 in
    others (Fig. 2) — there's no single "the" value; pick one and keep it
    fixed across your Phase 0 run and state it next to the number.

## Recommended fallback for the missing Year-MSD baseline

Since this repo has no Year-MSD loader, don't try to patch one into a
third-party clone (that reintroduces exactly the "did we implement it
the same way" risk Phase 0 was supposed to avoid). Instead:
- **Phase 0 (authors' code) covers CIFAR-10 only.** Run it there, where
  the repo actually supports the task.
- **For Year-MSD, cite our own `all-FO` baseline** (from
  `results/yearmsd/baseline/`) as the reference point instead — it
  already matches the paper's declared Year-MSD protocol exactly (model
  shape, loss, split, normalization, as confirmed above), so it's a
  faithful stand-in even though it's our implementation rather than
  theirs. State this explicitly in the report rather than presenting it
  as "the authors' number."

## Steps

1. **Clone it, somewhere outside this repo:**
   ```
   git clone https://github.com/ShayanTalaei/HDO.git
   cd HDO
   ```

2. **Install dependencies.**
   ```
   pip install -r requirements.txt
   ```
   If `mpi4py`/`openmpi` fails to build via pip (common on Windows),
   `conda install -c conda-forge mpi4py openmpi` is more reliable.
   Set `WANDB_MODE=offline` in your environment before running if you
   don't want to deal with a live W&B account.

3. **Get the smallest possible run working first.** Use `-np 4` (or
   whatever small node count) and `--steps 20` before trusting any real
   numbers — this isolates "does MPI/the repo even launch on my machine"
   from "does it produce a correct number."
   ```
   mpiexec --oversubscribe --allow-run-as-root -np 4 python main.py \
     --dataset cifar10 --model resnet --freeze_model \
     --fn 1 --z_grad zeroth_order_rge --lr0 0.001 --lr1 0.001 \
     --rv 10 --z_batch_size 128 --f_batch_size 128 --steps 20 --seed 0
   ```

4. **Run the real CIFAR-10 config**, matching our own hybrid population
   (n0=16 ZO / n1=4 FO, i.e. `-np 20 --fn 4`) so the comparison is
   apples-to-apples:
   ```
   mpiexec --oversubscribe --allow-run-as-root -np 20 python main.py \
     --dataset cifar10 --model resnet --freeze_model \
     --fn 4 --z_grad zeroth_order_rge --lr0 0.001 --lr1 0.001 \
     --rv 10 --z_batch_size 128 --f_batch_size 128 --steps 2000 --seed 0 \
     --wandb_group phase0_cifar10_hybrid
   ```
   Save stdout/W&B exports, and drop the final numbers (and the exact
   command you ran) into `phase0_author_repo/baseline_results.md`
   (create this file once you have real numbers — it's the reference
   every CIFAR-10 ablation plot in `src/` gets compared against).

5. **If a config won't run cleanly after a reasonable effort**, fall back
   to the closest one that does, and write down exactly what you changed
   and why in `baseline_results.md` — an honest documented deviation beats
   a silently-wrong "exact" number.

## Why this is safe to run in parallel with everything else

This track never touches `src/`, `main.py`, or the datasets you're
downloading for Phase 1/2 — it's a completely separate clone, separate
environment, separate task. Nothing here blocks nothing there.
