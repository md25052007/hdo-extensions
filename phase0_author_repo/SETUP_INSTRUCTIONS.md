# Phase 0 — Running the authors' own repo as the baseline

Goal: generate baseline numbers by running the paper's own code
(`github.com/ShayanTalaei/HDO`) directly, rather than reproducing it
ourselves. This track is independent of everything else in this repo —
it doesn't share any code with `src/`, and nothing else here depends on it.

This is **not** committed into `hdo-extensions` itself — clone it
somewhere else on your machine (e.g. a sibling folder), since it's a
separate, unmodified third-party codebase, not part of our submission.

## Steps

1. **Clone it, somewhere outside this repo:**
   ```
   git clone https://github.com/ShayanTalaei/HDO.git
   cd HDO
   ```

2. **Read `requirements.txt` and the README first.** It launches via MPI
   (`mpiexec`) and logs via Weights & Biases. Before installing anything,
   check:
   - Does it need `mpi4py` + a real MPI implementation (OpenMPI)? If so,
     `conda install -c conda-forge mpi4py openmpi` is usually more reliable
     than pip on Windows.
   - `wandb login` — you'll need a (free) W&B account, or check if there's
     an offline-logging flag (`WANDB_MODE=offline` as an env var usually
     works even if the repo doesn't expose a flag for it).

3. **Get the smallest possible run working FIRST**, before trusting any
   real numbers. Use the smallest `-np` (node count) the repo allows,
   and a tiny `--steps`. This isolates "does MPI/the repo even launch on my
   machine" from "does it produce a correct number" — don't skip this.

4. **Resolve open questions by reading the actual source, not just the
   README:**
   - Confirm which model `--model resnet` (or similar) actually
     instantiates in `models.py` — the paper's text says ResNet50 for the
     CIFAR-10 experiment; the README text we saw mentioned ResNet-18.
     Check `models.py` directly and note which one it really is.
   - Confirm whether `datasets/` actually has a Year-MSD / LibSVM loader
     and a CIFAR-10 loader (only "Brackets" was explicitly confirmed from
     the README before this).
   - Cross-reference the paper's Appendix B.2/B.3 hyperparameters
     (learning rate, `nu`, number of random vectors, population sizes)
     against this repo's CLI flags for the Year-MSD and CIFAR-10 tasks
     specifically, so your run config matches what a published figure
     actually used.

5. **Run the Year-MSD and CIFAR-10 configs**, save stdout/W&B exports, and
   drop the final numbers (and the exact command you ran) into
   `phase0_author_repo/baseline_results.md` (create this file once you
   have real numbers — it's the reference every ablation plot in `src/`
   gets compared against).

6. **If a config won't run cleanly after a reasonable effort**, fall back
   to the closest one that does, and write down exactly what you changed
   and why in `baseline_results.md` — an honest documented deviation beats
   a silently-wrong "exact" number.

## Why this is safe to run in parallel with everything else

This track never touches `src/`, `main.py`, or the datasets you're
downloading for Phase 1/2 — it's a completely separate clone, separate
environment, separate task. Nothing here blocks nothing there.
