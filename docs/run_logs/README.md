# Run logs

Raw console (stdout) transcripts of the post-fix ablation reruns — the
exact `python main.py --task <task> --mode ablation --seeds 0 1 2`
commands that produced the current committed results, kept verbatim as a
human-readable record of the actual runs.

These are a convenience/appendix, not a primary source: every number in
them is also in `results/<task>/ablation/aggregated.json` and
`raw_history_all_seeds.json`, which carry strictly more data (every eval
round rather than every 100th, plus gamma/var_ema/alpha diagnostics not
printed to console).

- `yearmsd_ablation_stdout.txt` — `python main.py --task yearmsd --mode ablation --seeds 0 1 2`
- `cifar10_ablation_stdout.txt` — `python main.py --task cifar10 --mode ablation --seeds 0 1 2`

Both are from the rerun after the `update_adaptive_budgets` fix in
`src/simulate.py` (see `PROGRESS.md`, "Fix adaptive-ZO-budget ratchet
bug; rerun both ablation suites").
