# Phase 4 Inventory

## Usable Evidence

- `revision_materials/results/validation_sweep_ramp100_results.jsonl`: 120 validation-only rows for H/r/lambda sensitivity.
- `revision_materials/results/w3_headcount_h1_ramp100_results.jsonl`: 12 validation-only H=1 rows.
- `revision_materials/results/phase3_main_ramp100_results.jsonl`: 144 final Phase 3 rows for checkpoint pairing.
- Phase 3 ramp100 checkpoints on disk: 144.

## Preliminary / Legacy Evidence

- `data/result_scan_head`, `data/results_ablation_heads_lambda`, and `data/result_scan_loss` contain recovered logs but lack verified seed/split provenance; use only for context.
- `img/figure_robustness.*` and `img/singular_value_spectrum-rank16.png` are legacy figures and must not be cited as regenerated evidence.

## Missing Evidence Before Phase 4 Runs

- Paired robustness manifest with deterministic corruption parameters.
- Batch spectrum manifest from Phase 3 paired checkpoints.
- ViT-L/14 pilot/subset manifest, or an explicit deferred report.
