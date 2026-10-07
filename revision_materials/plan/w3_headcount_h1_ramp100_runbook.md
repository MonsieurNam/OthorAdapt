# W3 H=1 Ramp100 Head-Count Add-On

Purpose: fill the missing H=1 cells needed for Reviewer 1 Weakness 3. These 12 validation runs complement `validation_sweep_ramp100_results.jsonl` so the head-count discussion can compare single-head and multi-head settings under ramp100.

## Scope

- Runs: 12 validation runs
- Backbone: CLIP ViT-B/16
- Split: validation only (`--selection_split val --sweep_mode`)
- Datasets: `eurosat`, `caltech101`
- Shots: 4
- Seeds: 1, 2, 3
- Adapter: `ohsinglora`
- Head count: `H=1`
- Ranks: `r=2`, `r=4`
- Orthogonality: `lambda_o=0.0`
- Ramp: `ramp_up_steps=100`

## Run and outputs

Run `revision_materials/scripts/w3_headcount_h1_ramp100_commands.sh`. It writes:

- `revision_materials/results/w3_headcount_h1_ramp100_results.jsonl`
- `revision_materials/logs/w3_headcount_h1_ramp100/*.log`
- `revision_materials/checkpoints/w3_headcount_h1_ramp100/**/*.pt`

## Merge path

Combine validation diagnostics from `validation_sweep_ramp100_results.jsonl` and
`w3_headcount_h1_ramp100_results.jsonl`. Compare both fixed total rank and fixed per-head rank, and describe these as validation diagnostics rather than final
test-set gains.
