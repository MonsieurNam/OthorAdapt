# Phase 3 Ramp100 Statistical Report

Source manifest: `D:/RESEARCH/CLIP-LoRA_group/OthorAdapt/revision_materials/results/phase3_main_ramp100_results.jsonl`.

This report uses the final ramp100 Phase 3 test manifest after validation-only hyperparameter selection. Accuracy values are percentages. Paired deltas are computed as OrthoAdapt minus CLIP-LoRA using the exact `(dataset, shot, seed)` match.

## Audit Result

- Coverage is complete: 8 datasets x 3 shots x 2 methods x 3 seeds = 144 completed rows.
- All rows use `selection_split=test` with `report_test=true`.
- Pairing is complete for 72 seed-level comparisons.
- Parameter counts are stable: CLIP-LoRA r=8 has 737,280 trainable parameters; OrthoAdapt H=2,r=8 has 460,800 trainable parameters.
- The LoRA rows store `ramp_up_steps=100` only as normalized metadata from the shared CLI schema. This value is not a LoRA mechanism and should not be interpreted as a LoRA training component.

## Overall Accuracy

| Method | n | Mean accuracy | Std | 95% CI | Params |
|---|---:|---:|---:|---:|---:|
| CLIP-LoRA r=8 | 72 | 79.157 | 18.100 | [74.905, 83.410] | 737,280 |
| OrthoAdapt H=2,r=8 | 72 | 79.514 | 17.883 | [75.312, 83.715] | 460,800 |

## Paired Comparison

Across 72 matched seed-level pairs, OrthoAdapt changes accuracy by 0.356 percentage points relative to CLIP-LoRA, with a 95% CI of [0.162, 0.551] pp. The paired standardized effect size dz is 0.430. OrthoAdapt wins/ties/losses are 48 / 3 / 21.

OrthoAdapt uses 37.5% fewer trainable parameters than CLIP-LoRA r=8 in this matrix.

## Paired Delta By Shot

| Shot | n pairs | Mean delta | 95% CI | dz | Wins/ties/losses |
|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 0.816 | [0.427, 1.205] | 0.887 | 19 / 3 / 2 |
| 4 | 24 | 0.402 | [0.094, 0.709] | 0.551 | 19 / 0 / 5 |
| 16 | 24 | -0.149 | [-0.359, 0.061] | -0.299 | 10 / 0 / 14 |

## Paired Delta By Dataset

| Dataset | n pairs | Mean delta | 95% CI | dz | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| Aircraft | 9 | 0.763 | [-0.100, 1.627] | 0.679 | 6 / 0 / 3 |
| EuroSAT | 9 | 1.071 | [0.400, 1.743] | 1.227 | 9 / 0 / 0 |
| Food | 9 | 0.478 | [0.091, 0.865] | 0.949 | 7 / 0 / 2 |
| Pets | 9 | 0.663 | [0.142, 1.185] | 0.978 | 7 / 0 / 2 |
| Flowers | 9 | -0.244 | [-0.846, 0.359] | -0.311 | 3 / 0 / 6 |
| Caltech | 9 | 0.126 | [-0.041, 0.293] | 0.582 | 6 / 2 / 1 |
| DTD | 9 | -0.131 | [-0.675, 0.413] | -0.186 | 5 / 0 / 4 |
| UCF | 9 | 0.123 | [-0.421, 0.667] | 0.174 | 5 / 1 / 3 |

## Claim Implications

- Keep: validation-only selection was enforced before test reporting.
- Keep: the final matrix is fully paired over datasets, shots, and seeds.
- Keep with precise wording: OrthoAdapt shows a modest positive average delta in this ramp100 final matrix while using fewer trainable parameters than CLIP-LoRA r=8.
- Weaken/remove: any broad claim of state-of-the-art performance, large gains, or consistent per-dataset superiority.
- Weaken/remove: claims that orthogonality causes robustness or that head-count behavior is explained by rank fragmentation; those require Phase 4 diagnostics.
