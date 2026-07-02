# Phase 4 Existing Diagnostics

This report uses validation-only diagnostics already available before new Phase 4 runs. These values support configuration-sensitivity discussion only; they do not prove rank-fragmentation or robustness causality.

## Coverage

- Validation sweep ramp100 rows used: 120
- W3 H=1 ramp100 rows used: 12
- All rows are validation split, 4-shot, EuroSAT/Caltech101, seeds 1/2/3.

## H/r/lambda Sensitivity

| H | r | lambda_o | n | Mean val acc | 95% CI | Params |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 0.03 | 6 | 91.667 | [87.033, 96.301] | 460800 |
| 2 | 8 | 0.05 | 6 | 91.583 | [87.006, 96.160] | 460800 |
| 1 | 2 | 0 | 6 | 91.333 | [86.150, 96.517] | 138240 |
| 2 | 8 | 0.01 | 6 | 91.208 | [86.028, 96.389] | 460800 |
| 2 | 8 | 0 | 6 | 91.167 | [86.018, 96.316] | 460800 |
| 2 | 4 | 0 | 6 | 91.125 | [85.986, 96.264] | 276480 |
| 4 | 8 | 0 | 6 | 91.083 | [86.567, 95.600] | 552960 |
| 4 | 8 | 0.01 | 6 | 91.042 | [86.550, 95.533] | 552960 |
| 2 | 2 | 0.05 | 6 | 91.000 | [86.619, 95.381] | 184320 |
| 2 | 2 | 0 | 6 | 90.958 | [86.620, 95.297] | 184320 |
| 2 | 2 | 0.03 | 6 | 90.958 | [86.620, 95.297] | 184320 |
| 2 | 4 | 0.05 | 6 | 90.792 | [84.890, 96.693] | 276480 |
| 4 | 4 | 0.01 | 6 | 90.792 | [84.458, 97.125] | 368640 |
| 4 | 4 | 0.03 | 6 | 90.792 | [84.450, 97.134] | 368640 |
| 4 | 4 | 0.05 | 6 | 90.792 | [84.458, 97.125] | 368640 |
| 4 | 4 | 0 | 6 | 90.750 | [84.435, 97.065] | 368640 |
| 2 | 4 | 0.03 | 6 | 90.708 | [84.868, 96.549] | 276480 |
| 2 | 4 | 0.01 | 6 | 90.667 | [84.861, 96.472] | 276480 |
| 4 | 8 | 0.03 | 6 | 90.625 | [85.603, 95.647] | 552960 |
| 4 | 8 | 0.05 | 6 | 90.625 | [85.603, 95.647] | 552960 |
| 2 | 2 | 0.01 | 6 | 90.125 | [84.080, 96.170] | 184320 |
| 1 | 4 | 0 | 6 | 89.750 | [81.613, 97.887] | 230400 |

## Claim Gate

- Keep: the selected ramp100 configuration is `H=2,r=8,lambda_o=0.03` under the frozen validation rule.
- Keep: orthogonality/lambda effects are configuration-dependent in the validation subset.
- Weaken/remove: any claim that higher head count universally helps or hurts.
- Weaken/remove: rank-fragmentation explanations until checkpoint diagnostics support them.
