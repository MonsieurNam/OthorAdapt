# Orthogonality Loss Validation Audit

Date: 2026-06-18

## Question

The validation sweep selected `lambda_o=0.0`. Does this mean the orthogonality mechanism is meaningless?

## Short Answer

No. The current evidence says something narrower:

- The orthogonality regularizer is active and does enforce near-orthogonal heads.
- Under the pre-registered global validation-selection rule, the best performing configuration is `H=2, r=4, lambda_o=0.0`.
- Under the fixed old-capacity setting `H=2, r=2`, `lambda_o=0.03` still improves mean validation accuracy over `lambda_o=0.0`.
- Therefore, the main paper should not claim that the orthogonality loss is the primary source of performance gains. It should frame it as an explored regularizer whose benefit is configuration-dependent.

## Evidence From Validation Sweep

Top validation configuration:

| H | r | lambda_o | mean_val_acc | eurosat_mean | caltech101_mean | params |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 0.0 | 91.541667 | 88.333333 | 94.750000 | 276480 |

Fixed `H=2, r=2` comparison:

| H | r | lambda_o | mean_val_acc | eurosat_mean | caltech101_mean | params |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.0 | 90.291667 | 85.833333 | 94.750000 | 184320 |
| 2 | 2 | 0.03 | 90.666667 | 86.666667 | 94.666667 | 184320 |

Fixed `H=2, r=4` comparison:

| H | r | lambda_o | mean_val_acc | eurosat_mean | caltech101_mean | params |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 0.0 | 91.541667 | 88.333333 | 94.750000 | 276480 |
| 2 | 4 | 0.01 | 91.000000 | 87.500000 | 94.500000 | 276480 |
| 2 | 4 | 0.03 | 90.708333 | 86.666667 | 94.750000 | 276480 |

## Code/Checkpoint Audit

The training path applies orthogonality loss in `lora.py` only when:

- `args.lambda_o > 0`
- `args.adapter in ['ohsinglora', 'gmhsinglora']`
- the module is an instance of `LinearOHsingLoRA`

The checkpoint-level diagnostic confirms the loss is not a no-op:

| Checkpoint | Total raw orthogonality loss across OH tensors |
|---|---:|
| EuroSAT seed2 `H=2,r=4,lambda_o=0.03` | about `6.23e-10` |
| EuroSAT seed2 `H=2,r=4,lambda_o=0.0` | about `1.25` |

So the regularizer drives head overlap almost to zero. The problem is not that the loss is disabled; the problem is that lower overlap does not always translate into higher validation accuracy.

## Interpretation

The old story likely mixed at least two effects:

1. Fixed-capacity effect: at `H=2,r=2`, orthogonality regularization can help.
2. Capacity/config effect: when `r=4` is allowed, the unregularized model has enough capacity and wins the validation metric.

This means the mechanism should be reframed, not discarded.

## Recommended Manuscript Framing

Avoid:

- "orthogonality loss is the main reason for the gain"
- "orthogonality regularization consistently improves performance"
- "orthogonality causes robustness"

Use:

- "The validation-selected configuration uses the adaptive multi-head symmetric adapter without an explicit orthogonality penalty."
- "Orthogonality regularization was beneficial under some fixed-capacity settings, but was not selected by the validation protocol for the main configuration."
- "This suggests that the benefit of the orthogonality penalty is configuration-dependent, while the main evidence supports the adaptive multi-head adapter structure."

## Action

Do not rewrite the project from scratch. Keep the code path, but revise the claims and separate:

1. main validation-selected performance configuration: `H=2,r=4,lambda_o=0.0`
2. fixed-capacity orthogonality ablation: e.g., `H=2,r=2,lambda_o=0.0` vs `0.03`
3. mechanism/diagnostic claims: only make them if supported by spectrum/gating/robustness diagnostics
