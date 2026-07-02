# Phase 4 Head Overlap Report

Source: `revision_materials/results/phase4_checkpoint_diagnostics.jsonl`.

- Completed checkpoints: 72
- Mean subspace overlap across checkpoints: 0.00253457
- Mean raw pair orthogonality loss per tensor: 0.000480021

## By Shot

| Shot | n | Mean subspace overlap | Std | Mean head norm |
|---:|---:|---:|---:|---:|
| 1 | 24 | 0.00627646 | 0.00607654 | 0.701168 |
| 4 | 24 | 0.00121391 | 0.00187467 | 0.825752 |
| 16 | 24 | 0.000113349 | 0.000242553 | 1.05669 |

## Claim Gate

- This report can support descriptive statements about learned head overlap.
- It does not by itself prove that head overlap causes accuracy or robustness changes.
