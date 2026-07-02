# Phase 4 Orthogonality Report

Source: `revision_materials/results/phase4_checkpoint_diagnostics.jsonl`.

The diagnostics below are computed directly from Phase 3 ramp100 OrthoAdapt checkpoints. They refresh the earlier orthogonality audit for the final selected configuration.

| Dataset | Shot | n | Raw ortho mean/tensor | Subspace overlap mean |
|---|---:|---:|---:|---:|
| caltech101 | 1 | 3 | 0.000473136 | 0.00390732 |
| caltech101 | 4 | 3 | 0.000226628 | 0.00091776 |
| caltech101 | 16 | 3 | 5.9232e-06 | 3.37764e-05 |
| dtd | 1 | 3 | 3.98847e-09 | 4.16876e-08 |
| dtd | 4 | 3 | 3.17185e-11 | 1.44503e-10 |
| dtd | 16 | 3 | 4.25535e-11 | 9.77776e-11 |
| eurosat | 1 | 3 | 6.43363e-11 | 5.12441e-10 |
| eurosat | 4 | 3 | 2.86388e-11 | 1.66464e-10 |
| eurosat | 16 | 3 | 9.81011e-12 | 3.67556e-11 |
| fgvc | 1 | 3 | 0.00289957 | 0.0162348 |
| fgvc | 4 | 3 | 0.0023692 | 0.00578674 |
| fgvc | 16 | 3 | 0.000255027 | 0.000732002 |
| food101 | 1 | 3 | 0.000738826 | 0.00630543 |
| food101 | 4 | 3 | 0.00015347 | 0.000745398 |
| food101 | 16 | 3 | 7.98553e-06 | 3.77569e-05 |
| oxford_flowers | 1 | 3 | 0.00168372 | 0.0107207 |
| oxford_flowers | 4 | 3 | 4.36244e-05 | 0.000352986 |
| oxford_flowers | 16 | 3 | 1.57803e-07 | 6.05497e-07 |
| oxford_pets | 1 | 3 | 5.44385e-05 | 0.000637697 |
| oxford_pets | 4 | 3 | 4.8852e-06 | 5.65203e-05 |
| oxford_pets | 16 | 3 | 1.62174e-07 | 8.87546e-07 |
| ucf101 | 1 | 3 | 0.00207129 | 0.0124056 |
| ucf101 | 4 | 3 | 0.000511268 | 0.00185187 |
| ucf101 | 16 | 3 | 2.11908e-05 | 0.000101762 |

## Claim Gate

- Keep: the final OrthoAdapt checkpoints contain measurable multi-head adapter tensors and can be audited for overlap.
- Weaken: orthogonality should be described as configuration-dependent unless paired ablations show a consistent accuracy or robustness effect.
