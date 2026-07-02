# Phase 4 Spectrum Report

Source: `revision_materials/results/phase4_spectrum_manifest.jsonl`.

Paired spectrum diagnostics use Phase 3 ramp100 checkpoints, vision layer 11, shot 4, seed 1, and parameters `q_proj` and `v_proj`. This is descriptive evidence only.

- Paired records: 16
- Figures directory: `revision_materials/results/figures/phase4_spectrum`

| Param | n | LoRA stable rank | OrthoAdapt stable rank | LoRA rank90 | OrthoAdapt rank90 |
|---|---:|---:|---:|---:|---:|
| q_proj | 8 | 1.003 | 1.140 | 1.000 | 1.125 |
| v_proj | 8 | 1.389 | 2.023 | 2.500 | 3.250 |

## Claim Gate

- Keep only descriptive spectrum language tied to the selected layer/param subset.
- Do not use this report alone as causal evidence for robustness or performance gains.
