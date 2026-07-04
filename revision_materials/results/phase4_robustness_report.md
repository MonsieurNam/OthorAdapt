# Phase 4 Robustness Report

Source: `revision_materials/results/phase4_robustness_manifest.jsonl`.

- Severity rows: 576 / 576
- Paired by dataset, shot, seed, and severity.
- Clean consistency gate: `fail` against `revision_materials/results/phase3_main_ramp100_results.jsonl`.

| Severity | n pairs | Mean delta OH-LoRA | 95% CI | Mean LoRA drop | Mean OH drop | Drop delta OH-LoRA |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 72 | -15.483 | [-18.912, -12.053] | 0.000 | 0.000 | 0.000 |
| 1 | 72 | -15.185 | [-18.534, -11.837] | 4.936 | 4.639 | -0.298 |
| 2 | 72 | -11.621 | [-13.783, -9.459] | 19.793 | 15.931 | -3.862 |
| 3 | 72 | -9.236 | [-10.762, -7.709] | 33.759 | 27.512 | -6.247 |

## Clean Consistency Audit

- Compared severity-0 rows against Phase 3 test accuracy with tolerance 1.0 pp: 144 comparisons, 63 failures, 0 missing references.

| Method | n | Mean clean-minus-Phase3 | Min | Max | Failures > tolerance |
|---|---:|---:|---:|---:|---:|
| lora | 72 | 0.001 | -0.120 | 0.236 | 0 |
| ohsinglora | 72 | -15.838 | -51.691 | 1.884 | 63 |

## Claim Gate

- Do not use this robustness manifest for scientific claims yet. Although coverage is structurally complete, the clean severity-0 check does not reproduce Phase 3 accuracy.
- The current failure pattern indicates an evaluator/checkpoint-loading mismatch, especially for OH-SingLoRA. Fix the evaluator and rerun robustness before reporting any robustness numbers.
- Reviewer-facing wording for now: paired robustness evaluation was audited but did not pass the clean-consistency gate, so robustness gains are removed/deferred.
