# Validation Sweep Protocol Audit

- Raw manifest: `revision_materials/results/validation_sweep_results.jsonl`
- Protocol-only manifest: `revision_materials/results/validation_sweep_results_protocol.jsonl`
- Expected protocol rows: 120
- Raw manifest rows: 121
- Protocol-valid rows: 120
- Unique protocol-valid keys: 120
- Missing protocol keys: 0
- Extra out-of-protocol rows preserved in raw manifest: 1

## Extra Rows

| line | dataset | shots | seed | H | r | lambda_o | filename | created_utc | git_revision | checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | eurosat | 4 | 1 | 1 | 2 | 0.0 | val_ohsinglora_h1_r2_lo0p0 | 2026-06-17T07:27:00+00:00 | ac99f948394bc91c1bb51e49e27925c685342dda | revision_materials/checkpoints/validation_sweep/ohsinglora/ViT-B16/eurosat/4shots/seed1/adapter_weights.pt |

## Decision

Use the protocol-only manifest for Phase 2 selection. Keep the raw manifest unchanged as evidence of the accidental pre-sweep manual run.
