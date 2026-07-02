# Phase 3 Ramp100 Audit

Source manifest: `D:/RESEARCH/CLIP-LoRA_group/OthorAdapt/revision_materials/results/phase3_main_ramp100_results.jsonl`.

## Coverage

- Expected rows: 144
- Manifest rows: 144
- Methods: {'lora': 72, 'ohsinglora': 72}
- Status counts: {'completed': 144}
- Selection split counts: {'test': 144}
- Report-test counts: {True: 144}
- Missing expected `(method, dataset, shot, seed)` keys: 0
- Extra keys: 0
- Duplicate keys: 0

## Parameter And Ramp Metadata

| Method | Parameter count | `ramp_up_steps` metadata | Note |
|---|---:|---:|---|
| CLIP-LoRA r=8 | 737,280 | 100 | Normalized metadata only; LoRA does not use ramp-up scheduling. |
| OrthoAdapt H=2,r=8 | 460,800 | 100 | Active ramp-up setting for OH-SingLoRA/OrthoAdapt. |

## Overall Method Means

| Method | n | Mean accuracy | Std | 95% CI | Runtime (h) | Params |
|---|---:|---:|---:|---:|---:|---:|
| CLIP-LoRA r=8 | 72 | 79.157 | 18.100 | [74.905, 83.410] | 12.113 | 737,280 |
| OrthoAdapt H=2,r=8 | 72 | 79.514 | 17.883 | [75.312, 83.715] | 27.170 | 460,800 |
