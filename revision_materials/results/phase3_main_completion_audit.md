# Phase 3 Main Completion Audit

Generated: 2026-06-20

## Completion Status

Phase 3 main matrix is complete.

- Expected commands: 144
- Manifest rows: 144
- Unique completed filenames: 144
- Missing filenames: 0
- Extra filenames: 0
- Duplicate filenames: 0
- Bad rows: 0
- Resume runner status: `completed=144/144, pending=0`

The three previously missing UCF101 16-shot OrthoAdapt runs are now present with
checkpoints and logs:

```text
ucf101_16shot_seed1_test_ohsinglora_h2_r4_lo0p0
ucf101_16shot_seed2_test_ohsinglora_h2_r4_lo0p0
ucf101_16shot_seed3_test_ohsinglora_h2_r4_lo0p0
```

## Aggregate Results

Overall rows:

| Method | n | Mean accuracy | Std accuracy | Trainable parameters |
|---|---:|---:|---:|---:|
| CLIP-LoRA | 72 | 79.2915 | 18.0492 | 368,640 |
| OrthoAdapt | 72 | 78.2552 | 18.4920 | 276,480 |

Paired seed-level comparison:

- Complete paired comparisons: 72
- Mean delta, OrthoAdapt minus CLIP-LoRA: -1.0364 pp
- Median delta: -0.3053 pp
- Bootstrap 95% CI for mean delta: [-1.6184, -0.5298] pp
- Normal-approx 95% CI for mean delta: [-1.5847, -0.4881] pp
- Wins / ties / losses for OrthoAdapt over CLIP-LoRA: 24 / 1 / 47

Mean paired delta by shot:

| Shot | n pairs | Mean delta |
|---:|---:|---:|
| 1 | 24 | -2.7118 |
| 4 | 24 | -0.1111 |
| 16 | 24 | -0.2863 |

Mean paired delta by dataset:

| Dataset | n pairs | Mean delta |
|---|---:|---:|
| caltech101 | 9 | +0.1307 |
| dtd | 9 | -1.2543 |
| eurosat | 9 | -0.9067 |
| fgvc | 9 | -1.0868 |
| food101 | 9 | -0.0117 |
| oxford_flowers | 9 | -3.6405 |
| oxford_pets | 9 | -0.6965 |
| ucf101 | 9 | -0.8253 |

Best dataset-shot deltas:

| Dataset | Shot | CLIP-LoRA | OrthoAdapt | Delta |
|---|---:|---:|---:|---:|
| caltech101 | 1 | 93.7390 | 94.1447 | +0.4057 |
| eurosat | 4 | 85.5185 | 85.7778 | +0.2593 |
| food101 | 4 | 83.5248 | 83.7613 | +0.2365 |
| ucf101 | 4 | 81.6283 | 81.7693 | +0.1410 |
| food101 | 16 | 84.3696 | 84.4994 | +0.1298 |
| eurosat | 16 | 92.3416 | 92.4609 | +0.1193 |
| oxford_pets | 16 | 92.8227 | 92.8500 | +0.0273 |
| caltech101 | 4 | 95.4293 | 95.4564 | +0.0270 |

Worst dataset-shot deltas:

| Dataset | Shot | CLIP-LoRA | OrthoAdapt | Delta |
|---|---:|---:|---:|---:|
| oxford_flowers | 1 | 83.9491 | 73.5282 | -10.4209 |
| eurosat | 1 | 74.0412 | 70.9424 | -3.0988 |
| dtd | 1 | 54.3144 | 51.5366 | -2.7778 |
| ucf101 | 1 | 76.4737 | 74.2532 | -2.2205 |
| oxford_pets | 1 | 91.1693 | 89.0888 | -2.0805 |
| fgvc | 16 | 56.8957 | 55.3355 | -1.5602 |
| fgvc | 1 | 29.6930 | 28.5929 | -1.1001 |
| fgvc | 4 | 38.2338 | 37.6338 | -0.6001 |

Runtime observed:

- Total runtime across 144 completed rows: 32.9367 hours
- CLIP-LoRA runtime: 12.9280 hours
- OrthoAdapt runtime: 20.0087 hours

## Parameter-Count Check

For ViT-B/16 with both encoders, all 12 text blocks and all 12 vision blocks,
and LoRA applied to `q`, `k`, and `v`, the adapter parameter counts are:

| Method/config | Trainable parameters | Notes |
|---|---:|---|
| CLIP-LoRA, r=4 | 368,640 | Standard LoRA has rank `r`; it has no head count and no `lambda_o`. |
| OrthoAdapt/OH-SingLoRA, H=1, r=2 | 138,240 | Formula: `46,080 * (r + H)`. |
| OrthoAdapt/OH-SingLoRA, H=2, r=2 | 184,320 | Matches Phase 2 manifest rows for H=2,r=2. |
| OrthoAdapt/OH-SingLoRA, H=1, r=4 | 230,400 | Not part of the 120-run Phase 2 sweep because H=1 was excluded. |
| OrthoAdapt/OH-SingLoRA, H=2, r=4 | 276,480 | Selected Phase 2 config used in Phase 3. |
| OrthoAdapt/OH-SingLoRA, H=4, r=4 | 368,640 | Same count as CLIP-LoRA r=4, but the mechanisms differ. |

The Phase 3 manifest stores `num_heads=2` under the raw CLI config for CLIP-LoRA
because the earlier command generator passed a uniform argument set to `main.py`.
This value is not used by the LoRA implementation. CLIP-LoRA should therefore
be reported as `r=4` only, not as a head-based method.

## Interpretation

The completed Phase 3 main matrix does not support a broad performance-gain
claim for OrthoAdapt over CLIP-LoRA. Across the fully paired 144-run matrix,
OrthoAdapt is lower on average by approximately 1.04 percentage points, with the
largest degradation in the 1-shot setting. The strongest negative outlier is
Oxford Flowers 1-shot.

The main positive interpretation available from this matrix is parameter
efficiency: OrthoAdapt uses 276,480 trainable parameters versus 368,640 for
CLIP-LoRA, approximately 25% fewer trainable parameters. However, this parameter
reduction comes with lower average accuracy in the current selected
configuration.

Because the selected Phase 2 configuration has `lambda_o=0.0`, final manuscript
claims must not attribute any effect to the orthogonality regularizer. The next
revision step should either:

1. Reframe OrthoAdapt as a parameter-efficiency trade-off rather than an
   accuracy-improving method, or
2. Revisit the selection/evaluation design before making any performance claim.

Do not write a broad SOTA or consistent-improvement claim from this Phase 3
matrix.
