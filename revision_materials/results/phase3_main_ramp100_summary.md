# Phase 3 Main Ramp100 Summary

Source manifest: `D:/RESEARCH/CLIP-LoRA_group/OthorAdapt/revision_materials/results/phase3_main_ramp100_results.jsonl`.

Accuracy values are test accuracies in percentage points. Intervals are 95% confidence intervals across the three seeds for each dataset-shot-method cell.

## Main Table Means

### 1-shot

| Method | Aircraft | EuroSAT | Food | Pets | Flowers | Caltech | DTD | UCF | Average | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP-LoRA r=8 | 28.95 | 73.47 | 83.97 | 90.46 | 84.45 | 93.73 | 54.55 | 76.29 | 73.23 | 737,280 |
| OrthoAdapt H=2,r=8 | 30.89 | 75.01 | 85.00 | 91.67 | 84.64 | 93.90 | 54.61 | 76.67 | 74.05 | 460,800 |

### 4-shot

| Method | Aircraft | EuroSAT | Food | Pets | Flowers | Caltech | DTD | UCF | Average | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP-LoRA r=8 | 38.19 | 84.36 | 83.20 | 90.61 | 94.38 | 95.08 | 65.70 | 81.05 | 79.07 | 737,280 |
| OrthoAdapt H=2,r=8 | 38.90 | 85.78 | 83.66 | 91.11 | 93.98 | 95.28 | 65.33 | 81.74 | 79.47 | 460,800 |

### 16-shot

| Method | Aircraft | EuroSAT | Food | Pets | Flowers | Caltech | DTD | UCF | Average | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP-LoRA r=8 | 56.99 | 92.42 | 84.51 | 92.69 | 98.27 | 96.34 | 73.29 | 86.85 | 85.17 | 737,280 |
| OrthoAdapt H=2,r=8 | 56.63 | 92.67 | 84.46 | 92.96 | 97.75 | 96.34 | 73.21 | 86.15 | 85.02 | 460,800 |

## Paired Delta Summary

Delta is OrthoAdapt minus CLIP-LoRA, paired by `(dataset, shot, seed)`.

- Overall seed-level pairs: n=72, mean delta=0.356 pp, 95% CI [0.162, 0.551] pp.
- Wins/ties/losses across seed-level pairs: 48 / 3 / 21.

| Shot | n pairs | Mean delta | 95% CI | Wins/ties/losses |
|---:|---:|---:|---:|---:|
| 1 | 24 | 0.816 | [0.427, 1.205] | 19 / 3 / 2 |
| 4 | 24 | 0.402 | [0.094, 0.709] | 19 / 0 / 5 |
| 16 | 24 | -0.149 | [-0.359, 0.061] | 10 / 0 / 14 |

## Dataset-Shot Paired Deltas

| Dataset | Shot | n pairs | Mean delta | 95% CI | Wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| Aircraft | 1 | 3 | 1.940 | [-0.150, 4.030] | 3 / 0 / 0 |
| Aircraft | 4 | 3 | 0.710 | [-0.643, 2.063] | 3 / 0 / 0 |
| Aircraft | 16 | 3 | -0.360 | [-1.031, 0.311] | 0 / 0 / 3 |
| EuroSAT | 1 | 3 | 1.543 | [-0.645, 3.732] | 3 / 0 / 0 |
| EuroSAT | 4 | 3 | 1.420 | [-0.639, 3.479] | 3 / 0 / 0 |
| EuroSAT | 16 | 3 | 0.251 | [-0.370, 0.872] | 3 / 0 / 0 |
| Food | 1 | 3 | 1.034 | [0.512, 1.557] | 3 / 0 / 0 |
| Food | 4 | 3 | 0.455 | [0.241, 0.670] | 3 / 0 / 0 |
| Food | 16 | 3 | -0.056 | [-0.714, 0.601] | 1 / 0 / 2 |
| Pets | 1 | 3 | 1.208 | [-0.802, 3.218] | 3 / 0 / 0 |
| Pets | 4 | 3 | 0.509 | [-0.669, 1.686] | 2 / 0 / 1 |
| Pets | 16 | 3 | 0.273 | [-0.970, 1.516] | 2 / 0 / 1 |
| Flowers | 1 | 3 | 0.189 | [-2.803, 3.182] | 2 / 0 / 1 |
| Flowers | 4 | 3 | -0.406 | [-2.179, 1.367] | 1 / 0 / 2 |
| Flowers | 16 | 3 | -0.514 | [-1.163, 0.134] | 0 / 0 / 3 |
| Caltech | 1 | 3 | 0.176 | [-0.581, 0.932] | 1 / 2 / 0 |
| Caltech | 4 | 3 | 0.203 | [-0.100, 0.505] | 3 / 0 / 0 |
| Caltech | 16 | 3 | -0.000 | [-0.524, 0.524] | 2 / 0 / 1 |
| DTD | 1 | 3 | 0.059 | [-1.781, 1.899] | 2 / 0 / 1 |
| DTD | 4 | 3 | -0.374 | [-2.233, 1.485] | 1 / 0 / 2 |
| DTD | 16 | 3 | -0.079 | [-2.228, 2.071] | 2 / 0 / 1 |
| UCF | 1 | 3 | 0.379 | [-0.513, 1.271] | 2 / 1 / 0 |
| UCF | 4 | 3 | 0.696 | [0.400, 0.992] | 3 / 0 / 0 |
| UCF | 16 | 3 | -0.705 | [-1.925, 0.515] | 0 / 0 / 3 |
