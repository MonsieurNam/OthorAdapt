# Missing Tier-A Matrix

Generated from `phase0_raw_artifact_manifest.csv`.

## Status Counts

| status | count |
| --- | --- |
| evidence_available_config_unclear_or_nonstandard | 16 |
| log_or_csv_evidence_available | 48 |
| strict_r2_h2_evidence_available | 8 |

## Missing Or Rerun Required

No dataset/shot/method cell is completely missing recovered log/CSV/checkpoint evidence.

## OrthoAdapt Evidence With Unclear/Nonstandard Config

| dataset | shot | log_count | csv_count | checkpoint_paths |
| --- | --- | --- | --- | --- |
| fgvc | 4 | 11 | 0 | 0 |
| fgvc | 16 | 6 | 4 | 14 |
| eurosat | 4 | 22 | 11 | 4 |
| eurosat | 16 | 17 | 15 | 25 |
| food101 | 4 | 11 | 0 | 0 |
| food101 | 16 | 6 | 4 | 12 |
| oxford_pets | 4 | 11 | 0 | 3 |
| oxford_pets | 16 | 6 | 4 | 8 |
| oxford_flowers | 4 | 11 | 0 | 0 |
| oxford_flowers | 16 | 6 | 4 | 10 |
| caltech101 | 4 | 22 | 11 | 2 |
| caltech101 | 16 | 17 | 15 | 24 |
| dtd | 4 | 11 | 0 | 3 |
| dtd | 16 | 6 | 4 | 19 |
| ucf101 | 4 | 10 | 0 | 0 |
| ucf101 | 16 | 6 | 4 | 6 |

## Interpretation

- `strict_r2_h2_evidence_available` means a recovered log or CSV row explicitly carries `rank=2` and `heads=2`.
- `log_or_csv_evidence_available` means baseline evidence exists, but seed/split hashes remain missing.
- All cells still require seed2/seed3 recovery or rerun before Tier-A mean/std claims are defensible.
