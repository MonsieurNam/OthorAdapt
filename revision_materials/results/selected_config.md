# Selected Validation Configuration

Status: FROZEN_FROM_VALIDATION_ONLY_SWEEP

## Selection Rule

- Split: val only
- Metric: unweighted mean validation accuracy across the pre-registered datasets, shots, and seeds
- Tie-break: lower parameter count, then lower lambda_o

## Winner

- num_heads: 2
- r: 4
- lambda_o: 0.0
- mean_validation_accuracy: 91.541667
- n: 6
- covered_datasets: caltech101, eurosat
- covered_shots: 4
- covered_seeds: 1, 2, 3

## Protocol

- schema_version: phase2.selection.v1
- selection_split: val
