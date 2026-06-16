# Phase 1B Run Manifest Schema

Purpose: define the single-source row format for validation sweeps, final reruns, table generation, and statistical reporting.

## Canonical Flat Row

Each final aggregation row must validate to this flat schema:

| field | type | required | description |
|---|---:|---:|---|
| `schema_version` | string | yes | Must be `phase1b.run.v1`. |
| `dataset` | string | yes | Dataset identifier, e.g. `eurosat`. |
| `method` | string | yes | Method/adapter identifier, e.g. `lora`, `singlora`, `ohsinglora`. |
| `shot` | integer | yes | Few-shot count. |
| `seed` | integer | yes | Random seed. |
| `split` | string | yes | One of `train`, `val`, `test`; final selection must use `val` before `test` reporting. |
| `accuracy` | float | yes | Accuracy for the row's split. |
| `runtime_seconds` | float | yes | End-to-end run or evaluation runtime in seconds. |
| `parameter_count` | integer | yes | Trainable adapter parameter count, including gating/scaler parameters. |
| `checkpoint_sha256` | string | yes | SHA256 of the adapter checkpoint used or produced by the run. |
| `git_revision` | string | yes | Git commit hash or explicit revision token for the code that produced the row. |

## Accepted Nested ECR3 Record

The aggregation loader also accepts ECR3 nested records emitted by `ecr3_provenance.build_run_record` and normalizes them to the flat schema. Required nested locations:

| flat field | nested source |
|---|---|
| `dataset` | `config.dataset` |
| `method` | `config.adapter` |
| `shot` | `config.shots` |
| `seed` | `config.seed` |
| `split` | `config.selection_split` or `metrics.selection_split` |
| `accuracy` | `metrics.<split>_accuracy` or `metrics.selection_accuracy` |
| `runtime_seconds` | `metrics.runtime_seconds` or `metrics.fine_tuning_seconds` |
| `parameter_count` | `metrics.trainable_parameters` or `metrics.parameter_count` |
| `checkpoint_sha256` | `checkpoint.sha256` |
| `git_revision` | top-level `git_revision` |

## Statistical Report Format

`aggregate_results.aggregate_manifest` returns:

- `groups`: keyed by `(dataset, shot, method, split)`, with `n`, `mean_accuracy`, sample `std_accuracy`, two-sided `ci95_half_width`, seed list, checkpoint hashes, git revisions, parameter counts, and total runtime.
- `paired`: keyed by `(dataset, shot, method, split)` for non-baseline methods when `baseline_method` is provided, with `paired_n`, paired seed list, `mean_delta`, `std_delta`, `ci95_delta_half_width`, and paired Cohen `effect_size_dz`.

Rows fail closed before aggregation if any required field is absent or invalid.
