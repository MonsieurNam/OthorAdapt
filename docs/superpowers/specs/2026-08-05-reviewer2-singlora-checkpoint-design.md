# Reviewer 2 SingLoRA Replication and Checkpoint V2 Design

## Objective

Add a reproducible 72-run SingLoRA `r=8` experiment matrix matching the frozen Phase 3 protocol, while making SingLoRA/OrthoAdapt checkpoints portable across training and evaluation machines. New checkpoints must preserve ramp-up state and reject incompatible evaluation configurations.

## Scope

The new experiment contains one method only: SingLoRA with `r=8`, `alpha=1`, and `ramp_up_steps=100`. It covers the eight Phase 3 datasets, 1/4/16-shot settings, and seeds 1/2/3, giving 72 training runs. It does not add the optional parameter-matched `r=10` baseline or the full OrthoAdapt `lambda_o=0` matrix.

The existing frozen Phase 3 protocol and its generated artifacts remain unchanged. Reviewer-2 replication artifacts use distinct protocol, manifest, checkpoint, and log paths.

## Experiment architecture

A dedicated generator, `phase3_reviewer2_singlora_experiment.py`, will validate a JSON-compatible protocol and emit two scripts. The training script will run the 72 fixed SingLoRA configurations, evaluate the validation split after training, save the final adapter checkpoint, and write a training manifest. The evaluation script will reconstruct the same adapter, load the transferred checkpoint, evaluate the test split with `--eval_only`, and write a separate evaluation manifest.

Both scripts will use the frozen Phase 3 settings: CLIP ViT-B/16, both encoders, all transformer blocks, Q/K/V projections, 500 iterations per shot, batch size 32, cross-entropy loss, and the same dataset/shot/seed ordering. The generator will require `ramp_up_steps=100`; this flag must be present in both training and evaluation commands. Filenames will include `singlora_r8_ramp100` to make the effective configuration auditable.

The existing resumable runner will be reused through its `--commands` and `--manifest` arguments. It will recognize a completed evaluation row only when the row has `status=eval_only` and a non-empty checkpoint SHA-256.

## Checkpoint v2 format

`save_adapter` will continue to store adapter parameters but will additionally include every adapter layer's `training_step` buffer. The top-level metadata will include a schema version and the following normalized fields:

- `adapter`
- `r`
- effective `num_heads` (`1` for SingLoRA; configured value for gated multi-head variants)
- `alpha`
- `ramp_up_steps`
- `params`
- `position`
- `encoder`
- `backbone`

Method-specific fields such as `lambda_o` and `ortho_reduction` may also be recorded for provenance, but they do not change the required compatibility keys for SingLoRA.

On load, checkpoint v2 metadata must exactly match the evaluation arguments after normalization. The loader must confirm that training-step buffers exist, load them, and verify that every adapted layer has reached `training_step >= ramp_up_steps`. A checkpoint with an incompatible architecture or an unsaturated ramp must fail with an actionable `ValueError`; silent fallback to an adapter-disabled evaluation is forbidden.

Legacy checkpoints lack both complete metadata and saved ramp state. They will not be silently treated as checkpoint v2. The loader will identify them explicitly and fail with guidance rather than producing an invalid evaluation. Existing training results remain usable as reported metrics, but legacy checkpoints require an explicit migration or rerun before evaluation-only use.

## Evaluation provenance

`load_adapter` will return the resolved checkpoint path. Evaluation-only manifest rows will record that path and its SHA-256, allowing the train and evaluation manifests to be joined and checked after checkpoint transfer. Dataset split hashes already recorded by the ECR3 manifest remain the authority for confirming identical data across machines.

## Error handling

Protocol validation will reject any dataset/shot/seed matrix other than the fixed Phase 3 matrix, multiple methods, adapters other than SingLoRA, ranks other than 8, alpha other than 1, or ramp-up values other than 100. Checkpoint loading will reject missing weights, missing v2 metadata, mismatched compatibility fields, missing training-step buffers, or unsaturated ramp state.

## Test strategy

Tests will be written before implementation. Generator tests will verify the 72-command count, exact dataset/shot/seed coverage, required flags, isolated artifact paths, and absence of OrthoAdapt/LoRA flags. Metadata tests will verify normalization and field-by-field mismatch failures. A checkpoint round-trip test, when PyTorch is available, will verify that adapter output before save equals output after loading into a fresh model and that `training_step` is preserved. Resumable-runner tests will verify that hashed `eval_only` rows are considered complete while unhashed rows are not.

Local verification will use the available standard-library unit tests and syntax compilation. Any PyTorch-dependent round-trip test that cannot run in the local environment must be run in the GPU environment before starting the 72-run matrix.

## Acceptance criteria

The change is ready for experiment execution when the generator deterministically emits exactly 72 training commands and 72 evaluation commands, all use the frozen Phase 3 configuration and ramp-up 100, checkpoint v2 rejects incompatible or unsaturated states, evaluation manifests contain the transferred checkpoint SHA-256, and all available regression tests pass.
