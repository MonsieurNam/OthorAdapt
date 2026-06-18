# Phase 3 Server Run Instructions

Use these steps after Phase 2 selection has been frozen. Phase 3 is the first test-set evaluation stage.

## 1. Prepare

```bash
export DATA_ROOT=/path/to/datasets
export PYTHON=/path/to/venv/bin/python
git checkout phase1b-experiment-infra
git pull
```

The frozen Phase 3 protocol is:

```text
revision_materials/plan/phase3_main_protocol.yaml
```

The generated commands are:

```text
revision_materials/scripts/phase3_main_commands.sh
```

The resume-safe runner is:

```text
revision_materials/scripts/phase3_resumable_runner.py
```

Expected command count: 144.

## 2. Estimate And Run Main Matrix

First, check pending work and ETA without launching training:

```bash
$PYTHON revision_materials/scripts/phase3_resumable_runner.py --dry-run
```

The runner estimates time from completed Phase 3 rows when available. Before any
Phase 3 row exists, it falls back to `validation_sweep_results_protocol.jsonl`
and scales by the expected training iterations in each command.

Run the matrix through the resume-safe runner:

```bash
$PYTHON revision_materials/scripts/phase3_resumable_runner.py
```

If the server is interrupted, rerun the same command. Completed runs are skipped
only when the manifest row has `status=completed`, `config.filename`, and a
non-empty checkpoint SHA256. A run interrupted before writing a completed
manifest row will be rerun from scratch.

Every command must include:

- `--selection_split test`
- `--report_test`
- `--run_manifest revision_materials/results/phase3_main_results.jsonl`
- `$PYTHON` as the interpreter, defaulting to `python3`
- `2>&1 | tee revision_materials/logs/phase3_main/<run>_${RUN_STAMP}.log`

The runner sets `RUN_STAMP` for the session if it is not already present. Set it
manually if you want a custom log suffix:

```bash
export RUN_STAMP=$(date +%Y%m%d_%H%M%S)
```

No command may include:

- `--sweep_mode`
- non-winning OrthoAdapt hyperparameters

The only OrthoAdapt configuration allowed in Phase 3 is:

```text
adapter=ohsinglora, num_heads=2, r=4, lambda_o=0.0
```

SingLoRA-CLIP is intentionally excluded from this main matrix because it is an internal unpublished CLIP adaptation, not a stable independently citable few-shot CLIP baseline. The reviewer-facing rationale is recorded in `revision_materials/plan/singlora_clip_baseline_note.md`.

## 3. Aggregate

After all 144 rows are present:

```bash
python aggregate_results.py \
  --manifest revision_materials/results/phase3_main_results.jsonl \
  --out_csv revision_materials/results/phase3_main_summary.csv \
  --baseline_method lora
```

Expected outputs:

- `revision_materials/results/phase3_main_results.jsonl`
- `revision_materials/results/phase3_main_summary.csv`

Do not rewrite manuscript tables until the manifest validates and every dataset/shot/method group has seeds `1 2 3`.
