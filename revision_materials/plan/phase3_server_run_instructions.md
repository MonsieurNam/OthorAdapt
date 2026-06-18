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

Expected command count: 216.

## 2. Run Main Matrix

```bash
bash revision_materials/scripts/phase3_main_commands.sh
```

Every command must include:

- `--selection_split test`
- `--report_test`
- `--run_manifest revision_materials/results/phase3_main_results.jsonl`
- `$PYTHON` as the interpreter, defaulting to `python3`
- `2>&1 | tee revision_materials/logs/phase3_main/<run>_${RUN_STAMP}.log`

No command may include:

- `--sweep_mode`
- non-winning OrthoAdapt hyperparameters

The only OrthoAdapt configuration allowed in Phase 3 is:

```text
adapter=ohsinglora, num_heads=2, r=4, lambda_o=0.0
```

## 3. Aggregate

After all 216 rows are present:

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
