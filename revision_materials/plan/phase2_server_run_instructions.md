# Phase 2 Server Run Instructions

Use these steps on the rented server. Do not run test-set evaluation during this phase.

## Current Status

Phase 2 ramp100 validation is complete.

- Manifest: `revision_materials/results/validation_sweep_ramp100_results.jsonl`
- Rows: 120/120 completed
- Split: all rows use `selection_split=val`
- Test reporting: all rows have `report_test=false`
- Ramp schedule: all rows use `ramp_up_steps=100`
- Frozen winner: `H=2`, `r=8`, `lambda_o=0.03`, mean validation accuracy `91.666667`
- Frozen selection file: `revision_materials/results/selected_config_ramp100.md`
- SHA256 sidecar: `revision_materials/results/selected_config_ramp100.md.sha256`

These instructions are retained for reproducibility or rerun only. The next stage is Phase 3 ramp100 test-set evaluation using the frozen winner and matched CLIP-LoRA baseline.

## 1. Prepare

```bash
export DATA_ROOT=/path/to/datasets
export PYTHON=python3
git checkout phase1b-experiment-infra
```

If you use conda or venv, point `PYTHON` to that interpreter instead, for example:

```bash
export PYTHON=/path/to/venv/bin/python
```

The frozen protocol is:

```text
revision_materials/plan/selection_protocol.yaml
```

The generated sweep commands are:

```text
revision_materials/scripts/validation_sweep_commands.sh
```

Expected command count: 120.

## 2. Run Validation Sweep

```bash
bash revision_materials/scripts/validation_sweep_commands.sh
```

Every command must include:

- `--selection_split val`
- `--sweep_mode`
- `--run_manifest revision_materials/results/validation_sweep_ramp100_results.jsonl`
- `--ramp_up_steps 100`
- `$PYTHON` as the interpreter, defaulting to `python3`
- `2>&1 | tee revision_materials/logs/validation_sweep_ramp100/<run>_${RUN_STAMP}.log` so stdout and stderr are saved per run without overwriting logs from a later rerun

No command may include `--report_test`.

## 3. Freeze Selected Config

After all 120 validation rows are present:

```bash
$PYTHON phase2_validation_sweep.py select \
  --protocol revision_materials/plan/selection_protocol.yaml \
  --manifest revision_materials/results/validation_sweep_ramp100_results.jsonl \
  --out revision_materials/results/selected_config_ramp100.md
```

The selector requires complete coverage for every pre-registered candidate. It will fail closed if a candidate is missing, if any row uses a non-validation split, or if a row is outside the frozen candidate grid.

Expected outputs:

- `revision_materials/results/validation_sweep_ramp100_results.jsonl`
- `revision_materials/results/selected_config_ramp100.md`
- `revision_materials/results/selected_config_ramp100.md.sha256`

Phase 3 test-set evaluation may start only after `selected_config_ramp100.md` and its SHA256 sidecar exist. They now exist, and the Phase 3 main protocol has been updated to the ramp100 winner `H=2,r=8,lambda_o=0.03`; use the regenerated Phase 3 command files rather than older `r=4,lambda_o=0.0` commands.
