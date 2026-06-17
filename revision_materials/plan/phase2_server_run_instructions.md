# Phase 2 Server Run Instructions

Use these steps on the rented server. Do not run test-set evaluation during this phase.

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
- `--run_manifest revision_materials/results/validation_sweep_results.jsonl`
- `$PYTHON` as the interpreter, defaulting to `python3`
- `2>&1 | tee revision_materials/logs/validation_sweep/<run>_${RUN_STAMP}.log` so stdout and stderr are saved per run without overwriting logs from a later rerun

No command may include `--report_test`.

## 3. Freeze Selected Config

After all 120 validation rows are present:

```bash
python phase2_validation_sweep.py select \
  --protocol revision_materials/plan/selection_protocol.yaml \
  --manifest revision_materials/results/validation_sweep_results.jsonl \
  --out revision_materials/results/selected_config.md
```

The selector requires complete coverage for every pre-registered candidate. It will fail closed if a candidate is missing, if any row uses a non-validation split, or if a row is outside the frozen candidate grid.

Expected outputs:

- `revision_materials/results/validation_sweep_results.jsonl`
- `revision_materials/results/selected_config.md`
- `revision_materials/results/selected_config.md.sha256`

Do not start Phase 3 test-set evaluation until `selected_config.md` and its SHA256 sidecar exist.
