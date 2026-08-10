# Reviewer-2 lambda=0 remote run

Frozen configuration: OrthoAdapt/OH-SingLoRA, `H=2`, `r=8`, `alpha=1`, `lambda_o=0`, `ramp_up_steps=100`, ViT-B/16, both encoders, all layers, Q/K/V, 500 base iterations scaled by shots, batch size 32.

Coverage: 8 datasets x 3 shots x 3 seeds = 72 train commands plus 72 checkpoint-based eval commands.

## Remote launch

From the repository root:

```bash
export DATA_ROOT=/absolute/path/to/datasets
export PYTHON=python3

# Optional one-run training smoke test. Re-running is resume-safe.
MAX_RUNS=1 bash revision_materials/scripts/phase3_reviewer2_lambda0_run.sh train

# Full training; the completed smoke run is skipped.
unset MAX_RUNS
bash revision_materials/scripts/phase3_reviewer2_lambda0_run.sh train

# Test evaluation from the saved checkpoints.
bash revision_materials/scripts/phase3_reviewer2_lambda0_run.sh eval
```

For an unattended session:

```bash
tmux new -s reviewer2-lambda0
export DATA_ROOT=/absolute/path/to/datasets
export PYTHON=python3
bash revision_materials/scripts/phase3_reviewer2_lambda0_run.sh all \
  2>&1 | tee revision_materials/logs/phase3_reviewer2_lambda0_launcher.log
```

Detach with `Ctrl-b d` and reconnect with `tmux attach -t reviewer2-lambda0`.

## Output paths

- Train manifest: `revision_materials/results/phase3_reviewer2_lambda0_train_results.jsonl`
- Eval manifest: `revision_materials/results/phase3_reviewer2_lambda0_eval_results.jsonl`
- Checkpoints: `revision_materials/checkpoints/phase3_reviewer2_lambda0_ramp100/`
- Train logs: `revision_materials/logs/phase3_reviewer2_lambda0_train/`
- Eval logs: `revision_materials/logs/phase3_reviewer2_lambda0_eval/`

Do not reuse the legacy `r=4`, `ramp_up_steps=1000`, `lambda_o=0` checkpoints for this comparison. The reviewer-facing orthogonality contrast requires the frozen `r=8`, ramp100 control above.
