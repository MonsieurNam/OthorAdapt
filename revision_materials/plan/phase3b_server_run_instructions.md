# Phase 3B Same-Parameter Server Run

## Purpose

Phase 3B tests the strongest legacy claim at matched parameter count:

| Method | Config | Trainable parameters |
|---|---|---:|
| CLIP-LoRA | `r=2` | 184,320 |
| OrthoAdapt/OH-SingLoRA | `H=2`, `r=2`, `lambda_o=0.03` | 184,320 |

This is a diagnostic same-parameter check. It does not replace the
validation-selected Phase 3 main matrix.

## Expected Workload

- Datasets: `fgvc`, `eurosat`, `food101`, `oxford_pets`, `oxford_flowers`,
  `caltech101`, `dtd`, `ucf101`
- Shots: `1`, `4`, `16`
- Seeds: `1`, `2`, `3`
- Methods: `lora`, `ohsinglora`
- Total: `8 * 3 * 3 * 2 = 144` runs

## Generate Commands

From repo root:

```bash
python phase3b_same_param_experiment.py generate \
  --protocol revision_materials/plan/phase3b_same_param_protocol.yaml \
  --out revision_materials/scripts/phase3b_same_param_commands.sh
```

## Run

```bash
export DATA_ROOT=/root/DATA
export PYTHON=python3
bash revision_materials/scripts/phase3b_same_param_commands.sh
```

Outputs:

- Manifest: `revision_materials/results/phase3b_same_param_ramp100_results.jsonl`
- Logs: `revision_materials/logs/phase3b_same_param_ramp100/`
- Checkpoints: `revision_materials/checkpoints/phase3b_same_param_ramp100/`

## Resume

Use the Phase 3B wrapper:

```bash
export DATA_ROOT=/root/DATA
export PYTHON=python3
python revision_materials/scripts/phase3b_resumable_runner.py
```

The runner skips commands whose `--filename` already appears as completed in
the manifest.

## Progress Report

```bash
python revision_materials/scripts/phase3b_checker_report.py
```

The report prints one JSON line with completed/pending counts, iteration-based
ETA, estimated finish time, and estimated remaining cost.

## Watchdog Loop

The checker loop periodically writes progress notes and resumes the run if no
Phase 3B process is active while work remains:

```bash
tmux new-session -d -s phase3b_checker
tmux send-keys -t phase3b_checker \
  'cd /root/OthorAdapt && DATA_ROOT=/root/DATA PYTHON=/opt/conda/bin/python bash revision_materials/scripts/phase3b_checker_loop.sh' C-m
```

Progress notes:

```text
revision_materials/logs/phase3b_progress_notes.md
```

## Google Drive Backup

One-time backup:

```bash
RCLONE_REMOTE=gdrive \
GDRIVE_DIR=RESEARCH/OHSinglora_CLIP/phase3b_same_param_ramp100_backups \
INCLUDE_CHECKPOINTS=1 \
bash revision_materials/scripts/phase3b_backup_to_gdrive.sh once
```

Looping backup:

```bash
tmux new-session -d -s phase3b_backup
tmux send-keys -t phase3b_backup \
  'cd /root/OthorAdapt && RCLONE_REMOTE=gdrive GDRIVE_DIR=RESEARCH/OHSinglora_CLIP/phase3b_same_param_ramp100_backups INCLUDE_CHECKPOINTS=1 bash revision_materials/scripts/phase3b_backup_to_gdrive.sh loop' C-m
```

## Cron Restore After Reboot

```bash
bash revision_materials/scripts/phase3b_install_cron.sh show
bash revision_materials/scripts/phase3b_install_cron.sh install
```

Remove managed cron entries:

```bash
bash revision_materials/scripts/phase3b_install_cron.sh remove
```

## Telegram Notes

Create `revision_materials/scripts/telegram.conf`:

```bash
TELEGRAM_TOKEN=...
CHAT_ID=...
```

Start watcher:

```bash
tmux new-session -d -s phase3b_telegram
tmux send-keys -t phase3b_telegram \
  'cd /root/OthorAdapt && bash revision_materials/scripts/phase3b_notes_telegram.sh watch >> revision_materials/logs/phase3b_notes_telegram.log 2>&1' C-m
```

## Notes

- CLIP-LoRA commands intentionally contain `--r 2` only. They must not contain
  `--num_heads` or `--lambda_o`.
- OrthoAdapt commands intentionally contain `--num_heads 2 --r 2 --lambda_o 0.03 --ramp_up_steps 100`.
- Keep this manifest separate from `phase3_main_ramp100_results.jsonl`.
