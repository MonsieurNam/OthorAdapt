# Kaggle Session Runbook - Fixed Robustness Rerun + Gating Figure

_Updated 2026-07-04._

## Current State

- The existing `revision_materials/results/phase4_robustness_manifest.jsonl` is structurally complete: 576/576 severity rows and 144/144 jobs.
- It is **not valid for robustness claims** because the clean-consistency audit fails for OH-SingLoRA: severity-0 accuracy does not reproduce the corresponding Phase 3 clean test accuracy.
- Root cause identified locally: the robustness evaluator could instantiate adapter branches not present in the checkpoint, leaving random un-loaded text-side OH-SingLoRA adapters active during evaluation.
- The evaluator has been patched to infer the effective encoder from checkpoint keys and to fail on unexpected adapter keys.
- The fixed rerun must write to a fresh manifest:
  `revision_materials/results/phase4_robustness_manifest_fixed.jsonl`

## What To Run On Kaggle

From `/kaggle/working/OthorAdapt`:

```bash
git pull
export ROBUST_MANIFEST=revision_materials/results/phase4_robustness_manifest_fixed.jsonl
export ROBUST_LOG_DIR=revision_materials/logs/phase4_robustness_fixed
bash revision_materials/scripts/kaggle_session_robustness_gating.sh
```

Expected fresh fixed rerun:

- 144 eval jobs.
- Each job appends 4 severity rows.
- Final fixed manifest should contain 576 rows.
- Aggregation command should be:

```bash
python revision_materials/scripts/phase4_aggregate_robustness.py \
  --manifest revision_materials/results/phase4_robustness_manifest_fixed.jsonl
```

## Required Download Files

Download the session zip and the standalone fixed manifest:

- `/kaggle/working/phase4_session_<stamp>.zip`
- `/kaggle/working/phase4_robustness_manifest_FIXED_SERVER_<stamp>.jsonl`

The zip should contain:

- `phase4_robustness_manifest_FIXED_SERVER_<stamp>.jsonl`
- `revision_materials/results/phase4_robustness_summary.csv`
- `revision_materials/results/phase4_robustness_report.md`
- `revision_materials/results/figures/gating_h2_r8_ramp100/`
- `revision_materials/logs/phase4_robustness_fixed/`

## Local Merge After Download

Merge into the fixed manifest, not the old failed manifest:

```bash
python revision_materials/scripts/phase4_merge_robustness_manifests.py \
  --into revision_materials/results/phase4_robustness_manifest_fixed.jsonl \
  --from-file phase4_robustness_manifest_FIXED_SERVER_<stamp>.jsonl

python revision_materials/scripts/phase4_aggregate_robustness.py \
  --manifest revision_materials/results/phase4_robustness_manifest_fixed.jsonl
```

Then inspect `revision_materials/results/phase4_robustness_report.md`.

## Acceptance Gate

The fixed rerun is usable only if all are true:

- Severity rows: 576/576.
- Paired by `(dataset, shot, seed, severity)`.
- Clean consistency gate: `pass`.
- For both methods, severity-0 accuracy differs from Phase 3 clean accuracy by no more than 1.0 pp.

If the clean gate fails again, do not report robustness numbers. Keep robustness claims removed/deferred.

## Runtime Budget

This is eval-only, no training.

| Scenario | Jobs | Expected time |
|---|---:|---:|
| Fresh fixed rerun | 144 | roughly 8-20 GPU-hours depending on dataset I/O |
| Resume after interruption | remaining jobs only | safe via fixed manifest |
| Gating figure | 1 figure job | about 5 minutes |

