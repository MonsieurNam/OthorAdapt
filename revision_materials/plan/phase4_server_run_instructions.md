# Phase 4 Server Run Instructions

These commands assume the project is at `/root/OthorAdapt` and datasets are at `/root/DATA`.

## 1. Already generated locally

These artifacts do not require GPU training/evaluation:

- `revision_materials/plan/phase4_inventory.md`
- `revision_materials/results/phase4_existing_diagnostics.md`
- `revision_materials/results/phase4_checkpoint_diagnostics.jsonl`
- `revision_materials/results/phase4_head_overlap_report.md`
- `revision_materials/results/phase4_orthogonality_report.md`
- `revision_materials/results/phase4_spectrum_manifest.jsonl`
- `revision_materials/results/phase4_spectrum_report.md`
- `revision_materials/results/figures/phase4_spectrum/`

## 2. Run paired robustness

```bash
cd /root/OthorAdapt
export PYTHON=${PYTHON:-python3}
export DATA_ROOT=${DATA_ROOT:-/root/DATA}
export RUN_STAMP=$(date +%Y%m%d_%H%M%S)

bash revision_materials/scripts/phase4_robustness_commands.sh
```

Expected raw output:

- `revision_materials/results/phase4_robustness_manifest.jsonl`
- `revision_materials/logs/phase4_robustness/*.log`

Expected manifest size after completion:

- 144 evaluation jobs
- 576 severity rows, because each job writes severity `0,1,2,3`

Aggregate after the run:

```bash
$PYTHON revision_materials/scripts/phase4_aggregate_robustness.py
```

Expected aggregate output:

- `revision_materials/results/phase4_robustness_summary.csv`
- `revision_materials/results/phase4_robustness_report.md`

## 3. Run ViT-L/14 pilot

Run pilot first:

```bash
cd /root/OthorAdapt
export PYTHON=${PYTHON:-python3}
export DATA_ROOT=${DATA_ROOT:-/root/DATA}
export RUN_STAMP=$(date +%Y%m%d_%H%M%S)

bash revision_materials/scripts/phase4_vitl14_pilot_commands.sh
```

Pilot output:

- `revision_materials/results/phase4_vitl14_pilot_results.jsonl`
- `revision_materials/logs/phase4_vitl14_pilot/*.log`

Continue only if both pilot runs complete cleanly and runtime is acceptable:

```bash
bash revision_materials/scripts/phase4_vitl14_commands.sh
```

Expected subset output:

- `revision_materials/results/phase4_vitl14_results.jsonl`
- `revision_materials/logs/phase4_vitl14/*.log`

If pilot fails or is too slow, keep `revision_materials/results/phase4_backbone_scaling_report.md` as deferred evidence and report ViT-L/14 as future work.

## 4. Claim gate

- Use robustness claims only after `phase4_robustness_report.md` exists.
- Use ViT-L/14 scaling claims only after `phase4_vitl14_results.jsonl` has 12 completed rows and a summary/report is generated.
- Do not use old robustness or spectrum figures from `img/` as regenerated evidence.
