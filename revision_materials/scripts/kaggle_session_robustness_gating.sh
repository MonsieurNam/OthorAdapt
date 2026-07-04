#!/usr/bin/env bash
# =============================================================================
# Kaggle session: fixed Phase 4 robustness rerun + regenerate gating figure
# =============================================================================
# One GPU session. The previous robustness manifest is retained only as an audit
# failure because severity-0 OH-SingLoRA did not reproduce Phase 3 clean test
# accuracy. This script writes to a fresh fixed manifest by default.
#
# Usage (Kaggle notebook cell):
#   !cd /kaggle/working/OthorAdapt && bash revision_materials/scripts/kaggle_session_robustness_gating.sh
#
# Prerequisites in /kaggle/working/OthorAdapt:
#   - repo synced (including revision_materials/scripts/ and this script)
#   - optional: revision_materials/results/phase4_robustness_manifest_fixed.jsonl
#     if resuming a fixed rerun. If absent, all 144 eval jobs are run.
#   - revision_materials/checkpoints/phase3_main_ramp100/  (both vitb16/ lora
#     and ohsinglora/ trees; 338 MB total)
#   - dataset input attached (data-image-classification-collection)
#   - raw EuroSAT class folders for the gating figure (eurosat/2750/Forest,
#     eurosat/2750/Highway) — usually inside the same Kaggle dataset input.
# =============================================================================
set -euo pipefail

REPO=${REPO:-/kaggle/working/OthorAdapt}
cd "$REPO"

PYTHON=${PYTHON:-python3}
export PYTHON
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
export RUN_STAMP

# --- locate dataset root (same auto-detect logic as phase4_robustness_commands.sh)
DATA_ROOT=${DATA_ROOT:-/root/DATA}
if [ ! -f "${DATA_ROOT}/Food101/split_zhou_Food101.json" ]; then
  for candidate in \
    /kaggle/input/datasets/nguyenngonhatnam/data-image-classification-collection \
    /kaggle/input/*/data-image-classification-collection \
    /kaggle/input/*; do
    if [ -f "${candidate}/Food101/split_zhou_Food101.json" ]; then
      DATA_ROOT="${candidate}"
      break
    fi
  done
fi
if [ ! -f "${DATA_ROOT}/Food101/split_zhou_Food101.json" ]; then
  echo "FATAL: DATA_ROOT not found (need Food101/split_zhou_Food101.json under it)" >&2
  exit 2
fi
export DATA_ROOT
echo "== DATA_ROOT=${DATA_ROOT}"

MANIFEST=${ROBUST_MANIFEST:-revision_materials/results/phase4_robustness_manifest_fixed.jsonl}
LOGDIR=${ROBUST_LOG_DIR:-revision_materials/logs/phase4_robustness_fixed}
mkdir -p "$LOGDIR"
export ROBUST_MANIFEST="$MANIFEST"
export ROBUST_LOG_DIR="$LOGDIR"

# =============================================================================
# STEP 0 — Snapshot the incoming manifest so nothing can be lost this session.
# =============================================================================
if [ -f "$MANIFEST" ]; then
  cp "$MANIFEST" "${MANIFEST%.jsonl}.session_start_${RUN_STAMP}.jsonl"
  echo "== manifest snapshot: ${MANIFEST%.jsonl}.session_start_${RUN_STAMP}.jsonl ($(wc -l < "$MANIFEST") rows)"
else
  echo "WARNING: no local manifest found — every job will be (re)run." >&2
fi

# =============================================================================
# STEP 1 — Generate the resume script from the MANIFEST ONLY.
# CRITICAL: --log-dir must point at a nonexistent directory. All 144 job logs
# already exist locally from earlier sessions, and phase4_make_resume_commands.py
# treats a finished log as "job complete" — a default resume would skip
# EVERYTHING and run nothing.
# =============================================================================
echo "== generating manifest-only resume script"
$PYTHON revision_materials/scripts/phase4_make_resume_commands.py \
    --manifest "$MANIFEST" \
    --log-dir /nonexistent_dir_ignore_logs \
    --out revision_materials/scripts/phase4_robustness_resume_${RUN_STAMP}.sh

PENDING=$(grep -c -- "--filename" revision_materials/scripts/phase4_robustness_resume_${RUN_STAMP}.sh || true)
echo "== pending jobs: ${PENDING} (expect 144 for a fresh fixed rerun)"
if [ "$PENDING" -eq 0 ]; then
  echo "== nothing pending; skipping to gating figure"
else
  # ===========================================================================
  # STEP 2 — Run the pending robustness evaluations (eval-only, no training).
  # Each job appends 4 rows (severities 0-3) to the manifest. ~4-10 min/job.
  # food101 is the slowest — do not be alarmed by long food101 jobs.
  # ===========================================================================
  echo "== running robustness resume (${PENDING} jobs)"
  bash revision_materials/scripts/phase4_robustness_resume_${RUN_STAMP}.sh
fi

# =============================================================================
# STEP 3 — Coverage check (dry-run merge tool doubles as a coverage report).
# =============================================================================
echo "== coverage after run:"
$PYTHON revision_materials/scripts/phase4_merge_robustness_manifests.py \
    --into "$MANIFEST" --dry-run | grep -E "^(----|rows|complete|missing)" || true

# =============================================================================
# STEP 4 — Aggregate (hard-fails unless exactly 576 unique completed rows).
# Writes phase4_robustness_summary.csv + phase4_robustness_report.md.
# =============================================================================
if $PYTHON revision_materials/scripts/phase4_aggregate_robustness.py --manifest "$MANIFEST"; then
  echo "== robustness aggregation COMPLETE"
else
  echo "== robustness aggregation still blocked (incomplete grid) — see coverage above" >&2
fi

# =============================================================================
# STEP 5 — Regenerate the gating specialization figure (UNR-013) with the
# FINAL selected configuration: H=2, r=8, lambda_o=0.03, ramp100, seed 1.
# The legacy figure in the paper is a rank-2 artifact and fails provenance.
# NOTE: visualize_gating.py defaults to --r 2. Passing --r 8 is REQUIRED.
# =============================================================================
GATING_CKPT=revision_materials/checkpoints/phase3_main_ramp100/ohsinglora/ViT-B16/eurosat/4shots/seed1/eurosat_4shot_seed1_test_ohsinglora_h2_r8_lo0p03_ramp100.pt

# locate raw EuroSAT class folders (Forest/, Highway/ under .../2750 or EuroSAT/2750)
EUROSAT_RAW=""
for candidate in \
  "${DATA_ROOT}/eurosat/2750" \
  "${DATA_ROOT}/EuroSAT/2750" \
  "${DATA_ROOT}/eurosat/EuroSAT/2750" \
  /kaggle/input/*/2750; do
  if [ -d "${candidate}/Forest" ] && [ -d "${candidate}/Highway" ]; then
    EUROSAT_RAW="${candidate}"
    break
  fi
done

if [ ! -f "$GATING_CKPT" ]; then
  echo "SKIP gating figure: checkpoint not found: $GATING_CKPT" >&2
elif [ -z "$EUROSAT_RAW" ]; then
  echo "SKIP gating figure: EuroSAT raw class folders (2750/Forest, 2750/Highway) not found" >&2
else
  echo "== regenerating gating figure from ${GATING_CKPT}"
  OUTDIR=revision_materials/results/figures/gating_h2_r8_ramp100
  mkdir -p "$OUTDIR"

  # ---- provenance BEFORE running: checkpoint hash + exact sample list
  # (visualize_gating.py takes the first 20 files from os.listdir per class;
  #  we record the same list, sorted flag noted, per UNR-013 requirements)
  $PYTHON - "$GATING_CKPT" "$EUROSAT_RAW" "$OUTDIR" <<'PYEOF'
import hashlib, json, os, sys
ckpt, raw, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
h = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
def sample(cls):
    d = os.path.join(raw, cls)
    files = [f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return files[:20]  # replicates load_images_from_folder(num_imgs=20)
prov = {
    "purpose": "UNR-013 gating specialization figure provenance",
    "checkpoint": ckpt,
    "checkpoint_sha256": h,
    "config": {"adapter": "ohsinglora", "num_heads": 2, "r": 8, "lambda_o": 0.03,
               "ramp_up_steps": 100, "backbone": "ViT-B/16", "dataset": "eurosat",
               "shots": 4, "seed": 1, "layer_idx": 11, "projection": "q_proj.gating_network"},
    "classes": {"Forest": sample("Forest"), "Highway": sample("Highway")},
    "num_imgs_per_class": 20,
    "sampling_rule": "first 20 image files from os.listdir order (visualize_gating.load_images_from_folder)",
}
path = os.path.join(outdir, "gating_provenance.json")
json.dump(prov, open(path, "w"), indent=2)
print("provenance written:", path)
PYEOF

  # ---- run the visualization (cuda hard-coded in script; fine on Kaggle GPU)
  $PYTHON visualize_gating.py \
      --checkpoint "$GATING_CKPT" \
      --data_root "$EUROSAT_RAW" \
      --class_a Forest --class_b Highway \
      --num_imgs 20 --layer_idx 11 \
      --r 8 --num_heads 2 --alpha 1 \
      2>&1 | tee "$LOGDIR/gating_regen_${RUN_STAMP}.log"

  # outputs land in CWD with fixed names; move + rename with full metadata
  mv gating_map_visualization.png "$OUTDIR/gating_map_layer11_eurosat_h2_r8_lo0p03_ramp100_seed1.png"
  mv head_specialization.png     "$OUTDIR/head_specialization_layer11_eurosat_h2_r8_lo0p03_ramp100_seed1.png"
  echo "== gating figures written to $OUTDIR"
fi

# =============================================================================
# STEP 6 — Package EVERYTHING for download. The manifest goes home as a
# separate timestamped file so the local merge is explicit (merge, never
# overwrite — the Jul-4 overwrite is why robustness was unpaired).
# =============================================================================
OUT_ZIP=/kaggle/working/phase4_session_${RUN_STAMP}.zip
cp "$MANIFEST" "/kaggle/working/phase4_robustness_manifest_FIXED_SERVER_${RUN_STAMP}.jsonl"
zip -r "$OUT_ZIP" \
    "/kaggle/working/phase4_robustness_manifest_FIXED_SERVER_${RUN_STAMP}.jsonl" \
    revision_materials/results/phase4_robustness_summary.csv \
    revision_materials/results/phase4_robustness_report.md \
    revision_materials/results/figures/gating_h2_r8_ramp100 \
    "$LOGDIR" \
    2>/dev/null || true
echo "== session package: $OUT_ZIP"
echo "== ALSO download the standalone manifest copy:"
echo "   /kaggle/working/phase4_robustness_manifest_FIXED_SERVER_${RUN_STAMP}.jsonl"
echo ""
echo "== NEXT (on the local machine): merge, never overwrite =="
echo "   python revision_materials/scripts/phase4_merge_robustness_manifests.py \\"
echo "       --into revision_materials/results/phase4_robustness_manifest_fixed.jsonl \\"
echo "       --from-file <downloaded phase4_robustness_manifest_FIXED_SERVER_${RUN_STAMP}.jsonl>"
echo "   python revision_materials/scripts/phase4_aggregate_robustness.py --manifest revision_materials/results/phase4_robustness_manifest_fixed.jsonl"
