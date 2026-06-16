# Artifact Inventory

Last updated: 2026-06-16

Purpose: record every recovered artifact that can support the revision and explicitly record remaining provenance gaps. This is the completed Phase 0 inventory pass after the checkpoint/raw-evidence update.

## Scan Summary

Searched workspace root `D:\RESEARCH\CLIP-LoRA_group\OthorAdapt` for:

- checkpoints: `.pt`, `.pth`, `.ckpt`, `.zip`
- logs and manifests: `.log`, `.out`, `.err`, `.json`, `.jsonl`, `.csv`, `.tsv`, `.yaml`, `.yml`
- notebooks: `.ipynb`
- result workbooks: `.xlsx`
- figure/source assets: `.png`, `.pdf`, `.drawio`
- paper sources, reviewer material, rendered figures, and code scripts

Recovered evidence now includes raw logs, CSV summaries, checkpoint archives, loadable checkpoints, one result workbook, and figure/source assets under `data/`. The machine-readable hash and metadata manifest is:

| Manifest | Rows | Coverage | Notes |
|---|---:|---|---|
| `revision_materials/plan/phase0_raw_artifact_manifest.csv` | 914 | All files currently found under `data/` with extensions `.pt`, `.zip`, `.xlsx`, `.png`, `.pdf`, `.drawio`, `.log`, `.csv` | Contains SHA256, size, timestamp, path-derived dataset/shot/seed/config hints, checkpoint metadata, log final-accuracy extraction, zero-shot accuracy extraction, and CSV headers where available. |

## Recovered Data Artifact Summary

| Artifact group | Count | Type(s) | Status | Evidence value | Remaining limitation |
|---|---:|---|---|---|---|
| `data/old_result/` | 286 logs + 1 zip | `.log`, `.zip` | Recovered raw/archived evidence | Historical run logs; 275 logs expose final test accuracy; includes LoRA, SingLoRA, GMHSingLoRA, and OHSingLoRA-style runs. | Log paths do not expose seed IDs; normalized into Phase 0 evidence manifests but not final verified rerun manifests. |
| `data/result_scan_loss/` | 232 logs + 1 CSV + 6 zips | `.log`, `.csv`, `.zip` | Recovered scan evidence | Loss-function scans; summary CSV has 54 rows and log filenames. | Loss scans are diagnostic, not final main-table evidence without validation/test protocol. |
| `data/result_scan_head/` | 104 logs + 1 CSV + 4 zips | `.log`, `.csv`, `.zip` | Recovered scan evidence | Head/rank scan logs; 103 logs expose final test accuracy; CSV has 32 rows with checkpoint paths. | Needs validation-only selection and exact seed/split metadata. |
| `data/results_ablation_heads_lambda/` | 66 logs + 3 CSVs | `.log`, `.csv` | Recovered ablation evidence | Ablation logs and summaries for 1/4/16-shot settings; all 66 logs expose final test accuracy. | Current evidence still appears sweep/test-result oriented; no split hashes or independent selection manifest. |
| `data/result_new/` | 36 logs + 8 zips | `.log`, `.zip` | Recovered main-run evidence | Main run logs for 8 datasets across 1/4/16-shot; all 36 logs expose final test accuracy. | Log path names do not encode seed; only saved checkpoint lines observed as `seed1`. |
| `data/checkpoints_16shot/` | 104 | `.pt` | Recovered and loadable | Seed-1 adapter checkpoints for 16-shot and nested cross-dataset paths; metadata includes adapter/r/alpha/params/backbone for most files. | No accuracy, command line, split, or seed-2/seed-3 provenance in checkpoint metadata. |
| `data/checkpoints_rank2/` | 19 `.pt` + 1 `.zip` | `.pt`, `.zip` | Recovered and loadable where `.pt` | Rank-2 OrthoAdapt/LoRA-style checkpoints for selected datasets/shots; zip hash recorded. | Mostly seed1; needs log/checkpoint pairing before use as table evidence. |
| `data/checkpoint_rank16/` | 16 | `.pt` | Recovered and loadable | Rank-16 checkpoints for selected datasets/shots. | Partial diagnostic coverage; not directly aligned with stated rank-2 main protocol. |
| `data/checkpoint_rank4/` | 8 | `.pt` | Recovered and loadable | Rank-4 checkpoints for selected datasets/shots. | Partial diagnostic coverage only. |
| `data/checkpoints_ablation_heads_lambda/` | 6 | `.pt` | Recovered and loadable | Checkpoints for EuroSAT/Caltech101 head/lambda ablation settings. | Does not prove validation-only selection; no split hash in metadata. |
| `data/clip_fewshot_results.xlsx` | 1 | `.xlsx` | Recovered derived numerical artifact | Contains main-table, ablation, sensitivity, loss, and baseline workbook sheets. | Derived workbook; must be reconciled against raw logs/CSV and regenerated manifests. |
| `data/*.png`, `data/*.pdf`, `data/*.drawio` | 11 | figure/source assets | Recovered | Figure source/rendered assets for ablation, spectrum, overview, dataset samples, corruption, and gating/comparison diagrams. | Rendered/source figures do not by themselves verify underlying numerical claims. |

## Raw Log and CSV Summary

| Property | Observed value |
|---|---|
| Total recovered `.log` files | 724 |
| Logs with parsed final test accuracy | 676 |
| Logs with parsed zero-shot CLIP accuracy | 714 |
| Log dataset hints | EuroSAT 122, Caltech101 113, DTD 91, FGVC 90, Food101 77, Oxford Flowers 77, Oxford Pets 77, UCF101 77 |
| Log method hints | OHSingLoRA 346, LoRA 156, GMHSingLoRA 128, SingLoRA 94 |
| Log shot hints | 1-shot 326, 4-shot 250, 16-shot 112, unparsed 36 |
| Parsed log seed hints | unparsed in log filenames: 724 |
| Recovered CSV summaries | 5 CSV files, 152 total data rows |
| CSV headers | `dataset,shots,adapter,rank,heads,lambda_o,params,time(s),accuracy,log_file,checkpoint_path`; `loss,dataset,shots,accuracy,log_file`; ablation CSVs without checkpoint path |

## Checkpoint Metadata Summary

All 153 `adapter_weights.pt` files were load-tested with PyTorch CPU loading. Every file loaded successfully and had the top-level keys `weights` and `metadata`.

| Checkpoint property | Observed value |
|---|---|
| Total loadable `.pt` checkpoints | 153 |
| Path-derived seeds | `seed1` only: 153 files |
| Path-derived shot coverage | `1shot`: 11, `4shot`: 22, `16shot`: 120 |
| Path-derived dataset coverage | EuroSAT 38, Caltech101 34, DTD 25, Oxford Pets 14, FGVC 14, Food101 12, Oxford Flowers 10, UCF101 6 |
| Path-derived method hints | `ohsinglora`: 138, `vitb16`: 15 |
| Metadata key sets | 135 files: `adapter;alpha;backbone;encoder;params;position;r`; 15 files: `alpha;encoder;params;position;r`; 3 files: `adapter;alpha;expansion_factor;params;r;r_res;vera_rank` |
| Checkpoint payload | `weights` dict plus `metadata` dict; no accuracy/test split/validation split/log-file payload observed in manifest-extracted metadata |

## Recovered Evidence Artifacts

| Artifact ID | Path | Type | SHA256 / hash source | Role | Provenance status | Notes |
|---|---|---|---|---|---|---|
| ART-RAW-MANIFEST | `revision_materials/plan/phase0_raw_artifact_manifest.csv` | CSV manifest | Generated Phase 0 manifest | Hash ledger for all recovered `data/` artifacts | Complete supporting inventory manifest | Use this file for per-artifact SHA256 values instead of duplicating 914 rows in this Markdown file. |
| ART-NORMALIZE-SCRIPT | `revision_materials/scripts/phase0_normalize_artifacts.py` | Python script | Local script; rerun verified on 2026-06-16 | Re-runnable Phase 0 normalization script | Complete supporting script | Regenerates all files in `revision_materials/results/` listed below from the raw artifact manifest and workbook. |
| ART-MAIN-RESULTS-MANIFEST | `revision_materials/results/main_results_manifest.jsonl`; `revision_materials/results/main_results_manifest.csv` | JSONL/CSV evidence manifest | Generated by `phase0_normalize_artifacts.py` | Canonical normalized recovered log/CSV evidence | Complete recovered-evidence manifest | 876 records: 724 log records and 152 CSV-summary records; seed/split still unresolved. |
| ART-TIER-A-MATRIX | `revision_materials/results/missing_tier_a_matrix.md`; `revision_materials/results/missing_tier_a_matrix.csv` | MD/CSV coverage matrix | Generated by `phase0_normalize_artifacts.py` | Tier-A dataset/shot/method coverage report | Complete recovered-evidence coverage matrix | 72 cells; no cell is completely missing recovered evidence, but only 8 OrthoAdapt cells have strict R2/H2 evidence and all cells still lack seed2/seed3/split hashes. |
| ART-WORKBOOK-XCHECK | `revision_materials/results/workbook_vs_logs_crosscheck.md`; `revision_materials/results/workbook_vs_logs_crosscheck.csv` | MD/CSV cross-check | Generated by `phase0_normalize_artifacts.py` | Workbook-vs-log reconciliation | Complete recovered-evidence cross-check | 782 workbook rows checked: 577 accuracy matches, 169 mismatches, 36 no matching log. |
| ART-NORMALIZATION-SUMMARY | `revision_materials/results/phase0_normalization_summary.md` | Markdown summary | Generated by `phase0_normalize_artifacts.py` | Normalization status summary | Complete supporting summary | Records input counts and generated-output counts. |
| ART-RAW-LOGS | `data/old_result/`, `data/result_new/`, `data/result_scan_head/`, `data/result_scan_loss/`, `data/results_ablation_heads_lambda/` | Run logs | See `phase0_raw_artifact_manifest.csv` and `main_results_manifest.*` | Raw final-accuracy and training-log evidence | Recovered and normalized into Phase 0 evidence manifests | Normalization does not resolve seed IDs, split hashes, validation-only selection, or seed2/seed3 coverage. |
| ART-CSV-SUMMARIES | `data/**/summary*.csv` | CSV summaries | See `phase0_raw_artifact_manifest.csv` | Run-summary evidence for scans/ablations | Recovered derived/summary evidence | Useful bridge from logs to tables; not a replacement for split/seed manifest. |
| ART-WORKBOOK-RESULTS | `data/clip_fewshot_results.xlsx` | Result workbook | `9AD16895CE4C8EE449581CD62B7F4FB11484AE0BEA1E191B49E117801C770B21` | Main tables, ablations, sensitivity data | Recovered derived numerical artifact | Must be cross-checked against raw logs/CSV and regenerated manifests. |
| ART-CKPT-ALL | `data/checkpoints_*`, `data/checkpoint_rank*` | Checkpoints | See `phase0_raw_artifact_manifest.csv` | Adapter checkpoints for selected datasets/shots/ranks/heads | Recovered checkpoint evidence | Useful for re-evaluation and diagnostics; not sufficient alone to verify accuracy claims. |
| ART-FIG-DATA-ASSETS | `data/*.png`, `data/*.pdf`, `data/*.drawio` | Figure assets | See `phase0_raw_artifact_manifest.csv` | Source/rendered figure artifacts | Recovered figure evidence | Supports figure provenance, not numerical verification without linked manifests/logs. |
| ART-MANUSCRIPT-TEX | `revision_materials/Latex_code/cas-sc-template.tex` | LaTeX source | `D95DD1F4162687A2EF6A4806A6EF92DAF0DFF35FD329B46766819EC05EBB9CC6` | Current manuscript source | Recovered source | Contains current table/figure labels and claims; not raw experimental evidence. |
| ART-PAPER-MD | `revision_codex/.analysis/paper.md` | Extracted paper text | `1B2B446F28F274ACC0F117D519884260BFBAEF9F4713E19BB0E9C446D9685A04` | Text extraction of submitted paper | Recovered derived text | Useful for claim search only. |
| ART-REVIEWER-MD | `revision_codex/.analysis/reviewer_respond.md` | Extracted reviewer text | `74E4933CF1C150ED64FCA80502D25FA7DE9BF0F589FA5DAE6C68F77AB394789F` | Reviewer comments | Recovered derived text | Used to map reviewer issues to paper artifacts. |
| ART-SCRIPT-MAIN | `main.py` | Code | `60D7E67B2BBC7A630CF5B04672DAFA117E5A3836F740DCF020C59F14A7FD36CA` | Entry point | Source-code inspection | Needs validation/test split repair before reruns. |
| ART-SCRIPT-LORA | `lora.py` | Code | `3B5653389C7C7F9A83B5CB4CC393840566CA442D3203297E67D8BA54F88A8287` | Adapter training/evaluation logic | Source-code inspection | Uses shot-scaled iteration semantics; needs split controls. |
| ART-SCRIPT-RUN-UTILS | `run_utils.py` | Code | `51E944976F44691B3A403F9A223845D792AAD3CAEA1C9BD148C8F377C2AD226A` | Dataset/run helper logic | Source-code inspection | Needs split count/hash recording. |
| ART-SCRIPT-ROBUSTNESS | `eval_robustness.py` | Code | `72CDD9BCBD67DA11D0F7CDD430D4A7E8E81DD7B8412F632D934C817F2632D23D` | Robustness evaluation | Source-code inspection | Needs deterministic paired corruption cache. |
| ART-SCRIPT-SPECTRUM | `analyze_spectrum.py` | Code | `0A17A823F91D0B8338136ED328F21D15A9C1EDB5F8E1DB259CF101B0853D7FF3` | Spectrum plotting | Source-code inspection | Needs fail-closed behavior and checkpoint metadata. |
| ART-SCRIPT-OH-LAYER | `loralib/layers_OH_singlora.py` | Code | `6032DF9F531133773A47B7A3CA157881C460F4F9710586524AE6AADB2ACFDE4E` | OrthoAdapt adapter layer | Source-code inspection | Needs orthogonality reduction normalization. |

## Workbook Sheet Inventory

| Sheet | Non-empty rows | Likely linked manuscript item | Status |
|---|---:|---|---|
| `ohsinglora` | 98 | Main comparison tables and current OrthoAdapt values | Derived artifact; can now be cross-checked against recovered logs/CSVs and re-evaluated checkpoints where coverage exists. |
| `num head scan ohsinglora` | 28 | Head-count ablation table/figure | Derived artifact; related logs, CSV, and some checkpoints recovered. |
| `Sensitivity` | 69 | Lambda/head sensitivity and timing/log filename references | Derived artifact; related ablation/scan logs recovered and normalized; remaining mismatches must be reconciled. |
| `compare loss ohsinglora` | 28 | CLIP-LoRA / loss comparison values | Derived artifact; related loss-scan logs and CSV recovered. |
| `Dynamic Gating loss` | 76 | Gating loss scans | Derived artifact; not directly mapped to main manuscript yet. |
| `scaning rank and alpha` | 42 | Rank/alpha scans and SingLoRA/Gated variants | Derived artifact; some related checkpoints/logs recovered. |
| `qlora` | 29 | Baseline comparison values | Derived artifact; raw evidence coverage still needs mapping. |
| `Singlora` | 43 | SingLoRA comparison values | Derived artifact; SingLoRA logs recovered in `old_result`, but not yet mapped to final tables. |

## Remaining Missing Original Artifacts

| Missing artifact type | Expected role | Current status | Consequence |
|---|---|---|---|
| Final rerun/evaluation manifests | Single-source table generation and statistical tests with seed/split provenance | Phase 0 recovered-evidence manifests now exist, but final seed/split-aware evaluation manifests do not | Main tables still need final manifests before claims can be marked verified. |
| YAML/JSON experiment configs | Freeze hyperparameters and split protocol | Not found, except local Claude settings | Selection protocol must still be created before validation sweeps. |
| Notebooks or Colab outputs | Recover original ad hoc analysis | Not found | Notebook-only provenance remains unavailable. |
| Split hashes / seed manifests | Validate reviewer seed/split concerns | Not found | Seed/split protocol still must be recreated and logged. |
| Seed-2 and seed-3 checkpoints/log identifiers | Support three-seed claims and paired statistics | Checkpoint paths are `seed1`; log paths do not parse seed IDs and saved checkpoint lines observed in samples point to `seed1` | Current raw evidence cannot satisfy the revision decision requiring Tier-A seeds `{1,2,3}`. |
| Final checkpoint-log-workbook mapping | Link each manuscript number to raw run and checkpoint | Partially normalized in `workbook_vs_logs_crosscheck.*`; 577 matches, 169 mismatches, 36 no-log workbook rows | Mismatches/no-match rows must be reconciled before replacing/rerunning table values. |
| Paired corruption cache/manifest | Verify robustness claims and severity wording | Not found | Robustness figure must be regenerated with deterministic paired corruptions. |

## Provenance Conclusion

Phase 0 file coverage is complete. The workspace now preserves raw logs, summary CSVs, checkpoints, archives, figures, a full hash manifest, and normalized recovered-evidence manifests. Checkpoint re-evaluation and targeted reruns can start from these artifacts where path coverage matches the manuscript claim.

Gate decision: do not write final response-letter claims as verified results yet. The recovered logs are now normalized into Phase 0 evidence manifests, but acceptance-critical numerical claims still need split hashes, seed identifiers, seed2/seed3 recovery or reruns, and final validation/test manifests according to the revised validation-only protocol.
