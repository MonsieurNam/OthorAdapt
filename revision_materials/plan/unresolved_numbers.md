# Unresolved Numbers

Last updated: 2026-06-16

This file lists manuscript numbers and claims that cannot currently be treated as verified. After the raw-evidence update, logs and checkpoints are no longer globally missing: 724 raw logs, 5 CSV summaries, 20 zip archives, 153 loadable checkpoint files, 1 workbook, and 11 figure/source assets are recorded in `phase0_raw_artifact_manifest.csv`.

The remaining blocker is evidence quality, not evidence absence: the recovered artifacts have now been normalized into Phase 0 evidence manifests, but they still lack explicit split hashes, seed2/seed3 provenance, paired robustness manifests, and final validation/test table-generation scripts.

## Critical Unresolved Numbers

| ID | Number / claim | Location | Current source | Status | Required resolution |
|---|---|---|---|---|---|
| UNR-001 | `Training Iterations = 500` | `tab:hyperparams`, LaTeX line 454 | Manuscript table; source code uses shot-scaled total iterations | SOURCE-CODE_INSPECTION_REQUIRED | Rewrite as 500 base steps per shot: 500/2000/8000 total steps for 1/4/16-shot. |
| UNR-002 | `Random Seeds = 1` | `tab:hyperparams`, LaTeX line 456 | Manuscript table; recovered checkpoints are path-derived `seed1`; logs do not expose parseable seed IDs in filenames | UNVERIFIED_RERUN_REQUIRED | Replace with seeds `{1,2,3}` only after Tier-A rerun or recovery of seed2/seed3 artifacts; do not claim current recovered evidence supports 3-seed averages. |
| UNR-003 | 1-shot avg `73.48` OrthoAdapt vs `73.26` CLIP-LoRA | `tab:1` / rendered Table 2 | Workbook-derived value; recovered 1-shot logs; some seed1 checkpoints recovered | UNVERIFIED_RERUN_REQUIRED | Normalize matching logs into manifest; re-evaluate checkpoints where needed; rerun/recover seeds 2-3; generate mean +/- std and paired statistics. |
| UNR-004 | 4-shot avg `79.96` OrthoAdapt vs `79.53` CLIP-LoRA | `tab:2` / rendered Table 3 | Workbook-derived value; recovered 4-shot logs/CSVs; some seed1 checkpoints recovered | UNVERIFIED_RERUN_REQUIRED | Normalize logs and CSVs; final table must use validation-selected config only and include 3-seed manifest. |
| UNR-005 | 4-shot EuroSAT `89.06` vs `87.49`, gain `+1.57` | `tab:2`, Figure combined chart, text line 524 | Workbook-derived value, rendered figure, recovered EuroSAT logs, and EuroSAT seed1 checkpoints | UNVERIFIED_RERUN_REQUIRED | Map exact log/checkpoint/config; verify split and seed; report statistical uncertainty or rerun. |
| UNR-006 | 16-shot avg `84.84` OrthoAdapt vs `84.52` CLIP-LoRA | `tab:3` / rendered Table 4 | Workbook-derived value; many 16-shot logs and seed1 checkpoints recovered | UNVERIFIED_RERUN_REQUIRED | Normalize recovered 16-shot logs; re-evaluate checkpoints where needed; rerun/recover seeds 2-3 before claiming mean/variance. |
| UNR-007 | Aircraft 16-shot `54.97` for both CLIP-LoRA and OrthoAdapt | `tab:3`, LaTeX lines 570 and 572 | Workbook/manuscript table; no aircraft checkpoint or log path observed in recovered manifest | UNVERIFIED_RERUN_REQUIRED | Recover/rerun Aircraft checkpoint/log; if identical, footnote; if copy error, correct. |
| UNR-008 | `H=2` best / `H=4` rank fragmentation | Figure 5a, `tab:4`, text lines 626-628 | Workbook-derived values, rendered figure, recovered head-scan logs/CSV, and some ablation checkpoints | UNVERIFIED_RERUN_REQUIRED | Normalize ablation evidence; move selection to validation split; add diagnostics before causal rank-fragmentation wording. |
| UNR-009 | EuroSAT lambda values `88.07`, `88.64`, `88.57`, `88.59`, `88.52` | `tab:5`, text line 633 | Workbook-derived sensitivity sheet plus recovered ablation logs/CSVs and some checkpoints | UNVERIFIED_RERUN_REQUIRED | Reconcile exact configs across table/figure/workbook/logs; rerun validation-only sweep as needed; fix decimal commas and labels. |
| UNR-010 | Robustness gain `+3.6%` / `46.3% vs 42.7%` | Robustness text line 644; Figure 6 | Rendered figure and checkpoint evidence, but no robustness log/corruption manifest | UNVERIFIED_RERUN_REQUIRED | Repair paired corruption protocol, re-evaluate from recovered checkpoints where possible, and reconcile medium vs severe wording. |
| UNR-011 | OxfordPets robustness gain `+1.6%` | Robustness text line 644 | Rendered figure and OxfordPets checkpoint evidence, but no robustness log/corruption manifest | UNVERIFIED_RERUN_REQUIRED | Same robustness re-evaluation; report exact severity and CI if possible. |
| UNR-012 | Spectrum singular values / flatter tail above `10^-1` | Spectral text lines 649-655; Figure 7 | Recovered spectrum PNG/drawio assets and checkpoint evidence | UNVERIFIED_RERUN_REQUIRED | Regenerate from exact selected checkpoint; record checkpoint hash, matrix definition, matrix shape/rank, layer, module, and SVD script version. |
| UNR-013 | Gating specialization: Highway vs Forest heads | Figure 8 and text line 670 | Recovered gating figure/drawio assets and checkpoint evidence | UNVERIFIED_RERUN_REQUIRED | Regenerate with selected checkpoint hash, sample manifest, class labels, layer/head metadata, and gating-weight export. |
| UNR-014 | Parameter counts such as `184320`, `276480`, `368640` | Workbook sheets and manuscript claims | Workbook-derived value, recovered CSV `params` columns, and source code | SOURCE-CODE_INSPECTION_REQUIRED | Recompute parameter counts including gating network and log in manifests. |
| UNR-015 | "11 state-of-the-art baselines" wording | Figure combined caption line 487 | Manuscript caption | SOURCE-CODE_INSPECTION_REQUIRED | Verify baseline count and remove SOTA wording unless evidence and citations support it. |
| UNR-016 | "significant" improvement wording | Abstract/introduction/results | Manuscript prose | UNVERIFIED_RERUN_REQUIRED | Use only after paired statistics support it; otherwise replace with "modest" or "observed". |
| UNR-017 | "orthogonality causes robustness" | Robustness and spectral sections | Manuscript prose | UNVERIFIED_RERUN_REQUIRED | Require orthogonality ablation under paired corruptions; otherwise weaken to correlation/possible explanation. |
| UNR-018 | PSD-escape / "overcomes PSD bottleneck" framing | Figure 2 and Method | Manuscript prose | SOURCE-CODE_INSPECTION_REQUIRED | Rewrite; softmax mixtures of PSD heads remain PSD. |
| UNR-019 | Missing ImageNet/SUN397/StanfordCars results | Experimental setup and reviewer R2-5d | Dataset loaders only; no recovered logs/checkpoints for these datasets | UNVERIFIED_RERUN_REQUIRED | Execute pilot gate; run if feasible or state limitation. |

## Recovered But Still Insufficient

| Recovered item | Why it helps | Why it is not enough |
|---|---|---|
| 724 raw `.log` files | Many logs expose zero-shot accuracy, training traces, final test accuracy, runtime, and saved-checkpoint lines; they are normalized into `main_results_manifest.*`. | Log filenames do not encode seeds; split hashes and validation/test selection protocol are absent. |
| `main_results_manifest.jsonl` / `.csv` | Canonical Phase 0 recovered-evidence manifest with 876 log/CSV records. | It is not a final rerun manifest and cannot establish 3-seed means or validation-only selection. |
| `missing_tier_a_matrix.md` / `.csv` | Shows all 72 Tier-A dataset/shot/method cells have some recovered evidence. | Only 8 OrthoAdapt cells have strict R2/H2 evidence; all cells still require seed2/seed3 and split provenance. |
| `workbook_vs_logs_crosscheck.md` / `.csv` | Identifies workbook/log agreement and reconciliation targets. | 169 mismatches and 36 no-log workbook rows must be resolved before table values are reused. |
| 5 summary CSV files | Preserve scan/ablation rows with accuracy, log filename, parameters, and sometimes checkpoint path. | They are summary artifacts; they do not prove 3-seed protocol or validation-only selection by themselves. |
| 153 `adapter_weights.pt` checkpoints | Enables re-evaluation and some diagnostics without retraining seed1 from scratch. | All path-derived checkpoint seeds are `seed1`; checkpoint metadata does not include accuracy, split, command line, or final logs. |
| 20 zip archives | Preserve compressed evidence bundles from result directories. | Archive contents are not separately expanded/normalized in Phase 0; use only after extraction and hashing if needed. |
| `data/clip_fewshot_results.xlsx` | Preserves table-level numerical values and log filename references. | It is a derived workbook; values must be cross-checked against raw logs/CSV and regenerated manifests. |
| Spectrum/gating/ablation/corruption figure assets | Preserves figure provenance and source/rendered artifacts. | They do not verify the numerical pipeline or exact checkpoint/sample provenance. |
| `phase0_raw_artifact_manifest.csv` | Provides SHA256 and metadata ledger for every recovered `data/` artifact. | It is an inventory, not an evaluation manifest. |

## Files Still Missing Or Not Yet Canonical

The following are still needed outside this Phase 0 inventory:

- Final rerun/evaluation manifest with seed IDs, split hashes, and validation/test role for every accepted table value.
- `validation_sweep_manifest.jsonl` with validation-only selection, not test-set selection.
- `split_manifest.json` or CSV with train/val/test counts, class lists, and hashes.
- Seed2/seed3 raw logs/checkpoints or fresh reruns for Tier-A claims.
- Exact checkpoint-log-workbook mapping for every value that remains in Tables 2-4 and ablations.
- Paired corruption sample/cache manifest for robustness.
- Spectrum and gating export manifests with checkpoint hash, layer/module, matrix definition, sample IDs, and script version.

## Phase 0 Gate Conclusion

The Phase 0 tracking files and normalization outputs are complete, and raw logs/checkpoint evidence has been recovered. Acceptance-critical numerical claims remain `UNVERIFIED_RERUN_REQUIRED` until final evaluation manifests with seed/split provenance are regenerated and missing seeds/configurations are rerun or recovered.
