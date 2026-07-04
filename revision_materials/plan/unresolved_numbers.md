# Unresolved Numbers

Last updated: 2026-07-04

This file lists manuscript numbers and claims that cannot currently be treated as verified. After the raw-evidence update, logs and checkpoints are no longer globally missing: 724 raw logs, 5 CSV summaries, 20 zip archives, 153 loadable checkpoint files, 1 workbook, and 11 figure/source assets are recorded in `phase0_raw_artifact_manifest.csv`.

The original Phase 0 blockers were evidence quality rather than evidence absence. Several acceptance-critical items have since been superseded by manifest-backed reruns:

- final validation selection: `validation_sweep_ramp100_results.jsonl` and `selected_config_ramp100.md`
- final main accuracy matrix: `phase3_main_ramp100_results.jsonl`, `statistical_report.md`, and `generated_tables.tex`
- spectrum diagnostics: `phase4_spectrum_manifest.jsonl` and `phase4_spectrum_report.md`
- same-parameter diagnostic: `phase3b_same_param_ramp100_report.md`
- bounded ViT-L/14 subset: `phase4_backbone_scaling_report.md`

Robustness remains unresolved for final claims: `phase4_robustness_manifest.jsonl` is structurally complete with 576/576 severity rows and paired method keys, but `phase4_robustness_report.md` marks the run invalid because severity-0 OH-SingLoRA accuracy does not reproduce the corresponding Phase 3 clean test accuracy. A fixed rerun is required before any robustness number can be reported.

## Critical Unresolved Numbers

| ID | Number / claim | Location | Current source | Status | Required resolution |
|---|---|---|---|---|---|
| UNR-001 | `Training Iterations = 500` | `tab:hyperparams`, LaTeX line 454 | Source-code inspection and final manuscript protocol text | RESOLVED_IN_MANUSCRIPT | Manuscript now describes inherited CLIP-LoRA-style base iterations and shot-scaled total steps rather than an ambiguous single `500` claim. |
| UNR-002 | `Random Seeds = 1` | `tab:hyperparams`, LaTeX line 456 | `phase3_main_ramp100_results.jsonl` | RESOLVED_BY_RERUN | Final main tables use seeds `{1,2,3}` from the 144-row Phase 3 ramp100 manifest. |
| UNR-003 | 1-shot avg `73.48` OrthoAdapt vs `73.26` CLIP-LoRA | `tab:1` / rendered Table 2 | `phase3_main_ramp100_results.jsonl`; `generated_tables.tex` | RESOLVED_BY_RERUN | Final 1-shot table now uses manifest-backed values: CLIP-LoRA 73.23, OrthoAdapt 74.05. |
| UNR-004 | 4-shot avg `79.96` OrthoAdapt vs `79.53` CLIP-LoRA | `tab:2` / rendered Table 3 | `phase3_main_ramp100_results.jsonl`; `generated_tables.tex` | RESOLVED_BY_RERUN | Final 4-shot table now uses manifest-backed values: CLIP-LoRA 79.07, OrthoAdapt 79.47. |
| UNR-005 | 4-shot EuroSAT `89.06` vs `87.49`, gain `+1.57` | `tab:2`, Figure combined chart, text line 524 | `phase3_main_ramp100_results.jsonl`; `generated_tables.tex` | RESOLVED_BY_RERUN | Final 4-shot EuroSAT values are CLIP-LoRA 84.36 and OrthoAdapt 85.78; legacy values are no longer final claims. |
| UNR-006 | 16-shot avg `84.84` OrthoAdapt vs `84.52` CLIP-LoRA | `tab:3` / rendered Table 4 | `phase3_main_ramp100_results.jsonl`; `generated_tables.tex` | RESOLVED_BY_RERUN | Final 16-shot table now uses manifest-backed values: CLIP-LoRA 85.17, OrthoAdapt 85.02. |
| UNR-007 | Aircraft 16-shot `54.97` for both CLIP-LoRA and OrthoAdapt | `tab:3`, LaTeX lines 570 and 572 | `phase3_main_ramp100_results.jsonl`; `generated_tables.tex` | RESOLVED_BY_RERUN | Final Aircraft 16-shot values are CLIP-LoRA 56.99 and OrthoAdapt 56.63. |
| UNR-008 | `H=2` best / `H=4` rank fragmentation | Figure 5a, `tab:4`, text lines 626-628 | `phase4_existing_diagnostics.md/csv`; `w3_headcount_h1_ramp100_results.jsonl`; checkpoint diagnostics | RESOLVED_WITH_WEAKENED_CLAIM | Manuscript uses configuration-dependent capacity/routing wording; no causal rank-fragmentation law is claimed. |
| UNR-009 | EuroSAT lambda values `88.07`, `88.64`, `88.57`, `88.59`, `88.52` | `tab:5`, text line 633 | Validation diagnostics and manuscript sensitivity table | RESOLVED_AS_DIAGNOSTIC | Decimal separators were fixed; retained table is labeled as validation/sensitivity diagnostic and not used for final EuroSAT test claims. |
| UNR-010 | Robustness gain `+3.6%` / `46.3% vs 42.7%` | Robustness text line 644; Figure 6 | `phase4_robustness_manifest.jsonl`; `phase4_robustness_report.md` clean-consistency audit | DOWNGRADED_CLEAN_GATE_FAILURE | Manuscript removes/caveats quantitative robustness claim. Need a fixed paired rerun that passes severity-0 reproduction before claiming robustness. |
| UNR-011 | OxfordPets robustness gain `+1.6%` | Robustness text line 644 | `phase4_robustness_manifest.jsonl` is unpaired | DOWNGRADED_PENDING_PAIRED_AGGREGATION | Same as UNR-010; no final robustness claim allowed. |
| UNR-012 | Spectrum singular values / flatter tail above `10^-1` | Spectral text lines 649-655; Figure 7 | `phase4_spectrum_manifest.jsonl`; `phase4_spectrum_report.md` | RESOLVED_DESCRIPTIVE_ONLY | Spectrum diagnostics were regenerated with checkpoint metadata and are used descriptively, not causally. |
| UNR-013 | Gating specialization: Highway vs Forest heads | Figure 8 and text line 670 | Recovered gating figure/drawio assets and checkpoint evidence | UNVERIFIED_RERUN_REQUIRED | Regenerate with selected checkpoint hash, sample manifest, class labels, layer/head metadata, and gating-weight export. |
| UNR-014 | Parameter counts such as `184320`, `276480`, `368640` | Workbook sheets and manuscript claims | Final manifests and reports | RESOLVED_BY_MANIFEST | Final main comparison reports CLIP-LoRA r=8 as 737,280 parameters and OrthoAdapt H=2,r=8 as 460,800 parameters; same-parameter diagnostic uses 184,320. |
| UNR-015 | "11 state-of-the-art baselines" wording | Figure combined caption line 487 | Manuscript caption | RESOLVED_BY_REWRITE | SOTA/broad baseline wording is removed or scoped; manuscript uses 8-dataset evidence and limitations. |
| UNR-016 | "significant" improvement wording | Abstract/introduction/results | Manuscript prose and `statistical_report.md` | RESOLVED_WITH_MODEST_WORDING | Main claim is framed as a modest average paired delta with uncertainty, not broad significant superiority. |
| UNR-017 | "orthogonality causes robustness" | Robustness and spectral sections | Manuscript prose | DOWNGRADED_PENDING_EVIDENCE | Manuscript explicitly avoids causal robustness claims; paired robustness evidence is still absent. |
| UNR-018 | PSD-escape / "overcomes PSD bottleneck" framing | Figure 2 and Method | Manuscript prose | RESOLVED_BY_REWRITE | Manuscript now states softmax-weighted PSD sums remain PSD and frames the contribution as input-conditioned routing. |
| UNR-019 | Missing ImageNet/SUN397/StanfordCars results | Experimental setup and reviewer R2-5d | No verified manifests for these datasets | SCOPED_AS_LIMITATION | Manuscript defines the benchmark as an 8-dataset suite and lists ImageNet/SUN397/StanfordCars as future work. |

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

The following are still needed outside this Phase 0 inventory or outside the current manuscript pass:

- Paired corruption sample/cache manifest and aggregate report for robustness.
- Final page/line mapping after PDF layout review.
- Release package packaging: anonymized repository, environment notes, scripts, manifests, and optional checkpoints.
- Optional regeneration of gating specialization exports if the current qualitative figure needs stronger provenance.

## Phase 0 Gate Conclusion

The Phase 0 tracking files and normalization outputs are complete, and raw logs/checkpoint evidence has been recovered. Acceptance-critical main accuracy claims are now superseded by the Phase 3 ramp100 rerun manifests. Robustness remains the only major numerical claim gate that is still unresolved for final quantitative reporting.
