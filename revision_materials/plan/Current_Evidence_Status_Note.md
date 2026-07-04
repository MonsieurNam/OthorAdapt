# Current Evidence Status Note

_Revision package status as of 2026-07-04._

---

## Executive Status

| Area | Current state | Revision decision |
|---|---|---|
| Evidence pipeline | Validation selection, Phase 3 main matrix, spectrum diagnostics, H/r/lambda diagnostics, same-parameter diagnostic, and bounded ViT-L/14 subset are available as manifest-backed artifacts. A paired robustness manifest exists but currently fails clean-consistency validation. | Use only clean-gated artifacts as the source of manuscript and response-letter claims. |
| Main manuscript | `revision_materials/Latex_code/cas-sc-template.tex` has been updated with yellow-highlighted revision text, manifest-backed main tables, paired statistics, downgraded robustness wording, descriptive spectrum wording, and reproducibility text. | Continue from this manuscript state; do not reintroduce legacy unsupported claims. |
| Reviewer 2 response | `Response_Letter_Reviewer2_Evidence_Draft.md` has been evidence-updated, then the RV2 section in `Response_Letter_Skeleton.md` was rewritten into 9 separate responses: M1-M4 and Minor a-e. | Treat RV2 response as current draft, pending final page/line references after final manuscript polish. |
| Build status | `latexmk -pdf -interaction=nonstopmode -halt-on-error cas-sc-template.tex` completed successfully on 2026-07-03/04 session; PDF is up to date. | Build gate is currently pass, with only layout warnings. |
| Robustness | `phase4_robustness_manifest.jsonl` has 576/576 completed severity rows, 144/144 jobs, 72 paired comparisons per severity, no missing pairs, and no missing checkpoint paths. However, the clean-consistency audit fails for OH-SingLoRA: severity-0 clean accuracy does not reproduce Phase 3 test accuracy. | Do not make robustness claims from this run. Treat it as an audit failure and rerun after the evaluator fix. |

---

## Completed Evidence

| Area | Status | Primary artifacts | Claim gate |
|---|---|---|---|
| Phase 0 inventory and normalization | Complete | `artifact_inventory.md`; `revision_traceability.csv`; `unresolved_numbers.md`; `reviewer_table_figure_mapping.md`; `main_results_manifest.*`; `workbook_vs_logs_crosscheck.*` | Use for provenance and legacy audit trail, not as final rerun accuracy evidence. |
| Phase 2 validation selection | Complete | `validation_sweep_ramp100_results.jsonl`; `w3_headcount_h1_ramp100_results.jsonl`; `selected_config_ramp100.md` | Frozen winner is OH-SingLoRA H=2,r=8,lambda_o=0.03,ramp100. |
| Phase 3 main matrix | Complete | `phase3_main_ramp100_results.jsonl`; `statistical_report.md`; `generated_tables.tex`; `phase3_main_ramp100_summary.*`; `phase3_main_ramp100_paired_summary.csv` | Keep modest average paired gain with fewer trainable parameters; remove broad SOTA/consistent-superiority wording. |
| Phase 3B same-parameter diagnostic | Complete for accuracy | `phase3b_same_param_ramp100_report.md`; `phase3b_same_param_ramp100_summary.csv`; `phase3b_same_param_ramp100_dataset_summary.csv`; `phase3b_same_param_ramp100_paired_results.csv` | Same-parameter OH-SingLoRA shows a small average paired gain, but dataset-level behavior is mixed. |
| Phase 4 H/r/lambda diagnostics | Complete | `phase4_existing_diagnostics.md/csv`; `w3_headcount_h1_ramp100_results.jsonl` | Use configuration-dependent wording; do not claim a causal head-count law. |
| Phase 4 checkpoint diagnostics | Complete | `phase4_checkpoint_diagnostics.jsonl`; `phase4_head_overlap_report.md`; `phase4_orthogonality_report.md` | Descriptive checkpoint-backed head overlap and orthogonality evidence only. |
| Phase 4 spectrum diagnostics | Complete | `phase4_spectrum_manifest.jsonl`; `phase4_spectrum_report.md`; `figures/phase4_spectrum/` | Descriptive spectral comparison only, not causal proof. |
| Phase 4 ViT-L/14 subset | Complete | `phase4_vitl14_results.jsonl`; `phase4_vitl14_summary.csv`; `phase4_vitl14_paired_results.csv`; `phase4_backbone_scaling_report.md` | Bounded larger-backbone applicability and parameter efficiency; no broad ViT-L/14 superiority claim. |
| Phase 4 robustness | Structurally complete but invalid for claims | `phase4_robustness_manifest.jsonl`; `phase4_robustness_summary.csv`; `phase4_robustness_report.md` | Clean-consistency gate fails for OH-SingLoRA; rerun with fixed evaluator before reporting robustness numbers. |
| Main manuscript revision pass | Current draft updated | `Latex_code/cas-sc-template.tex`; compiled `cas-sc-template.pdf` | Yellow-highlighted manuscript changes are present; final editorial polish and page/line references remain. |
| Response-letter RV2 pass | Current draft updated | `Response_Letter_Reviewer2_Evidence_Draft.md`; `Response_Letter_Skeleton.md` | RV2 has exactly 9 separate responses and current evidence gates. |

## Open Items

| Area | Current state | Next action |
|---|---|---|
| Phase 3B checkpoint sync | Accuracy evidence complete, but one local OH checkpoint is missing for `ucf101/16shots/seed3`. | Re-sync only if Phase 3B checkpoint-level diagnostics are needed. |
| Final response package | RV2 draft is current; R1 draft exists but may still need the same final page/line anchoring. | After manuscript final polish, add final page/line references and ensure every reviewer response points to the exact changed manuscript location. |
| Final manuscript polish | Main scientific edits are in place and compile. Some layout warnings remain in the LaTeX log. | Inspect PDF layout around tables/figures and resolve only if visible formatting issues affect readability. |
| Deferred scope | ImageNet/SUN397/StanfordCars, DoRA, multi-task learning, retrieval, and broader VLM families are not completed. | Keep as limitations/future work unless explicitly re-scoped. |

## Reviewer-Facing Wording Rules

- Main result: OrthoAdapt/OH-SingLoRA has a modest average paired gain while using fewer trainable parameters than CLIP-LoRA r=8.
- Same-parameter result: OH-SingLoRA H=2,r=2 has a small average paired gain over CLIP-LoRA r=2 at equal 184,320 trainable parameters, but the behavior is mixed by dataset.
- H/r/lambda behavior: report configuration-dependent validation behavior; avoid unsupported causal rank-fragmentation claims.
- Spectrum and checkpoint diagnostics: use as descriptive evidence only.
- ViT-L/14: report bounded applicability and parameter efficiency; do not claim statistically established superiority.
- Robustness: paired summary/report exist, but the current manifest fails clean-consistency validation for OH-SingLoRA; remove positive robustness claims and do not report the numeric robustness deltas until a fixed rerun passes the clean gate.
- Manuscript changes: all added/modified manuscript content for this revision pass should remain yellow-highlighted with `\revyellow{...}` or `\revyellowcaption{...}`.
