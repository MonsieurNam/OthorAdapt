# Phase 4 Inventory

## Usable Evidence

- `revision_materials/results/validation_sweep_ramp100_results.jsonl`: 120 validation-only rows for H/r/lambda sensitivity.
- `revision_materials/results/w3_headcount_h1_ramp100_results.jsonl`: 12 validation-only H=1 rows.
- `revision_materials/results/phase3_main_ramp100_results.jsonl`: 144 final Phase 3 rows for checkpoint pairing.
- Phase 3 ramp100 checkpoints on disk: 144.
- `revision_materials/results/phase4_existing_diagnostics.md/csv`: consolidated validation diagnostics for H/r/lambda behavior.
- `revision_materials/results/phase4_checkpoint_diagnostics.jsonl`: 72 OrthoAdapt Phase 3 ramp100 checkpoints processed for head/orthogonality diagnostics.
- `revision_materials/results/phase4_head_overlap_report.md` and `phase4_orthogonality_report.md`: descriptive checkpoint-backed head overlap and orthogonality summaries.
- `revision_materials/results/phase4_spectrum_manifest.jsonl`: 16 paired spectrum records over 8 datasets, shot=4, seed=1, layer 11, q/v projections.
- `revision_materials/results/phase4_spectrum_report.md` and `revision_materials/results/figures/phase4_spectrum/`: regenerated spectrum diagnostics from real Phase 3 ramp100 checkpoints.
- `revision_materials/results/phase4_vitl14_results.jsonl`: 12 completed ViT-L/14 rows for EuroSAT+Caltech101, 4-shot, seeds 1/2/3, CLIP-LoRA r=8 vs OH-SingLoRA H=2,r=8.
- `revision_materials/results/phase4_vitl14_summary.csv`, `phase4_vitl14_paired_results.csv`, and `phase4_backbone_scaling_report.md`: bounded larger-backbone summary and claim gate.
- `revision_materials/results/phase4_robustness_manifest.jsonl`: structurally complete paired robustness manifest with 576/576 severity rows, 144/144 jobs, and no missing method pairs or checkpoint paths.
- `revision_materials/results/phase4_robustness_summary.csv` and `phase4_robustness_report.md`: audit aggregation generated, but the report marks the current manifest invalid for scientific claims because severity-0 OH-SingLoRA does not reproduce Phase 3 clean accuracy.

## Related Phase 3B Same-Parameter Evidence

- `revision_materials/results/phase3b_same_param_results.jsonl`: 144 original same-parameter rows; final Phase 3B summary uses the 72 CLIP-LoRA r=2 rows from this file because LoRA has no orthogonality ramp mechanism.
- `revision_materials/results/phase3b_same_param_ramp100_results.jsonl`: 72 completed OH-SingLoRA H=2,r=2,lambda_o=0.03,ramp100 rows.
- `revision_materials/results/phase3b_same_param_ramp100_report.md`: final same-parameter diagnostic report. It supports a small average paired gain for OH-SingLoRA at equal trainable-parameter count, with mixed dataset-level behavior.

## Preliminary / Legacy Evidence

- `data/result_scan_head`, `data/results_ablation_heads_lambda`, and `data/result_scan_loss` contain recovered logs but lack verified seed/split provenance; use only for context.
- `img/figure_robustness.*` and `img/singular_value_spectrum-rank16.png` are legacy figures and must not be cited as regenerated evidence.

## Remaining Missing / Partial Evidence

- Robustness still needs a fixed rerun: the existing manifest is complete for coverage, but fails the clean-consistency gate. The evaluator has been patched to infer the effective encoder from checkpoint keys and avoid random un-loaded adapter branches. Do not claim a robustness gain or orthogonality-caused robustness until a rerun passes clean consistency.
- Phase 3B checkpoint coverage is incomplete locally by one file: `revision_materials/checkpoints/phase3b_same_param_ramp100/ohsinglora/ViT-B16/ucf101/16shots/seed3/ucf101_16shot_seed3_test_ohsinglora_h2_r2_lo0p03_ramp100.pt`. Accuracy evidence is complete; re-sync only if checkpoint-level diagnostics need all Phase 3B checkpoints.
- Broader scope items remain deferred unless explicitly re-scoped: ImageNet/SUN397/StanfordCars, DoRA, multi-task learning, retrieval, and additional VLM families beyond the bounded ViT-L/14 subset.

## Manuscript / Response Status

- `revision_materials/Latex_code/cas-sc-template.tex` has been updated with yellow-highlighted revision content for PSD reframing, validation/test separation, manifest-backed tables, paired statistics, descriptive spectrum wording, downgraded robustness wording, and reproducibility text.
- `revision_materials/plan/Response_Letter_Reviewer2_Evidence_Draft.md` is current for RV2 evidence, including M1-M4 and Minor a-e.
- `revision_materials/plan/Response_Letter_Skeleton.md` now contains the rewritten RV2 section as 9 separate responses. Final page/line anchors should be added only after final PDF layout review.
