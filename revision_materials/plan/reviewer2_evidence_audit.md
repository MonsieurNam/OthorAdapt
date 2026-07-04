# Reviewer 2 Evidence Audit

This audit records the current artifact state used by `reviewer2_response_matrix.md`, `Response_Letter_Reviewer2_Evidence_Draft.md`, and the RV2 section of `Response_Letter_Skeleton.md`.

Status date: 2026-07-04.

## Current RV2 Draft State

- `Response_Letter_Reviewer2_Evidence_Draft.md` is current and covers M1-M4 plus Minor a-e.
- `Response_Letter_Skeleton.md` contains the rewritten Reviewer 2 response section with exactly 9 separate responses.
- `cas-sc-template.tex` has yellow-highlighted manuscript edits for the RV2-relevant changes and compiles with `latexmk`.
- Final page/line references remain pending until final PDF layout review.

## M1 - PSD Claim Under Softmax Gating

Current usable evidence:

- Source-code/method inspection confirms the update is a non-negative softmax-weighted mixture of symmetric PSD low-rank heads.
- `cas-sc-template.tex` now reframes the mechanism as input-conditioned routing over symmetric PSD low-rank heads, not as escaping the PSD constraint.

Safe response wording:

- "The reviewer is correct: a non-negative mixture of PSD matrices remains PSD."
- "We revised the theory text and Fig. 2/method framing to describe input-conditioned PSD routing."

Unsafe wording:

- "OrthoAdapt escapes the PSD constraint."
- "The PSD issue is resolved by softmax gating."

## M2 - Validation Selection

Current final usable artifacts:

- `revision_materials/results/validation_sweep_ramp100_results.jsonl`
- `revision_materials/results/selected_config_ramp100.md`
- `revision_materials/results/selected_config_ramp100.md.sha256`

Completion:

- Rows: 120
- Status: all completed
- Split: all rows use `selection_split=val`
- Test reporting: all sweep rows have `report_test=false`
- Ramp schedule: all rows use `ramp_up_steps=100`
- Candidate scope: EuroSAT and Caltech101, 4-shot, seeds `{1,2,3}`
- Candidate grid: 20 candidates, each with complete 6-row coverage
- Selected ramp100 config: `H=2`, `r=8`, `lambda_o=0.03`
- Selected ramp100 mean validation accuracy: `91.666667`

Safe response wording:

- "We implemented a validation-only selection protocol and reran it with the corrected ramp schedule."
- "The final ramp100 validation sweep selects `H=2,r=8,lambda_o=0.03`."
- "No test metrics were used for selection; all 120 sweep rows use `selection_split=val` and do not report test accuracy."

Unsafe wording:

- "The validation sweep proves final test-set gains."
- "Test accuracy was used to choose the configuration."

## M3 - Seed Protocol and Accuracy Claims

Current usable artifacts:

- `revision_materials/results/phase3_main_ramp100_results.jsonl`
- `revision_materials/results/statistical_report.md`
- `revision_materials/results/generated_tables.tex`
- `revision_materials/results/phase3_main_ramp100_summary.csv`
- `revision_materials/results/phase3_main_ramp100_paired_summary.csv`

Completion:

- Rows: 144/144 completed
- Coverage: 8 datasets x 3 shots x 2 methods x 3 seeds
- Methods: 72 `lora` rows and 72 `ohsinglora` rows
- Split: all final rows use `selection_split=test` with `report_test=true`
- Pairing: 72 matched `(dataset, shot, seed)` comparisons
- Parameter counts: CLIP-LoRA r=8 has 737,280 trainable parameters; OrthoAdapt H=2,r=8 has 460,800 trainable parameters

Main paired result:

- Mean delta, OrthoAdapt minus CLIP-LoRA: `+0.356 pp`
- 95% CI: `[+0.162,+0.551] pp`
- Paired standardized effect size: `dz=0.430`
- Wins/ties/losses: `48/3/21`
- Shot-level deltas: 1-shot `+0.816`, 4-shot `+0.402`, 16-shot `-0.149`

Safe response wording:

- "The seed contradiction was corrected with a manifest-backed three-seed matrix."
- "The revised evidence supports a modest average paired gain with fewer trainable parameters."
- "The effect is mixed by dataset and shot, so we do not claim broad or uniform superiority."

Unsafe wording:

- "OrthoAdapt consistently outperforms CLIP-LoRA."
- "The main table demonstrates state-of-the-art accuracy."
- "The gains are large or uniformly significant."

## M4 - Spectral Figure

Current usable artifacts:

- `analyze_spectrum.py` has been repaired to fail closed.
- `revision_materials/results/phase4_spectrum_manifest.jsonl`
- `revision_materials/results/phase4_spectrum_report.md`
- `revision_materials/results/figures/phase4_spectrum/`

Completion:

- Spectrum manifest records: 16
- Scope: all 8 datasets, shot 4, seed 1, vision layer 11, q/v projections
- Artifacts record checkpoint hashes, matrix definitions, projection, rank/config metadata, and singular values

Safe response wording:

- "We rebuilt the spectrum diagnostic with checkpoint and matrix-definition metadata."
- "The revised spectrum analysis is descriptive and is not used as causal proof of accuracy or robustness."

Unsafe wording:

- "Figure 7 proves rank-collapse mitigation."
- "Spectrum diagnostics establish the robustness mechanism."

## Minor a - Numerical Inconsistencies

Current state:

- Main 1/4/16-shot tables in `cas-sc-template.tex` have been replaced with manifest-backed values from `phase3_main_ramp100_results.jsonl`.
- EuroSAT 4-shot final values are CLIP-LoRA `84.36` and OrthoAdapt `85.78`.
- Aircraft 16-shot final values are CLIP-LoRA `56.99` and OrthoAdapt `56.63`.
- Decimal separators in the retained sensitivity diagnostic table have been standardized.
- Robustness wording has been downgraded because `phase4_robustness_manifest.jsonl` fails the clean-consistency gate for OH-SingLoRA.

Robustness audit:

- Rows: 576/576 completed severity rows
- Jobs: 144/144; paired method keys exist for dataset/shot/seed/severity
- Checkpoint paths: no missing paths in the local audit
- Clean-consistency gate: fail for OH-SingLoRA; severity-0 clean accuracy does not reproduce Phase 3 clean test accuracy
- Action: rerun robustness with the fixed evaluator before reporting any robustness numbers

Safe response wording:

- "The conflicting legacy values are no longer used as final claims."
- "Final table values are regenerated from the Phase 3 ramp100 manifest."
- "Robustness is not claimed quantitatively because the paired robustness rerun did not pass the clean-consistency gate."

## Minor b - SingLoRA / OMoE

Current usable artifact:

- `revision_materials/results/w3_headcount_h1_ramp100_results.jsonl`

Completion:

- Rows: 12 completed validation-only H=1 diagnostic rows
- Scope: EuroSAT and Caltech101, 4-shot, seeds `{1,2,3}`
- `H=1,r=2,lambda_o=0.0`: `91.333333`
- `H=1,r=4,lambda_o=0.0`: `89.750000`
- Final selected ramp100 config for comparison: `H=2,r=8,lambda_o=0.03`, `91.666667`

Safe response wording:

- "We discuss SingLoRA as the conceptual starting point."
- "The H=1 controls are used as architecture diagnostics, not as a full external SingLoRA-CLIP benchmark."
- "OMoE and related MoE-LoRA methods are discussed as related but not directly matched settings."

Unsafe wording:

- "SingLoRA-CLIP is now a full baseline."
- "Multi-head gating is uniformly superior to single-head symmetric adaptation."

## Minor c - Ramp-Up and Epsilon

Current state:

- `cas-sc-template.tex` defines `u(t)=min(1,t/T)` near the method equations.
- Final OrthoAdapt ramp100 evidence uses `T=100`.
- The dangling epsilon reference is removed because the implemented orthogonality loss has no epsilon term.

Safe response wording:

- "The reviewer is correct; we added the missing ramp-up definition and removed the epsilon reference."

## Minor d - Missing ImageNet/SUN397/StanfordCars

Current state:

- No verified ImageNet/SUN397/StanfordCars result artifact is present for this revision.
- The manuscript now restricts empirical claims to the 8 manifest-backed datasets.

Safe response wording:

- "We clarify that the revised benchmark is an 8-dataset few-shot suite, not the complete 11-dataset CLIP suite."
- "ImageNet, SUN397, and StanfordCars are listed as future work rather than implied coverage."

## Minor e - Code Release and Exact Seeds

Current state:

- Manuscript Data/Code Availability text commits to code, exact seeds `{1,2,3}`, validation/test manifests, split hashes, command scripts, aggregation/statistical scripts, environment notes, and checkpoint release where storage/licensing allow.

Safe response wording:

- "We agree and have added a reproducibility statement covering exact seeds, manifests, split hashes, scripts, and environment notes."
