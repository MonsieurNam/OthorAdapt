# Reviewer 2 Response Matrix

This matrix is the evidence gate for the Reviewer 2 response. Use it before editing the final response letter. Planning files and the response-letter skeleton are not evidence.

Status date: 2026-07-04.

## Evidence Snapshot

| Artifact | Current status | Use in Reviewer 2 response |
|---|---|---|
| `revision_materials/Latex_code/cas-sc-template.tex` | Updated with yellow-highlighted RV2-relevant revisions: PSD reframing, ramp-up definition, validation/test protocol, manifest-backed main tables, paired summary table, descriptive spectrum wording, downgraded robustness wording, and data/code availability. Compiles with `latexmk`. | Safe to cite as the current manuscript draft. Final page/line references remain pending until final PDF layout review. |
| `revision_materials/results/validation_sweep_ramp100_results.jsonl` | Complete: 120 completed validation-only rows; all rows use `selection_split=val`, `report_test=false`, and `ramp_up_steps=100`. | Final evidence that hyperparameter selection was performed on validation data only. |
| `revision_materials/results/selected_config_ramp100.md` | Frozen ramp100 winner: `H=2`, `r=8`, `lambda_o=0.03`, mean validation accuracy `91.666667`; SHA256 sidecar exists. | Safe to state the final selected configuration. |
| `revision_materials/results/phase3_main_ramp100_results.jsonl` | Complete: 144 completed test rows, 8 datasets x 3 shots x 2 methods x 3 seeds, with 72 exact method pairs. | Final source for main 1/4/16-shot tables and paired accuracy claims. |
| `revision_materials/results/statistical_report.md` | Complete statistical report for Phase 3 ramp100: overall mean delta `+0.356 pp`, 95% CI `[+0.162,+0.551]`, wins/ties/losses `48/3/21`, mixed shot/dataset behavior. | Use for careful M3 response. Supports modest average paired gain with fewer trainable parameters, not broad superiority. |
| `revision_materials/results/generated_tables.tex` | Generated main result tables from the final Phase 3 ramp100 manifest. | Source of manifest-backed table values copied into the manuscript. |
| `revision_materials/results/w3_headcount_h1_ramp100_results.jsonl` | Complete H=1 validation diagnostic: 12 validation-only rows over EuroSAT/Caltech101, 4-shot, seeds `{1,2,3}`. | Use only as architectural diagnostic for single-head symmetric controls; not a full external SingLoRA baseline. |
| `revision_materials/results/phase4_spectrum_manifest.jsonl` and `phase4_spectrum_report.md` | Complete descriptive spectrum diagnostic: 16 records over all 8 datasets, shot 4, seed 1, layer 11, q/v projections. | Safe to say spectrum diagnostics were regenerated with checkpoint metadata, but only as descriptive evidence. |
| `revision_materials/results/phase4_robustness_manifest.jsonl` | Structurally complete but invalid for claims: 576/576 severity rows and paired method keys exist, but `phase4_robustness_report.md` shows the clean-consistency gate fails for OH-SingLoRA. | Use only to justify removing/defering quantitative robustness claims until a fixed rerun passes severity-0 reproduction. |
| `revision_materials/plan/Author_Decisions.md` | DoRA skipped; SingLoRA-CLIP excluded from Tier-A because it is internal exploratory adaptation; code release planned via anonymized GitHub/review package. | Use to answer missing-baseline and reproducibility comments honestly. |
| `Response_Letter_Reviewer2_Evidence_Draft.md` | Current evidence draft with separate M1-M4 and Minor a-e sections. | Drafting source for RV2 response language. |
| `Response_Letter_Skeleton.md` | RV2 section rewritten into exactly 9 separate responses. | Current response-letter draft; still needs final page/line references. |

## Reviewer 2 Items

| RV2 item | Reviewer concern | Response posture | Current evidence | Manuscript/action reference | Response-letter status |
|---|---|---|---|---|---|
| M1 | Softmax-gated PSD heads cannot escape the PSD constraint. | Full concede + reframe. | Manuscript now states that the method uses input-conditioned routing over symmetric PSD low-rank heads and no longer claims PSD escape. | Abstract/Introduction/Method/Fig. 2 caption; `cas-sc-template.tex` highlighted method text. | Can answer now. |
| M2 | Hyperparameters were selected on the test set. | Concede + describe validation-only protocol. | Ramp100 validation sweep is complete: 120 validation rows, selected `H=2,r=8,lambda_o=0.03`, no test reporting. | Evaluation protocol, validation-selection text, validation diagnostic table. | Can answer now. |
| M3 | Seed protocol contradiction and gains within seed noise. | Concede + report manifest-backed three-seed results and uncertainty. | Phase 3 ramp100 final matrix is complete: 144 rows and 72 pairs. Mean paired delta is `+0.356 pp`, 95% CI `[+0.162,+0.551]`, wins/ties/losses `48/3/21`, using fewer parameters. | Main Tables 1/2/3 and paired summary table in manuscript. | Can answer now with modest/mixed wording. |
| M4 | Spectrum figure is inconsistent with rank budget. | Concede + replace causal interpretation with descriptive diagnostics. | Spectrum manifest/report complete with 16 checkpoint-backed records. The analysis records matrix definition, checkpoint hash, projection, rank/config metadata, and singular values. | Spectrum caption and discussion. | Can answer now, descriptive only. |
| Minor a | Numerical inconsistencies: robustness severity, EuroSAT values, Aircraft value, decimal commas. | Concede + replace/caveat legacy values. | Main tables now use Phase 3 ramp100 manifest values; decimal separators standardized; robustness downgraded because the paired rerun fails clean-consistency validation. | Main result tables, sensitivity table note, robustness section. | Can answer now; do not claim robustness. |
| Minor b | SingLoRA should be full baseline; OMoE distinction needs sharpening. | Partial concede + scope/fairness explanation. | H=1 validation controls exist but are diagnostic only; no stable public SingLoRA-CLIP matched external baseline is claimed. | Related Work, diagnostics, limitations. | Can answer now without promising unsupported main baseline rows. |
| Minor c | Undefined ramp-up `u(t)` and dangling epsilon. | Concede + fix. | Manuscript defines `u(t)=min(1,t/T)`, states final OrthoAdapt uses `T=100`, and removes the epsilon reference. | Method equations and surrounding text. | Can answer now. |
| Minor d | Missing ImageNet, SUN397, StanfordCars from 11-dataset CLIP suite. | Concede + scope clarification. | No verified artifacts exist for these datasets; manuscript restricts claims to 8 manifest-backed datasets and lists missing datasets as future work. | Experimental Setup and Limitations. | Can answer now as scope clarification. |
| Minor e | Release code and exact seeds. | Agree + commit. | Manuscript Data/Code Availability now commits exact seeds `{1,2,3}`, manifests, split hashes, scripts, aggregation/statistical scripts, and environment notes. | Data/Code Availability section. | Can answer now. |

## Response Rules

- Use "the reviewer is correct" for M1, M2, M3, M4, and Minor a/c.
- Use "modest average paired gain with fewer trainable parameters" for M3.
- Use "mixed by dataset and shot" and avoid broad or uniform superiority claims.
- Use "validation-only selection" for M2, citing `validation_sweep_ramp100_results.jsonl` and `selected_config_ramp100.md`.
- Use "descriptive spectrum diagnostic" for M4, citing `phase4_spectrum_report.md`.
- Use "robustness remains pending/unpaired" for Minor a.
- Do not use: "state-of-the-art", "consistent gains", "orthogonality causes robustness", "escape PSD", or causal spectrum/robustness language.
