# Reviewer 2 Response QA Checklist

Use this checklist immediately before finalizing the Reviewer 2 response letter.

Status date: 2026-07-04.

## Artifact Gates

| Gate | Required check | Pass condition | Current expected status |
|---|---|---|---|
| M1 PSD wording | Search final response and manuscript for unsupported PSD-escape wording. | No claim that softmax gating escapes PSD; contribution is input-conditioned routing over symmetric PSD low-rank heads. | Should pass; current manuscript and RV2 draft use the corrected framing. |
| M2 validation selection | Inspect validation manifest used in final response. | All sweep rows use `selection_split=val`; no sweep row reports test accuracy; ramp schedule is disclosed. | Passes: ramp100 sweep has 120/120 completed validation-only rows, winner `H=2,r=8,lambda_o=0.03`. |
| H1 head-count diagnostic | Inspect H=1 diagnostic manifest. | All rows use `selection_split=val`; no row reports test accuracy; results are described as diagnostic only. | Passes: 12/12 completed validation-only rows; `H=1,r=2` averages `91.333333`, `H=1,r=4` averages `89.750000`. |
| M3 seed/statistics | Inspect final main-result source. | Values come from 3 seeds `{1,2,3}` and paired dataset/shot/method rows for the frozen selected configuration. | Passes: final ramp100 Phase 3 has 144/144 completed rows and 72 matched pairs. Use modest/mixed wording. |
| M4 spectrum | Inspect spectrum report files. | JSONL/report include checkpoint hash, matrix definition, layer, projection, rank/config metadata, and singular values. | Passes for descriptive diagnostics: `phase4_spectrum_manifest.jsonl` has 16 records and report exists. Do not claim causal mechanism. |
| Robustness | Inspect robustness evidence. | Paired method results exist on identical dataset/shot/seed/corruption/severity keys, aggregate uncertainty estimates exist, and severity-0 accuracy reproduces Phase 3 clean test accuracy. | Fails clean-consistency gate: current manifest has 576/576 severity rows, but OH-SingLoRA severity-0 accuracy does not reproduce Phase 3. Do not claim robustness until fixed rerun passes. |
| Dataset scope | Inspect final response. | Does not claim full 11-dataset CLIP suite unless ImageNet/SUN397/StanfordCars manifests exist. | Should be limitation wording; current RV2 draft restricts claims to 8 datasets. |
| Baselines | Inspect final response. | Does not claim DoRA/SingLoRA/MoRE empirical baselines were added unless matched artifacts exist. | Should be scope/limitation wording; H=1 controls are diagnostic only. |
| Manuscript highlights | Inspect LaTeX edits. | New/modified manuscript content is highlighted with `\revyellow{...}` or `\revyellowcaption{...}`. | Current manuscript has yellow-highlighted revision content; keep this convention in further edits. |
| Build | Compile LaTeX before final package. | `latexmk -pdf -interaction=nonstopmode -halt-on-error cas-sc-template.tex` exits 0. | Last run passed; rerun after any further LaTeX edits. |

## Forbidden Phrase Search

Run these searches on the final response draft and final manuscript:

```powershell
rg -n -S "escape(s|d)? the PSD|state-of-the-art|SOTA|consistent(ly)? (superior|improve|gain)|orthogonality causes|caused by orthogonality|spectral-rank recovery|responsible for the robustness|\\[X\\]|pending replacement" revision_materials/plan revision_materials/Latex_code
```

Any hit must be reviewed. It is allowed only if it is quoted from the reviewer or appears in a "forbidden wording" note, not as an author claim.

## Minimum Safe Final Claims

- OrthoAdapt uses input-conditioned routing over symmetric low-rank PSD heads.
- The original PSD-escape wording was incorrect and has been corrected.
- Validation-only ramp100 selection was implemented and completed; selected configuration is `H=2,r=8,lambda_o=0.03`.
- Final ramp100 Phase 3 evidence is complete: 144 rows, 72 pairs, 8 datasets, 3 shots, seeds `{1,2,3}`.
- The main empirical claim is a modest average paired gain with fewer trainable parameters, not broad SOTA or uniform superiority.
- H=1 results may be used as single-head architectural diagnostics, not as a full SingLoRA-CLIP external baseline.
- Spectrum diagnostics are regenerated and checkpoint-backed, but descriptive only.
- Robustness is pending because no paired method aggregation/report exists.
- Code, seeds, scripts, manifests, split hashes, and aggregation scripts will be released through an anonymized repository.
