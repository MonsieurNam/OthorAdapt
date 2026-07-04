# Response to Reviewer #2 - Evidence-Locked Draft

This file supersedes the Reviewer 2 section in `Response_Letter_Skeleton.md`. It is the controlling Reviewer 2 draft. Use this draft only with the current artifact gates below; planning files and the skeleton are not evidence.

## Current Evidence Snapshot

- Validation-only ramp100 selection is complete: `validation_sweep_ramp100_results.jsonl` has 120/120 completed rows, all on `selection_split=val`, with no test reporting. The frozen winner is `H=2,r=8,lambda_o=0.03`, mean validation accuracy `91.666667`, recorded in `selected_config_ramp100.md`.
- Final Phase 3 ramp100 test matrix is complete: `phase3_main_ramp100_results.jsonl` has 144/144 completed rows over 8 datasets, 3 shots, 2 methods, and seeds `{1,2,3}`.
- Final Phase 3 paired result: OrthoAdapt/OH-SingLoRA improves over CLIP-LoRA by a modest mean paired delta of `+0.356` percentage points, 95% CI `[+0.162,+0.551]`, with wins/ties/losses `48/3/21`, while using 37.5% fewer trainable parameters.
- Phase 3B matched-parameter diagnostic is complete for the accuracy comparison under the revised hybrid seed-1-original author decision: `phase3b_same_param_hybrid_seed1_original_report.md` reports 72/72 paired `r=2` comparisons, both methods use 184,320 trainable parameters, and OrthoAdapt has an overall paired delta of `+0.295` percentage points, 95% CI `[+0.102,+0.488]`.
- Spectrum diagnostics have been regenerated as descriptive diagnostics: `phase4_spectrum_manifest.jsonl` has 16 records, and `phase4_spectrum_report.md` summarizes layer-11, shot-4, seed-1, q/v projection comparisons over all 8 datasets.
- Robustness is not claim-ready. `phase4_robustness_manifest.jsonl` is structurally complete with 576/576 severity rows and paired method keys, but `phase4_robustness_report.md` marks it invalid for scientific claims because severity-0 OH-SingLoRA does not reproduce the corresponding Phase 3 clean test accuracy. Treat the current run as a clean-consistency audit failure, not as a robustness result.

## Major Concern M1 - PSD claim under softmax gating

> *"The central theoretical claim, namely that the gating mechanism escapes the PSD constraint, does not hold as stated. [...] A non-negative combination of PSD matrices is itself PSD. [...] What the gating genuinely contributes is input-dependence."*

**Response posture:** Full concede and reframe.

**Response draft:** The reviewer is correct. Under softmax gating, each coefficient is non-negative, so for any fixed input the update
`Delta W(x) = sum_h g_h(x) U_h U_h^T`
remains a non-negative combination of PSD matrices. The original wording that the gating mechanism "escapes" the PSD constraint was mathematically incorrect.

We have revised the manuscript accordingly. OrthoAdapt is now described as an input-conditioned family of symmetric PSD low-rank updates, not as a mechanism that removes the PSD constraint for an individual input. The contribution of the gate is routing: different inputs receive different PSD updates through learned head weights. This is a narrower claim than in the original submission, but it is the claim supported by the implemented architecture.

**Evidence used:** Source-code inspection of the softmax-gated symmetric heads; revised manuscript text in the Abstract, Introduction, Method, and Fig. 2 caption.

**Manuscript changes:** No new table or figure is needed. Revise theory prose and Fig. 2 caption. Highlight all added or changed text in yellow.

**Forbidden wording:** Do not say "escape PSD", "recovers spectral rank as the mechanism", or make a causal robustness claim from the orthogonality term.

## Major Concern M2 - Hyperparameter selection on the test set

> *"The number of heads and the regularization strength are chosen by test accuracy on EuroSAT. [...] I would ask the authors to perform this selection on validation data and to report the results again."*

**Response posture:** Concede the protocol flaw, then separate the protocol-defined matched-parameter diagnostic from the validation-selected final setting.

**Response draft:** The reviewer is correct that the original presentation did not adequately separate hyperparameter selection from test reporting. We corrected the code and protocol by adding an explicit validation-only selection path and by blocking test-set reporting during sweep mode.

The corrected ramp100 validation sweep contains 120 completed protocol rows over EuroSAT and Caltech101 in the 4-shot setting, seeds `{1,2,3}`, and the frozen candidate grid over head count, rank, and orthogonality weight. All protocol rows use `selection_split=val`, `ramp_up_steps=100`, and no sweep row reports test accuracy. Under the pre-registered rule, the selected configuration is `H=2`, `r=8`, and `lambda_o=0.03`, with mean validation accuracy `91.666667`. This selection is frozen in `selected_config_ramp100.md` with a SHA256 sidecar.

We now report two analyses with different roles. The first is a protocol-defined matched-parameter diagnostic at `r=2`, the default CLIP-LoRA baseline rank, where CLIP-LoRA and OrthoAdapt use the same 184,320 trainable-parameter budget. This `r=2` analysis is not presented as a test-selected optimum. The second is the validation-selected `r=8` setting, where `H=2` and `lambda_o=0.03` are selected from validation data and then frozen before final test reporting.

We keep the same `H=2` and `lambda_o=0.03` for the `r=2` matched-parameter diagnostic to avoid introducing a second test-driven selection loop. At `r=2`, validation sensitivity to `lambda_o` is weak: over `lambda_o` in `{0, 0.01, 0.03, 0.05}`, mean validation accuracy spans `90.125` to `91.000`, and the `{0, 0.03, 0.05}` settings lie within `0.05` points of each other. We therefore do not claim that `lambda_o=0.03` is separately optimal at `r=2`; we use it as the fixed protocol setting.

**Evidence used:** `validation_sweep_ramp100_results.jsonl`; `selected_config_ramp100.md`; `selected_config_ramp100.md.sha256`; `phase3b_same_param_ramp100_report.md`; code guards in `run_utils.py`, `ecr3_provenance.py`, and `lora.py`.

**Manuscript changes:** Add or retain a small validation-selection appendix table summarizing the selected `r=8` configuration and nearby H/r/lambda diagnostics, and state separately that `r=2` is a matched-parameter diagnostic. No new figure is required.

**Forbidden wording:** Do not imply the validation sweep itself proves test gains. Keep final test performance claims in M3.

## Major Concern M3 - Seed contradiction and gains within seed noise

> *"Section 4.1 states that results are averaged over three random seeds, while Table 1 states that a single seed was used. [...] Margins of this size cannot be interpreted without variance estimates."*

**Response posture:** Concede the contradiction, replace the evidence, and narrow the claim.

**Response draft:** The reviewer is correct. The original manuscript contained an inconsistency in the seed protocol, and small margins should not have been interpreted without uncertainty estimates.

The revised matched-parameter evidence separates the original seed-1 table values from the paired aggregate statistics. For the protocol-defined matched-parameter diagnostic, `phase3b_same_param_hybrid_seed1_original_report.md` contains 72 paired comparisons at `r=2`, with both methods using 184,320 trainable parameters. OrthoAdapt changes accuracy by `+0.295` percentage points relative to CLIP-LoRA, with a 95% CI of `[+0.102,+0.488]`. The shot-wise paired deltas are `+0.456` for 1-shot, `+0.428` for 4-shot, and `+0.001` for 16-shot, showing a small average gain with mixed behavior by shot.

For the validation-selected `r=8` setting, the final ramp100 Phase 3 manifest contains 144 completed test rows covering 8 datasets, 3 shot settings, 2 methods, and seeds `{1,2,3}`. Pairing is exact by `(dataset, shot, seed)`, yielding 72 matched comparisons between CLIP-LoRA r=8 and OrthoAdapt/OH-SingLoRA `H=2,r=8,lambda_o=0.03,ramp100`. Across these 72 matched comparisons, OrthoAdapt changes accuracy by `+0.356` percentage points relative to CLIP-LoRA, with a 95% CI of `[+0.162,+0.551]`. The paired standardized effect size is `dz=0.430`, and wins/ties/losses are `48/3/21`. The effect is modest and mixed by dataset and shot: the mean paired deltas are `+0.816` for 1-shot, `+0.402` for 4-shot, and `-0.149` for 16-shot. OrthoAdapt uses 460,800 trainable parameters compared with 737,280 for CLIP-LoRA r=8, a 37.5% reduction.

We therefore no longer claim broad benchmark-leading performance or uniform superiority. The revised conclusion is that OrthoAdapt shows a small positive average paired delta at equal parameters in the `r=2` diagnostic and a modest positive average paired delta at the validation-selected `r=8` setting while using fewer trainable parameters.

**Evidence used:** `phase3b_same_param_ramp100_report.md`; `phase3_main_ramp100_results.jsonl`; `statistical_report.md`; `generated_tables.tex`.

**Manuscript changes:** Revise the legacy 1-shot, 4-shot, and 16-shot tables into hybrid seed-1-original aggregate tables for the matched-parameter comparison. Add or retain a paired-delta table or concise paired-statistics table. Highlight all new captions, notes, and interpretation text in yellow.

**Forbidden wording:** Do not claim uniform per-dataset wins, benchmark-leading status, large effects, or unqualified significant improvement.

## Major Concern M4 - Spectral figure inconsistent with the rank budget

> *"With a total rank budget of r = 2, the update matrix can have at most two non-zero singular values [...] Yet the figure shows roughly fifteen smoothly decaying singular values at every layer."*

**Response posture:** Concede the legacy figure problem, then report the corrected descriptive diagnostic.

**Response draft:** The reviewer is correct. The legacy spectral figure could not support the stated low-rank claim unless the exact checkpoint, matrix definition, layer, projection, and rank budget were specified, and plotting more singular values than the declared rank budget made the figure easy to misread. We therefore removed the unsupported interpretation and rebuilt the spectrum analysis path so that it fails closed when required checkpoints, metadata, or rank-consistent plotting settings are missing.

The likely technical source of the roughly fifteen visible singular values in the old figure was the legacy visualization path, not a valid indication that the trained adapter used rank fifteen. The old plotting code computed SVD values from a checkpoint-derived matrix/proxy and displayed a fixed prefix of the spectrum controlled by a plotting parameter (`top_k`, previously defaulting to 50 in the analysis script) so that the decay shape was easy to inspect. That visualization choice was not tied to the declared rank budget. For a rank-budget diagnostic, only the first `r` singular values should be shown as the interpretable spectrum; otherwise the reviewer can reasonably read the extra displayed values as implying a larger update rank. We therefore treat the old figure as an unclear visualization for the `r=2` claim and replace it with rank-aligned plots.

The revised diagnostic records the matrix definition, layer, projection, rank/config metadata, checkpoint hashes, plotted rank budget, and singular values in a manifest. Plotting is now rank-aware: the `r=2` same-parameter diagnostic plots exactly the top two singular values, and the validation-selected `r=8` diagnostic plots exactly the top eight singular values. For the reviewer-facing manuscript figure, we selected the EuroSAT 4-shot, seed-1, vision-layer-11 `q_proj` panel from the matched-parameter `r=2` diagnostic (`revision_materials/results/figures/phase4_spectrum_r2_same_param/eurosat_4shot_seed1_layer11_q_proj.png`) because this is the configuration directly implicated by the reviewer concern and by the main matched-parameter comparison against CLIP-LoRA and the standard few-shot baselines. If a second panel is retained in the manuscript, it should come from the same diagnostic subset (EuroSAT, 4-shot, seed 1, layer 11, `r=2`) rather than from a validation-selected `r=8` run. The `r=8` diagnostic is retained as secondary traceability evidence for the validation-selected setting. In all cases, no post-rank numerical tail is shown or interpreted. The diagnostic is used only as descriptive evidence about the selected checkpoint subset. It is not presented as proof of a causal accuracy or robustness mechanism.

In the `r=2` manuscript diagnostic subset, the selected EuroSAT 4-shot, seed-1, layer-11 `q_proj` panel shows the first two normalized singular values only. Across the eight-dataset layer-11, seed-1 report, the 4-shot `q_proj` mean stable rank is 1.004 for CLIP-LoRA and 1.064 for OrthoAdapt; for `v_proj`, the mean stable rank is 1.281 for CLIP-LoRA and 1.563 for OrthoAdapt. These values support a bounded descriptive statement about the analyzed subset, not a causal mechanism claim.

**Evidence used:** `analyze_spectrum.py`; `phase4_spectrum_diagnostics.py`; `phase4_spectrum_manifest.jsonl`; `phase4_spectrum_report.md`; `phase4_spectrum_full_shots_manifest.jsonl`; `phase4_spectrum_full_shots_report.md`; `phase4_spectrum_r2_same_param_manifest.jsonl`; `phase4_spectrum_r2_same_param_report.md`; `revision_materials/results/figures/phase4_spectrum/`; `revision_materials/results/figures/phase4_spectrum_full_shots/`; `revision_materials/results/figures/phase4_spectrum_r2_same_param/`.

**Manuscript changes:** Replace the legacy spectrum interpretation with descriptive language tied to the exact regenerated diagnostic. The manuscript figure should explicitly identify the selected panel as EuroSAT 4-shot, seed 1, vision layer 11, `q_proj`, matched-parameter `r=2`, with the curve truncated to the first two singular values. If the `v_proj` companion panel is retained, it should use the same EuroSAT/4-shot/seed-1/layer-11/`r=2` subset. The report/manifest should be described as covering all eight datasets for this diagnostic setting. Keep figures only if captions state the exact checkpoint subset, layer/projection, rank budget, and diagnostic role.

**Forbidden wording:** Do not say the spectrum proves robustness, rank recovery, or a causal performance mechanism.

## Minor Comment a - Numerical inconsistencies

> *"The Abstract attributes the 3.6 point robustness gain to severe corruption, while Section 4.4 attributes it to the medium setting. The EuroSAT 4-shot accuracy appears as 88.64 in Figure 5, as 89.06 in Table 3, and as 88.57 in Table 6. In Table 4 the Aircraft result is identical to CLIP-LoRA at 54.97, which I suspect is a copy error. Table 6 uses commas as decimal separators."*

**Response posture:** Concede and resolve by replacing legacy values with manifest-backed values; downgrade robustness.

**Response draft:** We agree with the reviewer. The revised manuscript no longer defends the conflicting legacy values. The main accuracy tables have been regenerated from the final ramp100 manifest, and validation-sweep values are now clearly separated from final test values.

The EuroSAT 4-shot number is no longer allowed to take conflicting meanings across a figure, a main table, and an ablation table. At the matched-parameter `r=2` setting, the hybrid aggregate reports CLIP-LoRA `85.61` and OrthoAdapt `86.41`; the original `89.06` value is retained only as the seed-1 OrthoAdapt table entry. At the validation-selected `r=8` setting, EuroSAT is reported as CLIP-LoRA `84.36` and OrthoAdapt `85.78`.

The suspected Aircraft copy issue is also clarified. The original `54.97/54.97` value is retained as the seed-1 Aircraft 16-shot table entry rather than treated as a copy error. In the matched-parameter hybrid aggregate, Aircraft 16-shot is CLIP-LoRA `54.65` and OrthoAdapt `54.63`. At `r=8`, Aircraft 16-shot is CLIP-LoRA `56.99` and OrthoAdapt `56.63`.

The robustness wording has also been corrected. We attempted a paired robustness rerun over the final Phase 3 checkpoints, but the run did not pass the clean-consistency audit: severity-0 OH-SingLoRA accuracy did not reproduce the corresponding Phase 3 clean test accuracy, indicating an evaluator/checkpoint-loading mismatch. We therefore do not report the robustness numbers from that run and remove the original `+3.6` robustness statement rather than presenting it as a final revised claim.

The decimal-separator issue in Table 6 should be corrected wherever the legacy table is retained.

**Evidence used:** `generated_tables.tex`; `statistical_report.md`; `phase4_robustness_manifest.jsonl`; `phase4_robustness_report.md` clean-consistency audit.

**Manuscript changes:** Use manifest-backed main tables. Do not use a final robustness claim unless a fixed paired robustness rerun passes the clean-consistency gate. Fix decimal separators in any retained ablation table.

**Forbidden wording:** Do not describe the robustness result as final, paired, causal, or severe-corruption evidence.

## Minor Comment b - SingLoRA baseline and OMoE distinction

> *"Since SingLoRA is the architectural starting point, it deserves to appear as a full baseline in Tables 2 through 4. The distinction from recent work combining mixtures of LoRA experts with orthogonality, such as OMoE in reference 41, also needs a sharper comparison than the current Related Work provides."*

**Response posture:** Partial concede with scope clarification.

**Response draft:** We agree that SingLoRA is an important conceptual starting point, and the revised Related Work now explains this relationship more explicitly. Since the original submission, our prior SingLoRA-CLIP study has appeared as a published ICCSA/LNCS chapter. We now cite this work explicitly and position it as the closest predecessor and companion study: SingLoRA-CLIP evaluates a single-matrix symmetric update for few-shot CLIP adaptation, whereas OrthoAdapt studies input-conditioned multi-head routing over symmetric low-rank heads.

At the same time, we do not merge the published SingLoRA-CLIP numbers into the main revised tables. The SingLoRA-CLIP chapter reports a published single-seed literature result, while the revised OrthoAdapt tables use a manifest-controlled three-seed protocol with paired uncertainty estimates. Mixing a prior single-seed literature row with the rerun three-seed rows would make the uncertainty comparison unclear. We therefore cite and discuss SingLoRA-CLIP as the closest predecessor, while restricting paired statistical comparisons to methods rerun under the same revised protocol.

To address the architecture question without overstating the baseline evidence, we additionally use H=1 symmetric-head runs only as validation diagnostics within the current codebase. In the ramp100 H=1 validation diagnostic over EuroSAT and Caltech101, 4-shot, seeds `{1,2,3}`, `H=1,r=2,lambda_o=0.0` averages `91.333333`, and `H=1,r=4,lambda_o=0.0` averages `89.750000`. These diagnostic runs are not presented as a replacement for the published SingLoRA-CLIP study; they simply help interpret the effect of moving from one symmetric head to routed multi-head adaptation under the current validation protocol.

We also sharpened the OMoE/MoE-LoRA comparison by distinguishing target setting, base parameterization, and the role of orthogonality.

**Evidence used:** published SingLoRA-CLIP chapter (`https://doi.org/10.1007/978-3-032-30488-9_16`); `Author_Decisions.md`; `w3_headcount_h1_ramp100_results.jsonl`; revised Related Work and Limitations text.

**Manuscript changes:** Cite SingLoRA-CLIP in Related Work and position it as predecessor/companion work. No new main table row is needed unless the published single-seed row is clearly separated as a literature reference row or new three-seed SingLoRA-CLIP reruns are completed. Optionally include H=1 diagnostic in an appendix validation table. Strengthen Related Work and Limitations.

**Forbidden wording:** Do not claim a full SingLoRA-CLIP or DoRA empirical baseline was added.

## Minor Comment c - Undefined ramp-up function and epsilon

> *"The ramp-up function u(t) in Eqs. 3 and 6 is never defined, and the constant epsilon mentioned after Eq. 11 does not actually appear in the equation."*

**Response posture:** Concede and fix directly.

**Response draft:** The reviewer is correct. We added an explicit definition of the ramp-up factor: `u(t)=min(1,t/T)`, where `T` is the ramp-up step budget. In the final ramp100 experiments, `T=100`. We also removed the dangling epsilon reference because the implemented orthogonality loss does not include such a term.

**Evidence used:** `run_utils.py` defines `--ramp_up_steps`; the OH/SingLoRA layer uses the ramp-up budget; the orthogonality loss implementation has no epsilon term.

**Manuscript changes:** Add the ramp-up definition near Eqs. 3 and 6, and remove the epsilon sentence after the orthogonality loss.

**Forbidden wording:** Do not state `T=1000` for the final ramp100 evidence.

## Minor Comment d - Missing ImageNet, SUN397, and StanfordCars

> *"The standard CLIP few-shot suite comprises eleven datasets, including ImageNet, SUN397, and StanfordCars. The authors should explain why these three were omitted."*

**Response posture:** Concede broader-suite omission and define the revised scope.

**Response draft:** We thank the reviewer for pointing this out. We agree that ImageNet, SUN397, and StanfordCars are part of the broader standard CLIP few-shot suite. In the revised manuscript, we no longer describe our benchmark as the complete 11-dataset CLIP suite. We explicitly define it as an eight-dataset few-shot evaluation suite.

All empirical claims in the revised paper are restricted to the eight datasets with manifest-backed evidence: Aircraft/FGVC, EuroSAT, Food101, OxfordPets, OxfordFlowers, Caltech101, DTD, and UCF101. ImageNet, SUN397, and StanfordCars are now listed as future work rather than implied coverage.

**Evidence used:** Absence of verified ImageNet/SUN397/StanfordCars manifests; final Phase 3 manifest over 8 datasets.

**Manuscript changes:** Update Experimental Setup, figure captions, and Limitations so they do not imply full 11-dataset coverage.

**Forbidden wording:** Do not say "all 11 datasets" or "complete standard CLIP suite".

## Minor Comment e - Code release and exact seeds

> *"Given how small the margins are, releasing the code and the exact seeds would do a great deal for the credibility of the results, and I warmly encourage the authors to do so."*

**Response posture:** Agree and commit.

**Response draft:** We agree. The revised reproducibility statement commits to releasing the code, exact seeds `{1,2,3}`, run scripts, validation/test manifests, split hashes, aggregation/statistical scripts, environment information, and instructions needed to reproduce the reported tables. Adapter checkpoints will be released where storage and licensing constraints allow; otherwise, the scripts and manifests will be sufficient to reproduce them.

**Evidence used:** `Author_Decisions.md`; final manifests with seeds, split hashes, commands, metrics, and checkpoint hashes.

**Manuscript changes:** Update Data/Code Availability. No new table or figure is required.

**Forbidden wording:** Do not promise unavailable checkpoints unconditionally if storage or licensing constraints apply.

## Reviewer 2 Response Lock

Before copying this draft into the final response letter:

- Confirm RV2 has exactly nine separate entries: M1, M2, M3, M4, Minor a, Minor b, Minor c, Minor d, Minor e.
- Confirm main tables use manifest-backed values from `generated_tables.tex`.
- Confirm robustness is not claimed unless paired robustness aggregation exists.
- Search for forbidden wording and unresolved placeholders.
