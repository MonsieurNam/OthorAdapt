# Response Letter - ARRAY-D-26-02033
## Orthogonal Multi-Head Gated Low-Rank Adaptation for Robust Vision-Language Model Adaptation

Dear Dr. Bora and reviewers,

We thank you for the careful and constructive reviews and for inviting a revision. We have addressed each reviewer point separately below. Manuscript changes are highlighted in yellow in the revised PDF.

Several concerns, especially the seed protocol, validation/test separation, the spectral figure, the PSD framing, and the scope/positioning comments, were valid and led us to materially revise both the experiments and the narrative. Where the evidence supports a narrower claim than the original manuscript, we state that limitation explicitly.

At the same time, the retained contribution remains positive under the revised evidence standard. At the matched-parameter setting (`r=2`, identical 184,320 trainable-parameter budget), OrthoAdapt shows a small average paired gain over CLIP-LoRA (`+0.295` percentage points, 95% CI `[+0.102,+0.488]`), while the 16-shot subset is effectively tied. At the validation-selected `r=8` configuration, OrthoAdapt shows a modest average paired gain of `+0.356` percentage points (95% CI `[+0.162, +0.551]`) over 72 matched pairs while using 37.5% fewer trainable parameters than CLIP-LoRA r=8. We therefore believe the revision is not only a narrowing of claims but also a strengthening of the evidence standard behind the retained claims.

A summary of changes:
- The main results are now reported as two separate analyses with paired uncertainty estimates: a protocol-defined matched-parameter diagnostic at `r=2` (the CLIP-LoRA baseline rank), where the CLIP-LoRA and OrthoAdapt rows use an identical trainable-parameter budget and combine the original submitted Table 1-3 values as seed 1 with rerun seeds 2 and 3; and a validation-selected configuration at `r=8` compared head-to-head against CLIP-LoRA, which emphasizes parameter efficiency.
- Hyperparameters `H` and `lambda_o` are selected on held-out validation data before final test reporting; `r=2` is a fixed matched-parameter setting and `r=8` is the validation-selected setting, with a single fixed `lambda_o=0.03` and `H=2` applied at both.
- The PSD wording has been corrected: OrthoAdapt is framed as input-conditioned routing over symmetric PSD low-rank heads, not as escaping the PSD constraint.
- The spectral figure has been replaced with a regenerated, rank-consistent diagnostic from the revised checkpoints and is used descriptively rather than as causal proof.
- SingLoRA-CLIP and DoRA are discussed as related methods and limitations rather than presented as evaluated baselines.
- A bounded ViT-L/14 subset is added for EuroSAT and Caltech101, 4-shot, seeds `{1,2,3}`.
- Appendix diagnostics clarify that head count, rank budget, and orthogonality form a configuration-dependent capacity/routing trade-off rather than a proven universal mechanism.
- ImageNet, SUN397, and StanfordCars are explicitly scoped as future work; the revised benchmark is described as an 8-dataset suite.
- Code, exact seeds, run scripts, dataset split records, and aggregation/statistical scripts will be released through an anonymized repository.
- Main-table numerical inconsistencies are reconciled using the regenerated result tables. Robustness claims are removed as positive claims because the paired robustness rerun did not pass the clean-consistency audit needed for scientific reporting.

We now respond point by point.

---

# Response to Reviewer #1

We are grateful for the positive overall assessment and for the explicit recognition of the framework design, ablation analysis, and visualization effort. The four weaknesses and three questions raised by the reviewer led us to narrow the claims, clarify the scope, and separate evidence-supported findings from future work.

## Weakness W1 - Novelty vs. MoRE (ICLR 2025)

> *"The core idea of MoE-based LoRA with diversity constraints is highly similar to recent top-conference papers (e.g., MoRE at ICLR 2025), with incremental innovation only."*

**Response:** We thank the reviewer for pointing to this related line of work. We partially agree: the original manuscript framed the contribution too broadly relative to recent MoE-LoRA work. We have therefore narrowed the novelty claim and revised the Related Work discussion.

The revised manuscript positions OrthoAdapt as a few-shot VLM adaptation study rather than as a general MoE-LoRA framework. The distinction is threefold. First, methods such as MoRE, MoLE, and MoCLE primarily study multi-task or instruction-tuning settings, where routing separates tasks or instructions; our setting is single-task few-shot CLIP adaptation, where routing is used to model intra-task feature variation. Second, OrthoAdapt uses symmetric low-rank heads, whereas many MoE-LoRA methods compose standard asymmetric LoRA experts. Third, we now describe the orthogonality term as a configuration-dependent regularizer rather than as a universally causal mechanism.

We also added a more explicit comparison with OMoE and O-LoRA. These works motivate the relevance of orthogonality, but their experimental settings differ from our single-task few-shot VLM protocol.

Beyond narrowing the claim, we also state the retained positive contribution so that the revision is not read only as a set of concessions. After the theory correction (see Reviewer 2, M1), the contribution of OrthoAdapt is an input-conditioned family of symmetric PSD low-rank heads for few-shot CLIP adaptation. This narrower contribution is supported by two results. First, in the protocol-defined matched-parameter diagnostic at `r=2` (Tables 1-3), OrthoAdapt shows a small average paired gain over CLIP-LoRA at an identical trainable-parameter budget, with the advantage concentrated in 1-shot and 4-shot and a near-tie at 16-shot, evaluated alongside established few-shot CLIP baselines. Second, at the validation-selected `r=8` configuration, OrthoAdapt shows a modest average paired gain while using 37.5% fewer trainable parameters than CLIP-LoRA r=8. We therefore position OrthoAdapt as a parameter-efficient, input-conditioned adapter design for few-shot CLIP adaptation, rather than as a new general MoE-LoRA framework.

**Manuscript changes:** Introduction and Related Work were revised and highlighted in yellow, especially the CLIP-LoRA baseline paragraph, the LoRA/DoRA/rsLoRA positioning paragraph, and the MoRE/MoLE/MoCLE/O-LoRA comparison paragraphs. The matched-parameter comparison against the standard baselines (Tables 1-3) and the parameter-efficiency result at `r=8` were used to state the retained contribution.

---

## Weakness W2 - Narrow experimental scope

> *"It only focuses on single-task few-shot classification with ViT-B/16 backbone, lacking validation on large backbones (e.g., ViT-L/14), multi-task scenarios, or core VLM tasks (e.g., image retrieval)."*

**Response:** We agree that the original empirical scope was limited. We revised the manuscript to make the scope explicit: the reported experiments focus on single-task few-shot classification with CLIP ViT-B/16, with an additional bounded ViT-L/14 subset discussed separately in response to Q3. We do not present this as comprehensive coverage of all VLM adaptation settings.

We added a Limitations and Future Work discussion that identifies multi-task adaptation and image-text retrieval as important next steps. These settings require different training and evaluation protocols from the single-task few-shot classification setting used here, so we treat them as future work rather than as claims supported by the current experiments.

**Manuscript changes:** Experimental Setup, Backbone Scalability, Limitations, and Conclusion were revised and highlighted in yellow.

---

## Weakness W3 - Why H > 2 degrades

> *"The multi-head design is restricted to H=2 only, with clear performance degradation for H>2. The paper does not fully address the fundamental reasons for this limitation or propose effective solutions."*

**Response:** We thank the reviewer for asking us to make this limitation explicit. We added Appendix Table "Per-Head Rank Under Small Total-Rank Budgets" to clarify the capacity accounting behind the reviewer question. When the total rank is only `r=2` or `r=4`, increasing the number of heads can leave each head with only rank-1 or rank-2 capacity. Orthogonal regularization can encourage those heads to occupy different directions, but it cannot increase the capacity available inside each head. Under few-shot supervision, this creates a trade-off between head diversity, per-head expressiveness, and routing reliability.

We also added an appendix table reporting validation sensitivity to head count, and revised the interpretation from a definitive mechanism to a capacity/routing trade-off hypothesis. In the validation diagnostics, the selected `H=2, r=8, lambda_o=0.03` configuration obtains 91.67 mean validation accuracy, while the tested `H=4, r=8` settings remain close but lower, ranging from 90.63 to 91.08. We therefore describe this as configuration-dependent evidence rather than as proof of a universal degradation law for larger H.

The revised manuscript also points to the likely path for addressing this limitation: future studies should decouple total-rank effects from per-head-rank effects by running fixed-rank-per-head diagnostics and routing-statistics analyses. We therefore do not present H=2 as a universal optimum; we present it as the configuration supported by the current validation setting.

**Manuscript changes:** Appendix Table "Per-Head Rank Under Small Total-Rank Budgets" and Appendix Table "Ramp100 Validation Sensitivity to Head Count" were added. The main ablation discussion now points to both appendix tables, and the orthogonality-weight discussion, parameter-efficiency discussion, head-diversity diagnostics, and Limitations were revised and highlighted in yellow.

---

## Weakness W4 - Missing baselines (MoRE, O-LoRA, DoRA)

> *"It misses key state-of-the-art MoE-LoRA and orthogonal LoRA baselines from top conferences (e.g., MoRE, O-LoRA, DoRA), leading to an insufficiently comprehensive performance comparison."*

**Response:** We agree that the baseline discussion in the original manuscript was incomplete. We revised the Related Work and limitations text to explain the status of these baselines more clearly, and we retained a contextual empirical comparison against established few-shot CLIP baselines.

Tables 1-3 place OrthoAdapt in context by comparing it against established few-shot CLIP baselines: CLIP (zero-shot), CoOp, CoCoOp, CLIP-Adapter, Tip-Adapter-F, PLOT++, KgCoOp, TaskRes, MaPLe, ProGrad, and CLIP-LoRA. Within these tables, the CLIP-LoRA and OrthoAdapt rows are additionally matched at `r=2` and an equal trainable-parameter budget; the equal-parameter control applies to that pair, while the remaining methods are standard baselines reported at their published settings.

MoRE and O-LoRA are important related methods, but their published settings do not directly match our single-task few-shot CLIP protocol. MoRE focuses on adaptive multi-task learning, while O-LoRA is designed around continual-learning interference. A faithful comparison would require a careful port and matched protocol rather than inserting their original numbers into our tables. We therefore discuss them as related work and scope limitations.

For DoRA, we do not report an empirical row because a faithful comparison would require implementing and validating DoRA within the same few-shot CLIP attention-adaptation protocol used for our main experiments. We therefore discuss DoRA as an important related parameterization and as a limitation, rather than claiming that it has been added as a completed baseline. For SingLoRA-CLIP, we now cite the recently published ICCSA/LNCS chapter as the closest predecessor and companion work. We do not merge its published single-seed literature row into the main revised tables because those tables use rerun three-seed paired evidence; instead, we discuss SingLoRA-CLIP in Related Work and use single-head symmetric controls only as within-protocol architectural diagnostics.

**Manuscript changes:** Related Work and Limitations were revised and highlighted in yellow. No new empirical rows were added for baselines that were not evaluated under the matched protocol.

---

## Question Q1 - Low total rank, orthogonality, and number of heads

> *"Could you explain in more detail how the extremely low total rank (r=2/4) and strong orthogonal regularization together limit the number of heads?"*

**Response:** We revised the manuscript to make this interaction explicit and added Appendix Table "Per-Head Rank Under Small Total-Rank Budgets." With a fixed low total-rank budget, increasing the number of heads reduces the capacity assigned to each head. For example, when `r=2` and `H=2`, each head is rank-1; when `r=4` and `H=4`, each head is also rank-1. This can create an under-capacity regime in which each head has too little rank to model useful task-specific variation, while the gating network must still learn reliable routing from very few examples.

We also clarified that the effect of orthogonal regularization is configuration-dependent. Under the corrected validation protocol, the selected configuration is `H=2`, `r=8`, `lambda_o=0.03`. Nearby settings can be competitive, and different regularization strengths are preferred under different rank settings. We therefore do not claim that stronger orthogonality universally improves performance or that it is the primary cause of the observed head-count behavior. The revised text describes the interaction among total rank, per-head capacity, gating reliability, and regularization strength as a trade-off that requires explicit validation.

**Manuscript changes:** Head-count ablation, the new per-head-rank appendix table, Appendix Table "Ramp100 Validation Sensitivity to Head Count", orthogonal-regularization sensitivity, head-diversity diagnostics, and Limitations were revised and highlighted in yellow.

---

## Question Q2 - Multi-task extension

> *"Recent top-conference works (e.g., MoRE) use MoE-LoRA for multi-task adaptation. Why does your work only focus on single-task few-shot classification, and do you plan to extend OrthoAdapt to multi-task scenarios in future work?"*

**Response:** The scope is deliberate. This manuscript studies single-task few-shot CLIP adaptation because it isolates the low-data adaptation behavior of the adapter without introducing multi-task sampling, task-balancing, or retrieval-specific evaluation choices. We agree that multi-task adaptation is a natural extension, especially given the relationship to MoE-LoRA methods, but it requires a separate protocol and is now stated explicitly as future work rather than implied by the present experiments.

**Manuscript changes:** Limitations and Conclusion were revised and highlighted in yellow.

---

## Question Q3 - Larger backbones such as ViT-L/14

> *"Your experiments only use the ViT-B/16 backbone. Would the performance gains of OrthoAdapt still hold when scaling to larger backbones like ViT-L/14 or other VLMs?"*

**Response:** We added a bounded larger-backbone study with CLIP ViT-L/14 on EuroSAT and Caltech101 in the 4-shot setting over three seeds. The comparison uses the same paired protocol as the main revision evidence: CLIP-LoRA r=8 versus OrthoAdapt `H=2, r=8, lambda_o=0.03`. We also added a bounded CLIP ViT-L/14 backbone-scaling table to make this evidence explicit.

The completed ViT-L/14 subset contains 12 experimental rows and 6 paired comparisons. OrthoAdapt uses 1.01M trainable parameters compared with 1.62M for CLIP-LoRA, a 37.5% reduction. The overall paired accuracy difference is +0.21 percentage points, with a 95% CI of [-1.04, +1.46]. The confidence interval crosses zero, so we do not claim that the accuracy gains broadly hold on larger backbones. Instead, we use this result as bounded evidence that the method can be applied to a larger CLIP backbone with substantially fewer trainable parameters. Multi-task learning, retrieval, and broader VLM-family evaluation remain future work.

**Manuscript changes:** Experimental Setup, Backbone Scalability, the new table "Bounded CLIP ViT-L/14 Backbone Scaling Study", Limitations, and Conclusion were revised and highlighted in yellow.

---

# Response to Reviewer #2

We thank the reviewer for the careful technical reading. Several points identified genuine problems in the original submission. We have corrected those issues and narrowed the claims where the revised evidence does not support the original wording.

## Major Concern M1 - The PSD claim does not hold under softmax gating

> *"The central theoretical claim, namely that the gating mechanism escapes the PSD constraint, does not hold as stated. [...] A non-negative combination of PSD matrices is itself PSD. [...] What the gating genuinely contributes is input-dependence."*

**Response:** The reviewer is correct. Under softmax gating, the per-input update `Delta W(x) = sum_h g_h(x) U_h U_h^T` remains a non-negative combination of PSD matrices. The original wording that the gating mechanism "escapes" the PSD constraint was therefore mathematically incorrect.

We have revised the manuscript accordingly. OrthoAdapt is now described as an input-conditioned family of symmetric PSD low-rank updates, not as a mechanism that removes the PSD constraint for an individual input. The value of the gate is input-conditioned routing: different inputs can receive different PSD updates through the learned head weights. This is narrower than the original claim, but it is the claim supported by the architecture actually implemented.

**Manuscript changes:** Abstract, Introduction, Method, and Fig. 2 caption were revised and highlighted in yellow. No new table or figure is needed for this item.

---

## Major Concern M2 - Hyperparameters selected on the test set

> *"The number of heads and the regularization strength are chosen by test accuracy on EuroSAT. [...] Selecting hyper-parameters on the very test set where the main improvements are claimed sits uneasily with the fixed hyper-parameter framing the paper otherwise adopts. I would ask the authors to perform this selection on validation data and to report the results again."*

**Response:** The reviewer is correct that the original presentation did not adequately separate hyperparameter selection from test reporting. We corrected the code and protocol by adding an explicit validation-only selection path and by blocking test-set reporting during sweep mode.

To avoid the concern the reviewer raised, we now report two separate analyses at two rank settings, and we perform all hyper-parameter selection on held-out validation data.

The first is a protocol-defined matched-parameter diagnostic at `r=2`, the default rank of the CLIP-LoRA baseline. Here the goal is a controlled comparison in which OrthoAdapt and CLIP-LoRA use an identical trainable-parameter budget, so that any difference between the two cannot be attributed to a larger adapter. The number of heads (`H=2`) and orthogonality weight (`lambda_o=0.03`) used at this setting are fixed rather than chosen from test accuracy.

The second is the validation-selected configuration. Following a predefined validation rule, we swept head count, rank, and orthogonality weight over EuroSAT and Caltech101 in the 4-shot setting, seeds `{1,2,3}`, evaluating only the validation split and never reporting test accuracy during the sweep. Under this rule the selected configuration is `H=2`, `r=8`, `lambda_o=0.03`, with mean validation accuracy `91.67`. We then report this configuration on the test set. Non-winning candidates were not evaluated on the test set for selection purposes.

We keep a single fixed `H=2` and `lambda_o=0.03` across both rank settings. This is defensible because, at `r=2`, the orthogonality weight has a weak, within-noise effect on validation accuracy: across `lambda_o` in `{0, 0.01, 0.03, 0.05}` the mean validation accuracy spans only `90.125` to `91.000`, and the `{0, 0.03, 0.05}` values lie within `0.05` points of each other. We therefore do not claim that `lambda_o=0.03` was separately validation-optimal at `r=2`; we hold it fixed and note that the specific value in this range does not change the matched-parameter conclusion.

**Manuscript changes:** The evaluation protocol and validation-selection discussion were revised and highlighted in yellow, presenting the `r=2` matched-parameter diagnostic and the `r=8` validation-selected configuration as two separate analyses. We also added validation diagnostics, including the `r=2` lambda-sensitivity range, so the selected configuration and nearby settings are transparent.

---

## Major Concern M3 - Seed protocol contradiction and gains within seed noise

> *"Section 4.1 states that results are averaged over three random seeds, while Table 1 states that a single seed was used. These two statements cannot both be true. [...] Margins of this size cannot be interpreted without variance estimates."*

**Response:** The reviewer is correct. The original manuscript contained an inconsistency in the seed protocol, and small margins should not have been interpreted without uncertainty estimates.

We first fixed the contradiction by separating the original seed-1 table values from the revised paired aggregation. For the matched-parameter literature comparison, we retain the original submitted Table 1-3 CLIP-LoRA and OrthoAdapt values as seed 1 and aggregate them with rerun seeds 2 and 3. For the validation-selected `r=8` comparison, all three seeds come from the ramp100 rerun matrix. We report paired uncertainty estimates for both comparisons.

In the protocol-defined matched-parameter diagnostic at `r=2` (Tables 1-3), OrthoAdapt and CLIP-LoRA are compared at an identical 184,320 trainable-parameter budget using the hybrid 3-seed aggregation described above. The paired deltas are `+0.456` for 1-shot, `+0.428` for 4-shot, and `+0.001` for 16-shot, with an overall paired delta of `+0.295` points (95% CI `[+0.102,+0.488]`) over 72 matched pairs. We report the 16-shot paired delta's 95% CI of `[-0.192,+0.193]` explicitly because it crosses zero, making the 16-shot behavior transparent and showing that OrthoAdapt's advantage concentrates in the lowest-shot regimes. This shows a small average gain at equal parameters, with mixed dataset-level behavior.

At the validation-selected configuration `r=8`, the comparison covers 8 datasets, 3 shot settings, 2 methods, and seeds `{1,2,3}` (144 completed test runs). Pairing is exact by `(dataset, shot, seed)`, yielding 72 matched comparisons between CLIP-LoRA r=8 and OrthoAdapt `H=2, r=8, lambda_o=0.03`. Across these 72 pairs, OrthoAdapt changes accuracy by `+0.356` percentage points, with a 95% CI of `[+0.162,+0.551]`, paired effect size `dz=0.430`, and wins/ties/losses `48/3/21`. The mean paired deltas are `+0.816` for 1-shot, `+0.402` for 4-shot, and `-0.149` for 16-shot. We report the 16-shot paired delta's 95% CI of `[-0.359, +0.061]` explicitly because it crosses zero, making the 16-shot behavior transparent and showing that OrthoAdapt's advantage concentrates in the lowest-shot regimes. OrthoAdapt uses 460,800 trainable parameters compared with 737,280 for CLIP-LoRA r=8, a 37.5% reduction.

We therefore no longer claim broad benchmark-leading performance or uniform superiority. The revised conclusion is that OrthoAdapt shows a modest positive average paired delta while using no more parameters than CLIP-LoRA at the matched setting, and substantially fewer parameters at the validation-selected `r=8` configuration.

**Manuscript changes:** The original 1-shot, 4-shot, and 16-shot literature-comparison values are retained as the seed-1 entries for the matched-parameter table, and paired statistics are added by aggregating them with rerun seeds 2 and 3. The validation-selected `r=8` analysis is reported from the full three-seed ramp100 rerun matrix. Paired statistics, including per-shot confidence intervals, are reported in the manuscript for both the `r=2` and `r=8` analyses. These changes are highlighted in yellow.

---

## Major Concern M4 - Spectral figure inconsistent with the rank budget

> *"With a total rank budget of r = 2, the update matrix can have at most two non-zero singular values [...] Yet the figure shows roughly fifteen smoothly decaying singular values at every layer."*

**Response:** The reviewer is correct that the previous spectral figure was not sufficiently tied to the stated rank budget. The issue was a visualization-choice issue rather than evidence that the trained adapter used rank fifteen. The earlier plotting path computed an SVD on a checkpoint-derived matrix/proxy and displayed a longer prefix of the singular-value curve so that the decay shape could be seen more clearly. However, because this display was not explicitly constrained by the declared rank budget, the figure was easy to read as implying more active singular directions than an `r=2` configuration can have. We therefore agree that the submitted presentation was unclear for a rank-budget diagnostic.

We have regenerated the spectrum diagnostics with rank-aligned plotting. In the corrected diagnostic, an `r=2` comparison displays exactly the first two normalized singular values, matching the rank used in the matched-parameter CLIP-LoRA comparison and the standard-baseline tables. The plotting pipeline now fails closed if a requested `top_k` does not match the declared rank budget. Each diagnostic record also stores the matrix definition, layer, projection, checkpoint metadata, checkpoint hash, configuration, rank budget, and plotted singular values, so the figure can be traced back to the exact analyzed checkpoint.

The revised manuscript figure is anchored on the EuroSAT 4-shot, seed-1, vision-layer-11 `q_proj` panel from the same-parameter `r=2` diagnostic (`eurosat_4shot_seed1_layer11_q_proj.png`). We selected this panel because it directly matches the rank budget raised in the reviewer's concern and the matched-parameter comparison against CLIP-LoRA and the standard few-shot baselines. If a companion `v_proj` panel is retained, it is drawn from the same EuroSAT/4-shot/seed-1/layer-11/`r=2` diagnostic subset. We also regenerated the validation-selected `r=8` diagnostic for internal traceability and appendix-level evidence, but the response to this concern and the manuscript figure are anchored on `r=2`. In both settings, no post-rank numerical tail is shown or interpreted.

To summarize the manuscript diagnostic subset, the selected EuroSAT 4-shot, seed-1, layer-11 `q_proj` panel displays exactly the first two normalized singular values. Across the eight-dataset layer-11, seed-1 report for the same diagnostic setting, the 4-shot query projection has mean stable rank 1.004 for CLIP-LoRA and 1.064 for OrthoAdapt, and the value projection has mean stable rank 1.281 for CLIP-LoRA and 1.563 for OrthoAdapt. These values are reported only as bounded descriptive statistics about the analyzed checkpoints, not as causal evidence for accuracy or robustness.

**Manuscript changes:** The previous spectrum figure was replaced with a regenerated EuroSAT 4-shot, seed-1, vision-layer-11 `q_proj` panel from the matched-parameter `r=2` setting; any retained companion projection panel uses the same diagnostic subset. The corrected curves stop at the rank budget, and the caption and discussion were revised and highlighted in yellow to state the exact checkpoint subset, matrix definition, layer/projection, rank budget, and descriptive role. The manuscript no longer uses the spectrum plot as causal proof; it is treated as a traceable diagnostic of how the analyzed checkpoint uses its low-rank budget.

---

## Minor Comment a - Numerical inconsistencies

> *"The Abstract attributes the 3.6 point robustness gain to severe corruption, while Section 4.4 attributes it to the medium setting. The EuroSAT 4-shot accuracy appears as 88.64 in Figure 5, as 89.06 in Table 3, and as 88.57 in Table 6. In Table 4 the Aircraft result is identical to CLIP-LoRA at 54.97, which I suspect is a copy error. Table 6 uses commas as decimal separators."*

**Response:** We agree with the reviewer. The revised manuscript no longer mixes single-seed table values, exploratory sensitivity values, and final aggregate values as if they were the same quantity. The original Table 1-3 values are now treated consistently as seed-1 entries for the matched-parameter literature comparison, while the final claims use paired aggregate statistics. Legacy exploratory sensitivity tables are explicitly labeled as diagnostics and separated from the final test values.

The EuroSAT 4-shot number no longer takes conflicting meanings. At the matched-parameter setting `r=2`, the hybrid 3-seed aggregate is OrthoAdapt `86.41` and CLIP-LoRA `85.61`; the original `89.06` value is retained only as the seed-1 OrthoAdapt entry in the matched-parameter table. At the validation-selected `r=8`, the full rerun aggregate is OrthoAdapt `85.78` and CLIP-LoRA `84.36`. The other legacy EuroSAT values (88.64 and 88.57) are now labeled as diagnostics rather than final aggregate test results.

The suspected Aircraft copy issue is also clarified. The original `54.97/54.97` value is retained as the seed-1 Aircraft 16-shot entry rather than treated as a copy error. In the matched-parameter hybrid 3-seed aggregate, Aircraft 16-shot is CLIP-LoRA `54.65` and OrthoAdapt `54.63`; at `r=8`, Aircraft 16-shot is CLIP-LoRA `56.99` and OrthoAdapt `56.63`.

The robustness wording has also been corrected. We attempted the paired robustness evaluation over the final Phase 3 checkpoints, but the run did not pass the clean-consistency audit: severity-0 OH-SingLoRA accuracy did not reproduce the corresponding Phase 3 clean test accuracy, indicating an evaluator/checkpoint-loading mismatch. We therefore do not report the robustness numbers from that run and removed the original `+3.6` robustness statement from the main conclusions.

The legacy robustness visualization is retained in the manuscript only as an explicitly labeled qualitative/exploratory illustration, with a caption stating that no quantitative robustness gain is claimed from it. Robustness-specific evaluation is treated as future work unless a clean-gated rerun is completed.

The decimal-separator issue in the legacy ablation table is corrected wherever that table is retained.

**Manuscript changes:** Main result tables, ablation captions, robustness text, the retained qualitative/exploratory robustness visualization caption, and table formatting were revised and highlighted in yellow.

---

## Minor Comment b - SingLoRA baseline and OMoE distinction

> *"Since SingLoRA is the architectural starting point, it deserves to appear as a full baseline in Tables 2 through 4. The distinction from recent work combining mixtures of LoRA experts with orthogonality, such as OMoE in reference 41, also needs a sharper comparison than the current Related Work provides."*

**Response:** We agree that SingLoRA is an important conceptual starting point, and the revised Related Work now explains this relationship more explicitly. Since the original submission, our prior SingLoRA-CLIP study has appeared as a published ICCSA/LNCS chapter. We now cite this work explicitly and position it as the closest predecessor and companion study: SingLoRA-CLIP evaluates a single-matrix symmetric update for few-shot CLIP adaptation, whereas OrthoAdapt studies input-conditioned multi-head routing over symmetric low-rank heads.

At the same time, we do not merge the published SingLoRA-CLIP numbers into the main revised tables. The SingLoRA-CLIP chapter reports a published single-seed literature result, while the revised OrthoAdapt tables use a manifest-controlled three-seed protocol with paired uncertainty estimates. Mixing a prior single-seed literature row with the rerun three-seed rows would make the uncertainty comparison unclear. We therefore cite and discuss SingLoRA-CLIP as the closest predecessor, while restricting paired statistical comparisons to methods rerun under the same revised protocol.

To address the architecture question without overstating the baseline evidence, we additionally use H=1 symmetric-head runs only as validation diagnostics within the current codebase. In the H=1 validation diagnostic over EuroSAT and Caltech101, 4-shot, seeds `{1,2,3}`, `H=1, r=2, lambda_o=0.0` averages `91.33`, and `H=1, r=4, lambda_o=0.0` averages `89.75`. These diagnostic runs are not presented as a replacement for the published SingLoRA-CLIP study; they simply help interpret the effect of moving from one symmetric head to routed multi-head adaptation under the current validation protocol.

We also sharpened the OMoE/MoE-LoRA comparison by distinguishing target setting, base parameterization, and the role of orthogonality.

**Manuscript changes:** Related Work now cites and positions the published SingLoRA-CLIP chapter as the closest predecessor/companion work, and the OMoE/MoE-LoRA comparison was sharpened by distinguishing target setting, base parameterization, and the role of orthogonality. Architectural diagnostics, including the `H=1` validation diagnostic values, and Limitations were revised and highlighted in yellow. No additional main-table baseline row was added for results not evaluated under the revised three-seed matched protocol.

---

## Minor Comment c - Undefined ramp-up function and epsilon

> *"The ramp-up function u(t) in Eqs. 3 and 6 is never defined, and the constant epsilon mentioned after Eq. 11 does not actually appear in the equation."*

**Response:** The reviewer is correct. We added an explicit definition of the ramp-up factor: `u(t)=min(1,t/T)`, where `T` is the ramp-up step budget. In the final experiments, `T=100`. We also removed the dangling epsilon reference because the implemented orthogonality loss does not include such a term.

**Manuscript changes:** The method equations and surrounding text were revised and highlighted in yellow. No new table or figure is required.

---

## Minor Comment d - Missing ImageNet, SUN397, and StanfordCars

> *"The standard CLIP few-shot suite comprises eleven datasets, including ImageNet, SUN397, and StanfordCars. The authors should explain why these three were omitted."*

**Response:** We thank the reviewer for pointing this out. We agree that ImageNet, SUN397, and StanfordCars are part of the broader standard CLIP few-shot suite. In the revised manuscript, we no longer describe our benchmark as the complete 11-dataset CLIP suite. We explicitly define it as an eight-dataset few-shot evaluation suite.

All empirical claims in the revised paper are restricted to the eight datasets with complete rerun evidence: Aircraft/FGVC, EuroSAT, Food101, OxfordPets, OxfordFlowers, Caltech101, DTD, and UCF101. ImageNet, SUN397, and StanfordCars are now listed as future work rather than implied coverage.

**Manuscript changes:** Experimental Setup, figure captions, and Limitations were revised and highlighted in yellow, and ImageNet, SUN397, and StanfordCars are now explicitly listed as future work in the Limitations section. No new table or figure is required.

---

## Minor Comment e - Code release and exact seeds

> *"Given how small the margins are, releasing the code and the exact seeds would do a great deal for the credibility of the results, and I warmly encourage the authors to do so."*

**Response:** We agree. The revised reproducibility statement commits to releasing the code, exact seeds `{1,2,3}`, run scripts, validation/test run records, dataset split records, aggregation/statistical scripts, environment information, and instructions needed to reproduce the reported tables. Adapter checkpoints will be released where storage and licensing constraints allow; otherwise, the released scripts and run records will be sufficient to reproduce them.

**Manuscript changes:** Data/Code Availability was revised and highlighted in yellow. No new table or figure is required.

---

We hope these revisions satisfactorily address the reviewers' concerns. We are particularly grateful to Reviewer 2 for the precise technical comments on the PSD framing, seed protocol, validation/test separation, and spectral figure. These comments identified genuine flaws that the revision now corrects.

Sincerely,

Hoang Ngoc Tran (on behalf of all authors)
Department of Artificial Intelligence, FPT University, Can Tho, Vietnam
hoang2531992@gmail.com
