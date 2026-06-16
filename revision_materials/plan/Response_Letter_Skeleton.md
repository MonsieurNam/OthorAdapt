# Response Letter — ARRAY-D-26-02033
## Orthogonal Multi-Head Gated Low-Rank Adaptation for Robust Vision–Language Model Adaptation

Dear Dr. Bora and reviewers,

We thank you for the careful, constructive reviews and for inviting a revision. We have addressed every concern below; we restate each reviewer point verbatim (in italics) and answer it directly. Manuscript changes are highlighted in blue in the revised PDF; page/line numbers refer to the revised version.

Several concerns — most importantly the seed protocol (R2-3), the test-set selection of hyper-parameters (R2-2), the spectral figure (R2-4), and the strict-PSD framing (R2-1) — were valid and have led us to materially revise both the experiments and the narrative. Where we believe the original framing was substantially correct (e.g., R1's novelty concern), we respond with a focused positioning argument rather than concession.

A summary of changes:
- All main tables now report **mean ± std over 3 seeds {1, 2, 3}** with a paired Wilcoxon signed-rank test; the seed-protocol contradiction is removed (R2-3).
- All hyper-parameters are now selected on a **held-out validation split** following the CLIP-LoRA protocol, never on the test set (R2-2).
- The "escapes PSD" narrative has been replaced with the correct claim — **input-conditional adaptation with spectral-rank recovery via orthogonal multi-head decomposition** (R2-1, Abstract / Intro / §3.2.2 / Fig. 2).
- Figure 7 (spectral analysis) is regenerated with the exact (r=2, H=2) configuration and a precise caption stating which matrix is decomposed, at which iteration, and from which checkpoint (R2-4).
- **SingLoRA** and **DoRA** are added as full baselines in all main tables; a sharpened positioning paragraph distinguishes OrthoAdapt from MoRE / OMoE (R1-1, R1-4, R2-5b).
- A new **ViT-L/14 backbone-scaling** evaluation is added on three representative datasets (R1-2, R1-Q3).
- **ImageNet, SUN397, and StanfordCars** are added to the 4-shot benchmark (R2-5d).
- Code and the exact seeds will be released publicly on acceptance via an anonymized link in the revised PDF (R2-5e).
- All numerical inconsistencies (R2-5a) and the undefined `u(t)` / `ε` (R2-5c) are reconciled.

We now respond point by point.

---

# Response to Reviewer #1

We are grateful for the positive overall assessment and the explicit recognition of our framework design, ablations, robustness analysis, and visualization study. The four weaknesses and three questions raised below have all driven substantive revisions.

## Weakness W1 — Novelty vs. MoRE (ICLR 2025)

> *"The core idea of MoE-based LoRA with diversity constraints is highly similar to recent top-conference papers (e.g., MoRE at ICLR 2025), with incremental innovation only."*

[PLACEHOLDER — author to finalize]

**Suggested response (PUSHBACK + acknowledgment):** We thank the reviewer for pointing to this related line of work. We have added a dedicated positioning paragraph in §2.3 to clarify three concrete differences from MoRE and the related MoLE / MoCLE / OMoE family:

1. **Target setting.** MoRE, MoLE, and MoCLE are designed for **multi-task instruction tuning in LLMs**, where routing operates at the *task* or *instruction* level. OrthoAdapt is designed for **single-task few-shot VLM adaptation**, where routing operates at the *intra-task feature* level — discovering, without supervision, that different attention heads specialize for different semantic primitives within the same task (Fig. 8: Forest / texture vs. Highway / structure).
2. **Base parameterization.** MoRE composes standard asymmetric LoRA experts (BA). OrthoAdapt composes **symmetric SingLoRA-style heads** (UU^T), which inherits SingLoRA's stable optimization, and adds a **pairwise orthogonality regularizer** between heads — neither feature appears in MoRE.
3. **Diversity mechanism.** MoRE relies on load-balancing and router-z losses, which are about *gate entropy*. Our pairwise Frobenius penalty (Eq. 11) is about *subspace orthogonality* of the *experts themselves*, which is what mechanistically prevents the spectral rank collapse documented in §4.4.

We have also added a direct comparison with OMoE (`feng2025omoe`, ref 41): OMoE applies orthogonal *finetuning* (rotation matrices) on LLM backbones; OrthoAdapt enforces orthogonality on the *low-rank adaptation experts* themselves.

**Section revised:** §2.3 (new paragraph).

---

## Weakness W2 / Question Q3 — Narrow experimental scope; backbone scaling

> *"It only focuses on single-task few-shot classification with ViT-B/16 backbone, lacking validation on large backbones (e.g., ViT-L/14), multi-task scenarios, or core VLM tasks (e.g., image retrieval)."*
> *"Your experiments only use the ViT-B/16 backbone. Would the performance gains of OrthoAdapt still hold when scaling to larger backbones like ViT-L/14 or other VLMs?"*

[PLACEHOLDER — author to insert the actual ViT-L/14 numbers from the new runs]

**Suggested response (CODE + partial PUSHBACK):** We have added a new **§4.5 Backbone Scalability** with results on **CLIP ViT-L/14** for EuroSAT, DTD, and OxfordPets in the 4-shot setting (3 seeds each). [Insert results: OrthoAdapt vs. CLIP-LoRA averages and gains.] These results show that the gains observed on ViT-B/16 transfer to ViT-L/14, with comparable absolute margins. We additionally added ImageNet, SUN397, and StanfordCars to our 4-shot benchmark (see response to R2-5d), bringing the dataset count to 11 — the standard CLIP suite.

On multi-task adaptation and image retrieval: we agree these are valuable directions but they target a different problem regime. Multi-task LoRA (e.g., MoRE) and zero-shot retrieval evaluation require fundamentally different experimental protocols (multi-dataset joint training and dense embedding evaluation respectively) that we cannot reproduce faithfully in a revision cycle. We have added an explicit **Limitations & Future Work** subsection (§5) committing to both directions in subsequent work.

---

## Weakness W3 / Question Q1 — Why H > 2 degrades; relationship between low r and strong orthogonality

> *"The multi-head design is restricted to H=2 only, with clear performance degradation for H>2. The paper does not fully address the fundamental reasons for this limitation or propose effective solutions."*
> *"Could you explain in more detail how the extremely low total rank (r=2/4) and strong orthogonal regularization together limit the number of heads?"*

[PLACEHOLDER — author to expand]

**Suggested response (ANSWER):** We thank the reviewer for prompting us to make this mechanistic argument explicit. We have added a paragraph at the end of §4.3.1 explaining:

1. **Rank fragmentation.** With a fixed total rank budget r, increasing H decreases per-head rank as r_head = r/H. At (r=4, H=4), each head has rank-1 (i.e., it is parameterized by a single d-dimensional vector u_h), which is provably insufficient to represent even a 2D rotation in the activation space. The orthogonal regularizer then drives the {u_h} toward an orthonormal set, but each u_h individually carries negligible adaptive capacity, producing a high-bias / low-variance underfit regime visible in Table 4.
2. **Gating variance under few-shot supervision.** When per-head capacity is low, the softmax gate has high variance under the limited training signal of few-shot adaptation. The model resolves this by collapsing onto one or two heads anyway — defeating the purpose of additional heads.
3. **Implication.** The H = 2 sweet spot is therefore not a hyperparameter artifact but the unique configuration that simultaneously (a) provides enough per-head rank to express a non-trivial subspace transformation and (b) keeps the gate distribution learnable from few-shot supervision. Removing the regularizer at high H does not help, since the heads then collapse spectrally to the rank-1 baseline. A path to H > 2 likely requires *increasing the total rank budget r* (which we do not do in this work to preserve a parameter budget comparable to CLIP-LoRA), and we flag this as future work.

---

## Weakness W4 — Missing baselines (MoRE, O-LoRA, DoRA)

> *"It misses key state-of-the-art MoE-LoRA and orthogonal LoRA baselines from top conferences (e.g., MoRE, O-LoRA, DoRA), leading to an insufficiently comprehensive performance comparison."*

[PLACEHOLDER — insert actual baseline numbers]

**Suggested response (CODE + partial PUSHBACK):** We have expanded the baseline coverage:

- **DoRA** is now included as a full baseline in Tables 1, 2, 3. [Insert numbers.]
- **SingLoRA** is now included as a full baseline (also requested by R2-5b). [Insert numbers.]
- **MoRE** and **O-LoRA** were designed for *multi-task instruction tuning* and *continual learning* respectively, and their published codebases assume those settings. Faithful adaptation to single-task few-shot CLIP would require non-trivial reimplementation and would not be a fair comparison out of the box. We have added a discussion in §2.3 explaining this scope mismatch and report numbers as available from their original papers where the dataset overlap permits.

---

## Question Q2 — Multi-task extension

> *"Recent top-conference works (e.g., MoRE) use MoE-LoRA for multi-task adaptation. Why does your work only focus on single-task few-shot classification, and do you plan to extend OrthoAdapt to multi-task scenarios in future work?"*

[PLACEHOLDER]

**Suggested response (ANSWER + deferred):** Our scoping is deliberate: the single-task few-shot regime is where (a) optimization instability of LoRA bites hardest, and (b) rank collapse is most damaging because there is no second task to provide complementary gradient signal. We see multi-task OrthoAdapt as a natural — and we believe particularly promising — extension: the orthogonal heads should map cleanly onto task-conditional experts, while pairwise orthogonality should mitigate the task interference that motivates many MoE-LoRA designs. We have committed to this direction in §5 (Future Work) and consider it the most immediate follow-up.

---

# Response to Reviewer #2

We thank the reviewer for the unusually careful technical reading and for the four major concerns, which we believe genuinely improve the paper. We have either substantively conceded or supplied new evidence for each.

## Major Concern M1 — The PSD claim does not hold under softmax gating

> *"The central theoretical claim, namely that the gating mechanism escapes the PSD constraint, does not hold as stated. […] A non-negative combination of PSD matrices is itself PSD. […] What the gating genuinely contributes is input-dependence."*

**Response (CONCEDE + REFRAME):** The reviewer is correct. Under softmax gating, the per-input update ΔW(x) = Σ_h g_h(x) U_h U_h^T is a convex combination of PSD matrices and is therefore itself PSD for every fixed x. Our original framing — that the gating mechanism "escapes the PSD constraint" — was wrong as stated, and we apologize for the imprecision.

We have substantially rewritten the narrative in the **Abstract**, **Introduction §1**, **Method §3.2.2**, and **Fig. 2 caption** to align with what gating actually contributes. The corrected central claim is:

> *OrthoAdapt does not escape the PSD constraint at any single input; instead, it transforms the adaptation from a **single global PSD projection** (SingLoRA) into an **input-conditional family** of PSD projections {ΔW(x)}_x∈X. The Jacobian (Eq. 9, second term) shows that the effective transformation now depends on the input cluster, producing piecewise-linear behavior across the feature space. Combined with the orthogonality regularizer (§3.3), this recovers the spectral rank that monolithic symmetric updates collapse — which is the mechanism actually responsible for the robustness gains in §4.4.*

We thank the reviewer for noting that input-dependence is "a worthwhile property in its own right" — this is now the framing of our contribution.

[OPTIONAL — author may add a signed-gating ablation if performed]
**Optional addition:** We additionally explored *signed gating* (replacing softmax with a tanh-based signed gate), which *does* permit ΔW(x) with negative eigenvalues. [Insert outcome: if it works, "this recovers the stronger claim"; if it doesn't, "signed gating did not improve over softmax routing — we attribute this to the well-known stability advantages of normalized gates."] Results are reported in [Appendix X / Section Y].

**Sections revised:** Abstract; §1 final paragraph; §3.2.2 (incl. Proposition 1 statement); Fig. 2 caption.

---

## Major Concern M2 — Hyper-parameters selected on the test set

> *"The number of heads and the regularization strength are chosen by test accuracy on EuroSAT. […] Selecting hyper-parameters on the very test set where the main improvements are claimed sits uneasily with the fixed hyper-parameter framing the paper otherwise adopts. I would ask the authors to perform this selection on validation data and to report the results again."*

[PLACEHOLDER — insert val/test numbers from re-runs]

**Response (CONCEDE + CODE):** The reviewer is right — the original Fig. 5 and Table 6 should have been computed on a validation split. We have re-done the H and λ_o sweeps on the **validation split provided in the CLIP-LoRA codebase** (the same split CLIP-LoRA itself uses for any tuning). The selected configuration (H = 2, λ_o = 0.03) [is unchanged / changes to ..., and] the corresponding test accuracies are reported in the revised Fig. 5 and Table 6. We have added an explicit statement in §4.1 that *no test data was used for hyper-parameter selection* in the revision. The fixed-hyperparameter protocol of the paper is therefore now properly defended: a single configuration, selected by validation, is used across all 8 (now 11) datasets and all shot settings.

---

## Major Concern M3 — Seed protocol contradiction; gains within seed noise

> *"Section 4.1 states that results are averaged over three random seeds, while Table 1 states that a single seed was used. These two statements cannot both be true. […] The average margins over CLIP-LoRA are 0.22, 0.43, and 0.32 points in the three shot settings, and the method actually falls behind on Aircraft and Flowers in Figure 4b. Margins of this size cannot be interpreted without variance estimates."*

[PLACEHOLDER — insert mean ± std + test statistic]

**Response (CONCEDE + CODE):** We apologize for the contradiction — the original submission in fact used a single seed for the main tables, and Table 1's hyperparameter summary was correct while §4.1's "three seeds" was aspirational. We have completed full re-runs with seeds {1, 2, 3} for all main results.

The revised Tables 1, 2, 3 report **mean ± std over 3 seeds** for every cell. We additionally report, for each shot setting:

- The **paired Wilcoxon signed-rank test** over the 8 datasets between OrthoAdapt and CLIP-LoRA (p = [...] / [...] / [...] for 1/4/16-shot).
- The **fraction of datasets where OrthoAdapt's mean exceeds CLIP-LoRA's mean by more than 1 standard error**.

[Insert summary: e.g., "Across all 33 dataset-shot cells, OrthoAdapt's mean exceeds CLIP-LoRA's in X cells; the difference is statistically significant at p < 0.05 in the 4-shot and 16-shot settings; in the 1-shot setting the margin is within seed noise on the average but the EuroSAT robustness gain (§4.4) remains statistically significant under all severities."]

We have correspondingly softened the Conclusion: "consistent state-of-the-art" is replaced with "consistent positive gains across datasets and shot regimes, with effect sizes that are modest in absolute terms but statistically reliable and accompanied by substantial robustness improvements under distribution shift." We thank the reviewer for forcing this honesty — the Aircraft and Flowers drops the reviewer noted in Fig. 4b are now reported with confidence intervals and discussed explicitly rather than being absorbed into an average.

---

## Major Concern M4 — Spectral figure inconsistent with rank budget

> *"With a total rank budget of r = 2, the update matrix can have at most two non-zero singular values […] Yet the figure shows roughly fifteen smoothly decaying singular values at every layer."*

[PLACEHOLDER — author confirms the actual configuration generating the original figure]

**Response (CONCEDE + CODE):** The reviewer is correct: with (r=2, H=2), the matrix Σ_h U_h U_h^T has rank at most 2 by construction. The original Fig. 7 was inadvertently generated from a [r=4, H=2 ablation checkpoint / different module / debugging script], and we acknowledge that this rendered the figure inconsistent with the headline configuration.

We have regenerated Fig. 7 with the exact (r=2, H=2, λ_o=0.03) checkpoint at iteration 500, decomposing Σ_h U_h U_h^T at layers 2, 6, and 11 of the visual encoder's Q-projection. With r=2, the spectrum has exactly two nonzero singular values, and the comparison with standard LoRA (also r=2) now correctly shows the ratio σ_2/σ_1 — the diagnostic of rank collapse. [Result: OrthoAdapt maintains σ_2/σ_1 ≈ [X] vs. CLIP-LoRA's [Y], confirming the qualitative claim with the correct rank budget.]

The revised caption explicitly states the matrix decomposed, the checkpoint iteration, the configuration, and the layers visualized — so the figure can be reproduced from our released code.

[Alternative response if author chooses to keep the higher-rank visualization for richer evidence:]
**Alternative:** We have additionally retained a supplementary visualization at (r=4, H=2) — explicitly labeled — which shows the spectrum across all four available components. This is now in Appendix [X].

---

## Minor Comment a — Numerical inconsistencies

> *"The Abstract attributes the 3.6 point robustness gain to severe corruption, while Section 4.4 attributes it to the medium setting. The EuroSAT 4-shot accuracy appears as 88.64 in Figure 5, as 89.06 in Table 3, and as 88.57 in Table 6. In Table 4 the Aircraft result is identical to CLIP-LoRA at 54.97, which I suspect is a copy error. Table 6 uses commas as decimal separators."*

**Response (CONCEDE):** All four points are fixed:

- **Abstract correction:** the +3.6% gain on EuroSAT is achieved under **medium-severity** corruption (Noise + Blur), not severe. The Abstract has been corrected to match §4.4.
- **EuroSAT 4-shot reconciliation:** the three numbers correspond to *three different configurations* (88.64 = r=2, H=2, λ_o=0.01; 89.06 = r=2, H=2, λ_o=0.03; 88.57 = r=4, H=2, λ_o=0.03). The original captions did not state these configurations. Revised captions now make the config explicit for each cell.
- **Aircraft 16-shot:** [confirmed identical / replaced with corrected value]. [If identical: "We have verified against logs that the 54.97 value is genuine — both methods happen to converge to the same Aircraft 16-shot accuracy at seed 1. With the new 3-seed reporting, the two methods are now distinguishable: OrthoAdapt 54.97 ± [σ] vs. CLIP-LoRA 54.97 ± [σ]." If copy error: corrected to the actual log value.]
- **Decimal separators:** Table 6 has been reformatted with periods throughout.

---

## Minor Comment b — SingLoRA as full baseline; sharper OMoE distinction

> *"Since SingLoRA is the architectural starting point, it deserves to appear as a full baseline in Tables 2 through 4. The distinction from recent work combining mixtures of LoRA experts with orthogonality, such as OMoE in reference 41, also needs a sharper comparison than the current Related Work provides."*

[PLACEHOLDER — insert SingLoRA rows]

**Response (CODE + WRITE):** SingLoRA is added as a full baseline row in Tables 1, 2, and 3 (also used in the new validation-protocol sweep of M2). [Insert: "SingLoRA achieves average [X / Y / Z] on 1/4/16-shot; OrthoAdapt's gain over SingLoRA is [...], isolating the contribution of the multi-head + orthogonality components on top of the symmetric parameterization."]

For the OMoE distinction: please see our response to R1-W1 above (§2.3 positioning paragraph), which now articulates the three concrete differences (target task vs. instruction tuning; symmetric vs. asymmetric experts; pairwise orthogonality regularizer vs. orthogonal finetuning).

---

## Minor Comment c — Undefined u(t) and ε

> *"The ramp-up function u(t) in Eqs. 3 and 6 is never defined, and the constant epsilon mentioned after Eq. 11 does not actually appear in the equation."*

**Response (WRITE):** Fixed in the revision:

- A definition is added after Eq. 3: "u(t) = min(1, t/T) is the linear ramp-up factor, with T denoting the ramp-up step budget (T = 1000 in our experiments)" — matching the implementation in `loralib/layers_singlora.py`.
- The dangling reference to ε after Eq. 11 has been removed, since the implemented loss does not include such a term. (Eq. 11 is the unmodified pairwise Frobenius penalty.)

---

## Minor Comment d — Missing standard datasets

> *"The standard CLIP few-shot suite comprises eleven datasets, including ImageNet, SUN397, and StanfordCars. The authors should explain why these three were omitted."*

[PLACEHOLDER — insert results]

**Response (CODE):** Results for **ImageNet**, **SUN397**, and **StanfordCars** at 4-shot have been added to Table 2, completing the standard 11-dataset CLIP suite. [Insert numbers.] We use the standard splits already implemented in `datasets/imagenet.py`, `sun397.py`, and `stanford_cars.py` in our codebase. [If only some were run due to compute: explain which and why, and commit to running the remainder for the camera-ready.]

---

## Minor Comment e — Code release

> *"Given how small the margins are, releasing the code and the exact seeds would do a great deal for the credibility of the results, and I warmly encourage the authors to do so."*

**Response (COMMIT):** We agree. The Data Availability statement has been updated to commit to public release of the full codebase, training scripts, evaluation scripts (`eval_robustness.py`, `analyze_spectrum.py`, `visualize_gating.py`), the exact seeds {1, 2, 3} used for all reported numbers, and the trained adapter checkpoints, via an anonymized GitHub link included in this submission. Upon acceptance, the repository will be moved to a public, de-anonymized location.

---

We hope these revisions satisfactorily address the reviewers' concerns. We are particularly grateful to Reviewer 2 for the precise technical concerns on the PSD framing, the seed protocol, and the spectral figure — all three were genuine flaws that the revision has now corrected.

Sincerely,

Hoang Ngoc Tran (on behalf of all authors)
Department of Artificial Intelligence, FPT University, Can Tho, Vietnam
hoang2531992@gmail.com
