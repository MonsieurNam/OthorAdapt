# Author Decisions

This file records author-level decisions that gate the Master Revision Plan. Do not start experiments that depend on these choices until the relevant decision is filled.

## 1. Technical Decisions

### 1.1 Validation Rule For Hyperparameter Selection
- **Decision:** Selection metric is the unweighted mean validation accuracy across EuroSAT and Caltech101, averaged over seeds {1,2,3} when feasible; if only one validation seed is used for compute reasons, this will be explicitly documented as a limitation.
- **Candidates:** Executed frozen sweep uses `H` in {2,4}, `r` in {2,4,8} where divisible by H, and `lambda_o` in {0, 0.01, 0.03, 0.05}; this yields 20 candidate configurations and 120 validation runs across EuroSAT/Caltech101, 4-shot, seeds {1,2,3}.
- **Selected configuration:** The previous frozen selection was `H=2`, `r=4`, `lambda_o=0.0` in `revision_materials/results/selected_config.md`, but it was produced before the ramp schedule mismatch was found. The revised ramp100 validation sweep selected `H=2`, `r=8`, `lambda_o=0.03` in `revision_materials/results/selected_config_ramp100.md`; Phase 3 main must use this ramp100-selected winner.
- **Tie-break rule:** Ties will be broken by lower parameter count, then lower `lambda_o`.
- **Rationale:** EuroSAT represents a domain-shifted remote-sensing dataset, while Caltech101 represents general object diversity. Using only validation splits explicitly avoids test-set overfitting. Pre-registering candidates and tie-breaks prevents post-hoc selection accusations.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Must be coded into `selection_protocol.yaml` and frozen before testing. No test metrics for non-winning candidates will be computed or logged.

### 1.2 Iteration Semantics
- **Decision:** Keep internal logic as `500 * shots` (i.e., 500, 2000, 8000 steps).
- **Rationale:** This follows the inherited CLIP-LoRA-style training-budget convention implemented in the current base code (`total_iters = args.n_iters * args.shots`, with `--n_iters 500`). Because OrthoAdapt is positioned as a drop-in extension / upgrade of CLIP-LoRA, the revision should preserve the same shot-scaled optimization budget for CLIP-LoRA, SingLoRA, and OrthoAdapt rather than introducing a new method-specific schedule. This is a protocol-control decision for fair comparison, not a new OrthoAdapt hyperparameter. If the manuscript explicitly cites the original CLIP-LoRA implementation as the source of this convention, the corresponding upstream code or paper text must be verified and referenced; otherwise use the safer wording "inherited CLIP-LoRA-style training loop used by our codebase."
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** The manuscript must be explicitly rewritten to clear the ambiguity: "All methods were trained using the inherited CLIP-LoRA-style training-budget convention, where the base iteration count is 500 and total steps scale with shot count, yielding 500, 2000, and 8000 optimization steps for the 1-, 4-, and 16-shot settings, respectively."

### 1.3 Seed Count
- **Decision:** 3 seeds {1, 2, 3} for all Tier-A main-table runs; Tier-B diagnostics will use 3 seeds where feasible and otherwise be explicitly marked partial.
- **Rationale:** Balances the minimum requirement for paired statistical tests with the realistic 30-day compute budget.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** All Tier-A tables will report mean +/- std and paired uncertainty estimates over matched seed/dataset splits. Claims of consistency or superiority will be weakened if CIs do not support them.

### 1.4 DoRA Inclusion
- **Decision:** Skip DoRA implementation; add to Limitations.
- **Rationale:** DoRA will be discussed conceptually and listed as a limitation because a matched implementation in the custom CLIP attention path is not available within the revision window. The main empirical baseline remains CLIP-LoRA, which is public, citable, and directly matched to the few-shot CLIP setting.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Reframe the response letter to address DoRA via literature comparison rather than direct empirical comparison.

### 1.5 ImageNet Inclusion
- **Decision:** Defer ImageNet, SUN397, and StanfordCars for this revision package; define the revised benchmark as an 8-dataset few-shot suite.
- **Rationale:** No verified manifest-backed results exist for ImageNet, SUN397, or StanfordCars within the current revision evidence package. Adding partial or unsupported rows would weaken the response. The stronger and more honest answer is to restrict claims to the 8 datasets with complete Phase 3 ramp100 evidence and list the missing standard-suite datasets as future work.
- **Date / Decided by:** 2026-07-04 / HN Tran
- **Impact:** The manuscript must not claim the complete 11-dataset CLIP suite. Reviewer 2 Minor d should be answered as a scope clarification and limitation, not as newly completed evidence.

### 1.6 Code Release Plan
- **Decision:** Anonymized GitHub link on submission.
- **Rationale:** Directly addresses R2's credibility concern during the review phase.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Release will include cleaned scripts, exact configs, seed list, split hashes, environment file, result manifests, aggregation/statistical scripts, and instructions to reproduce all tables/figures. Adapter checkpoints will be released if storage and licensing constraints permit; otherwise scripts and manifests will be sufficient to reproduce them.

### 1.7 SingLoRA-CLIP Baseline Inclusion
- **Decision:** Exclude SingLoRA-CLIP from the Tier-A Phase 3 main matrix. Phase 3 will compare CLIP-LoRA and OrthoAdapt only: 8 datasets x 3 shots x 3 seeds x 2 methods = 144 runs.
- **Rationale:** The citable SingLoRA work is a general PEFT preprint, while the SingLoRA-CLIP variant in this repository is an internal unpublished CLIP adaptation still under investigation. Including it as a main reviewer-facing baseline would mix a non-public, unstable research variant into the acceptance-critical table and could reduce reproducibility. CLIP-LoRA is the appropriate direct baseline because it is a public few-shot VLM/CLIP method and explicitly positions itself as a strong baseline for evaluating progress in few-shot VLM adaptation.
- **Date / Decided by:** 2026-06-18 / HN Tran
- **Impact:** Update `phase3_main_protocol.yaml`, `phase3_main_commands.sh`, and server instructions to 144 runs. Mention SingLoRA-CLIP only as internal exploratory/future work unless it is separately published with stable code, protocol, and citations.

### 1.8 Dual-Configuration Main-Table Strategy (r=2 matched-rank vs r=8 validation-selected)
- **Decision:** Present two distinct main-result tables that serve two different purposes, and distinguish them explicitly throughout the manuscript:
  - **Table A - Matched-rank literature comparison (r=2, H=2, lambda_o=0.03).** Restore the full literature suite (CLIP zero-shot, CoOp M=4/16, CoCoOp, CLIP-Adapter, Tip-Adapter-F, PLOT++, KgCoOp, TaskRes, MaPLe, ProGrad, CLIP-LoRA) exactly as in the original submission's Tables 1-3. The `r=2` setting is the rank that **matches the CLIP-LoRA baseline's default rank**; it is a comparison axis, not a tuned hyper-parameter. This table answers R1-W1 (novelty/positioning) and R1-W4 (baseline completeness) and preserves the "we compete against the field at equal rank" story that the revised head-to-head-only tables had removed.
  - **Table B - Validation-selected configuration + parameter efficiency (r=8, H=2, lambda_o=0.03).** Keep the current head-to-head CLIP-LoRA r=8 vs OrthoAdapt r=8 tables and emphasize the 37.5% trainable-parameter reduction. This answers R2-M2 (validation-only selection) and supports Q3/backbone-scaling.
- **Data sourcing:**
  - Table A literature rows keep the **original cited numbers** from the previous submission (CoOp...PLOT++, etc.).
  - Table A CLIP-LoRA r=2 and OrthoAdapt r=2,H=2,lambda_o=0.03 rows use a **hybrid 3-seed aggregation**: seed 1 is retained from the original submitted Tables 1-3, while seeds 2 and 3 come from Phase 3B same-parameter reruns (`phase3b_same_param_results.jsonl` for CLIP-LoRA r=2; `phase3b_same_param_ramp100_results.jsonl` for OrthoAdapt r=2). Both methods remain at 184,320 trainable parameters. This preserves the author-approved original table as the seed-1 record while adding paired uncertainty from two additional matched seeds.
  - Table B rows use the Phase 3 main ramp100 manifest (`phase3_main_ramp100_results.jsonl`).
- **Rationale:** `r=2` and `r=8` are two points on a **rank axis**, not two independently tuned models. The original design fixed `r=2` to match CLIP-LoRA's rank so that gains could not be attributed to a larger adapter budget; the validation scan later identified `r=8` as the higher-rank configuration. Under the hybrid Table A aggregation, the r=2 overall paired delta is +0.295 pp with 95% CI [0.102,0.488]; under the full ramp100 rerun, the r=8 overall paired delta is +0.356 pp with 95% CI [0.162,0.551]. Keeping r=2 for the literature table therefore preserves the comprehensive comparison R1 asked for while keeping the claim modest and paired.
- **Framing rule for R2-M2 defense:** State that **Table B uses the validation-selected configuration** `H=2,r=8,lambda_o=0.03` from the ramp100 validation sweep. For **Table A**, `r=2` is the baseline-matching rank axis, while `H=2` and `lambda_o=0.03` are held fixed from the original/frozen protocol for consistency with the matched-rank literature comparison. Do **not** claim that `lambda_o=0.03` was independently selected at `r=2`; the r=2 lambda sensitivity is weak/within-noise and is documented separately in Decision 1.9. Do not describe r=2 as separately tuned.
- **Date / Decided by:** 2026-07-04 / HN Tran
- **Impact:** Restore Table A (literature suite, r=2) alongside the existing Table B (r=8) in `cas-sc-template.tex`; update the Reviewer 1 (W1/W4) and Reviewer 2 (M2/a) responses to reference both tables, the matched-rank framing, and the hybrid seed-1-original aggregation for Table A. The original Aircraft 16-shot `54.97/54.97` value is retained as the seed-1 table value rather than treated as a copy error; the corresponding 3-seed hybrid means are CLIP-LoRA 54.65 and OrthoAdapt 54.63.

### 1.9 Fixed lambda_o Across Rank Points (no r=2 re-tuning; lambda is weak/configuration-dependent at r=2)
- **Decision:** Use a single fixed `lambda_o=0.03` for both the r=2 and r=8 configurations. Do **not** rerun the r=2 matrix with `lambda_o=0.05`, even though 0.05 is nominally the top validation value at r=2.
- **Evidence (ramp100 validation, r=2, H=2, EuroSAT+Caltech101, 4-shot, seeds {1,2,3}, mean validation accuracy):**
  - lambda_o=0.05 -> 91.000
  - lambda_o=0.00 -> 90.958
  - lambda_o=0.03 -> 90.958
  - lambda_o=0.01 -> 90.125
- **Rationale:** At `r=2,H=2` the orthogonality weight is **weak and configuration-dependent**: the full sweep spans only 90.125-91.000 (< 0.9 points), and the 0.00 / 0.03 / 0.05 values lie within 0.05 points of each other, well inside seed noise. The `lambda_o=0.03` value is the one selected by the validation protocol at the `r=8` grid, where it is the genuine winner; applying the same fixed value at r=2 is consistent with the fixed-hyper-parameter framing and avoids per-rank cherry-picking. Re-running the entire r=2 matrix to switch 0.03 -> 0.05 would change the matched-rank comparison by a statistically meaningless margin and is therefore not justified.
- **Honesty guardrail:** Do **not** claim that validation selected `lambda_o=0.03` at r=2. The correct statement is that 0.03 is the validation-selected value at r=8 and is held fixed across rank points, and that at r=2 the lambda effect is weak/within-noise so the specific choice among {0, 0.03, 0.05} does not affect the conclusion.
- **Date / Decided by:** 2026-07-04 / HN Tran
- **Impact:** Add a short validation-sensitivity sentence (with the four r=2 lambda values or a range statement) to the ablation/appendix text; ensure the Reviewer 2 M2 response uses the fixed-lambda / weak-effect wording rather than an r=2 selection claim.


## 2. Administrative / Publishing Decisions

### 2.1 AI Disclosure
- **Decision:** Broad disclosure of AI assistance.
- **Rationale:** AI assistants were used to support manuscript revision planning, language polishing, response-letter organization, and code/document inspection. All experimental design decisions, code changes, numerical results, statistical analyses, and scientific claims were reviewed, verified, and approved by the human authors, who take full responsibility for the content.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Add to cover letter and manuscript declaration section, following Elsevier AI disclosure requirements.

### 2.2 CRediT Statement
- **Decision:** Update to specific granular roles.
- **Rationale:** CRediT statements should specify concrete contributor roles rather than relying only on a blanket equal-contribution sentence.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Review with co-authors before final submission.

### 2.3 Funding / Acknowledgement
- **Decision:** Acknowledge FPT University facilities; verify explicit grant numbers.
- **Rationale:** Transparency and Elsevier compliance.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Update Acknowledgement section in `.tex` if a grant number exists.

### 2.4 Conflict Of Interest
- **Decision:** Verify with all co-authors.
- **Rationale:** COI requires consensus.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Confirm with all co-authors before final submission; keep the current statement only if all authors explicitly agree.

### 2.5 Highlights Box
- **Decision:** Include 3-5 bullet points in the revised `.tex`, drafted post-results.
- **Rationale:** Often requested or supported by Elsevier CAS workflows.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Highlights must be drafted after final evidence tables are regenerated; avoid SOTA/significance claims unless statistically supported.

### 2.6 Robustness Corruption Protocol
- **Decision:** Keep custom set (Gaussian Noise, Blur, Color Jitter), but strictly enforce paired corruptions.
- **Rationale:** The corruption RNG seed will be fixed and identical corrupted samples will be used across methods to ensure valid paired comparisons.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Update `eval_robustness.py` to accept `--corruption_seed`. Primary implementation will cache corrupted tensors per severity and sample index, then reuse the cache across methods to ensure strictly identical corrupted inputs.

### 2.7 Submission Package
- **Decision:** Submit Response Letter, Cover Letter, Clean PDF, and Marked-up PDF.
- **Rationale:** Required for this resubmission package.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Phase 8 requires a clean manuscript and a marked-up manuscript. The current LaTeX revision uses yellow-highlight macros (`\revyellow{...}` and `\revyellowcaption{...}`), so further manuscript edits should preserve the same yellow-highlight convention unless the journal requires a different marked-up format.
