# Author Decisions

This file records author-level decisions that gate the Master Revision Plan. Do not start experiments that depend on these choices until the relevant decision is filled.

## 1. Technical Decisions

### 1.1 Validation Rule For Hyperparameter Selection
- **Decision:** Selection metric is the unweighted mean validation accuracy across EuroSAT and Caltech101, averaged over seeds {1,2,3} when feasible; if only one validation seed is used for compute reasons, this will be explicitly documented as a limitation.
- **Candidates:** Executed frozen sweep uses `H` in {2,4}, `r` in {2,4,8} where divisible by H, and `lambda_o` in {0, 0.01, 0.03, 0.05}; this yields 20 candidate configurations and 120 validation runs across EuroSAT/Caltech101, 4-shot, seeds {1,2,3}.
- **Selected configuration:** `H=2`, `r=4`, `lambda_o=0.0`, selected by the frozen validation rule in `revision_materials/results/selected_config.md`.
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
- **Rationale:** SingLoRA will be used as the matched same-family symmetric baseline; DoRA will be discussed conceptually and listed as a limitation because a matched implementation in the custom CLIP attention path is not available within the revision window.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Reframe the response letter to address DoRA via literature comparison rather than direct empirical comparison.

### 1.5 ImageNet Inclusion
- **Decision:** Go/No-Go based on Jun 24 Pilot.
- **Rationale:** Go if pilot wall-clock projection shows Tier-A matrix plus ImageNet/SUN397/StanfordCars can finish by July 3 with available GPU resources; otherwise run SUN397/StanfordCars first and defer ImageNet.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** If ImageNet is deferred, the manuscript will remove "standard 11-dataset comprehensive suite" wording and explicitly state that ImageNet is left for future work due to compute constraints. SUN397/StanfordCars are prioritized before ImageNet if compute is constrained.

### 1.6 Code Release Plan
- **Decision:** Anonymized GitHub link on submission.
- **Rationale:** Directly addresses R2's credibility concern during the review phase.
- **Date / Decided by:** 2026-06-16 / HN Tran
- **Impact:** Release will include cleaned scripts, exact configs, seed list, split hashes, environment file, result manifests, aggregation/statistical scripts, and instructions to reproduce all tables/figures. Adapter checkpoints will be released if storage and licensing constraints permit; otherwise scripts and manifests will be sufficient to reproduce them.


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
- **Impact:** Phase 8 requires `latexdiff` (preferred) or manual color highlighting (blue text) for revisions.
