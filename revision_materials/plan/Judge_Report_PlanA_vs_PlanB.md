# Strict Revision Judge — Plan A vs Plan B Comparison

> **Subject:** ARRAY-D-26-02033 "Orthogonal Multi-Head Gated Low-Rank Adaptation for Robust Vision–Language Model Adaptation"
> **Date:** 2026-06-15
> **Reviewed plans:**
> - **Plan A** — `D:/RESEARCH/CLIP-LoRA_group/OthorAdapt/revision_codex/docs/superpowers/plans/2026-06-15-reviewer-response-revision-plan.md`
> - **Plan B** — `D:/RESEARCH/CLIP-LoRA_group/OthorAdapt/revision_materials/plan/` (Revision_Roadmap.md + Response_Letter_Skeleton.md + Triage_Summary.md)

---

## A. Executive Summary

**Verdict on plans:** Plan A is a defensible scientific revision plan. Plan B is a presentation/triage layer mistakenly written as if it were a scientific plan. The two are complementary in *function* (evidence pipeline vs. communication template) but Plan B contains structural failures that would harm the submission if used as the master plan.

**Core asymmetry:**
- Plan A treats every claim as **needs evidence first** (provenance gate, unit tests, factorial diagnostics, manifest-driven LaTeX). It refuses to commit to a number until the number is reproducible.
- Plan B treats every claim as **needs explaining first** (posture mapping, response letter, section mapping). It writes responses as if corrections were already made and includes 6+ paragraphs of speculative mechanistic argument as if proven.

**Single overriding risk:** if Plan B's response letter were submitted as drafted, it would contain four factually unsupported assertions (original-paper used 1 seed; H>2 fails because of rank fragmentation; the three EuroSAT numbers correspond to three specific configs; orthogonality directly causes robustness). Any of these can be falsified by the reviewer with a follow-up question — and submitting unverifiable claims in a rebuttal letter is materially worse than not addressing the original issue.

**Recommended structure:**
- **Master plan = Plan A** (scientific spine)
- **Triage dashboard = Plan B's `Triage_Summary.md`** (operational view)
- **Response letter = Plan B's skeleton, but only filled after Plan A's Phase 0–4 produce verified results**
- **Both plans require additions** (cover letter, AI disclosure, Highlights box, reviewer-figure-table number mapping)

Neither plan, standalone, is sufficient. The Master Revision Plan in §F below merges Plan A's scientific tasks with Plan B's communication assets and adds 9 reviewer-adjacent items that neither plan covers.

---

## B. Plan A — Strengths and Weaknesses

### Strengths (load-bearing)

| # | Strength | Why it matters |
|---|---|---|
| A-S1 | **Provenance gate (Task 1)** | Repo has no original logs/checkpoints; without this gate every "we corrected to X" is fabrication. |
| A-S2 | **Unit tests for R2 math claims (Task 2)** | Converts R2-M1 concession from rhetoric into reproducible artifact. `rank(ΔW(x)) ≤ r` test makes Fig 7 honesty enforceable. |
| A-S3 | **Iteration semantics catch (`500 × shots`)** | Manuscript says 500 iter, code does 500×shots. Both plans must address, but **only A finds this**. If reviewer 3 (in a later round) catches this, the paper is dead. |
| A-S4 | **Pair-count `H(H-1)/2` confounder in ortho loss** | Reveals that R1-W3/Q1 has at least 3 confounders, not 1. Speculating the mechanism without a normalized-loss diagnostic is what got the paper into trouble. |
| A-S5 | **`analyze_spectrum.py` silently generates synthetic spectra when checkpoints missing** | A publication script doing this is a *research integrity* issue. Must be removed before code release. |
| A-S6 | **Statistical design uses paired permutation / hierarchical bootstrap** | Statistically defensible for 8-dataset × 3-seed paired design; Wilcoxon over 8 datasets has near-zero power. |
| A-S7 | **Tier A/B/C decomposition + Stop/Go decisions** | Provides graceful degradation if compute/time runs out. Plan B has no fallback. |
| A-S8 | **Validation-aggregate pre-registration (EuroSAT + Caltech101 4-shot)** | Locks the selection rule before seeing test, eliminating the "you just picked another split that helps you" rebuttal. |
| A-S9 | **Result manifest + single-source LaTeX generation** | Mechanically prevents the 88.64/89.06/88.57 inconsistency from recurring. |
| A-S10 | **Identifies non-reviewer issues (Section 3): 8 latent problems** | Most importantly text/vision encoder gating discrepancy, checkpoint metadata gaps, robustness pairing. |
| A-S11 | **DoRA marked "only if validated in same protocol"** | Repo uses custom CLIP attention, not HF — DoRA port is non-trivial. Conditional inclusion is correct. |
| A-S12 | **Signed-gate marked Tier C ("out of scope")** | Architecture change requires full re-evaluation; cannot be a fortnight addendum. |
| A-S13 | **Explicit warning that MoRE venue is ACL Findings, not ICLR** | Notes the reviewer's metadata error but instructs *not* to weaponize it. Good political judgement. |

### Weaknesses

| # | Weakness | Severity |
|---|---|---|
| A-W1 | **Infrastructure scope vs. 30-day deadline.** Building `scripts/run_revision_matrix.py`, `aggregate_results.py`, `statistical_tests.py`, `generate_latex_results.py`, two test suites, `manifest.jsonl`, plus the YAML config tree is a 1–2 week engineering effort before any experiment runs. | High |
| A-W2 | **No compute budget benchmarking.** Section 6 schedule allocates June 24–July 3 (10 days) for "main three-seed baseline matrix, additional datasets, and ViT-L/14 subset" without a single GPU-hour estimate or pilot run. | High |
| A-W3 | **Validation aggregate "EuroSAT + Caltech101 4-shot" is asserted but not justified.** Reviewers could ask why these two datasets and not others. Must be pre-registered *with rationale*. | Medium |
| A-W4 | **Iteration-semantics fix proposal is ambiguous** — Task 4 lists two options ("rename CLI" vs "preserve historical 500×shots and disclose") without picking one. Indecision here propagates everywhere. | Medium |
| A-W5 | **No response-letter skeleton.** Task 13 specifies format but provides no template. Author has to draft from scratch. | Medium |
| A-W6 | **No reviewer-figure / reviewer-table → tex-label mapping.** Reviewer cites "Figure 7", "Table 6" — these don't match the tex labels (`fig:6`, `tab:5`). Plan never tabulates this; risks correcting the wrong artifact. | Medium |
| A-W7 | **No editor-facing artifacts** (cover letter, marked-up vs clean manuscript split, traceability CSV format). | Low |
| A-W8 | **No AI-disclosure consideration.** Elsevier policy may require declaration of AI tools used in revision; both plans silent. | Low |
| A-W9 | **No mention of FGVC-Aircraft evaluation-protocol idiosyncrasies.** Aircraft has multiple Top-1 conventions; reviewer noted exact equality with CLIP-LoRA which could be protocol-related not copy error. | Low |
| A-W10 | **No mention of the `\begin{highlights}` block** currently commented out in `cas-sc-template.tex`. Elsevier CAS papers often require Highlights. | Low |

### Net assessment

Plan A is a correct scientific plan that is **at the edge of feasible** for the 30-day window. Without aggressive scoping (drop Tier B if needed), the infrastructure work alone could consume two weeks. But the priority ordering — provenance → infrastructure → validation → main matrix → analysis → manuscript → letter — is sound.

---

## C. Plan B — Strengths and Weaknesses

### Strengths

| # | Strength | Why it matters |
|---|---|---|
| B-S1 | **Posture mapping (ANSWER / PUSHBACK / CODE / WRITE)** | Operational language usable in author meetings; Plan A lacks this granularity. |
| B-S2 | **Section-by-section change map** (Abstract / §3.2.2 / Fig 2 / etc.) | Speeds the editing phase; Plan A doesn't tabulate this. |
| B-S3 | **Response letter skeleton with 16-comment coverage** | Verbatim quote → response → changes-made structure. Plan A has no template. |
| B-S4 | **Cross-reviewer pattern recognition** (both reviewers want larger backbone; both flag variance) | Lets the author bundle two responses into one experimental investment. |
| B-S5 | **Triage dashboard for meeting/standup use** | Single-page operational view. |
| B-S6 | **P1/P2/P3 prioritization** | Easier to communicate scope to coauthors than Plan A's task numbering. |
| B-S7 | **Acknowledges positive comments explicitly** (R1 opening, R2's praise for Fig 8 + candor) | Tone management — a response letter that doesn't thank reviewers reads as combative. |
| B-S8 | **Honest concession framing for R2-M1** | "the reviewer is correct" + "we apologize for the imprecision" — correct register. |
| B-S9 | **Reviewer-figure mapping at least partially noted** | (Though incomplete — see weakness B-W6.) |

### Weaknesses (critical)

| # | Weakness | Severity |
|---|---|---|
| B-W1 | **Writes results as fact before any rerun** ("all main tables now report mean ± std…", "Figure 7 is regenerated…", "DoRA is added…"). If submitted as-is the letter contains 12+ unverified assertions. | **CRITICAL** |
| B-W2 | **Asserts "original submission used a single seed"** as historical fact, without log evidence. Could be wrong (e.g., one table averaged, another not). Correct framing: "we audited and re-ran." | **CRITICAL** |
| B-W3 | **Asserts mechanism for H>2 degradation** as conclusion (rank fragmentation + gate variance + collapse + "unique configuration"). All four claims are speculation. Plan A correctly flags this requires factorial diagnostic. | **HIGH** |
| B-W4 | **Uses "spectral-rank recovery" as causal claim for robustness.** Total rank is bounded by r — orthogonality merely uses the budget; cannot "recover" rank. And robustness has not been shown to follow *from* rank rather than from the multi-head input-conditioning. | **HIGH** |
| B-W5 | **Speculates EuroSAT 88.64/89.06/88.57 → three specific configs** (λ=0.01, 0.03, 0.03+r=4) without log evidence. Triage line 64 says "most likely"; response letter line 174 promotes to "the three numbers correspond to". | **HIGH** |
| B-W6 | **Reviewer-table-number ↔ tex-label mapping is wrong/inconsistent.** Plan B says "Table 4 OrthoAdapt = CLIP-LoRA = 54.97" while triaging it; but the latex `tab:4` is the ablation table (Head Count), not the 16-shot results. Reviewer's "Table 4" = rendered table number = tex `tab:3` (16-shot). The Aircraft 54.97 equality is in `tab:3`, not `tab:4`. Plan B's section mapping is therefore unreliable. | **HIGH** |
| B-W7 | **Statistical method (Wilcoxon over 8 datasets + "1 SE" criterion)** is statistically weak and partially ad-hoc. No multiple-comparison correction. | High |
| B-W8 | **DoRA listed as "drop-in via PEFT lib"** — repo uses custom CLIP attention; PEFT doesn't trivially work. | High |
| B-W9 | **Signed-gating ablation marked "recommended (optional)"** — architecture change in last 4 weeks of revision is high-risk and may demand re-doing all main tables. | Medium |
| B-W10 | **Inconsistent compute estimates** (40–60h in Roadmap, 80h in Triage). No pilot run cited. | Medium |
| B-W11 | **Misses iteration semantics issue, pair-count scaling, `analyze_spectrum.py` synthetic fallback** — three repo-level issues Plan A catches. | Medium |
| B-W12 | **No provenance step.** Implicitly assumes original results are reproducible. They may not be. | **CRITICAL (compounds B-W1)** |
| B-W13 | **No fallback / Tier structure.** All 7 experimental items are P1; if compute slips, no plan to scale back. | Medium |
| B-W14 | **No mention of pre-registering the validation aggregate.** Even if val split is used, "we picked EuroSAT + DTD + Pets" after seeing results is the same problem in a different costume. | Medium |
| B-W15 | **R1-W1 (novelty) posture is too combative** ("PUSHBACK + WRITE"). Reviewer has a legitimate point — OMoE in particular is conceptually close. Correct posture: "we partially agree and have narrowed the novelty claim." | Medium |

### Net assessment

Plan B is **valuable as a presentation/communication layer** but **dangerous as a scientific plan**. Its skeleton, triage, and section-map are real contributions. Its response letter content, mechanistic explanations, and "drop-in" engineering assumptions must not be used as-is.

---

## D. Conflict Analysis

Where the plans give incompatible recommendations, classify and choose:

| # | Topic | Plan A position | Plan B position | Resolution |
|---|---|---|---|---|
| C1 | **Original-paper seed history** | Treat as unknown until logs found | Asserts "used 1 seed" | **Use Plan A.** Authorial admission of fault sounds responsible but creates a verifiable claim that may itself be false. |
| C2 | **H>2 mechanism** | Run factorial + diagnostics first (fixed-r, fixed-r/head, normalized loss, head norms, principal angles) | Already writes the explanation | **Use Plan A.** Diagnostic results must precede the explanation. |
| C3 | **PSD rewrite scope** | Rewrite Abstract / Intro / §3 / Fig 2 + add unit test + explicit "ΔW(x) is PSD for every x" proof | Same rewrite, plus optional signed-gating ablation | **Use Plan A (no signed gate).** Architecture change in last 4 weeks is reckless. |
| C4 | **Spectral figure** | Rewrite script, fail-closed on missing ckpt, prefer summary metrics (σ₂/σ₁, stable rank, effective rank, subspace overlap) over a 15-value curve | Regenerate with (r=2,H=2) checkpoint, exactly 2 non-zero SVs | **Use Plan A.** A 2-value bar is uninformative; summary scalars across layers/seeds with CI is more credible. |
| C5 | **Statistical test** | Paired permutation / hierarchical bootstrap, effect size + CI, multiple-comparison correction | Wilcoxon signed-rank + "1 SE" rule | **Use Plan A.** Wilcoxon on 8 datasets has ~no power; "1 SE" is undefined as a test. |
| C6 | **DoRA inclusion** | Conditional — only if validated in matched protocol | "Drop-in via PEFT lib" | **Use Plan A.** Repo uses custom layers; PEFT-DoRA is not drop-in. |
| C7 | **Novelty posture (R1-W1)** | "Partially agree; do not strongly rebut; narrow the claim" | "PUSHBACK + WRITE" (positioning paragraph) | **Use Plan A's posture, Plan B's positioning paragraph.** Lead the response with concession ("we partially agree and have narrowed the contribution language to …"), then the differentiation points. |
| C8 | **MoRE / O-LoRA reproduction** | Conceptual comparison only; don't put their numbers in our main tables | Same as Plan A | **Both agree.** No conflict. |
| C9 | **Signed gating ablation** | Tier C — out of scope | Optional but "recommended" | **Use Plan A.** Out of scope. |
| C10 | **EuroSAT 88.64/89.06/88.57** | Resolve from logs; if not, rerun | Speculates three config explanations | **Use Plan A.** Speculation in response letter is the same epistemic error as the original test-set tuning. |
| C11 | **Iteration semantics (`500 × shots`)** | Address; pick one definition; apply to all baselines | Not mentioned | **Plan A only — adopt.** |
| C12 | **Pair-count `H(H-1)/2`** | Normalize / report both | Not mentioned | **Plan A only — adopt.** |
| C13 | **Code-release scope** | Configs + seeds + split hashes + manifest + release tag + dataset hashes | Anonymized GitHub link + seed list | **Use Plan A's scope (richer); Plan B's release-letter language.** |
| C14 | **Validation aggregate justification** | EuroSAT + Caltech101 4-shot, pre-registered | Not addressed | **Use Plan A; add written rationale (see §H Q1).** |
| C15 | **Schedule/compute** | Tier A/B/C with Stop/Go decisions; no hard total | 40–60h or 80h (inconsistent) | **Use Plan A's Tier structure; benchmark first.** |
| C16 | **Spectral causation claim** | Avoid "rank recovery causes robustness" — temper causal language | "Spectral-rank recovery → robustness" | **Use Plan A.** Correlation, not causation, unless ablation supports. |

**Conflicts where Plan B wins:**

| # | Topic | Reason |
|---|---|---|
| C17 | **Communication structure of response letter** | Verbatim quote → response → changes-made format is essential; Plan A only specifies format abstractly. |
| C18 | **Posture vocabulary (ANSWER/PUSHBACK/CODE/WRITE)** | Operationally useful for team coordination. |
| C19 | **Explicit acknowledgment of positive reviewer comments** | Plan A is silent on this; bad form to skip thanks. |
| C20 | **P1/P2/P3 communication scaffolding** | Easier to map for co-authors than 14-task list. |

---

## E. Missing Coverage (Neither Plan Addresses)

### E1. Reviewer concerns silently dropped

All 16 explicit reviewer concerns are nominally covered by both plans. However, the following **sub-aspects** are underweighted or missed:

| # | Missing item | Source | Why it matters |
|---|---|---|---|
| E1.1 | **Specific dataset losses (Aircraft 4-shot −0.54, Flowers 4-shot −0.37)** must be explicitly discussed | R2-M3 "method actually falls behind on Aircraft and Flowers" | Both plans bundle this into "report variance" but reviewer asked specifically about these two datasets. Response must explicitly acknowledge per-dataset losses and discuss whether they survive 3-seed averaging. |
| E1.2 | **Mapping reviewer's table/figure references → tex labels** is not tabulated | All R2 minors | Reviewer's "Table 6" = `tab:5`, "Table 4" = `tab:3`, "Table 3" = `tab:2`, "Figure 7" = `fig:6`, "Figure 8" = `fig:7`, "Figure 4b" = `fig:combined_chart` panel b, "Figure 5" = `fig:4`. Plan B mis-maps Table 4 once. |
| E1.3 | **Robustness corruption set choice** (Gaussian Noise + Blur + Color Jitter, custom severity) | Adjacent to R2-M3 | Manuscript cites `hendrycks2019benchmarking` but does *not* use ImageNet-C standard set. A skeptical reviewer can ask why; both plans should pre-empt with a methods-paragraph upgrade. |
| E1.4 | **The original paper's "highly compact parameter budget" / "comparable to SingLoRA" claim** | Adjacent to R2-M1 | Not quantified anywhere with actual parameter counts including gating network. Easy to add; closes a credibility gap. |

### E2. Submission-package items neither plan addresses

| # | Missing item | Required? | Plan |
|---|---|---|---|
| E2.1 | **Cover letter to editor** | Strongly recommended for major revision | Add to master plan. |
| E2.2 | **AI-usage disclosure for revision** (Elsevier policy) | Required if AI tools were used in revision drafting | Add to master plan. |
| E2.3 | **Highlights box** (`\begin{highlights}` currently commented out) | Elsevier CAS journals typically require 3–5 highlights | Add to manuscript edits. |
| E2.4 | **Marked-up vs. clean manuscript split** for resubmission | Elsevier requires both (manuscript with tracked changes + clean version) | Add to Task 14 / final integrity pass. |
| E2.5 | **CRediT statement specificity** | Current statement: "all authors contributed equally to ... conceptualization, methodology, ..." — CRediT requires more granular role assignment | Suggest review with co-authors. |
| E2.6 | **Data Availability statement upgrade** | Both plans address but neither writes the specific text | Draft proposed wording. |
| E2.7 | **Funding / acknowledgement section** | If FPT University provided funding beyond facilities, declare | Confirm with corresponding author. |
| E2.8 | **Conflict of interest declaration matches reality** | Current: "no competing interests" | Confirm with authors. |
| E2.9 | **References sanity audit (especially `ref7` at `arXiv:2512.23165`)** | This arXiv ID is plausible (Dec 2025 numbering) but should be verified existence — and Yin et al. RLVR is reused as authority on "spectral collapse" | Verify. |
| E2.10 | **`ref8` SingLoRA `arXiv:2507.05566`** | Verify existence and that the paper actually contains the claims attributed (especially the gradient-imbalance analysis cited) | Verify. |

### E3. Code-base items neither plan addresses

| # | Missing item | Source | Why |
|---|---|---|---|
| E3.1 | **`run_all.sh`, `scan_head.sh`, `scan_loss.sh`, `scan_NumHead.sh` hardcoded paths and configs** | Repository scan | Plan A flags but lists only `run_all.sh` and `scan_head.sh`; the others need same treatment. |
| E3.2 | **Confirm `loralib/layers.py` standard-LoRA path is the actual baseline used** | Repo has 4 layer files: `layers.py`, `layers_singlora.py`, `layers_OH_singlora.py`, `easymultiheadattention.py` | If `layers.py` is not the actual baseline LoRA used in published numbers, the "CLIP-LoRA baseline" provenance is broken. |
| E3.3 | **Memory of the `text_features_train` issue** (Plan A §3 item 6) is only when text encoder adapted; what does the actual run script do? | Repo `lora.py` reading | Pin down whether the published numbers come from vision-only, text-only, or both — `args.encoder` default in `run_utils.py:46` is "both". |

### E4. Statistical / methodological items neither plan addresses fully

| # | Missing item | Severity |
|---|---|---|
| E4.1 | **Family-wise error correction for the 33 (8 dataset × 3 shot + ablations) confirmatory tests** | High |
| E4.2 | **Pre-registration of which dataset/shot combinations are "primary" vs "exploratory"** | High |
| E4.3 | **Reporting both effect size AND p-value (not just one)** | Medium |
| E4.4 | **Distinguishing seed-variance from few-shot-split variance** — both vary per seed | Medium |
| E4.5 | **Confidence-interval bands on the ablation plots (Fig. 4)** in addition to point estimates | Medium |

---

## F. Consolidated Master Revision Plan

Format: each row = one action.
- **Reviewer**: which reviewer concern(s) it addresses (or `INTERNAL` if it's a repo-level fix that isn't a reviewer concern but is required for credibility / scientific integrity).
- **Source**: `A` = Plan A only, `B` = Plan B only, `A+B` = both, `JUDGE` = neither, judge-added.
- **Classification**: MUST_FIX / SHOULD_FIX / OPTIONAL / RISKY.
- **Deps**: blocking dependencies (other action IDs).

### Phase 0 — Provenance & Integrity Gate (June 15–18)

| ID | Action | Reviewer | Source | Class | Deps |
|---|---|---|---|---|---|
| P0.1 | Search all storage (local, Drive, Colab, lab cluster) for original checkpoints, logs, CSVs, configs used in current Tables 1–6 and Figs 4–8. Hash and record everything. | INTERNAL (foundational) | A | MUST_FIX | — |
| P0.2 | Build `revision_traceability.csv`: every numerical claim in paper → source artifact (or `UNVERIFIED_RERUN_REQUIRED`). | INTERNAL | A | MUST_FIX | P0.1 |
| P0.3 | Decide artifact-recovery cutoff: if any acceptance-critical number lacks provenance by June 18, schedule full rerun. | INTERNAL | A | MUST_FIX | P0.2 |
| P0.4 | **Tabulate reviewer's table/figure references → tex labels** (e.g., reviewer Table 4 = tex `tab:3`). | All R2-minor | JUDGE | MUST_FIX | — |
| P0.5 | Confirm which `layers_*.py` produced the baseline numbers and which produced OrthoAdapt numbers. | INTERNAL | JUDGE | MUST_FIX | P0.1 |

### Phase 1 — Code & Test Infrastructure (June 16–20, parallel with P0)

| ID | Action | Reviewer | Source | Class | Deps |
|---|---|---|---|---|---|
| P1.1 | Write unit tests: softmax non-negativity, `ΔW(x)` PSD, `rank(ΔW(x)) ≤ r`, ortho loss zero on orthogonal heads, pair-count scaling regression. | R2-M1 | A | MUST_FIX | — |
| P1.2 | Add `--selection_split {val,test}` (default `val`) and `--report_test` flag. Forbid test eval during sweep. | R2-M2 | A | MUST_FIX | — |
| P1.3 | Resolve `n_iters * shots` semantics: pick one definition, apply uniformly to all baselines, document in manuscript. | INTERNAL (per Plan A §3.1) | A | MUST_FIX | — |
| P1.4 | Rewrite `analyze_spectrum.py`: remove synthetic-fallback DEMO branch; fail-closed on missing checkpoint; validate matrix dimensions. | INTERNAL (per Plan A §3.5) | A | MUST_FIX | — |
| P1.5 | Add `reduction={sum,mean}` option to `calculate_ortho_loss`; log raw + pair-normalized + head norms + principal angles. | R1-W3/Q1 (diagnostic) | A | MUST_FIX | — |
| P1.6 | Expand checkpoint metadata to record seed, H, λ, ramp-up, iter-budget, split hash, git revision, CUDA/torch version. | INTERNAL | A | MUST_FIX | — |
| P1.7 | Build `scripts/aggregate_results.py` → JSONL manifest → `scripts/generate_latex_results.py`. | R2-M3, R2-5a | A | SHOULD_FIX | P1.6 |
| P1.8 | Build `scripts/statistical_tests.py`: paired permutation + hierarchical bootstrap + effect size + CI + multiple-comparison correction. | R2-M3 | A (Plan B: weak Wilcoxon — rejected) | MUST_FIX | P1.7 |
| P1.9 | Add consistency tests: prose number = table number = figure number (fail CI if mismatched). | R2-5a | A | SHOULD_FIX | P1.7 |
| P1.10 | Pin robustness corruption seed; record exact transform parameters; ensure pairing across methods. | INTERNAL (per Plan A §3.8) | A | SHOULD_FIX | — |
| P1.11 | Clean up `run_all.sh`, `scan_head.sh`, `scan_loss.sh`, `scan_NumHead.sh` hardcoded paths. | INTERNAL | JUDGE (Plan A names only 2) | SHOULD_FIX | — |
| P1.12 | Verify references `ref7` (Yin et al. arXiv:2512.23165) and `ref8` (SingLoRA arXiv:2507.05566) actually exist with the cited content. | INTERNAL | JUDGE | MUST_FIX | — |

### Phase 2 — Pre-Registered Validation Sweep (June 19–23)

| ID | Action | Reviewer | Source | Class | Deps |
|---|---|---|---|---|---|
| P2.1 | **Write and freeze** the validation-aggregate definition (proposed: mean val accuracy over EuroSAT + Caltech101 at 4-shot) **and the rationale** for choosing these two datasets before any sweep runs. Save to `configs/revision/selection_protocol.yaml`. | R2-M2 + JUDGE (rationale missing in A) | A + JUDGE | MUST_FIX | P1.2 |
| P2.2 | Sweep `H ∈ {1,2,4}`, `r ∈ {2,4,8}`, `λ_o ∈ {0, 0.01, 0.03, 0.05, 0.1}` on validation only. | R2-M2, R1-W3/Q1 | A | MUST_FIX | P2.1 |
| P2.3 | Also run a **fixed-rank-per-head** axis (H = {1,2,4} with r_head = 2 each) and a **pair-normalized loss** axis to disentangle R1-W3 confounders. | R1-W3/Q1 | A | MUST_FIX | P1.5, P2.2 |
| P2.4 | Select winning config by frozen rule (P2.1) only; record selection trace. | R2-M2 | A | MUST_FIX | P2.2 |
| P2.5 | Run selected config **once** on test sets; close the test-set selection loophole. | R2-M2 | A | MUST_FIX | P2.4 |

### Phase 3 — Main Experiment Matrix (Tier A) (June 22–July 3)

| ID | Action | Reviewer | Source | Class | Deps |
|---|---|---|---|---|---|
| P3.1 | Pilot benchmark: 1 dataset × 1 shot × 1 seed for OrthoAdapt, SingLoRA, CLIP-LoRA on the target GPU. Measure wall-clock, project full matrix. | INTERNAL | JUDGE | MUST_FIX | P1.3, P1.6 |
| P3.2 | **Tier A**: CLIP-LoRA + SingLoRA + OrthoAdapt × 8 datasets × {1,4,16}-shot × 3 seeds (seeds 1,2,3 — preferably 5). | R2-M3, R2-5b, R1-W4 | A | MUST_FIX | P3.1, P2.5 |
| P3.3 | **Tier A**: Add SUN397 and StanfordCars at same protocol. | R2-5d | A | MUST_FIX | P3.2 |
| P3.4 | **Tier A**: ImageNet only if compute permits; otherwise drop "comprehensive 11-dataset" language. | R2-5d | A | SHOULD_FIX | P3.2 |
| P3.5 | Aggregate via P1.7, generate tables via P1.7 → `cas-sc-template.tex`. | R2-M3 | A | MUST_FIX | P1.7, P3.2 |
| P3.6 | Run paired permutation + bootstrap CI via P1.8; report effect sizes per shot. | R2-M3 | A | MUST_FIX | P1.8, P3.5 |

### Phase 4 — Scope & Diagnostic Experiments (Tier B) (July 1–6)

| ID | Action | Reviewer | Source | Class | Deps |
|---|---|---|---|---|---|
| P4.1 | ViT-L/14 on EuroSAT + OxfordPets + DTD × 4-shot × 3 seeds for CLIP-LoRA, SingLoRA, OrthoAdapt. | R1-W2, R1-Q3 | A+B | MUST_FIX | P3.5 |
| P4.2 | DoRA baseline: only if portable to the custom CLIP attention path within ~3 days. Otherwise document as Tier C with conceptual comparison only. | R1-W4 | A (Plan B's "drop-in" rejected) | SHOULD_FIX (gated) | P3.2 |
| P4.3 | Three-seed robustness eval with deterministically paired corruptions on EuroSAT + OxfordPets, all three methods. | R2-M3 + INTERNAL pairing | A | SHOULD_FIX | P1.10, P3.2 |
| P4.4 | Gating-only (λ=0) and orthogonality-only diagnostics under matched parameter budget. | R1-W3/Q1 | A | SHOULD_FIX | P3.2 |
| P4.5 | Replace spectral analysis: report σ₂/σ₁, stable rank, entropy effective rank, head-subspace overlap with CIs across layers/seeds — not the 15-value curve. | R2-M4 | A | MUST_FIX | P1.4, P3.2 |
| P4.6 | Regenerate Fig 7 (= reviewer's "Figure 7") with named matrix, checkpoint hash, layer, configuration, layer-by-layer summary scalars + uncertainty. | R2-M4 | A+B | MUST_FIX | P4.5 |

### Phase 5 — Tier C (Explicitly Deferred to Future Work)

| ID | Action | Reviewer | Source | Class | Deps |
|---|---|---|---|---|---|
| P5.1 | Multi-task adaptation extension | R1-Q2 | A | OPTIONAL — future work | — |
| P5.2 | Image-text retrieval benchmark | R1-W2 | A | OPTIONAL — future work | — |
| P5.3 | Signed-gate architecture replacing softmax | R2-M1 (defensive) | Plan B suggested — JUDGE: **RISKY**, defer | RISKY — DO NOT DO | — |
| P5.4 | Full reproduction of MoRE / O-LoRA in their original settings | R1-W4 | A | RISKY — out of scope | — |

### Phase 6 — Manuscript Rewrite (July 7–11)

| ID | Action | Reviewer | Source | Class | Deps |
|---|---|---|---|---|---|
| P6.1 | Abstract: remove "escape PSD" framing; describe input-conditioned adaptation; fix "severe" → "medium" (3.6%); audit every number against P3.5/P4.6. | R2-M1, R2-5a | A+B | MUST_FIX | P3.5, P4.6 |
| P6.2 | Intro: replace incorrect "weighted combination of PSD is not PSD" statement; update contributions list. | R2-M1 | A | MUST_FIX | P6.1 |
| P6.3 | Fig 2 (architecture comparison): relabel OrthoAdapt as input-conditioned PSD mixture; redraw if needed. | R2-M1 | A+B | MUST_FIX | P6.1 |
| P6.4 | §3.2.2: prove `ΔW(x)` PSD for every x under softmax gating; retain Jacobian non-linearity argument as the genuine contribution. | R2-M1 | A | MUST_FIX | P1.1 |
| P6.5 | §3.2.2 Proposition 2: rewrite "batch-level rank recovery" — total rank is bounded by r; correct claim is about effective rank within budget when heads are orthogonal. Remove "each sample uses only rank r/H" (false under softmax). | R2-M1 | A | MUST_FIX | — |
| P6.6 | §3.2.1 / §3.3.1: define `u(t) = min(1, t/T)` with T=1000; remove ε reference. | R2-5c | A+B | MUST_FIX | — |
| P6.7 | §2.3 (Related Work): new "Position vs. MoE-LoRA / OMoE" paragraph framed as concession ("we partially agree and have narrowed the claim") + concrete differentiation table. Drop "novel framework" language where unsupported. | R1-W1, R2-5b | B's positioning + A's posture | MUST_FIX | — |
| P6.8 | §4.1: state explicit hyperparameter selection protocol (val-only), seed protocol (≥3 seeds), iteration budget (resolved per P1.3). | R2-M2, R2-M3 | A | MUST_FIX | P2.5, P1.3 |
| P6.9 | §4.3.1: new mechanistic discussion of H>2 degradation **driven by P2.3 diagnostics, not speculation**. State which confounder(s) dominate. | R1-W3/Q1 | A (Plan B's speculative draft rejected) | MUST_FIX | P2.3 |
| P6.10 | §4.4: new spectral analysis prose matching new figures (P4.5/P4.6); temper causal claims to "correlates with" unless P4.3 + P4.4 ablations support causation. | R2-M4 | A | MUST_FIX | P4.5 |
| P6.11 | New §4.5 Backbone Scalability with P4.1 results. | R1-W2, R1-Q3 | A+B | MUST_FIX | P4.1 |
| P6.12 | Update Tables 2–4 (4-shot, 16-shot, 1-shot) to include SingLoRA row; add DoRA row if P4.2 succeeded. | R1-W4, R2-5b | A+B | MUST_FIX | P3.5 |
| P6.13 | Conclusion: replace "consistent state-of-the-art" with language defensible from CIs (e.g., "consistent positive mean gains across datasets with statistically detectable improvement at 4-shot and 16-shot"). | R2-M3 | A | MUST_FIX | P3.6 |
| P6.14 | **New Limitations section** (or expand §5) explicitly committing to multi-task, retrieval, larger backbones, and acknowledging dataset losses (Aircraft, Flowers at 4-shot) with CIs. | R1-Q2, R1-W2, R2-M3 | A + JUDGE | MUST_FIX | P3.6 |
| P6.15 | Fix Table 5 (`tab:5`, reviewer's Table 6) decimal commas → periods. | R2-5a | A+B | MUST_FIX | — |
| P6.16 | Resolve Aircraft 16-shot 54.97 equality from logs (not assume copy error). | R2-5a | A | MUST_FIX | P0.2 |
| P6.17 | Resolve EuroSAT 88.64 / 89.06 / 88.57 by stating exact config of each cell from manifest. | R2-5a | A | MUST_FIX | P3.5 |
| P6.18 | Fill `\begin{highlights}` block (3–5 highlights). | INTERNAL — Elsevier CAS format | JUDGE | SHOULD_FIX | P6.13 |
| P6.19 | Strengthen Data Availability statement: commit to anonymized GitHub link in submission, public release on acceptance, list seeds and split hashes. | R2-5e | A+B | MUST_FIX | — |
| P6.20 | Add quantitative parameter-count table (OrthoAdapt vs CLIP-LoRA vs SingLoRA vs DoRA) including gating-network params. | INTERNAL (credibility) | JUDGE | SHOULD_FIX | P3.2 |

### Phase 7 — Response Letter, Cover Letter & Submission Package (July 12–13)

| ID | Action | Reviewer | Source | Class | Deps |
|---|---|---|---|---|---|
| P7.1 | Draft cover letter to Editor (Dr. Bora): summary of major changes, statement of completeness, AI-disclosure (if applicable). | E2.1 | JUDGE | MUST_FIX | P6.* done |
| P7.2 | Adopt Plan B's response-letter skeleton as **template only**; replace every placeholder with values verified from P3.5/P4.5/P4.6 manifests. Remove all "we have", "is regenerated", "now reports" claims that lack manifest backing. | All reviewer items | B (gutted of unsupported claims) | MUST_FIX | P6.* done |
| P7.3 | Rewrite R1-W1 (novelty) response: lead with "We partially agree and have narrowed the contribution claim" → then differentiation. | R1-W1 | A's posture + B's content | MUST_FIX | P6.7 |
| P7.4 | Rewrite R1-W3/Q1 (H>2) response: present P2.3 diagnostic results; state which confounder(s) the data identifies; do not assert "unique configuration" unless validated. | R1-W3/Q1 | A | MUST_FIX | P2.3, P6.9 |
| P7.5 | Rewrite R2-M3 (seeds) response: avoid asserting historical seed count; use "the original manuscript contained an inconsistency; we audited records and re-ran with seeds {1,2,3}". Discuss Aircraft and Flowers specifically. | R2-M3 | A | MUST_FIX | P3.6 |
| P7.6 | Rewrite R2-5a (EuroSAT three numbers): cite manifest-backed configs only; do not retro-fit a "most likely" explanation. | R2-5a | A | MUST_FIX | P3.5 |
| P7.7 | Build traceability table: every reviewer comment ↔ response paragraph ↔ manuscript change ↔ manifest evidence. | All | A | MUST_FIX | P7.2 |
| P7.8 | Quote exact revised page / line / table / figure locations after PDF rebuild. | All | A | MUST_FIX | P7.2 |
| P7.9 | AI-usage disclosure per Elsevier policy (e.g., "Claude Code / GPT was used in revision drafting; all technical claims verified by authors against manifest-tracked experimental evidence"). | E2.2 | JUDGE | MUST_FIX (if AI used) | — |
| P7.10 | Verify CRediT, COI, Funding statements with all authors. | E2.5, E2.7, E2.8 | JUDGE | MUST_FIX | — |

### Phase 8 — Final Integrity Pass (July 14)

| ID | Action | Reviewer | Source | Class | Deps |
|---|---|---|---|---|---|
| P8.1 | Clean LaTeX rebuild; check warnings; verify all figures embed correctly. | INTERNAL | A | MUST_FIX | P7.* |
| P8.2 | Run P1.9 consistency tests; resolve any mismatch. | R2-5a | A | MUST_FIX | P1.9, P6.* |
| P8.3 | Recompute every table average from row values. | R2-5a | A | MUST_FIX | — |
| P8.4 | Verify every reviewer comment has exactly one response paragraph + at least one manuscript change OR explicit rebuttal rationale. | All | A | MUST_FIX | P7.7 |
| P8.5 | Generate both marked-up (tracked-changes) PDF and clean PDF. | E2.4 | JUDGE | MUST_FIX | P8.1 |
| P8.6 | Independent re-review focused on R2-M1 through R2-M4 (the four major concerns). | R2-M1..M4 | A | MUST_FIX | P8.1 |
| P8.7 | Confirm anonymized repo is accessible from the URL listed in Data Availability. | R2-5e | A | MUST_FIX | P6.19 |

### Classification summary

| Class | Count | Notes |
|---|---|---|
| MUST_FIX | 49 | All reviewer-tied + infrastructure required for them |
| SHOULD_FIX | 11 | Improve credibility, gate on time/compute |
| OPTIONAL | 2 | Future-work commitments only |
| RISKY | 4 | Tier C — do not undertake (signed gate, MoRE reproduction, multi-task in this revision, image retrieval in this revision) |

---

## G. Recommended Execution Order

**Critical-path principle:** Provenance (P0) gates *everything*. Infrastructure (P1) and validation (P2) gate the main matrix (P3). Main matrix gates rewrite (P6) and response (P7). Do not parallelize across these boundaries.

```
Day 1–4  (Jun 15–18)  ┃ P0.1–P0.5  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                       ┃                                    ──  PROVENANCE GATE  ──
                       ┃ P1.1, P1.2, P1.3, P1.4, P1.5, P1.6 (parallel with P0)
Day 5–9  (Jun 19–23)  ┃ P1.7, P1.8, P1.9, P1.10, P1.11, P1.12   (parallel infra)
                       ┃ P2.1 (FREEZE protocol)
                       ┃ P2.2, P2.3, P2.4  (validation sweeps)
Day 10        (Jun 24) ┃ P3.1 (pilot benchmark — REQUIRED before P3.2)
Day 11–19 (Jun 24–Jul 3) ┃ P2.5 + P3.2 + P3.3 + (P3.4 if compute)
                       ┃              ──  TIER A MATRIX  ──
Day 17–22 (Jul 1–6)   ┃ P4.1–P4.6  (Tier B, partial overlap with late P3)
Day 23–27 (Jul 7–11)  ┃ P6.1–P6.20  (manuscript rewrite — no exp runs)
Day 28–29 (Jul 12–13) ┃ P7.1–P7.10  (response letter, cover letter, traceability)
Day 30        (Jul 14) ┃ P8.1–P8.7  (final integrity pass)
Day 31        (Jul 15) ┃ SUBMISSION
```

### Stop/Go decision points

| Date | Decision | Threshold for "STOP and re-scope" |
|---|---|---|
| Jun 18 EOD | Did P0.1 recover ≥80% of original artifacts? | If no: skip reconciliation, plan full Tier A rerun |
| Jun 23 EOD | Is `aggregate_results.py` + `statistical_tests.py` working on the validation-sweep output? | If no: postpone Tier A by 1 day, refactor |
| Jun 24 PM | Does P3.1 pilot suggest Tier A completes by Jul 3? | If no: drop ImageNet from Tier A, drop DoRA from Tier B |
| Jul 3 EOD | Are Tier A tables complete with CIs? | If no: drop SUN397 / StanfordCars, narrow "comprehensive suite" language |
| Jul 6 EOD | Did Tier B (ViT-L/14) finish? | If no: report negative-result / partial-result framing |
| Jul 11 EOD | Manuscript rewrite complete? | If no: request 1-week deadline extension from editor on Jul 12 |

### Concurrent vs sequential

- **Parallelizable**: P1 infra tasks among themselves; P3.2 / P3.3 / P3.4 across datasets; P6 manuscript subsections after P3.5 done.
- **Strictly sequential**: P0 → P1.2/P1.3 → P2.1 → P2.2 → P2.4 → P3.2 → P3.5/P3.6 → P6.* → P7.* → P8.*

---

## H. Questions Requiring Clarification (NEEDS_CLARIFICATION)

Before any experiment runs, please resolve the following with the corresponding author:

### Q1. Validation-aggregate justification (required for P2.1)
Plan A proposes "mean val accuracy over EuroSAT + Caltech101 at 4-shot" as the selection rule. **Why these two datasets?** Suggested rationales (pick one and pre-register):
- (a) EuroSAT for domain shift + Caltech101 for general objects = diversity proxy.
- (b) These two have the largest validation splits in the CLIP-LoRA codebase.
- (c) Other (please specify).

**Risk if not pre-registered**: a reviewer can argue the selection is post-hoc.

### Q2. Iteration semantics (P1.3)
Two options:
- (a) **Keep historical `500 × shots`** and document as "500 steps per shot, totaling 500/2000/8000 for 1/4/16-shot." Apply same to all baselines.
- (b) **Switch to literal 500 total steps** as the paper states; rerun everything.

**Recommendation**: (a), since it preserves the headline budget; but the manuscript must be unambiguous.

### Q3. Seed count
- (a) 3 seeds {1,2,3} — minimum, fits 30-day window.
- (b) 5 seeds {1,2,3,4,5} — better statistical power; ~67% more compute.

**Recommendation**: (a) for Tier A, (b) for the headline EuroSAT 4-shot cell only.

### Q4. DoRA inclusion (P4.2)
- (a) **Skip** — defend with "concurrent method targeting different parameterization; conceptual comparison in §2.3".
- (b) **Port** — ~3 person-days; risk of misaligned protocol.

**Recommendation**: decision by Jun 24 based on P3.1 pilot timing.

### Q5. ImageNet inclusion
- (a) **Include** — ~12 GPU-hours added.
- (b) **Skip** and remove "comprehensive standard suite" wording.

**Recommendation**: include only if Tier A is on schedule by Jul 1.

### Q6. AI-usage disclosure (P7.9)
Will the revision letter and any manuscript prose be drafted with AI tool assistance (Claude / GPT / etc.)?
- (a) Yes — add disclosure paragraph per Elsevier policy.
- (b) No — no disclosure needed.

### Q7. CRediT statement (E2.5)
Current statement says "all authors contributed equally". Confirm with co-authors whether this is literally true for all 14 CRediT roles, or whether the revision should disaggregate (e.g., experimental work vs. writing).

### Q8. Funding statement (E2.7)
Manuscript Acknowledgement mentions FPT University facilities. Was there *grant funding* (e.g., research project ID) that must be declared per Elsevier policy? Currently no funding statement is in the manuscript.

### Q9. Code release platform & timeline (P6.19)
- (a) Anonymized GitHub link in submission, de-anonymize on acceptance.
- (b) Promise release on acceptance only.

**Recommendation**: (a) — R2-5e is explicit, and an anonymized repo eliminates the credibility doubt at review time, not after.

### Q10. Robustness corruption methodology (E1.3)
Current corruptions are "Gaussian Noise, Blur, Color Jitter" with custom severity labels. Should we:
- (a) Switch to standard ImageNet-C corruption set + severities to match `hendrycks2019benchmarking`.
- (b) Keep custom set but document parameters precisely.

**Recommendation**: (b) for revision-cycle pragmatism, plus an Appendix with the exact transform code.

### Q11. Highlights box (P6.18)
Elsevier CAS template supports `\begin{highlights}` for journal Highlights. Should we fill it (3–5 bullet points)? If yes, suggested draft:
- We address optimization instability and rank collapse in low-rank VLM adaptation under few-shot settings.
- We introduce input-conditioned PSD mixture updates with cross-head orthogonality regularization.
- ...

### Q12. Per-dataset loss discussion (E1.1)
At 4-shot, OrthoAdapt loses to CLIP-LoRA on Aircraft (−0.54) and Flowers (−0.37). After 3-seed re-run with CIs:
- (a) If the losses are within noise, discuss as "indistinguishable" and add to Limitations.
- (b) If the losses persist with CIs, do *not* hide them — explicitly discuss why the orthogonal multi-head doesn't help here (e.g., fine-grained tasks may benefit from concentrated rather than dispersed updates).

**Recommendation**: (b) — honest reporting is what R2 praised in the original paper.

---

## Summary recommendation

**Adopt Plan A as the master scientific plan with the additions in §F. Use Plan B exclusively as the response-letter template, and gut its unsupported assertions before any sentence is sent to the editor.**

The single most important behavioral change between the two plans is **evidence-first vs. explanation-first**. Plan A makes evidence the prerequisite for every claim; Plan B treats the explanation as the deliverable. For a major-revision response, only Plan A's posture is defensible — reviewers will check, and a rebuttal letter caught making false historical claims (e.g., "we used 1 seed") is a fast track to rejection.

Final decision items requiring author input: Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12. Please answer before Jun 18 to keep Phase 0 on schedule.
