# Revision Roadmap — ARRAY-D-26-02033 "OrthoAdapt"

**Editor verdict:** Major Revision; resubmission due **2026-07-15**.
**Reviewers:** R1 (substantive but mostly positive, 4 weaknesses + 3 questions), R2 (positive tone but **4 hard technical concerns** + 5 minor items).
**Effort estimation:** **Substantial** (≈3–4 weeks) — at least one concern (R2-1, PSD claim) requires either a narrative rewrite *or* an architecture change; R2-3 (multi-seed) is non-negotiable for a journal resubmission.

> **Triage legend:** Each comment is labeled with a recommended posture:
> - **ANSWER** = the paper is already (mostly) right; explain and add a clarification in the manuscript.
> - **PUSHBACK** = the reviewer's premise is partially incorrect; respectfully disagree with evidence — but always concede whatever sub-point *is* valid.
> - **CODE** = new experiments / re-runs / new figures must be produced before the response can be written.
> - **WRITE** = a textual change only (definition, citation, narrative reframing, table fix).
>
> Most P1 items are **CODE + WRITE** combined.

---

## Overview

| Item | Count |
|------|-------|
| Total reviewer concerns | **15** (R1: 4 weaknesses + 3 questions; R2: 4 major + 5 minor sub-items) |
| Major (P1) | 7 (R1-1, R1-2, R1-3, R2-1, R2-2, R2-3, R2-4) |
| Minor (P2) | 5 (R1-4, R2-5b, R2-5c, R2-5d, R2-5e) |
| Editorial (P3) | 3 (R2-5a numerical inconsistencies — 4 sub-items) |
| Positive acknowledgments | 2 (R1 opening; R2 opening on candor + Fig 8) |

**Reviewer consensus signals (auto-escalated):**
- **Larger backbone / scope expansion** raised by R1-2, R1-Q3 → must address.
- **Multi-seed variance / honest reporting** raised implicitly by R2-3 and R2-5e → **central credibility issue**.

---

## P1 — Must Fix

| # | Comment summary | Reviewer | Posture | Section to revise | Suggested action |
|---|---|---|---|---|---|
| **P1-1** | PSD claim is wrong — softmax gating preserves PSD (Eq. 6 → 8) | R2-1 | **PUSHBACK partial + WRITE major** | Abstract, Intro §1, Method §3.2.2, Fig. 2 | Concede the strict-PSD point. Reframe contribution as **input-dependent (piecewise / context-conditional) update + spectral-rank recovery via orthogonal heads**, not "PSD escape." Optionally add a **signed-gating ablation** (tanh / signed softmax) to recover the stronger claim — this is the only way to defend the original framing. |
| **P1-2** | Hyper-parameters (H, λ_o) selected on EuroSAT test set | R2-2 | **ANSWER + CODE** | §4.1 Protocol, Tables 6 & ablation | Re-run head-count and λ_o sweeps using the **CLIP-LoRA validation split protocol** (the few-shot val set CLIP-LoRA itself uses) instead of test. Report selected config + test accuracy. Add a short paragraph stating the val protocol explicitly. |
| **P1-3** | Seed protocol contradiction: §4.1 says "3 seeds", Table 1 says "1 seed". Gains (0.22/0.43/0.32) are within noise; method loses on Aircraft & Flowers | R2-3 | **ANSWER (concede) + CODE major** | All main tables (1, 2, 3), §4.1, Conclusion | This is the single largest credibility risk. **Re-run all main experiments with ≥3 seeds** (the codebase already supports `--seed`; see `run_utils.py:20`). Report **mean ± std** + a paired test (Wilcoxon signed-rank or paired t over the 8 datasets) for each shot setting. Soften "consistent state-of-the-art" in Conclusion to language that the variance evidence actually supports. |
| **P1-4** | Spectral figure (Fig. 7) shows ~15 singular values but rank budget is r=2 — figure inconsistent with stated config | R2-4 | **ANSWER + CODE** | §4.4 Spectral Analysis, Fig. 7 caption | `analyze_spectrum.py` computes `Σ A_h A_h^T` of shape (d, d) — its rank is at most r=2 for the (r=2, H=2) config; the 15 nonzero singular values must come from a different setting (likely r=4 + r/head=2 + numerical noise, or a different checkpoint). **Either** re-generate Fig. 7 using the exact (r=2, H=2) checkpoint with rank capped at 2, **or** explicitly state the figure uses the r=4, H=2 ablation config and clarify in caption "Stable rank / effective rank visualization, not full spectrum." Add the precise matrix being decomposed (e.g., `Σ_h U_h U_h^T` of one attention block at iteration 500). |
| **P1-5** | Novelty vs. MoRE (ICLR 2025) and similar MoE-LoRA concurrent work | R1-1 | **PUSHBACK + WRITE** | §2.3, Intro contributions list | Add a "Position vs. MoE-LoRA" paragraph in §2.3 explicitly contrasting: (i) MoRE / MoLE / MoCLE target **multi-task instruction tuning in LLMs** with task-level routing; OrthoAdapt targets **single-task feature-level routing in few-shot VLMs**. (ii) OrthoAdapt uses **symmetric (SingLoRA-style) heads + pairwise orthogonality**, neither of which appears in MoRE. (iii) Cite OMoE (already in refs as ref 41 / `feng2025omoe`) and articulate the difference: OMoE uses orthogonal *fine-tuning* on the LLM; OrthoAdapt enforces head-level orthogonality on low-rank adapters in VLMs. |
| **P1-6** | Narrow experimental scope (ViT-B/16 only, no large backbone, no multi-task, no retrieval) | R1-2, R1-Q3 | **PARTIAL CODE + PUSHBACK + WRITE** | §4 + new §4.5 (Backbone Scalability) | **Required:** Run OrthoAdapt + CLIP-LoRA on **ViT-L/14** for at least 3 representative datasets (EuroSAT, DTD, OxfordPets) at 4-shot, 3 seeds. The codebase passes `--backbone` directly to `clip.load`, so no code changes needed (only RAM/VRAM). **Pushback:** politely note that multi-task / retrieval are out of scope for a *few-shot classification* paper, but **add an explicit Limitations & Future Work paragraph** committing to those directions. |
| **P1-7** | Missing baselines: MoRE, O-LoRA, DoRA | R1-4 | **CODE + WRITE** | Tables 2–4 (or a new comparison table) | Add at minimum: **DoRA** (PEFT lib has it — drop-in; cited in paper but not used as baseline), **SingLoRA** (R2-5b also requests this as a full baseline — it is already implemented in `loralib/layers_singlora.py`). MoRE and O-LoRA are designed for multi-task / continual; reproducing them on this single-task few-shot benchmark is non-trivial — defend with a paragraph + cite their reported numbers if available; if not, add to Limitations. **Bare minimum**: SingLoRA + DoRA as full rows. |

---

## P2 — Should Fix

| # | Comment summary | Reviewer | Posture | Section | Suggested action |
|---|---|---|---|---|---|
| **P2-1** | Why does performance drop for H > 2 with r=2/4? Reviewer wants mechanistic explanation. | R1-Q1 | **ANSWER + WRITE** | §4.3.1 Multi-Head ablation | Add 2-paragraph explanation: (a) **Rank fragmentation**: r=4, H=4 ⇒ each head rank-1 — provably cannot represent any 2D rotation; the orthogonality constraint then makes the heads near-orthonormal vectors, which collectively span at most a 4-D subspace but each individually carries almost no transformation capacity. (b) **Optimization noise**: with so little capacity per head, the softmax gate has high variance under few-shot supervision. Tie to Table 4 numbers already in the paper. |
| **P2-2** | Multi-task extension future work | R1-Q2 | **ANSWER** (deferred) | Conclusion / Future Work | Already partially in §5; expand to one paragraph explicitly committing to a multi-task evaluation with MoRE-style routing, and explain why this paper deliberately scopes to single-task few-shot. |
| **P2-3** | SingLoRA as full baseline in Tables 2–4; sharper distinction from OMoE | R2-5b | **CODE + WRITE** | Tables 2–4, §2.3 | SingLoRA runs are required anyway for P1-7. Reuse them. For OMoE, see P1-5. |
| **P2-4** | Standard CLIP suite has 11 datasets (ImageNet, SUN397, StanfordCars omitted) | R2-5d | **CODE preferred, otherwise PUSHBACK + WRITE** | §4.1 Datasets | Run the missing three on at least 4-shot (codebase already has dataset loaders: `datasets/imagenet.py`, `sun397.py`, `stanford_cars.py`). ImageNet 4-shot/3 seeds is the most expensive but most important — reviewers will weight its absence. If time permits, run all three; if not, run ImageNet + SUN397 and defend StanfordCars omission. |
| **P2-5** | Release code + exact seeds | R2-5e | **ANSWER (commit)** | Data/code availability statement | Promise public release on acceptance (anonymized GitHub link in revision). The `Data availability` section currently says "available upon reasonable request" — strengthen to commit to public release with seeds (1, 2, 3) listed explicitly. |

---

## P3 — Editorial / Quick Fixes

| # | Comment summary | Reviewer | Posture | Section | Suggested action |
|---|---|---|---|---|---|
| **P3-1a** | Abstract says "severe corruption +3.6%", §4.4 says "medium" | R2-5a | **WRITE** (fix abstract) | Abstract, §4.4 | Reconcile — the actual code (`eval_robustness.py`) sets the severity; pick one and use it consistently. From §4.4 text "Medium severity (Noise + Blur)" the **+3.6% is on medium**; fix Abstract to "medium-severity corruption." |
| **P3-1b** | EuroSAT 4-shot: 88.64 (Fig 5) vs 89.06 (Tab 3 — actually Tab 2) vs 88.57 (Tab 6 — actually Tab 5) | R2-5a | **WRITE + verify** | Fig 5, Tables 2, 5 | Re-verify against logs. Most likely: 89.06 = (r=2, H=2, λ=0.03); 88.64 = (r=2, H=2, λ=0.01); 88.57 = (r=4, H=2, λ=0.03) — i.e. three different configs, not three reports of the same number. State the config of each in captions. |
| **P3-1c** | Table 4 Aircraft 16-shot OrthoAdapt = CLIP-LoRA = 54.97 (copy error?) | R2-5a | **WRITE + verify** | Table 3 (the 16-shot table) | Cross-check log; if genuinely identical, mark with a footnote; if a copy error, replace. |
| **P3-1d** | Table 6 (= Table 5 in tex) uses commas as decimal separators | R2-5a | **WRITE** | Table 5 (Latex_code/cas-sc-template.tex L719–723) | Replace `73,27` → `73.27` throughout the table. |
| **P3-2** | u(t) ramp-up function undefined; ε after Eq. 11 doesn't appear in equation | R2-5c | **WRITE** | §3.2.1 (around Eq. 3), §3.3.1 (around Eq. 11) | Add a one-line definition: `u(t) = min(1, t/T)` (verified in `loralib/layers_singlora.py` — confirm exact form). Remove the dangling "ε" sentence after Eq. 11 *or* add ε to Eq. 11 if it actually is in the code. |

---

## Positive Acknowledgments (mention briefly in response letter)

- R1 opening: "well-designed framework", "thorough ablation", "promising performance".
- R2: praised candor on modest gains, found Fig. 8 (gating specialization) "genuinely interesting mechanistic evidence."

Thank both reviewers explicitly; quote R2's "I enjoyed reading the paper" to signal genuine engagement.

---

## Cross-Reviewer Patterns

1. **Both reviewers** flag **scope concerns** (R1-2/R1-Q3 backbone & multi-task; R2-5d missing datasets). Treat backbone scaling (ViT-L/14) and ImageNet as **shared P1 evidence** addressing both at once.
2. **Both reviewers** flag **missing baselines** (R1-4 wants MoRE/O-LoRA/DoRA; R2-5b wants SingLoRA + OMoE distinction). Running SingLoRA + DoRA satisfies both.
3. **Both reviewers** are skeptical of the **performance margin**. R2-3 explicitly (variance); R1 implicitly (calls gains "incremental"). Mean ± std + significance test addresses both.

---

## Suggested Revision Order (work plan)

**Week 1 — Data integrity & non-negotiables (P1-3, P3-1, P1-4)**
1. Re-run all main tables with seeds {1, 2, 3}. ~24 GPU-hours assuming 8 datasets × 3 shots × 3 seeds × 2 methods (Ours + SingLoRA) on a single A100. Use existing `run_all.sh` modified for seed loop.
2. Fix numerical inconsistencies (P3-1a–d) by cross-referencing log files (P3 closes once Week-1 logs are in).
3. Regenerate Fig. 7 with the exact stated config and updated caption (P1-4).

**Week 2 — New baselines & validation protocol (P1-2, P1-7, P2-3, P2-5)**
4. Add SingLoRA and DoRA as full rows in Tables 1–3.
5. Re-do H and λ_o sweeps on validation split (P1-2). Confirm the chosen (H=2, λ=0.03) still wins on val.
6. Prepare anonymous code release (P2-5).

**Week 3 — Scope expansion (P1-6, P2-4)**
7. ViT-L/14 on EuroSAT + DTD + OxfordPets, 4-shot, 3 seeds. ~16 GPU-hours.
8. ImageNet + SUN397 (+ optionally StanfordCars) on ViT-B/16, 4-shot, 3 seeds.

**Week 4 — Writing & framing (P1-1, P1-5, P2-1, P2-2, P3)**
9. Rewrite Abstract, Intro §1, Method §3.2.2, Fig. 2 caption to drop the "PSD escape" framing. New framing: **"input-conditional adaptation + spectral-rank recovery via orthogonal multi-head."** (P1-1)
10. Add §2.3 paragraph positioning vs. MoRE / OMoE (P1-5).
11. Add mechanistic explanation for H > 2 degradation (P2-1).
12. Expand Future Work — multi-task, retrieval, ViT-L/14 (P2-2, P1-6 pushback).
13. Assemble Response Letter using `Response_Letter_Skeleton.md`.
14. Run final consistency pass: numbers, citations, captions.

**Optional defensive experiment (recommended)**: ablation with **signed gating** (gate g_h(x) ∈ [-1, 1], e.g. tanh or signed softmax) — if it works, you can keep something close to the original "PSD escape" claim. If it doesn't (likely — softmax routing has good optimization properties), you have evidence justifying the design choice and turning R2-1 into a *strength* of the paper.

---

## Risk Flags

- **Compute budget**: With multi-seed × ViT-L/14 × multi-baseline this is a non-trivial GPU spend. Estimate **40–60 A100-hours total**. Plan for the deadline.
- **R2-1 is load-bearing**: If you only do the narrative rewrite (no signed-gating experiment), the response letter must be unusually careful. R2 may push back if the rewrite looks like rhetorical fence-sitting rather than a clean concession.
- **Mean ± std might reduce the headline margin further**. Prepare to defend "consistent improvement with modest effect size" as a legitimate finding given the few-shot regime — paper already partially does this in the Abstract.
