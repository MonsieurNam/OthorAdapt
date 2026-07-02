# OrthoAdapt Master Revision Plan

> **For agentic workers:** Use this file as the single execution entrypoint. Plan A is the scientific backbone. Plan B is only a communication layer. Do not write final response-letter claims until the evidence artifacts named below exist.

**Goal:** Produce a defensible revised OrthoAdapt manuscript, response letter, and submission package with every scientific claim traceable to verified artifacts, rerun manifests, source-code inspection, author decisions, or explicit limitations.

**Architecture:** Treat the revision as an evidence pipeline. First establish provenance and repair validity-breaking code paths, then freeze validation-only selection, rerun acceptance-critical experiments, generate statistics/tables from manifests, rewrite the manuscript, and only then write the response letter.

**Tech Stack:** Python, PyTorch, CLIP, NumPy/SciPy, pytest, LaTeX, CSV/JSONL manifests, Markdown tracking files.

---

## 0. Source Hierarchy & Evidence Rules

### Source Hierarchy

| Source | Role | Use |
|---|---|---|
| `revision_codex/docs/superpowers/plans/2026-06-15-reviewer-response-revision-plan.md` | Plan A: master scientific backbone | Primary task list, evidence gates, rerun logic, theory corrections |
| `revision_materials/plan/Revision_Roadmap.md` | Plan B: communication / triage layer | Co-author-facing roadmap only |
| `revision_materials/plan/Triage_Summary.md` | Plan B: triage layer | Quick issue dashboard only |
| `revision_materials/plan/Response_Letter_Skeleton.md` | Template only | Response structure only; not evidence |
| `revision_materials/plan/Judge_Report_PlanA_vs_PlanB.md` | Conflict-resolution authority | Missing items, Stop/Go gates, safety rules, Plan A vs Plan B resolution |

### Evidence Priority

Scientific evidence priority:
1. verified original artifacts,
2. rerun manifests,
3. source-code inspection,
4. explicit author decisions,
5. limitation or future-work statements.

Planning files, response-letter skeletons, and speculative explanations are not evidence.

### Unsupported-Claim Placeholders

Use these placeholders until evidence exists:

- `[UNVERIFIED - RERUN REQUIRED]`
- `[PENDING MANIFEST]`
- `[NEEDS AUTHOR DECISION]`
- `[SOURCE-CODE INSPECTION REQUIRED]`
- `[CLAIM MUST BE WEAKENED OR REMOVED]`

### Forbidden Claims Without Evidence

Do not write these phrases, or equivalent claims, unless supported by verified artifacts:

- "we have re-run"
- "Table X now shows"
- "Figure Y has been regenerated"
- "the results demonstrate"
- "H>2 degradation is caused by rank fragmentation"
- "orthogonality causes robustness"
- "the original submission used one seed"

---

## 1. Author_Decisions.md

**File:** `revision_materials/plan/Author_Decisions.md`

Author decisions must be recorded before running experiments that depend on them.

### 1.1 Technical Decisions

- [x] Validation rule for hyperparameter selection.
- [x] Iteration semantics: keep `500 x shots` or change to 500 total steps.
- [x] Seed count: three seeds or five seeds.
- [x] DoRA inclusion.
- [x] ImageNet inclusion.
- [x] Code release: anonymized GitHub or release on acceptance.

### 1.2 Administrative / Publishing Decisions

- [x] AI disclosure.
- [x] CRediT statement: who did what.
- [x] Funding / acknowledgement: confirm whether an FPT project ID exists.
- [x] Conflict of Interest: confirm with all authors.
- [x] Highlights box.
- [x] Robustness protocol: keep custom corruption set but record exact parameters.
- [x] Cover letter / clean manuscript / marked-up manuscript package.

---

## 2. Phase 0 - Provenance & Integrity Gate

**Purpose:** Establish what original evidence exists and what must be rerun.

**Required outputs:**
- `artifact_inventory.md`
- `revision_traceability.csv`
- `unresolved_numbers.md`
- `reviewer_table_figure_mapping.md`

**Tasks:**
- [x] Search for original checkpoints, logs, CSVs, notebooks, Colab outputs, and seed records. **Complete after raw-evidence update: 724 logs, 5 CSV summaries, 20 zip archives, 153 loadable checkpoints, 1 workbook, and figure/source assets recovered; notebooks, split hashes, final rerun/evaluation manifests, and seed2/seed3 evidence remain missing.**
- [x] Hash every recovered artifact and record path, timestamp, command/config, seed, split, and linked manuscript value. **Complete via `artifact_inventory.md`, `phase0_raw_artifact_manifest.csv`, and `revision_materials/results/main_results_manifest.*`; log final accuracies, zero-shot accuracies, CSV rows, checkpoint metadata, and path-derived hints were extracted where available, but split/seed/config metadata remain incomplete.**
- [x] Mark each manuscript number as `VERIFIED`, `UNVERIFIED_RERUN_REQUIRED`, or `SOURCE-CODE_INSPECTION_REQUIRED`. **Complete in `revision_traceability.csv`, `unresolved_numbers.md`, `missing_tier_a_matrix.*`, and `workbook_vs_logs_crosscheck.*`; no acceptance-critical numerical claim is currently verified because seed2/seed3/split-hash evidence and final validation/test manifests are still absent.**
- [x] Create `reviewer_table_figure_mapping.md` before editing any response paragraph. **Complete after raw-log/checkpoint evidence update.**

**Reviewer table/figure mapping format:**

| Reviewer reference | Rendered paper number | LaTeX label | Source `.tex` location | Issue | Action ID | Status |
|---|---|---|---|---|---|---|

**Gate:** If original artifacts are not found or are incomplete, schedule full rerun for acceptance-critical results.

---

## 3. Phase 1A - Emergency Code Repair / Validity Fixes

**Purpose:** Fix validity-breaking code paths before any wide experiment.

**Required outputs:**
- [x] code repair commits or patches. **Implemented in `analyze_spectrum.py`, `loralib/layers_OH_singlora.py`, `run_utils.py`, `main.py`, `lora.py`, `loralib/utils.py`, and `ecr3_provenance.py`.**
- [x] test logs. **Verified on 2026-06-16 with `.venv\Scripts\python.exe -m unittest discover -s tests -v`: 12 tests passed before Phase 1B; after Phase 1B additions the expanded suite must remain green before experiments.**
- [x] fixed scripts. **ECR1 spectrum script fails closed; ECR2 orthogonality loss has explicit reduction; ECR3 run path blocks test-set sweep tuning and records split provenance.**
- [x] ECR completion note. **ECR1-ECR3 gate closed after installing project-local CPU PyTorch in `.venv`; `analyze_spectrum.py --lora_path missing_lora.pt --oh_path missing_oh.pt` exits non-zero with a missing-checkpoint error.**

### ECR1. Remove Synthetic Spectrum Fallback

**File:** `analyze_spectrum.py`

- [x] Remove synthetic/demo output. **Implemented in `analyze_spectrum.py`; both `--lora_path` and `--oh_path` are required.**
- [x] Make missing checkpoints fail closed with a non-zero exit code. **`validate_spectrum_args` raises `SpectrumError`; CLI exits non-zero before loading scientific stack.**
- [x] Require checkpoint metadata and matrix definition before plotting. **`require_checkpoint_metadata` and `matrix_definition` are required before report/plot generation.**
- [x] Add or update tests proving missing checkpoints fail. **Covered by `tests/test_ecr1_spectrum.py`.**

### ECR2. Fix Orthogonality Loss Scaling

**File:** `loralib/layers_OH_singlora.py`

- [x] Add `reduction={sum,mean}`. **Implemented in `calculate_ortho_loss(lora_A_heads, reduction=...)`.**
- [x] Use pair-normalized mean for validation and diagnostic runs. **Implemented as `--ortho_reduction`, default `mean`.**
- [x] Log raw pair-sum loss and normalized mean loss. **Training logs now include objective reduction and `Ortho-raw-sum`.**
- [x] Add tests covering H=2 vs H=4 pair-count scaling. **Added `tests/test_ecr2_ortho_loss.py`; numeric assertions run when PyTorch is installed.**

### ECR3. Prevent Test-Set Tuning

**Files:** `main.py`, `run_utils.py`, `lora.py`

- [x] Add `--selection_split {val,test}` with default `val`. **Implemented in `run_utils.py`; default selection path in `lora.py` now evaluates the validation split.**
- [x] Forbid `test` split during sweep mode. **Implemented by `validate_reporting_policy`; `--sweep_mode` rejects test selection and test reporting.**
- [x] Add explicit test-reporting flag for final selected config only. **Implemented as `--report_test`; test accuracy is not evaluated/reported unless explicitly requested.**
- [x] Record split counts and split hashes. **Implemented via `ecr3_provenance.py`; `main.py` records train/val/test split hashes, and run records/checkpoints include ECR3 provenance metadata.**

**Hard gate:** Do not run validation sweeps, main matrix, figure regeneration, or response-letter claim writing until ECR1-ECR3 are complete.

---

## 4. Phase 1B - Experiment Infrastructure

**Purpose:** Make reruns reproducible and table generation single-source.

**Required outputs:**
- [x] metadata schema. **Defined in `revision_materials/plan/phase1b_run_manifest_schema.md`; nested ECR3 records normalize to the canonical flat row.**
- [x] run manifest schema. **Implemented by `experiment_manifest.py` with `phase1b.run.v1` validation and fail-closed required fields.**
- [x] documented `statistical_tests.py` output format. **Documented in `statistical_tests.py` and `phase1b_run_manifest_schema.md`: mean, sample std, 95% CI, paired deltas, paired CI, and paired Cohen dz.**
- [x] aggregation dry-run logs. **Dry-run fixture: `revision_materials/fixtures/phase1b_dry_run_manifest.jsonl`; generated summary: `revision_materials/results/phase1b_dry_run_summary.csv`.**

**Tasks:**
- [x] Define JSONL row schema for run results. **Canonical flat schema is `phase1b.run.v1`; nested ECR3 records are accepted and normalized.**
- [x] Add config, seed, split, dataset, method, shot, accuracy, runtime, parameter count, checkpoint hash, and git revision. **`ecr3_provenance.build_run_record` now records `git_revision`; `lora.py` writes runtime and trainable parameter count into manifest metrics; `experiment_manifest.validate_run_record` enforces all aggregation fields.**
- [x] Implement or repair aggregation script. **Implemented `aggregate_results.py` with JSONL loader, fail-closed validation, grouped aggregation, and CSV export.**
- [x] Implement statistical report format with means, standard deviations, CIs, paired tests, and effect sizes. **Implemented in `aggregate_results.aggregate_manifest`; `statistical_tests.py` documents the output contract.**
- [x] Add dry-run fixtures so the aggregation pipeline can be tested before expensive experiments. **Added `revision_materials/fixtures/phase1b_dry_run_manifest.jsonl` and verified aggregation to `revision_materials/results/phase1b_dry_run_summary.csv`.**

---

## 5. Phase 2 - Pre-Registered Validation Sweep

**Purpose:** Close the test-set tuning issue before final test evaluation.

**Required outputs:**
- [x] `selection_protocol.yaml`. **Frozen in `revision_materials/plan/selection_protocol.yaml`; JSON-compatible YAML to avoid extra parser dependency; uses validation split only.**
- [x] `validation_sweep_results.jsonl`. **Server run completed with 121 raw rows: 120 protocol-valid validation rows plus one preserved pre-sweep manual row outside the frozen grid. Protocol-only manifest is `revision_materials/results/validation_sweep_results_protocol.jsonl`; audit is `revision_materials/results/validation_sweep_protocol_audit.md`.**
- [x] `selected_config.md`. **Generated from the 120-row protocol-only manifest after fail-closed selector rejected the raw manifest's out-of-protocol row. Winner: `num_heads=2`, `r=4`, `lambda_o=0.0`, mean validation accuracy `91.541667`.**
- [x] `validation_sweep_ramp100_results.jsonl`. **Ramp100 validation rerun complete: 120/120 rows, all `completed`, all `selection_split=val`, all `report_test=false`, all `ramp_up_steps=100`; 20 candidates each have full 6-row coverage over EuroSAT/Caltech101, 4-shot, seeds {1,2,3}.**
- [x] `selected_config_ramp100.md`. **Freeze complete. Winner changed to `num_heads=2`, `r=8`, `lambda_o=0.03`, mean validation accuracy `91.666667`; Phase 3 main protocol and command generation must use this winner.**

**Tasks:**
- [x] Freeze validation aggregate and rationale before running sweeps. **Protocol selects by unweighted mean validation accuracy over EuroSAT and Caltech101, 4-shot, seeds {1,2,3}; candidate grid and tie-breaks match `Author_Decisions.md`.**
- [x] Sweep only validation data. **`phase2_validation_sweep.py generate` produced 120 commands in `revision_materials/scripts/validation_sweep_commands.sh`; every command uses `--selection_split val` and `--sweep_mode`, with no `--report_test`.**
- [x] Record all candidates in JSONL. **Server manifest contains all 120 frozen candidate rows exactly once; the raw manifest also preserves one accidental pre-sweep manual row for auditability.**
- [x] Select winner by frozen rule only. **Selector failed closed on the raw 121-row manifest, then succeeded on `validation_sweep_results_protocol.jsonl` after excluding the documented out-of-protocol row. Output: `revision_materials/results/selected_config.md`.**
- [x] Hash and freeze `selected_config.md`. **SHA256 sidecar exists at `revision_materials/results/selected_config.md.sha256`.**
- [x] Do not evaluate test data for non-winning candidates. **Ramp100 validation manifest confirms all 120 sweep rows are validation-only with no test reporting. Phase 3 test-set evaluation is now allowed only for the frozen ramp100 winner `H=2,r=8,lambda_o=0.03` and the matched CLIP-LoRA baseline in `phase3_main_protocol.yaml`.**

---

## 6. Phase 3 - Main Experiment Matrix

**Purpose:** Produce acceptance-critical performance evidence.

**Required outputs:**
- [x] `main_results_manifest.jsonl` / final rerun manifest. **Final Tier-A rerun evidence is `revision_materials/results/phase3_main_ramp100_results.jsonl`, with 144/144 completed test rows after validation-only ramp100 selection. Phase 0 recovered evidence remains in `revision_materials/results/main_results_manifest.jsonl` and is not used as the final Tier-A table source.**
- [x] `statistical_report.md`. **Generated at `revision_materials/results/statistical_report.md` from the ramp100 Phase 3 manifest; includes full coverage audit, mean/std/95% CI, paired OrthoAdapt-minus-CLIP-LoRA deltas, parameter counts, and claim implications.**
- [x] `generated_tables.tex`. **Generated at `revision_materials/results/generated_tables.tex`; includes 1-shot, 4-shot, 16-shot main tables and a paired-delta table from the ramp100 manifest.**
- [x] summary CSV/Markdown. **Generated `revision_materials/results/phase3_main_ramp100_summary.csv`, `revision_materials/results/phase3_main_ramp100_summary.md`, `revision_materials/results/phase3_main_ramp100_paired_summary.csv`, and `revision_materials/results/phase3_main_ramp100_audit.md`.**
- [x] Phase 3 run protocol and server command script. **Started with `revision_materials/plan/phase3_main_protocol.yaml`, `revision_materials/scripts/phase3_main_commands.sh`, and `revision_materials/plan/phase3_server_run_instructions.md`. Updated after ramp100 selection to 144 test-set commands: 8 datasets x 3 shots x 2 methods x 3 seeds. The OrthoAdapt arm uses the frozen validation winner `H=2,r=8,lambda_o=0.03,ramp_up_steps=100`; the CLIP-LoRA arm uses `r=8` for a selected-rank matched comparison. SingLoRA-CLIP is excluded from the main matrix by author decision because it is an internal unpublished CLIP adaptation, not a stable independently citable few-shot CLIP baseline.**

**Minimum Tier A tasks:**
- [x] Run CLIP-LoRA and OrthoAdapt. SingLoRA-CLIP is documented as excluded from the main baseline matrix; do not present it as Tier-A reviewer evidence. **Complete in `phase3_main_ramp100_results.jsonl`: CLIP-LoRA r=8 has 72 rows and OrthoAdapt/OH-SingLoRA H=2,r=8,lambda_o=0.03,ramp100 has 72 rows.**
- [x] Use all original main datasets and shots. **Complete: 8 datasets (`fgvc`, `eurosat`, `food101`, `oxford_pets`, `oxford_flowers`, `caltech101`, `dtd`, `ucf101`) x 3 shots (`1`, `4`, `16`).**
- [x] Use at least three seeds. **Complete: seeds `{1,2,3}` for every dataset-shot-method cell.**
- [x] Add paired statistics and confidence intervals. **Complete in `statistical_report.md` and `phase3_main_ramp100_paired_summary.csv`; pairing is exact by `(dataset, shot, seed)`, with 72/72 matched pairs.**
- [x] Include all parameter counts including gating network parameters. **Complete in manifest/report/table outputs: CLIP-LoRA r=8 has 737,280 trainable parameters; OrthoAdapt H=2,r=8 has 460,800 trainable parameters. LoRA rows store `ramp_up_steps=100` only as normalized metadata from the shared CLI schema; ramp-up is not a LoRA mechanism.**

**Phase 3 claim gate after ramp100 evidence:**
- [x] Keep: validation-only hyperparameter selection was enforced before final test reporting.
- [x] Keep: final Tier-A evidence is a complete paired 8-dataset, 3-shot, 3-seed matrix for CLIP-LoRA r=8 versus OrthoAdapt H=2,r=8,lambda_o=0.03,ramp100.
- [x] Keep with precise wording: OrthoAdapt shows a modest positive average paired delta in the ramp100 final matrix while using fewer trainable parameters than CLIP-LoRA r=8.
- [x] Weaken/remove: broad state-of-the-art, large-gain, or consistent-superiority claims.
- [x] Weaken/remove until Phase 4 evidence exists: causal robustness claims, rank-fragmentation explanations, and mechanistic claims about head-count behavior.

**Scope additions if feasible:**
- [ ] Add SUN397 and StanfordCars.
- [ ] Add ImageNet only if pilot benchmark says schedule can absorb it.
- [ ] Add DoRA only if feasibility gate passes.

---

## 7. Phase 4 - Diagnostics / Tier B

**Purpose:** Support mechanistic claims only where diagnostics justify them.

**Required outputs:**
- [x] diagnostic manifests. **Generated existing-diagnostic and checkpoint-diagnostic artifacts: `phase4_existing_diagnostics.csv`, `phase4_checkpoint_diagnostics.jsonl`, and `phase4_spectrum_manifest.jsonl`. Robustness and ViT-L/14 manifests are generated by server commands and remain pending until those jobs finish.**
- [x] spectral-analysis report. **Generated `revision_materials/results/phase4_spectrum_report.md` and figures in `revision_materials/results/figures/phase4_spectrum/` from paired Phase 3 ramp100 checkpoints. This is descriptive evidence only, not causal proof.**
- [ ] robustness report. **Infrastructure generated: `revision_materials/scripts/phase4_eval_robustness.py`, `phase4_robustness_commands.sh`, and `phase4_aggregate_robustness.py`. Server run is still required to produce `phase4_robustness_manifest.jsonl`, `phase4_robustness_summary.csv`, and `phase4_robustness_report.md`.**
- [ ] backbone-scaling report. **ViT-L/14 pilot/subset commands generated in `phase4_vitl14_pilot_commands.sh` and `phase4_vitl14_commands.sh`; `phase4_backbone_scaling_report.md` currently records pending/deferred status until pilot/subset results exist.**
- [x] Phase 4 run instructions. **Created `revision_materials/plan/phase4_server_run_instructions.md` with robustness and ViT-L/14 server commands and claim gates.**

**Tasks:**
- [x] Run fixed-total-rank and fixed-rank-per-head H studies. **Existing validation evidence was consolidated in `phase4_existing_diagnostics.md/csv`: ramp100 validation sweep covers H/r/lambda over EuroSAT+Caltech101, 4-shot, seeds {1,2,3}; W3 adds H=1 for r={2,4}. This supports only configuration-sensitivity language.**
- [x] Compare raw orthogonality sum vs pair-normalized mean. **Checkpoint diagnostics over 72 final OrthoAdapt checkpoints compute raw orthogonality and normalized subspace-overlap metrics; summarized in `phase4_orthogonality_report.md`.**
- [x] Report head norms and subspace overlap metrics. **Generated `phase4_head_overlap_report.md` from `phase4_checkpoint_diagnostics.jsonl`; 72/72 OrthoAdapt Phase 3 ramp100 checkpoints processed.**
- [x] Replace invalid spectral figure with rank-consistent metrics. **Generated paired spectrum diagnostics for all 8 datasets, shot=4, seed=1, vision layer 11, q/v projections. Old spectrum figures remain legacy only.**
- [ ] Run ViT-L/14 subset if feasible. **Pilot and subset command scripts are ready; run pilot first and continue only if runtime is acceptable.**
- [ ] Run robustness with deterministic paired corruptions and exact transform parameters. **Deterministic evaluator and 144-job command script are ready; server run remains pending.**

**Claim rule:** Do not claim rank fragmentation or orthogonality-caused robustness unless these diagnostics support it.

**Phase 4 claim gate after local diagnostics:**
- [x] Keep: H/r/lambda behavior is configuration-dependent under validation-only diagnostics.
- [x] Keep: final OrthoAdapt checkpoints learn low-overlap head tensors, descriptively.
- [x] Keep with caution: spectrum diagnostics may be used only as descriptive, checkpoint-backed evidence for the selected subset.
- [x] Weaken/remove until robustness server run completes: robustness claims and orthogonality-causes-robustness wording.
- [x] Weaken/remove until ViT-L/14 subset completes: backbone-scaling claims.

---

## 8. Phase 5 - Deferred / Out-of-Scope

**Purpose:** Keep the revision focused.

**Do not undertake in this revision cycle unless the user explicitly re-scopes the project:**
- signed-gating architecture,
- full MoRE/O-LoRA reproduction in their original settings,
- multi-task adaptation benchmark,
- image-text retrieval benchmark.

**Required output:**
- explicit limitation/future-work text for any deferred reviewer request.

---

## 9. Phase 6 - Manuscript Rewrite

**Purpose:** Make manuscript claims match verified evidence.

**Required outputs:**
- revised manuscript sections,
- updated tables/figures,
- limitations section,
- data/code availability text.

**Tasks:**
- [ ] Rewrite PSD/rank theory: softmax-weighted PSD sums remain PSD.
- [ ] Keep input-conditioned nonlinearity claim only where mathematically correct.
- [ ] Replace invalid spectral discussion.
- [ ] Replace "state-of-the-art" and "significant" wording unless statistically supported.
- [ ] Add honest per-dataset loss discussion.
- [ ] Add exact seed, split, validation, and iteration protocols.
- [ ] Add release and reproducibility details.

---

## 10. Phase 7 - Response Letter + Cover Letter + Publishing Package

**Purpose:** Respond only after manuscript and evidence are ready.

**Required outputs:**
- `reviewer_response_matrix.md`
- response letter
- cover letter
- clean manuscript
- marked-up manuscript

**Start gate:**
Response-letter drafting may begin only after:
- `revision_traceability.csv` exists,
- `main_results_manifest.jsonl` exists or unresolved claims are explicitly marked,
- `reviewer_response_matrix.md` maps every comment to evidence/action,
- manuscript sections have been updated or marked pending.

**Tasks:**
- [ ] Use `Response_Letter_Skeleton.md` only as structure.
- [ ] Replace every placeholder with evidence-backed text.
- [ ] Link every response to manuscript location and evidence artifact.
- [ ] Do not use "we have..." language without manifest support.
- [ ] Include cover letter summary and publishing disclosures.

---

## 11. Phase 8 - Final Integrity Pass

**Required outputs:**
- final consistency report,
- LaTeX build logs,
- submission checklist.

**Tasks:**
- [ ] Verify every reviewer comment has one response and one evidence/action link.
- [ ] Recompute all table averages from manifests.
- [ ] Cross-check abstract, tables, figures, and conclusion against generated artifacts.
- [ ] Check references and citation metadata.
- [ ] Build clean and marked manuscripts.
- [ ] Confirm AI disclosure, CRediT, funding, conflict of interest, highlights, and data availability.

---

## 12. Stop/Go Decision Gates

| Target date | Gate | Decision | Stop or rescope condition |
|---|---|---|---|
| Jun 18 EOD | After Phase 0 | Use recovered artifacts or full rerun | If original artifacts are missing, schedule full rerun |
| Jun 19 EOD | After Emergency Code Repair | Continue to experiment infrastructure | If ECR1-ECR3 are incomplete, do not run experiments |
| Jun 23 EOD | After validation infrastructure | Continue to validation sweep | If aggregation/statistical scripts fail, pause and fix infrastructure |
| Jun 24 PM | After pilot benchmark | Run full Tier A or rescope | If Tier A cannot finish on time, drop ImageNet and DoRA first |
| Jun 23-24 | After validation sweep | Freeze selected configuration | If selection artifacts are incomplete, do not run test-set evaluation |
| Jul 3 EOD | After Tier A | Keep, narrow, or weaken claims | If three-seed results are incomplete, narrow claims and update limitations |
| Jul 6 EOD | After Tier B | Keep or downgrade scope-expansion claims | If diagnostics or ViT-L/14 are incomplete, mark partial or defer |
| Jul 11 EOD | Before response package | Proceed to response package | If manuscript rewrite is incomplete, request extension or submit minimum viable revision only |
| Jul 12-13 | Before response letter | Draft response or hold | If evidence/action mapping is incomplete, do not write final response paragraphs |

---

## 13. Agent System Rules

### Evidence-First Revision Rule

The agent must treat scientific evidence in this priority order:
1. verified original artifacts,
2. rerun manifests,
3. source-code inspection,
4. explicit author decisions,
5. limitation/future-work statements.

The agent must not treat planning files, response-letter skeletons, or speculative explanations as evidence.

### Forbidden Actions

- Do not use `Response_Letter_Skeleton.md` as evidence.
- Do not use Plan B as scientific evidence.
- Do not write "we have re-run", "Table X now shows", "Figure Y has been regenerated", or "the results demonstrate" without manifest evidence.
- Do not assert original submission used one seed unless provenance confirms it.
- Do not assert H>2 fails due to rank fragmentation unless diagnostics support it.
- Do not claim orthogonality causes robustness unless ablations support causality.
- Do not add signed-gating architecture in this revision cycle.
- Do not port DoRA unless the feasibility gate passes.

### Minimum Viable Revision

If time or compute is constrained, prioritize:
1. fix PSD/rank theoretical claim,
2. fix test-set tuning with validation-only protocol,
3. rerun CLIP-LoRA and OrthoAdapt with three seeds,
4. add statistical tests and confidence intervals,
5. fix table/figure inconsistencies,
6. add ViT-L/14 subset if feasible,
7. add limitations and honest per-dataset losses,
8. provide anonymized code/reproducibility package or concrete release plan.
