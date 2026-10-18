# Matched compute refresh

Implementation comparison: CLIP-LoRA, SingLoRA, dynamic OrthoAdapt dense/manual, and dynamic OrthoAdapt factorized/SDPA. Both OrthoAdapt implementations use the same checkpoint; no ramp-tensor optimization. Defaults retain the full protocol for backward compatibility.

## Explicit economy profile

User approved18 October2026 to reduce rental cost, after inspecting partial full-profile progress. Not chosen by accuracy or favourable speed results. The stopped full run is immutable and excluded from the economy matrix.

```text
--timing-profile economy
```

Eight datasets × four methods × four scopes × three repetitions =384 timing rows. Each has10 warm-ups and30 measured calls:11,520 measured calls total. Smoke separately uses16 cells,2 warm-ups and3 measured calls. Full profile remains5 repetitions/20 warm-ups/100 measured calls,640 rows.

Scopes are forward with autograd; backward with a fresh graph prepared outside timing; directly measured forward+backward; cached-text inference. Training-compute batch32, inference batch256; all class prompts. No optimizer/GradScaler/clipping/scheduler update, no training, no accuracy rerun. Gradient zeroing, input transfers, finite checks and classifier construction are outside timed scopes. Keep existing per-call finite gates, memory snapshots before validation, source/checkpoint/split/input audits and fused-SDPA checks.

Use the first32 records of the frozen4-shot training split and first256 test records, deterministic CLIP preprocessing, seed1. This measures eight workloads, not full-test accuracy. Save all rows including regressions. Median latency, throughput and allocated peak memory are primary descriptive compute metrics. P95 is secondary and less stable at30 calls. Three repetitions are timing repeats, not independent model-training seeds.

## Controller

Stage source from the exact committed revision into a fresh directory, with `source_inventory.json`,24 immutable original checkpoint entries/manifest hashes in `original_inventory.json`, resource detector and approved protocol named `IMPLEMENTATION_PLAN.md`. Never modify the main checkout or old evidence. The legacy conversion remains explicit/user-authorized and separately audited; same original/derived input hashes may be reused, never old timing rows.

```bash
ORTHO_FROZEN_SOURCE=/path/to/immutable/predecessor \
python -u compute_refresh_control.py --run-root /path/to/fresh/run \
  --timing-profile economy --git-revision VERIFIED_COMMIT_SHA
```

Controller runs full tests, resource/environment capture, eight-dataset split/file checks, explicit legacy conversion,32 strict loads,16-cell smoke, then384-cell fresh main matrix. Test dependencies must be available before launch. Hardware identity currently pins the authorized3090 UUID and250W in the harness; a different rental requires a separately audited hardware amendment, not a silent override.

Each main row labels timing profile/warm-up/call count. Coverage validation must use the same explicit profile; do not mix full/economy evidence. Source commit is recorded in protocol metadata; portable sources must be hash-verified against that commit, not merely labelled with it. No automatic restart after failed smoke/main gates. Keep failure/user-stop evidence distinct from scientific failure or successful completion.

## Interpretation

F+B is compute-only, not optimizer-step, epoch or end-to-end training time. Do not sum separately measured F/B to infer the directly measured total. Backward memory includes the prepared graph; reserved memory includes warmed allocator cache. Supported profiler FLOPs are not total FLOPs because fused operators may be unsupported. Do not add inclusive operator times or single-change speedups.

Earlier numerical and all-arm accuracy gates remain failed; combined checkpoint agreement was verified only on a limited two-dataset subset. Compute profiling does not establish optimized training equivalence or preservation of all-eight-dataset accuracy. Do not replace historical actual training-time or accuracy evidence with these measurements.
