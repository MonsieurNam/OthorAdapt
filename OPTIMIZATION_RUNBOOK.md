# Bounded factorized + SDPA verification

This worktree is based on `ce709fa5b32b7392db532a3776ab7ecbda8500bc` and retains legacy execution by default.

Opt-in production flags:

```text
--adapter ohsinglora --params q k v
--adapter_compute factorized --attention_backend sdpa
```

Execution choices do not change adapter parameter keys, v3 checkpoint schema, or valid v2 loading. Static and dynamic routing remain distinct model identities. Nondefault backends are limited to the square OrthoAdapt q/k/v configuration.

## Verification

Run `python -m pytest tests -q` locally; CUDA tests must also pass on the target stack with `ORTHO_FROZEN_SOURCE` pointing to the immutable source import. Do not use archive/snapshot directories as pytest roots.

`optimization_experiment.py all --evidence-root <frozen-source-root> --data-root /root/DATA --out <new-output-directory>` performs, in order:

1. Actual-batch FP32/FP16 numerical preflight for both datasets and routing modes.
2. Thirty-two existing-checkpoint full-test evaluations (eight checkpoints, four execution arms).
3. Four paired fixed-iteration training runs, with every augmented batch hashed before optimizer updates; eight new-checkpoint evaluations.
4. Production-path profiling (160 rows), separate traces, strict raw-data audit and report.

`optimization_job.py` runs target-stack tests before the experiment, preserves the log and output in an archive even on failure, and writes `archive_ready.json` with exit code, completion flag, size and SHA-256. A standalone stage is explicitly partial and cannot claim full completion. Tests or accuracy gate failure stops subsequent stages; no automatic full matrix rerun occurs.

The old AMP sequence clips scaled gradients before `GradScaler.step`. It is intentionally preserved for this isolated optimization test, not endorsed as a best-practice change. Observer hooks read detached values only and never call `unscale_()` or modify optimizer inputs.

## Interpretation

Numeric tolerances are FP32 1e-5/1e-4 and FP16 0.02/0.02. Real prediction gates are stricter: <=0.05 pp accuracy difference and >=99.9% prediction agreement for every checkpoint/arm. The four-run training screen uses <=0.5 pp, not a claim of statistical equivalence. Speed gates concern paired forward/backward and cached-text inference wall latency, not optimizer-inclusive training time. Timing repetitions are not dataset/seed confidence intervals.

Use the pinned RTX 3090 physical UUID at 250 W, one GPU job at a time. Do not compare these absolute latencies with previous 320 W measurements. Existing scientific evidence and manuscript are not replaced by this bounded experiment.
