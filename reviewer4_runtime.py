"""Pure runtime-accounting helpers used by the Reviewer #4 instrumentation."""

from __future__ import annotations

import statistics


def summarize_training_runtime(
    complete_epoch_seconds,
    partial_epoch_seconds,
    total_steps,
    total_seconds,
    peak_allocated_bytes,
    peak_reserved_bytes,
    warmup_epochs=5,
):
    complete = [float(value) for value in complete_epoch_seconds]
    partial = [float(value) for value in partial_epoch_seconds]
    if total_steps <= 0 or total_seconds < 0:
        raise ValueError("total_steps must be positive and total_seconds non-negative")
    if warmup_epochs < 0 or any(value < 0 for value in complete + partial):
        raise ValueError("epoch timings and warmup_epochs must be non-negative")
    steady = complete[int(warmup_epochs):]
    return {
        "complete_epoch_seconds": complete,
        "partial_epoch_seconds": partial,
        "complete_epochs": len(complete),
        "partial_epochs": len(partial),
        "epoch_timing_warmup_epochs": int(warmup_epochs),
        "steady_epoch_count": len(steady),
        "mean_complete_epoch_seconds_after_warmup": statistics.fmean(steady) if steady else None,
        "median_complete_epoch_seconds_after_warmup": statistics.median(steady) if steady else None,
        "seconds_per_optimizer_step": float(total_seconds) / int(total_steps),
        "peak_gpu_allocated_bytes": int(peak_allocated_bytes),
        "peak_gpu_reserved_bytes": int(peak_reserved_bytes),
    }
