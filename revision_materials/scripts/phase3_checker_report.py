#!/usr/bin/env python3
"""Print one-line JSON progress for the Phase 3 main matrix.

ETA is always iteration-based: Phase 3 mixes 1/4/16-shot runs whose cost scales
with n_iters * shots (1-shot=500 iters, 16-shot=8000 iters). A flat mean
runtime/run would badly underestimate the remaining time once the easy 1-shot
runs finish first, so we estimate seconds-per-iteration from completed runs
(Phase 3 first, validation sweep as fallback) and multiply by the total pending
iterations.
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_resumable_runner as runner  # noqa: E402


VN_TZ = timezone(timedelta(hours=7))



def format_finish_time(dt):
    return dt.strftime("%Y-%m-%d %H:%M")


def build_report(
    cost_per_hour_vnd=5000,
    now=None,
    commands_path=runner.DEFAULT_COMMANDS,
    manifest_path=runner.DEFAULT_MANIFEST,
    runtime_sources=None,
):
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    commands = runner.load_phase3_commands(commands_path)
    completed, runtimes = runner.completed_filenames(manifest_path)
    pending = [c for c in commands if runner.extract_filename(c) not in completed]
    done = len(commands) - len(pending)

    # Seconds per iteration from completed rows (Phase 3 manifest first, then the
    # validation sweep protocol manifest). Both expose runtime + iterations.
    sources = list(runtime_sources or [str(manifest_path), str(runner.DEFAULT_RUNTIME_SOURCE)])
    seconds_per_iter = runner.seconds_per_iteration(sources)
    pending_iterations = sum(runner.command_iterations(c) or 0 for c in pending)

    if seconds_per_iter is not None and pending_iterations > 0:
        eta_seconds = seconds_per_iter * pending_iterations
        rate = f"{seconds_per_iter:.4f}s/iter"
    elif runtimes:
        # Last-resort fallback if iteration metadata is missing: flat mean/run.
        mean_runtime = sum(runtimes) / len(runtimes)
        eta_seconds = mean_runtime * len(pending)
        rate = f"{runner.format_duration(mean_runtime)}/run (flat)"
    else:
        eta_seconds = 0
        rate = "unknown"

    eta_seconds_int = int(round(eta_seconds))
    eta_hours = eta_seconds_int / 3600.0
    finish_utc = now + timedelta(seconds=eta_seconds_int)
    finish_vn = finish_utc.astimezone(VN_TZ)
    return {
        "done": done,
        "total": len(commands),
        "pending": len(pending),
        "commands": str(commands_path),
        "manifest": str(manifest_path),
        "pending_iterations": pending_iterations,
        "eta_seconds": eta_seconds_int,
        "eta_human": runner.format_duration(eta_seconds_int),
        "eta_hours": round(eta_hours, 1),
        "cost_remaining_vnd": int(round(eta_hours) * cost_per_hour_vnd),
        "estimated_finish": f"{format_finish_time(finish_vn)} VN",
        "estimated_finish_vn": f"{format_finish_time(finish_vn)} VN",
        "rate": rate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", default=str(runner.DEFAULT_COMMANDS))
    parser.add_argument("--manifest", default=str(runner.DEFAULT_MANIFEST))
    parser.add_argument(
        "--runtime-source",
        action="append",
        default=[],
        help="Additional JSONL manifest for runtime estimation; can be repeated",
    )
    args = parser.parse_args()
    cost = int(os.environ.get("COST_PER_HOUR_VND", "5000"))
    runtime_sources = [args.manifest, *args.runtime_source]
    if runner.DEFAULT_RUNTIME_SOURCE.exists() and str(runner.DEFAULT_RUNTIME_SOURCE) not in runtime_sources:
        runtime_sources.append(str(runner.DEFAULT_RUNTIME_SOURCE))
    print(
        json.dumps(
            build_report(
                cost,
                commands_path=args.commands,
                manifest_path=args.manifest,
                runtime_sources=runtime_sources,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
