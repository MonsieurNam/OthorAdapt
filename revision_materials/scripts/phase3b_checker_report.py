#!/usr/bin/env python3
"""Print one-line JSON progress for the Phase 3B same-parameter matrix."""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_resumable_runner as runner  # noqa: E402


DEFAULT_COMMANDS = Path("revision_materials/scripts/phase3b_same_param_commands.sh")
DEFAULT_MANIFEST = Path("revision_materials/results/phase3b_same_param_ramp100_results.jsonl")
DEFAULT_RUNTIME_SOURCE = Path("revision_materials/results/validation_sweep_ramp100_results.jsonl")
VN_TZ = timezone(timedelta(hours=7))


def format_finish_time(dt):
    return dt.strftime("%Y-%m-%d %H:%M")


def build_report(
    cost_per_hour_vnd=5000,
    now=None,
    commands_path=DEFAULT_COMMANDS,
    manifest_path=DEFAULT_MANIFEST,
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

    sources = list(runtime_sources or [str(manifest_path), str(DEFAULT_RUNTIME_SOURCE)])
    seconds_per_iter = runner.seconds_per_iteration(sources)
    pending_iterations = sum(runner.command_iterations(c) or 0 for c in pending)

    if seconds_per_iter is not None and pending_iterations > 0:
        eta_seconds = seconds_per_iter * pending_iterations
        rate = f"{seconds_per_iter:.4f}s/iter"
    elif runtimes:
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
        "pending_iterations": pending_iterations,
        "eta_seconds": eta_seconds_int,
        "eta_human": runner.format_duration(eta_seconds_int),
        "eta_hours": round(eta_hours, 1),
        "cost_remaining_vnd": int(round(eta_hours) * cost_per_hour_vnd),
        "estimated_finish_utc": f"{format_finish_time(finish_utc)} UTC",
        "estimated_finish_vn": f"{format_finish_time(finish_vn)} VN",
        "rate": rate,
    }


def main():
    cost = int(os.environ.get("COST_PER_HOUR_VND", "5000"))
    print(json.dumps(build_report(cost), sort_keys=True))


if __name__ == "__main__":
    main()
