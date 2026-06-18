#!/usr/bin/env python3
"""Resume-safe runner for the Phase 3 main experiment matrix."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_COMMANDS = Path("revision_materials/scripts/phase3_main_commands.sh")
DEFAULT_MANIFEST = Path("revision_materials/results/phase3_main_results.jsonl")
DEFAULT_RUNTIME_SOURCE = Path("revision_materials/results/validation_sweep_results_protocol.jsonl")
FILENAME_RE = re.compile(r"(?:^|\s)--filename\s+(\"[^\"]+\"|'[^']+'|\S+)")


@dataclass
class RunResult:
    returncode: int
    elapsed_seconds: float


def extract_filename(command):
    match = FILENAME_RE.search(command)
    if not match:
        raise ValueError(f"Command is missing --filename: {command}")
    value = match.group(1)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def load_phase3_commands(path):
    commands = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(": ") or stripped.startswith("set "):
            continue
        if "--filename" in stripped:
            commands.append(stripped)
    return commands


def _nested(record, path, default=None):
    current = record
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flag_value(command, flag):
    match = re.search(rf"(?:^|\s){re.escape(flag)}\s+(\"[^\"]+\"|'[^']+'|\S+)", command)
    if not match:
        return None
    value = match.group(1)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def command_iterations(command):
    n_iters = _float_or_none(_flag_value(command, "--n_iters"))
    shots = _float_or_none(_flag_value(command, "--shots"))
    if n_iters is None or shots is None:
        return None
    return int(n_iters * shots)


def _iter_manifest_rows(path):
    manifest = Path(path)
    if not manifest.exists():
        return
    with manifest.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row in {manifest} at line {lineno}") from exc


def seconds_per_iteration(manifest_paths):
    samples = []
    for manifest_path in manifest_paths:
        for record in _iter_manifest_rows(manifest_path) or []:
            runtime = _float_or_none(
                _nested(record, "metrics.runtime_seconds")
                or _nested(record, "metrics.fine_tuning_seconds")
            )
            iterations = _float_or_none(_nested(record, "metrics.train_total_iterations"))
            if runtime is not None and iterations is not None and runtime >= 0 and iterations > 0:
                samples.append(runtime / iterations)
    if not samples:
        return None
    return sum(samples) / len(samples)


def completed_filenames(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        return set(), []

    completed = set()
    runtimes = []
    for record in _iter_manifest_rows(path) or []:
        filename = _nested(record, "config.filename")
        checkpoint_sha = _nested(record, "checkpoint.sha256")
        status = record.get("status")
        if status == "completed" and filename and checkpoint_sha:
            completed.add(str(filename))
            runtime = _float_or_none(
                _nested(record, "metrics.runtime_seconds")
                or _nested(record, "metrics.fine_tuning_seconds")
            )
            if runtime is not None and runtime >= 0:
                runtimes.append(runtime)

    return completed, runtimes


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def default_executor(command):
    env = os.environ.copy()
    env.setdefault("PYTHON", "python3")
    env.setdefault("RUN_STAMP", time.strftime("%Y%m%d_%H%M%S"))
    start = time.monotonic()
    result = subprocess.run(
        ["bash", "-lc", f"set -o pipefail; {command}"],
        env=env,
        check=False,
    )
    return RunResult(returncode=result.returncode, elapsed_seconds=time.monotonic() - start)


def run_pending_commands(
    commands,
    completed,
    historical_runtimes=None,
    historical_seconds_per_iteration=None,
    executor=default_executor,
    max_runs=None,
    dry_run=False,
    output=None,
):
    out = output or sys.stdout
    historical_runtimes = list(historical_runtimes or [])
    pending = [(extract_filename(command), command) for command in commands if extract_filename(command) not in completed]
    total = len(commands)
    completed_before = total - len(pending)
    run_limit = len(pending) if max_runs is None else min(max_runs, len(pending))

    def write(message):
        print(message, file=out, flush=True)

    write(f"Phase 3 resume: completed={completed_before}/{total}, pending={len(pending)}")
    # Prefer iteration-based ETA: Phase 3 mixes 1/4/16-shot runs whose cost scales
    # with n_iters * shots, so a flat mean runtime/run badly underestimates the
    # remaining time once the cheap 1-shot runs finish first. Fall back to flat
    # mean only when iteration metadata is unavailable.
    if historical_seconds_per_iteration is not None:
        pending_iterations = sum(command_iterations(command) or 0 for _, command in pending)
        write(
            "Historical mean seconds/iteration="
            f"{historical_seconds_per_iteration:.6f}; ETA for pending="
            f"{format_duration(historical_seconds_per_iteration * pending_iterations)}"
        )
    elif historical_runtimes:
        mean_runtime = sum(historical_runtimes) / len(historical_runtimes)
        write(
            "Historical mean runtime/run (flat fallback)="
            f"{format_duration(mean_runtime)}; ETA for pending={format_duration(mean_runtime * len(pending))}"
        )
    else:
        write("Historical ETA unavailable until one run completes")

    executed = 0
    observed_runtimes = []
    for index, (filename, command) in enumerate(pending[:run_limit], start=1):
        remaining_iterations = sum(
            command_iterations(command) or 0 for _, command in pending[index - 1 :]
        )
        if historical_seconds_per_iteration is not None and remaining_iterations > 0:
            eta = format_duration(historical_seconds_per_iteration * remaining_iterations)
        else:
            elapsed_samples = historical_runtimes + observed_runtimes
            mean_runtime = (sum(elapsed_samples) / len(elapsed_samples)) if elapsed_samples else None
            eta = format_duration(mean_runtime * (len(pending) - index + 1)) if mean_runtime else "unknown"
        write(f"[{completed_before + index}/{total}] {filename} ETA={eta}")

        if dry_run:
            continue

        result = executor(command)
        if result.returncode != 0:
            write(f"FAILED {filename}: returncode={result.returncode}")
            raise SystemExit(result.returncode)
        executed += 1
        observed_runtimes.append(result.elapsed_seconds)
        write(f"completed {filename} runtime={format_duration(result.elapsed_seconds)}")

    remaining_after = len(pending) - executed if not dry_run else len(pending)
    return {
        "total": total,
        "completed_before": completed_before,
        "pending_before": len(pending),
        "executed": executed,
        "remaining_after": remaining_after,
        "dry_run": dry_run,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", default=str(DEFAULT_COMMANDS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--runtime-source",
        action="append",
        default=[],
        help="Additional JSONL manifest for runtime estimation; can be repeated",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Run at most N pending commands")
    parser.add_argument("--dry-run", action="store_true", help="Print pending work without executing commands")
    args = parser.parse_args(argv)

    commands = load_phase3_commands(args.commands)
    completed, runtimes = completed_filenames(args.manifest)
    runtime_sources = [args.manifest, *args.runtime_source]
    if DEFAULT_RUNTIME_SOURCE.exists() and str(DEFAULT_RUNTIME_SOURCE) not in runtime_sources:
        runtime_sources.append(str(DEFAULT_RUNTIME_SOURCE))
    seconds_per_iter = seconds_per_iteration(runtime_sources)
    summary = run_pending_commands(
        commands,
        completed=completed,
        historical_runtimes=runtimes,
        historical_seconds_per_iteration=seconds_per_iter,
        max_runs=args.max_runs,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
