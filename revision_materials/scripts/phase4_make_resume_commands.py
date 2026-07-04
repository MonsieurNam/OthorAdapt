"""Create a Phase 4 robustness resume script by skipping completed jobs.

Completion is determined from the selected Phase 4 robustness manifest. A job is
considered complete only when it has completed rows for severities 0,1,2,3.
"""

from __future__ import annotations

import argparse
import json
import shlex
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMANDS = ROOT / "revision_materials" / "scripts" / "phase4_robustness_commands.sh"
DEFAULT_MANIFEST = ROOT / "revision_materials" / "results" / "phase4_robustness_manifest_fixed.jsonl"
DEFAULT_OUT = ROOT / "revision_materials" / "scripts" / "phase4_robustness_resume.sh"
DEFAULT_LOG_DIR = ROOT / "revision_materials" / "logs" / "phase4_robustness_fixed"
EXPECTED_SEVERITIES = {0, 1, 2, 3}


def completed_jobs(manifest: Path) -> set[str]:
    completed_by_job: dict[str, set[int]] = defaultdict(set)
    if not manifest.exists():
        return set()

    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"warning: skipping malformed manifest line {line_no}")
            continue

        if row.get("schema_version") != "phase4.robustness.v1":
            continue
        if row.get("status") != "completed":
            continue

        job_id = row.get("job_id")
        severity = row.get("severity")
        if job_id is None or severity is None:
            continue

        try:
            completed_by_job[str(job_id)].add(int(severity))
        except (TypeError, ValueError):
            continue

    return {job for job, severities in completed_by_job.items() if severities == EXPECTED_SEVERITIES}


def completed_jobs_from_logs(log_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not log_dir.exists():
        return completed

    for path in log_dir.glob("*.log"):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("job_id") and int(row.get("rows", 0)) == len(EXPECTED_SEVERITIES):
                completed.add(str(row["job_id"]))
            break
    return completed


def filename_from_command(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None

    for idx, token in enumerate(tokens):
        if token == "--filename" and idx + 1 < len(tokens):
            return tokens[idx + 1]
    return None


def make_resume(
    commands_path: Path,
    manifest_path: Path,
    out_path: Path,
    log_dir: Path = DEFAULT_LOG_DIR,
) -> tuple[int, int, int]:
    completed = completed_jobs(manifest_path) | completed_jobs_from_logs(log_dir)
    lines = commands_path.read_text(encoding="utf-8").splitlines()

    header: list[str] = []
    pending: list[str] = []
    total_commands = 0
    skipped_commands = 0

    for line in lines:
        job_id = filename_from_command(line)
        if job_id is None:
            header.append(line)
            continue

        total_commands += 1
        if job_id in completed:
            skipped_commands += 1
        else:
            pending.append(line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header + pending) + "\n", encoding="utf-8")
    try:
        out_path.chmod(0o755)
    except OSError:
        pass

    return len(completed), skipped_commands, total_commands - skipped_commands


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", default=str(DEFAULT_COMMANDS), help="Original full Phase 4 command script.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Phase 4 JSONL manifest.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output resume shell script.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Existing Phase 4 log directory used to skip completed jobs.")
    args = parser.parse_args(argv)

    completed_count, skipped_count, pending_count = make_resume(
        Path(args.commands),
        Path(args.manifest),
        Path(args.out),
        log_dir=Path(args.log_dir),
    )

    out = Path(args.out)
    try:
        out_display = out.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        out_display = str(out)

    print(f"completed_jobs={completed_count}")
    print(f"skipped_commands={skipped_count}")
    print(f"pending_commands={pending_count}")
    print(f"resume_script={out_display}")


if __name__ == "__main__":
    main()
