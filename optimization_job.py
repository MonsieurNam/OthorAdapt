"""Single fail-closed remote job with an archive receipt, even on failure."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

from optimization_evidence import sha256, write_json


def experiment_command(options, output):
    script = 'optimization_amended.py' if options.amended_protocol else 'optimization_experiment.py'
    stage = [] if options.amended_protocol else [options.stage]
    return [sys.executable, '-u', script, *stage, '--evidence-root', options.evidence_root,
            '--data-root', options.data_root, '--out', str(output)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--evidence-root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--stage', default='all', choices=['all', 'old-eval', 'train', 'profile'])
    parser.add_argument('--amended-protocol', action='store_true')
    options = parser.parse_args()
    root = Path(options.root).resolve()
    output = root / 'output'
    log = root / 'run.log'
    if output.exists() or log.exists():
        raise FileExistsError('job output/log already exists; select a new root')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               ORTHO_FROZEN_SOURCE=options.evidence_root)
    code = 1
    with log.open('x', encoding='utf-8') as stream:
        try:
            tests = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
            if tests.returncode:
                code = tests.returncode
            else:
                code = subprocess.run(experiment_command(options, output), env=env,
                                      stdout=stream, stderr=subprocess.STDOUT).returncode
        except Exception:
            traceback.print_exc(file=stream)
    archive = root / 'optimization_evidence.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as package:
        package.write(log, 'run.log')
        for base in [output, root / 'source']:
            if base.exists():
                for path in sorted(base.rglob('*')):
                    if path.is_file() and '__pycache__' not in path.parts:
                        package.write(path, path.relative_to(root).as_posix())
    write_json(root / 'archive_ready.json', dict(exit_code=code, stage=options.stage,
        full_experiment=not options.amended_protocol and options.stage == 'all' and code == 0 and (output / 'completed.json').is_file(),
        amended_bounded_workflow_completed=options.amended_protocol and code == 0 and (output / 'amended_completed.json').is_file(),
        original_numerical_gate_passed=False if options.amended_protocol else None,
        archive=archive.name, sha256=sha256(archive), bytes=archive.stat().st_size,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    sys.exit(code)


if __name__ == '__main__':
    main()
