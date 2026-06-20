#!/usr/bin/env python3
"""Resume-safe runner for the Phase 3B same-parameter experiment matrix."""

import sys

import phase3_resumable_runner as runner


DEFAULT_COMMANDS = "revision_materials/scripts/phase3b_same_param_commands.sh"
DEFAULT_MANIFEST = "revision_materials/results/phase3b_same_param_results.jsonl"


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--commands" not in args:
        args.extend(["--commands", DEFAULT_COMMANDS])
    if "--manifest" not in args:
        args.extend(["--manifest", DEFAULT_MANIFEST])
    runner.main(args)


if __name__ == "__main__":
    main()
