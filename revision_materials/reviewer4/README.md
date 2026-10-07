# Reviewer #4 routing-control evidence

This directory is the immutable evidence root for the input-conditioned versus
input-independent OrthoAdapt control. The scientific protocol is frozen in
`protocol/reviewer4_gating_protocol.yaml` (JSON-compatible YAML).

No accuracy result is valid until all of the following are true:

1. the eight-dataset split identities match `raw/frozen_dynamic_reference.jsonl`;
2. the machine is one physical NVIDIA RTX 3090 and its GPU UUID/software stack
   are recorded;
3. all repository tests pass;
4. the smoke reload difference is at most 0.05 percentage points;
5. the six-run dynamic sentinel gate passes, or a fresh 72-run dynamic matrix is
   completed instead;
6. `reviewer4_audit.py matrix` passes for the 72 static train and reload rows.

Generated commands, manifests, checkpoints, raw profiler output, analysis files,
and paper-ready artifacts belong under this directory. Results are intentionally
not committed before the RTX 3090 run is complete and audited.
