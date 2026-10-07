# RTX 3090 execution runbook

## Immutable setup

Use one idle physical RTX 3090 for every new accuracy and compute-profile run.
Record the GPU UUID; do not substitute a 4090. The intended environment is
Ubuntu 22.04, Python 3.11.10, PyTorch 2.5.1+cu124, CUDA runtime 12.4, and cuDNN
90100. A compatible driver is acceptable only when its exact version is saved.

Install PyTorch from the CUDA 12.4 index, install the repository dependencies,
then install `requirements-reviewer4.txt`. Copy this worktree/commit and the
frozen dynamic reference to the server. Set:

```bash
export PYTHON=/path/to/python
export DATA_ROOT=/path/to/datasets
```

## Mandatory order

1. Run `bash revision_materials/reviewer4/protocol/run_preflight_rtx3090.sh`.
   Training must not start unless the RTX 3090 environment, all tests, and all 72
   split identities pass.
2. Run `bash revision_materials/reviewer4/protocol/run_accuracy_pipeline.sh`.
   It performs the smoke, six dynamic sentinels, automatic 66-run fallback when
   required, static 72-run matrix, reloads, and coverage audit.
3. Run `bash revision_materials/reviewer4/protocol/run_compute_profile_pipeline.sh`.
   It executes 24 timed training profiles, audits the common RTX 3090 UUID and
   software stack, performs fixed cached-text inference/FLOP microprofiles, and
   writes the compute tables. Do not run any competing GPU process and do not
   change clocks, power limit, software, or batch size.
4. Inspect the raw operator tables. Unsupported operator FLOPs remain unfilled;
   the scripts never infer them.
5. Run `bash revision_materials/reviewer4/protocol/run_analysis_pipeline.sh`.
   It uses the selected dynamic path, the static reload manifest, 100,000
   dataset-cluster bootstrap replicates with the frozen seed, the secondary
   mixed-effects model, and emits provenance sidecars for derived artifacts.

After all GPU work, save a second `nvidia-smi`, GPU identity, `pip freeze`, git
revision, and environment JSON. Do not report any result until coverage,
checkpoint SHA-256, split, environment, and derived-artifact provenance audits
all pass.
