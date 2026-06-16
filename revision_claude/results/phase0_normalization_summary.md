# Phase 0 Normalization Summary

## Inputs

| artifact type | count |
| --- | --- |
| .csv | 5 |
| .drawio | 4 |
| .log | 724 |
| .pdf | 1 |
| .png | 6 |
| .pt | 153 |
| .xlsx | 1 |
| .zip | 20 |

## Generated Outputs

| file | purpose |
| --- | --- |
| revision_claude/results/main_results_manifest.jsonl | Canonical log/CSV evidence manifest in JSONL |
| revision_claude/results/main_results_manifest.csv | Same manifest in CSV |
| revision_claude/results/missing_tier_a_matrix.csv | Tier-A coverage matrix |
| revision_claude/results/missing_tier_a_matrix.md | Human-readable Tier-A coverage report |
| revision_claude/results/workbook_vs_logs_crosscheck.csv | Workbook rows matched against recovered logs |
| revision_claude/results/workbook_vs_logs_crosscheck.md | Human-readable workbook/log cross-check |
| revision_claude/results/phase0_normalization_summary.md | This normalization summary |

## Key Counts

| metric | count |
| --- | --- |
| canonical log records | 724 |
| canonical CSV summary records | 152 |
| Tier-A cells with strict OrthoAdapt R2/H2 evidence | 8 |
| Tier-A cells completely missing recovered evidence | 0 |
| workbook cross-check rows | 782 |
| workbook rows with accuracy match | 577 |
| workbook rows with accuracy mismatch | 169 |
| workbook rows without matching log | 36 |

## Gate Interpretation

Recovered logs and CSVs materially improve provenance, but they do not by themselves satisfy the revised Tier-A protocol because seed identifiers, split hashes, validation-only selection evidence, and seed2/seed3 coverage remain unresolved.

Recommended next step: inspect `missing_tier_a_matrix.md` and rerun/recover only the cells that remain missing or have unclear/nonstandard configuration evidence.
