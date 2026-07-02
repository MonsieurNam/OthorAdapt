# Phase 4 Backbone Scaling Report

Status: pending pilot; no ViT-L/14 results have been run yet.

Protocol:
- Pilot: EuroSAT 4-shot seed1, CLIP-LoRA r=8 and OrthoAdapt H=2,r=8,lambda_o=0.03,ramp100.
- If pilot completes within available compute, run EuroSAT + Caltech101, 4-shot, seeds 1/2/3, both methods.

Claim gate:
- Do not claim backbone scaling until `phase4_vitl14_results.jsonl` has 12 completed rows and the summary is generated.
- If pilot is too slow or fails, mark ViT-L/14 as deferred in limitations.
