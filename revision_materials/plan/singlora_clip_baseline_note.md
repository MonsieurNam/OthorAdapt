# SingLoRA-CLIP Baseline Note

## Decision

SingLoRA-CLIP is not included in the Tier-A Phase 3 main matrix.

The revised Phase 3 matrix is:

```text
8 datasets x 3 shots x 3 seeds x 2 methods = 144 runs
methods = {CLIP-LoRA, OrthoAdapt}
```

## Reviewer-Facing Explanation

We do not include SingLoRA-CLIP as a main baseline because it is an internal,
unpublished CLIP adaptation that is still under investigation, not a stable
independently citable few-shot CLIP method. The main empirical comparison is
therefore kept to CLIP-LoRA versus OrthoAdapt under matched datasets, shots,
seeds, rank budget, backbone, encoder scope, and training schedule.

This choice improves reproducibility and interpretability. A reviewer can trace
the baseline to a public few-shot VLM method, rerun the protocol, and evaluate
whether OrthoAdapt improves over the closest published low-rank CLIP baseline.
Adding an internal SingLoRA-CLIP variant to the acceptance-critical table would
mix unpublished exploratory work into the main evidence and could make the
comparison harder to audit.

Suggested response-letter wording:

```text
We thank the reviewer for the baseline suggestion. In the revised experiments,
we focus the Tier-A main table on CLIP-LoRA and OrthoAdapt under a fully paired
protocol. We did not include SingLoRA-CLIP as a main baseline because that
variant is an internal unpublished CLIP adaptation rather than a stable,
independently citable few-shot CLIP method. The public SingLoRA preprint studies
a general single-matrix PEFT formulation, whereas CLIP-LoRA is directly proposed
and evaluated for few-shot adaptation of vision-language models. We therefore
use CLIP-LoRA as the primary low-rank CLIP baseline and document SingLoRA-CLIP
as internal exploratory work/future work unless a separately published and
reproducible CLIP-specific version becomes available.
```

## Evidence and Citations

1. CLIP-LoRA is the direct public baseline for this paper's setting. Zanella and
   Ben Ayed introduce LoRA for few-shot vision-language models, evaluate it on
   11 datasets, and explicitly position CLIP-LoRA as a strong baseline for
   progress in few-shot VLM adaptation.

   Source: Maxime Zanella and Ismail Ben Ayed, "Low-Rank Few-Shot Adaptation of
   Vision-Language Models", arXiv:2405.18541, 2024.
   https://arxiv.org/abs/2405.18541

2. The citable SingLoRA work is a general PEFT method, not a CLIP few-shot
   baseline. Its abstract describes a single-matrix low-rank adaptation method
   with experiments on tasks such as common-sense reasoning and image generation.
   It does not establish the repository's SingLoRA-CLIP variant as a published
   few-shot CLIP baseline.

   Source: David Bensaid, Noam Rotstein, Roy Velich, Daniel Bensaid, and Ron
   Kimmel, "SingLoRA: Low Rank Adaptation Using a Single Matrix",
   arXiv:2507.05566, 2025.
   https://arxiv.org/abs/2507.05566

3. LoRA remains the common PEFT foundation behind both CLIP-LoRA and OrthoAdapt.
   The original LoRA paper proposes freezing pretrained weights and injecting
   trainable low-rank matrices, motivating the use of a CLIP-LoRA-style baseline
   for a low-rank adaptation paper.

   Source: Edward J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language
   Models", arXiv:2106.09685, 2021.
   https://arxiv.org/abs/2106.09685

## Protocol Consequence

The Phase 3 protocol uses only the public direct baseline and the proposed
method:

- `lora`: CLIP-LoRA baseline, rank `r=4`.
- `ohsinglora`: OrthoAdapt selected configuration, `H=2`, `r=4`,
  `lambda_o=0.0`.

No Phase 3 command should contain `--adapter singlora`.
