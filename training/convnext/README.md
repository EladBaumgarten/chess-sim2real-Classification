# ConvNeXt-Tiny architecture comparison

Replicates our three ResNet-18 training styles on a **ConvNeXt-Tiny** backbone, to compare
architecture-vs-architecture on the **same data, splits, crop/warp pipeline, and games-2/6
eval harness**. Only the backbone changes (plus an architecture-appropriate AdamW/cosine
recipe instead of SGD — intentional and logged per run).

## Scope

Exactly **3 runs**. We do NOT re-run the five-lever investigation (weighting / L2-SP /
BN-freeze / rehearsal / TTA) — that stays a ResNet-only story. ConvNeXt uses LayerNorm
(no BatchNorm running stats), so the BN-freeze lever is moot; forgetting Δ is still logged
as free corroboration of the BN-stat finding.

| run | style | source weights | data | selection |
|---|---|---|---|---|
| `convnext_zeroshot` | synth-only | ImageNet | full dataset_v1 synth | synth val acc |
| `convnext_stage3` | sequential FT | `convnext_zeroshot/best_synth.pt` | 30 manual + game4 + game5 (~323 frames) | game7 real_val |
| `convnext_stage5` | combined | ImageNet | synth + 30 manual + game4 + game5 | game7 real_val |

## Recipe (AdamW + cosine + weight decay)

- AdamW (betas 0.9/0.999), weight_decay 0.05, cosine LR (eta_min = 0.01·lr_head).
- Two-phase freeze adapted to ConvNeXt's stage structure: **Phase A** freezes
  `model.features` (stem + 4 stages + downsamplers), trains `model.classifier` only
  (lr 1e-4); **Phase B** unfreezes all with discriminative LRs (head 1e-4; backbone
  **1e-5 for zeroshot, 3e-5 for stage3/stage5** — the larger ConvNeXt backbone would
  underfit the ~20k-square real set at 1e-5).
- ConvNeXt-Tiny ≈ 27.8M params (vs ResNet-18 11.2M, 2.49×). Head: `classifier[2]` swapped
  to `Linear(768, 13)`. No BatchNorm.
- Fixed `--seed 42`. 100×100 crops fed directly (no resize). ImageNet normalize at the
  model boundary — identical to the ResNet runs.
- Augmentation matches ResNet per stage: zero-shot = color-jitter only; stage3/stage5 =
  jitter@0.7 → shear@0.8(±8°) → noise@0.5(std=0.015).

> **Resolution caveat.** At our 100×100 input the ConvNeXt-Tiny backbone produces a
> **3×3** feature map (9 spatial tokens, 768 ch) vs **7×7** at its native 224×224
> pretraining resolution. We keep 100×100 for input-consistency with the ResNet-18 runs
> (the published 0.5138/0.9085/0.9160 all live on the 100×100 path; changing input size
> would break the single-variable comparison). This is well below pretraining resolution
> and may understate ConvNeXt's potential — a native-resolution comparison is left to
> future work.

The exact recipe per run is written to `results/<run_name>/recipe.json`.

## Write discipline

All outputs are routed through `--run_name` under `convnext/`. A hard
write-guard (in `train.py` and `eval_games_2_6.py`) asserts every output path resolves
under `convnext/` and names no frozen baseline (`zero_shot`, `stage1_10`,
`stage2_30`, `stage3_323`, `stage3_improved`, `stage5_combined_323`). Those dirs are
READ-ONLY references.

## Run order

```bash
PY=~/.conda/envs/chess/bin/python
cd /home/eladbaum/chess_project/training/convnext

# Step 0 — confirm ConvNeXt loads / head swap / param count
$PY training_scripts/confirm_convnext.py

# Run 1 (longest — full synth train) → produces best_synth.pt
$PY training_scripts/train.py --mode zeroshot --run_name convnext_zeroshot

# Run 2 — sequential FT from run 1
$PY training_scripts/train.py --mode stage3 --run_name convnext_stage3

# Run 3 — combined from ImageNet
$PY training_scripts/train.py --mode stage5 --run_name convnext_stage5

# Report
$PY build_report.py
```

`eval_games_2_6.py --run_name <run>` re-runs the verbatim games-2/6 eval on any run's best
checkpoint (the same harness that reproduced ResNet s00's 0.9085).

## Outputs per run

- `checkpoints/<run_name>/`: `best_real.pt` (or `best_synth.pt` for zeroshot),
  `best_synth_monitor.pt`, `latest.pt`.
- `results/<run_name>/`: `training_log.csv`, `recipe.json`, `games_2_6_eval.json`
  (the headline comparable number), `game7_results.json`, `synth_monitor_results.json`
  (forgetting Δ), `predictions/*.npy`.
- `plots/<run_name>/`: `training_curves.png`, `*_cm.png`.
