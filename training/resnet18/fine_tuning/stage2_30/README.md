# Fine-tuning — stage 2 (30 real images, all manual labels)

Stage 2 of three planned fine-tuning experiments. **Single experimental
variable vs. stage 1: training data 10 → 30 images. Everything else is
identical.**

| stage | training data | size | start from |
|------|---------------|-----:|------------|
| zero-shot (v1 baseline) | synthetic dataset_v1 only | 6,132 imgs | ImageNet |
| stage1_10 | v1 baseline weights + 10 real frames | 10 imgs | v1 baseline |
| **stage2_30 (this)** | v1 baseline weights + 30 real frames | 30 imgs | **v1 baseline (cold-start, NOT stage 1)** |
| stage3_combined | synth + 30 real, trained from scratch | 6,132+30 imgs | ImageNet |

## Why cold-start from v1 baseline (not stage 1 weights)

The partition is kept clean so that stages 1 and 2 differ only in "amount
of real data," not in starting weights. Warm-starting stage 2 from stage 1
would conflate two variables and make per-class deltas vs. stage 1
uninterpretable.

## Source checkpoint

`zero_shot/results/best_synth.pt` — v1 baseline (mild aug, no shear,
dataset_v1). Peak game7 real_val during training was **0.5923** (epoch 5);
the saved checkpoint is epoch 7 (synth_val=0.9987, real_val@ep7=0.5670).
**NOT** zero_shot_v1.5, **NOT** stage1_10/checkpoints/best_real.pt.

## Training recipe

Identical to stage 1 except:
- **No scheduler in phase B.** Stage 1's StepLR drop at epoch 21 produced
  no measurable improvement (real_val plateaued at 0.7634 from epoch 20).
  Stage 2 skips that perturbation entirely.

Otherwise unchanged: phase-A head warmup (5 epochs @ lr=1e-3), phase-B
full FT (25 epochs @ lr=1e-4), heavy aug (jitter @0.7, shear @0.8 ±8°,
noise @0.5 std=0.015), shuffle=True, best-by-`real_val_acc` on game7,
early-stop patience=8.

## Data splits (identical across stages)

- **Train:** all 30 manual labels from games 8-11 in
  `data/real_labels.csv` (game8=8, game9=7, game10=8, game11=7 — total 30).
  ~1,920 samples per epoch.
- **Monitor:** game7 (55 frames / 3,520 squares). Gates checkpoint
  selection.
- **Held-out test:** games 2, 4, 5, 6 (462 frames / 29,568 squares).
  Evaluated once at end on `best_real.pt`.
- **Synth-forgetting probe:** 5% slice of `dataset_v1` (NOT v1.5),
  SEED=42, sampled at script start. Identical slice to stage 1 (same
  seed + same source list), so the forgetting numbers are directly
  comparable.

## What to look for

Headline metrics to compare against stage 1 (reference numbers):
- **Held-out per-square acc:** stage 1 = 0.8295. Expectation: 0.85–0.90.
- **Held-out piece-only acc:** stage 1 = 0.5012. Expectation: bigger lift
  than per-square (most gains hide in non-empty classes).
- **Per-class wN/wK/bK/bB/wQ/bQ:** all <0.10 in stage 1. Did more data
  move them off zero?
- **Game7 real_val:** stage 1 = 0.7634 at epoch 20.
- **Catastrophic-forgetting Δ on 5% v1 slice:** stage 1 = -0.0897. With
  more real data and stronger gradient on the FT task, the Δ may worsen
  modestly.

## Outputs

- `checkpoints/best_real.pt` — headline checkpoint, by real_val_acc.
- `checkpoints/best_synth_monitor.pt` — forgetting probe checkpoint.
- `checkpoints/latest.pt`.
- `results/stage2_train_manifest.csv` — all 30 frames used.
- `results/training_log.csv` — per-epoch + 13 per-class real_val cols.
- `results/{synth_test,game7,game{2,4,5,6},held_out_aggregate}_results.json`.
- `results/predictions/*.npy`.
- `results/summary.md` — comparison table vs. stage1_10 AND vs. v1 baseline.
- `plots/{aug_smoke_check,stage2_train_samples,training_curves,
  per_class_real_val,synth_test_cm,game{7,2,4,5,6}_cm,aggregate_cm,
  game{2,4,5,6}_qualitative}.png`.
