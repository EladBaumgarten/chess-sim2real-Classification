# Fine-tuning — stage 3 (~323 real frames: 30 manual + game4 + game5 PGN)

Stage 3 of five planned fine-tuning / training experiments. **Single
experimental variable vs. stage 2: training data expanded from 30 manual
labels to 30 manual + game4 PGN + game5 PGN (~323 frames total).**

| stage | training data | size | start from |
|------|---------------|-----:|------------|
| zero-shot (v1 baseline) | synthetic dataset_v1 only | 6,132 imgs | ImageNet |
| stage1_10 | v1 baseline weights + 10 manual frames | 10 imgs | v1 baseline |
| stage2_30 | v1 baseline weights + 30 manual frames | 30 imgs | v1 baseline |
| **stage3_323 (this)** | v1 baseline weights + 30 manual + game4 + game5 PGN | ~323 imgs | **v1 baseline (cold-start)** |
| stage5_joint | synth + same 323 real, trained from scratch | 6,132+323 imgs | ImageNet |

## Why cold-start from v1 baseline (not stage 2 weights)

Same logic as stage 2: stages 1/2/3 differ only in "amount of real
data," not in starting weights. Cold-starting keeps the variable
isolated and the deltas interpretable.

## Source checkpoint

`zero_shot/results/best_synth.pt` — v1 baseline (mild aug, no shear,
dataset_v1). Same source as stages 1 and 2. **NOT** stage1/stage2
weights.

## ⚠ TEST PARTITION CHANGE — NOT DIRECTLY COMPARABLE TO STAGES 1/2

Stage 3 trains on game4 + game5, so they cannot be in the test set.

- **Monitor (val):** game7 (unchanged from stages 1, 2).
- **Test:** games 2 + 6 ONLY (~169 frames / ~10,816 squares) — a
  strict subset of stage 1/2's held-out (games 2/4/5/6).

To bridge the comparison, **Cell 22** loads `stage2_30/checkpoints/best_real.pt`
and re-evaluates it on games 2/6, writing
[results/stage2_reeval_on_games_2_6.json](results/stage2_reeval_on_games_2_6.json)
and appending a matched-partition table to summary.md. That is the
cross-stage comparison to use, not the (caveated) full-partition table.

Stage 3's test partition is also identical to **stage 5**
(joint-from-scratch on the same training data), enabling **Comparison B**
in the report.

## Training recipe

Identical to stage 2 except:
- **Training data:** 30 manual + game4 PGN (184 frames) + game5 PGN
  (109 frames). Combined via `ConcatDataset`. Same augmentation pipeline
  applied to BOTH manual and PGN samples.
- **`print_every`** raised from 50 → 100 batches (epoch is ~325 batches
  vs. stage 2's 30; keeps printout cadence roughly the same).

Otherwise unchanged: phase-A head warmup (5 epochs @ lr=1e-3), phase-B
full FT (25 epochs @ lr=1e-4, no scheduler), heavy aug (jitter @0.7,
shear @0.8 ±8°, noise @0.5 std=0.015), shuffle=True, best-by-`real_val_acc`
on game7, early-stop patience=8.

## Data splits

- **Train:** 30 manual labels (`data/real_labels.csv`) + full
  `data/game4_per_frame/gt.csv` + full `data/game5_per_frame/gt.csv`.
  ~323 frames × 64 squares = ~20,700 samples per epoch.
- **Monitor:** game7 (55 frames / 3,520 squares). Gates checkpoint
  selection.
- **Held-out test:** games 2, 6 (~169 frames / ~10,816 squares).
  Evaluated once at end on `best_real.pt`.
- **Synth-forgetting probe:** 5% slice of `dataset_v1`, SEED=42 — same
  slice as stages 1 and 2.

## What to look for

Headline metrics on the **matched partition** (games 2/6) — see Cell 22
bridge:
- **per-sq:** stage 2 = TBD on 2/6 (Cell 22 fills this in). Goal:
  push toward 0.90+.
- **piece-only:** stage 2 = TBD on 2/6. Goal: 0.65–0.75.
- **wN, wK:** still <0.10 in stage 2. Game4 and game5 PGNs contain
  early-game frames with all knights and kings present — should finally
  give the model gradient on these classes.

Catastrophic-forgetting Δ on 5% v1 slice: expected to widen modestly vs.
stage 2 (more real-data gradient pulls the backbone further from synth
features).

## Expected runtime

- ~325 batches/epoch × ~0.5s ≈ 165s train pass.
- synth_monitor + game7 eval ≈ 105s/epoch.
- Per-epoch ≈ 4.5 min. 30 epochs ≈ 135 min worst case; early-stop
  likely fires around epoch 12-18 → 60-90 min realistic.

## Outputs

- `checkpoints/best_real.pt` — headline checkpoint, by real_val_acc.
- `checkpoints/best_synth_monitor.pt` — forgetting probe checkpoint.
- `checkpoints/latest.pt`.
- `results/stage3_manual_manifest.csv` — the 30 manual frames.
- `results/training_log.csv` — per-epoch + 13 per-class real_val cols.
- `results/{synth_test,game7,game{2,6},held_out_aggregate}_results.json`.
- `results/stage2_reeval_on_games_2_6.json` — **matched-partition bridge** (Cell 22).
- `results/predictions/*.npy`.
- `results/summary.md` — caveated stage-2 comparison + bridge section.
- `plots/{aug_smoke_check,stage3_manual_samples,training_curves,
  per_class_real_val,synth_test_cm,game{7,2,6}_cm,aggregate_cm,
  game{2,6}_qualitative}.png`.
