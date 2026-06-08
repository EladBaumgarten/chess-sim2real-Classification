# Stage 4 — Joint synth+real training from ImageNet (30 manual labels)

Stage 4 of five. **Single experimental variable vs. stage 2: training
procedure (joint synth+real from ImageNet vs. sequential FT from the v1
synth-trained baseline).** Same 30 real frames, same test partition,
same augmentation.

This is **Comparison A** in the report: joint vs sequential, holding
training data and test partition constant.

| stage | start | training data | size |
|------|-------|---------------|-----:|
| zero-shot (v1 baseline) | ImageNet | synth dataset_v1 only | 6,132 imgs |
| stage1_10 | v1 baseline | + 10 manual frames (FT) | 10 imgs |
| stage2_30 | v1 baseline | + 30 manual frames (FT) | 30 imgs |
| stage3_323 | v1 baseline | + 30 manual + game4 + game5 PGN (FT) | ~323 imgs |
| **stage4_combined_30 (this)** | **ImageNet** | synth + 30 manual (joint, balanced) | ~390k + 30 |
| stage5_joint (planned) | ImageNet | synth + 323 real (joint, balanced) | ~390k + 323 |

## Why ImageNet (not v1 baseline)

Stage 4 is meant to answer: **does joint training match sequential FT on
real-domain accuracy while preserving synth knowledge?** Starting from
the v1 baseline would conflate the comparison with "we already have a
synth-good init." ImageNet is the natural shared starting point for both
the joint-training stages (4, 5) and the FT chain's grandparent.

## Training recipe

- **Source weights:** ImageNet pretrained via torchvision
  (`ResNet18_Weights.IMAGENET1K_V1`). NO checkpoint loaded.
- **Training data:** `ConcatDataset([synth_train_dataset,
  manual_train_dataset])` — the full dataset_v1 manifest (~390k squares)
  plus all 30 manual-label real frames (~1,920 squares).
- **Sampler:** `WeightedRandomSampler` at 50% synth / 50% real per
  batch. `num_samples=100,000` per epoch ⇒ ~1,560 batches/epoch.
- **Single phase**: all params trainable from epoch 1.
  `SGD(lr=1e-4, momentum=0.9, weight_decay=1e-4)`, no scheduler, no
  freeze.
- **Aug:** color jitter @0.7, shear @0.8 (±8°), noise @0.5 std=0.015 —
  identical to stages 1/2/3, applied to BOTH synth and real samples.
- **Checkpoint by:** real_val_acc on game7. **Early stop:** patience=8.

## Data splits

- **Train:** dataset_v1 (full) + 30 manual labels from
  `data/real_labels.csv`.
- **Monitor:** game7 (55 frames / 3,520 squares). Gates checkpoint
  selection.
- **Held-out test:** games 2, 4, 5, 6 (~462 frames / ~29,568 squares).
  **Identical** to stages 1 and 2 — direct, matched comparison.
- **Synth-forgetting probe:** 5% slice of dataset_v1 (SEED=42) — same
  slice as stages 1/2/3 for cross-stage Δ.

## What to look for

Headline metrics:
- **Held-out per-sq:** stage 2 reached 0.8582 on games 2/4/5/6. Stage 4
  should be in a similar ballpark (0.85–0.90), possibly better on
  piece-only since synth keeps feature coverage of underrepresented
  classes alive.
- **Catastrophic forgetting probe (5% v1 slice):** stage 2 ended at
  ~0.92 (Δ = −0.08 from the v1 baseline's ~0.999). Stage 4 should hold
  this much higher (≥ 0.95) because synth is in the training mix every
  batch.
- **Dead classes (wN, wK, bK, bB):** stage 2 left these very low. With
  synth always present in the gradient, stage 4 should keep them
  visible.

## Expected runtime

- ~1,560 batches/epoch × ~0.5s/batch = ~780s train pass per epoch
- synth_monitor eval ≈ ~100s/epoch
- game7 eval ≈ ~5s/epoch
- Per-epoch ≈ ~15 min
- 30 epochs worst case ≈ 7.5 hours; realistic with early stop ≈ 3–4 hours

## Outputs

- `checkpoints/best_real.pt` — headline checkpoint, by real_val_acc.
- `checkpoints/best_synth_monitor.pt` — forgetting probe checkpoint.
- `checkpoints/latest.pt`.
- `results/stage4_real_manifest.csv` — the 30 manual frames (real side).
- `results/training_log.csv` — per-epoch log + 13 per-class real_val cols.
- `results/{synth_test,game7,game{2,4,5,6},held_out_aggregate}_results.json`.
- `results/stage2_reeval_on_games_2_4_5_6.json` — sanity-check bridge
  (Cell 22; reproduces stage 2's published numbers on the same
  partition).
- `results/predictions/*.npy`.
- `results/summary.md` — Comparison A table + per-class deltas +
  catastrophic-forgetting Δ headline.
- `plots/{aug_smoke_check,stage4_real_samples,training_curves,
  per_class_real_val,synth_test_cm,game{7,2,4,5,6}_cm,aggregate_cm,
  game{2,4,5,6}_qualitative}.png`.
