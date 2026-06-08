# Fine-tuning — stage 1 (10 real images)

Stage 1 of three planned fine-tuning experiments. Loads the v1 zero-shot
baseline checkpoint and adapts it to real photos using **10 stratified
real images** drawn from the 30 manual labels across games 8–11.

| stage | training data | size |
|------|---------------|-----:|
| zero-shot (v1 baseline) | synthetic dataset_v1 only | 6,132 imgs |
| **stage1_10 (this)** | v1 baseline weights + 10 real frames | 10 imgs |
| stage2_30 | v1 baseline weights + all 30 real frames | 30 imgs |
| stage3_combined | synth + 30 real, trained from scratch | 6,132+30 imgs |

All four runs share an identical test partition (games 2/4/5/6 = 462
frames / 29,568 squares), enabling a clean four-row comparison table.

## Source checkpoint

`zero_shot/results/best_synth.pt` — the **v1 baseline** (mild aug, no
shear, dataset_v1 only). Peak game7 real_val during training was
**0.5923** (epoch 5); the saved checkpoint is epoch 7 (synth_val_acc =
0.9987, real_val_acc-at-that-epoch = 0.5670). **NOT** zero_shot_v1.5 —
that run was 1pp worse on real_val.

## Training recipe

Two-phase, following Wölflein 2021's head-warmup pattern.

- **Phase A** (epochs 1–5): freeze everything except `model.fc`.
  SGD(lr=1e-3, momentum=0.9, wd=1e-4).
- **Phase B** (epochs 6–30): unfreeze all. New SGD(lr=1e-4, momentum=0.9,
  wd=1e-4) + StepLR(step_size=15, gamma=0.1) — drops to 1e-5 at absolute
  epoch 21.
- **Augmentation:** heavier than zero-shot. Color jitter @0.7 (b/c/s=0.3,
  h=0.08) → shear @0.8 (±8°, ±4% translate, ±5% scale) → noise @0.5
  (std=0.015). Wölflein's #1 finding for fine-tuning was strong shear.
- **Sampler:** none (shuffle=True). 640 squares is too small for class-
  balanced sampling.
- **Loss:** plain `CrossEntropyLoss`.
- **Early stopping:** patience=8 on real_val_acc (game7).
- **Checkpoint:** best by **real_val_acc on game7** → `best_real.pt`.
  Also saves `best_synth_monitor.pt` (5% slice of dataset_v1) as a
  catastrophic-forgetting probe.

## Data splits (identical across stages)

- **Train pool:** 30 manual labels from games 8–11, in
  `data/real_labels.csv`. Stage 1 picks 10 stratified by game (SEED=42);
  the other 20 are reserved untouched.
- **Monitor:** game7 (55 frames / 3,520 squares). Gates checkpoint
  selection.
- **Held-out test:** games 2, 4, 5, 6 (462 frames / 29,568 squares).
  Evaluated once at end on `best_real.pt`.
- **Synth-forgetting probe:** 5% slice of `dataset_v1` (NOT v1.5),
  seed-42 sampled at script start.

## Outputs

- `checkpoints/best_real.pt` — headline checkpoint, by real_val_acc.
- `checkpoints/best_synth_monitor.pt` — forgetting probe checkpoint.
- `checkpoints/latest.pt`.
- `results/stage1_train_manifest.csv` — the 10 selected frames.
- `results/training_log.csv` — per-epoch + 13 per-class real_val cols.
- `results/{synth_test,game7,game{2,4,5,6},held_out_aggregate}_results.json`.
- `results/predictions/*.npy`.
- `results/summary.md` — comparison vs. v1 zero-shot baseline.
- `plots/{aug_smoke_check,stage1_train_samples,training_curves,
  per_class_real_val,synth_test_cm,game{7,2,4,5,6}_cm,aggregate_cm,
  game{2,4,5,6}_qualitative}.png`.
