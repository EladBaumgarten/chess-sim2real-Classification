# Zero-shot v1.5 — final zero-shot run

## What changed vs. v1 baseline
- **Dataset:** dataset_v1.5 (7,665 imgs = v1 6,132 + legacy 1,533).
- **Augmentation:** mild shear added to baseline (color jitter @0.5 → shear @0.5 → noise @0.5).
- Everything else (ResNet18+ImageNet, SGD lr=0.001, StepLR(7,0.1), no class weights,
  sqrt-inv-freq sampler, batch 64, 10 epochs, seed 42) is identical to baseline.

## Training
- Total training time: **216.5 min** (10 epochs, 416,640 synth train samples).
- Best synth_val_acc: **0.9998** at epoch **9**.
- Peak real_val_acc on game7 during training: **0.5798** at epoch **5** (monitor only).

## Synth test (held-out FENs from dataset_v1.5)
- **Overall accuracy: 0.9997**
- **Piece-only accuracy (exclude empty class 12): 0.9989**
- Per-view: overhead=0.9996 (n=12,480), west=0.9995 (n=12,480), east=0.9999 (n=12,480)

## Game7 monitor (NOT held-out — used only as in-training real signal)
- Per-square accuracy at best-synth checkpoint: **0.5213**
- Per-board accuracy: 0/55 = 0.0000
- Mean squares correct/board: 33.36/64
- Peak game7 real_val_acc anywhere in training: 0.5798 (epoch 5)

## Held-out real test (games 2, 4, 5, 6)

| game | n_frames | n_squares | per-sq acc | per-board acc | piece-only acc | mean correct |
|------|---------:|----------:|-----------:|--------------:|---------------:|-------------:|
| game2 | 77 | 4928 | 0.5467 | 0.0000 | 0.1121 | 34.99/64 |
| game4 | 184 | 11776 | 0.5845 | 0.0000 | 0.1579 | 37.41/64 |
| game5 | 109 | 6976 | 0.6302 | 0.0000 | 0.1677 | 40.33/64 |
| game6 | 92 | 5888 | 0.4604 | 0.0000 | 0.3171 | 29.47/64 |
| **agg** | 462 | 29568 | **0.5643** | **0.0000** | **0.1926** | 36.11/64 |

## Sim-to-real gap
- synth_test per-square: **0.9997**
- held-out aggregate per-square: **0.5643**
- **Gap: +0.4354**

## Comparison to v1 baseline (zero_shot/, mild aug, dataset_v1)
- v1 baseline peak real_val on game7: **0.5923**
- this run peak real_val on game7:    **0.5798**  (Δ = -0.0125)

## Per-class shear effect (the diagnostic this run was designed to test)
- Per-class effect: wB, wK, bB crossed >5% on real_val; wN, bN, bK stayed effectively at zero. Mixed evidence — shear helped some position-bound classes but not all.
- See plots/per_class_real_val.png for the full 13-class trajectory.

## Artifacts
- `results/training_log.csv` (per-epoch + 13 per-class real_val columns)
- `results/synth_test_results.json`
- `results/game7_results.json`
- `results/game{2,4,5,6}_results.json`
- `results/held_out_aggregate.json`
- `results/predictions/*.npy` (raw pred/target tensors per eval set)
- `checkpoints/best_synth.pt` (epoch 9, synth_val_acc=0.9998)
- `checkpoints/best_real_monitor.pt` (epoch 5, real_val_acc=0.5798) — monitor-only artifact
- `checkpoints/latest.pt`
- `plots/aug_smoke_check.png`, `training_curves.png`, `per_class_real_val.png`
- `plots/synth_test_cm.png`, `game7_cm.png`, `game{2,4,5,6}_cm.png`, `aggregate_cm.png`
- `plots/game{2,4,5,6}_qualitative.png`