# Fine-tuning stage 1 — 10 real images from games 8-11

## Recipe (vs. v1 zero-shot baseline)
- **Source weights:** v1 baseline (zero_shot/results/best_synth.pt).
- **Training data:** 10 stratified real frames (3/3/2/2 from game8/9/10/11).
- **Phase A** (epochs 1-5): freeze conv1/bn1/layer1-4; train fc only @ lr=0.001.
- **Phase B** (epochs 6-30): unfreeze all; lr=0.0001; StepLR(step=15, gamma=0.1).
- **Aug:** color jitter @0.7 → shear @0.8 (±8°) → noise @0.5 (std=0.015).
- **Sampler:** shuffle=True (NO weighted sampler, 640 samples is too small).
- **Checkpoint by:** real_val_acc on game7 (NOT synth_val).
- **Early stop:** patience=8 on real_val_acc.

## Training
- Ran **28** epochs in **48.7 min**.
- Stop reason: `early_stop_patience_8_no_improve_real_val (best=0.7634 at epoch 20)`.
- Best real_val_acc (game7):  **0.7634** at epoch 20.
- Best synth_monitor (5% v1): **0.9873** at epoch 1.

## Catastrophic-forgetting probe (5% slice of dataset_v1)
- Synth-monitor acc BEFORE FT (loaded baseline): **0.9997**
- Synth-monitor acc AFTER FT  (best_real ckpt):  **0.9101**
- **Catastrophic-forgetting Δ: -0.0897**

## Game7 monitor (NOT held-out — used for checkpoint selection)
- Per-square at best_real:  **0.7634**  (before FT: 0.5670; v1 ckpt-epoch real_val: 0.5670; v1 peak real_val: 0.5923)
- Improvement over loaded baseline: **+0.1963**
- Per-board acc: 0/55 = 0.0000
- Mean squares correct/board: 48.85/64

## Held-out real test (games 2, 4, 5, 6) — same partition as zero-shot

| game | n_frames | n_squares | per-sq acc | per-board acc | piece-only acc | mean correct |
|------|---------:|----------:|-----------:|--------------:|---------------:|-------------:|
| game2 | 77 | 4928 | 0.7790 | 0.0000 | 0.4791 | 49.86/64 |
| game4 | 184 | 11776 | 0.8525 | 0.0000 | 0.4374 | 54.56/64 |
| game5 | 109 | 6976 | 0.7974 | 0.0000 | 0.4243 | 51.04/64 |
| game6 | 92 | 5888 | 0.8636 | 0.0000 | 0.6599 | 55.27/64 |
| **agg** | 462 | 29568 | **0.8295** | **0.0000** | **0.5012** | 53.09/64 |

**Verdict:** stage1_10 per-square acc on held-out (games 2/4/5/6) is **0.8295**. v1 zero-shot held-out aggregate not available on disk — compare via `zero_shot/results/games_2_4_5_6_eval/` if present, or re-evaluate the baseline checkpoint on this exact partition.

## Per-class deltas on aggregate held-out (vs. v1 zero-shot baseline if available)

| class | v1 baseline | stage1_10 | Δ |
|-------|------------:|----------:|---:|
| wP | n/a | 0.7938 | n/a |
| wR | n/a | 0.6114 | n/a |
| wN | n/a | 0.0134 | n/a |
| wB | n/a | 0.4658 | n/a |
| wQ | n/a | 0.0645 | n/a |
| wK | n/a | 0.0952 | n/a |
| bP | n/a | 0.6313 | n/a |
| bR | n/a | 0.5441 | n/a |
| bN | n/a | 0.2404 | n/a |
| bB | n/a | 0.0729 | n/a |
| bQ | n/a | 0.2279 | n/a |
| bK | n/a | 0.0043 | n/a |
| empty | n/a | 0.9646 | n/a |

## Per-class real_val trajectory analysis
- wB, bN, bB crossed >5% on real_val; wN, wK, bK stayed effectively at zero. Mixed evidence.
- See plots/per_class_real_val.png for the 13-class trajectory.

## Artifacts
- `checkpoints/best_real.pt` (epoch 20, real_val_acc=0.7634) — headline ckpt
- `checkpoints/best_synth_monitor.pt` (epoch 1, synth_monitor_acc=0.9873) — monitor-only
- `checkpoints/latest.pt`
- `results/stage1_train_manifest.csv` — the 10 chosen frames
- `results/training_log.csv` — per-epoch log + 13 per-class real_val columns
- `results/synth_test_results.json` (catastrophic-forgetting probe)
- `results/game7_results.json`
- `results/game{2,4,5,6}_results.json`
- `results/held_out_aggregate.json`
- `results/predictions/*.npy`
- `plots/aug_smoke_check.png`, `stage1_train_samples.png`
- `plots/training_curves.png`, `per_class_real_val.png`
- `plots/synth_test_cm.png`, `game7_cm.png`, `game{2,4,5,6}_cm.png`, `aggregate_cm.png`
- `plots/game{2,4,5,6}_qualitative.png`