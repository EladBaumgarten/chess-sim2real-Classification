# Fine-tuning stage 2 — 30 real images from games 8-11 (all manual labels)

## Recipe (vs. v1 zero-shot baseline)
- **Source weights:** v1 baseline (zero_shot/results/best_synth.pt). Cold-start (NOT stage 1 weights).
- **Training data:** all 30 manual-label frames from game8/9/10/11.
- **Phase A** (epochs 1-5): freeze conv1/bn1/layer1-4; train fc only @ lr=0.001.
- **Phase B** (epochs 6-30): unfreeze all; lr=0.0001; no scheduler (stage 2 design choice — stage 1's LR step at epoch 21 produced no improvement).
- **Aug:** color jitter @0.7 → shear @0.8 (±8°) → noise @0.5 (std=0.015).
- **Sampler:** shuffle=True (NO weighted sampler, 1,920 samples is still small).
- **Checkpoint by:** real_val_acc on game7 (NOT synth_val).
- **Early stop:** patience=8 on real_val_acc.

## Training
- Ran **30** epochs in **53.5 min**.
- Stop reason: `completed_all_epochs`.
- Best real_val_acc (game7):  **0.8037** at epoch 27.
- Best synth_monitor (5% v1): **0.9377** at epoch 1.

## Catastrophic-forgetting probe (5% slice of dataset_v1)
- Synth-monitor acc BEFORE FT (loaded baseline): **0.9997**
- Synth-monitor acc AFTER FT  (best_real ckpt):  **0.9202**
- **Catastrophic-forgetting Δ: -0.0796**

## Game7 monitor (NOT held-out — used for checkpoint selection)
- Per-square at best_real:  **0.8037**  (before FT: 0.5670; v1 ckpt-epoch real_val: 0.5670; v1 peak real_val: 0.5923)
- Improvement over loaded baseline: **+0.2366**
- Per-board acc: 0/55 = 0.0000
- Mean squares correct/board: 51.44/64

## Held-out real test (games 2, 4, 5, 6) — same partition as zero-shot

| game | n_frames | n_squares | per-sq acc | per-board acc | piece-only acc | mean correct |
|------|---------:|----------:|-----------:|--------------:|---------------:|-------------:|
| game2 | 77 | 4928 | 0.8320 | 0.0000 | 0.5517 | 53.25/64 |
| game4 | 184 | 11776 | 0.8702 | 0.0000 | 0.4474 | 55.69/64 |
| game5 | 109 | 6976 | 0.8387 | 0.0000 | 0.4804 | 53.68/64 |
| game6 | 92 | 5888 | 0.8794 | 0.0000 | 0.6935 | 56.28/64 |
| **agg** | 462 | 29568 | **0.8582** | **0.0000** | **0.5408** | 54.93/64 |

**Verdict:** stage2_30 per-square acc on held-out (games 2/4/5/6) is **0.8582**. v1 zero-shot held-out aggregate not available on disk — compare via `zero_shot/results/games_2_4_5_6_eval/` if present, or re-evaluate the baseline checkpoint on this exact partition.

## Comparison to stage1_10 (10 real images)

| metric | stage1_10 | stage2_30 | Δ |
|--------|----------:|----------:|---:|
| game7 real_val_acc | 0.7634 | 0.8037 | +0.0403 |
| held-out per-sq | 0.8295 | 0.8582 | +0.0287 |
| held-out piece-only | 0.5012 | 0.5408 | +0.0397 |
| forgetting Δ on 5% v1 | -0.0897 | -0.0796 | +0.0101 |

### Per-class delta vs. stage1_10 (aggregate held-out)

| class | stage1_10 | stage2_30 | Δ |
|-------|----------:|----------:|---:|
| wP | 0.7938 | 0.8223 | +0.0285 |
| wR | 0.6114 | 0.5578 | -0.0536 |
| wN | 0.0134 | 0.0201 | +0.0067 |
| wB | 0.4658 | 0.5918 | +0.1260 |
| wQ | 0.0645 | 0.1797 | +0.1152 |
| wK | 0.0952 | 0.0866 | -0.0087 |
| bP | 0.6313 | 0.7163 | +0.0850 |
| bR | 0.5441 | 0.4729 | -0.0712 |
| bN | 0.2404 | 0.3066 | +0.0662 |
| bB | 0.0729 | 0.1025 | +0.0296 |
| bQ | 0.2279 | 0.2744 | +0.0465 |
| bK | 0.0043 | 0.1147 | +0.1104 |
| empty | 0.9646 | 0.9889 | +0.0243 |

## Per-class deltas on aggregate held-out (vs. v1 zero-shot baseline if available)

| class | v1 baseline | stage2_30 | Δ |
|-------|------------:|----------:|---:|
| wP | n/a | 0.8223 | n/a |
| wR | n/a | 0.5578 | n/a |
| wN | n/a | 0.0201 | n/a |
| wB | n/a | 0.5918 | n/a |
| wQ | n/a | 0.1797 | n/a |
| wK | n/a | 0.0866 | n/a |
| bP | n/a | 0.7163 | n/a |
| bR | n/a | 0.4729 | n/a |
| bN | n/a | 0.3066 | n/a |
| bB | n/a | 0.1025 | n/a |
| bQ | n/a | 0.2744 | n/a |
| bK | n/a | 0.1147 | n/a |
| empty | n/a | 0.9889 | n/a |

## Per-class real_val trajectory analysis
- wB, bN, bB, bK crossed >5% on real_val; wN, wK stayed effectively at zero. Mixed evidence.
- See plots/per_class_real_val.png for the 13-class trajectory.

## Artifacts
- `checkpoints/best_real.pt` (epoch 27, real_val_acc=0.8037) — headline ckpt
- `checkpoints/best_synth_monitor.pt` (epoch 1, synth_monitor_acc=0.9377) — monitor-only
- `checkpoints/latest.pt`
- `results/stage2_train_manifest.csv` — the 30 manual-label frames
- `results/training_log.csv` — per-epoch log + 13 per-class real_val columns
- `results/synth_test_results.json` (catastrophic-forgetting probe)
- `results/game7_results.json`
- `results/game{2,4,5,6}_results.json`
- `results/held_out_aggregate.json`
- `results/predictions/*.npy`
- `plots/aug_smoke_check.png`, `stage2_train_samples.png`
- `plots/training_curves.png`, `per_class_real_val.png`
- `plots/synth_test_cm.png`, `game7_cm.png`, `game{2,4,5,6}_cm.png`, `aggregate_cm.png`
- `plots/game{2,4,5,6}_qualitative.png`