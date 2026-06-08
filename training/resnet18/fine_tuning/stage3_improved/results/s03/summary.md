# Fine-tuning stage 3 — 30 manual + game4 PGN + game5 PGN (~323 real frames)

## Recipe (vs. v1 zero-shot baseline)
- **Source weights:** v1 baseline (zero_shot/results/best_synth.pt). Cold-start (NOT stage 1 or stage 2 weights).
- **Training data:** 30 manual-label frames (games 8-11) + full game4 PGN (184 frames) + full game5 PGN (109 frames). ~323 total real frames, ~20,700 squares.
- **Phase A** (epochs 1-5): freeze conv1/bn1/layer1-4; train fc only @ lr=0.001.
- **Phase B** (epochs 6-30): unfreeze all; lr=0.0001; no scheduler.
- **Aug:** color jitter @0.7 → shear @0.8 (±8°) → noise @0.5 (std=0.015), applied to BOTH manual and PGN samples.
- **Sampler:** shuffle=True (NO weighted sampler; natural PGN class distribution preserved on purpose).
- **Checkpoint by:** real_val_acc on game7 (NOT synth_val).
- **Early stop:** patience=8 on real_val_acc.

## Training
- Ran **30** epochs in **60.5 min**.
- Stop reason: `completed_all_epochs`.
- Best real_val_acc (game7):  **0.9389** at epoch 28.
- Best synth_monitor (5% v1): **0.8848** at epoch 1.

## Catastrophic-forgetting probe (5% slice of dataset_v1)
- Synth-monitor acc BEFORE FT (loaded baseline): **0.9997**
- Synth-monitor acc AFTER FT  (best_real ckpt):  **0.8685**
- **Catastrophic-forgetting Δ: -0.1312**

## Game7 monitor (NOT held-out — used for checkpoint selection)
- Per-square at best_real:  **0.9389**  (before FT: 0.5670; v1 ckpt-epoch real_val: 0.5670; v1 peak real_val: 0.5923)
- Improvement over loaded baseline: **+0.3719**
- Per-board acc: 0/55 = 0.0000
- Mean squares correct/board: 60.09/64

## Held-out real test (games 2/6 — subset of stage 2 partition; games 4/5 are training data in stage 3)

| game | n_frames | n_squares | per-sq acc | per-board acc | piece-only acc | mean correct |
|------|---------:|----------:|-----------:|--------------:|---------------:|-------------:|
| game2 | 77 | 4928 | 0.9543 | 0.0390 | 0.8754 | 61.08/64 |
| game6 | 92 | 5888 | 0.8668 | 0.0000 | 0.6563 | 55.48/64 |
| **agg** | 169 | 10816 | **0.9067** | **0.0178** | **0.7519** | 58.03/64 |

**Verdict:** stage3_323 per-square acc on held-out (games 2/6) is **0.9067**. v1 zero-shot held-out aggregate not available on disk — compare via `zero_shot/results/games_2_4_5_6_eval/` if present, or re-evaluate the baseline checkpoint on this exact partition.

## Comparison to stage2_30 (different test partition — see re-eval below)

⚠️ Stage 2 numbers below are on games 2/4/5/6 (full held-out). Stage 3 numbers are on games 2/6 only. NOT directly comparable. See 'Stage 2 reevaluated on games 2/6' section for the matched-partition bridge.

| metric | stage2_30 (2/4/5/6) | stage3_323 (2/6) | Δ (NOT VALID) |
|--------|--------------------:|-----------------:|--------------:|
| game7 real_val_acc | 0.8037 | 0.9389 | +0.1352 |
| held-out per-sq | 0.8582 | 0.9067 | +0.0485 |
| held-out piece-only | 0.5408 | 0.7519 | +0.2110 |
| forgetting Δ on 5% v1 | -0.0796 | -0.1312 | -0.0517 |

### Per-class on aggregate held-out (different test partition — caveat applies)

| class | stage2_30 (2/4/5/6) | stage3_323 (2/6) | Δ |
|-------|--------------------:|-----------------:|---:|
| wP | 0.8223 | 0.9636 | +0.1414 |
| wR | 0.5578 | 0.9762 | +0.4184 |
| wN | 0.0201 | 0.3609 | +0.3408 |
| wB | 0.5918 | 0.3161 | -0.2757 |
| wQ | 0.1797 | 0.1894 | +0.0097 |
| wK | 0.0866 | 0.4379 | +0.3513 |
| bP | 0.7163 | 0.9580 | +0.2417 |
| bR | 0.4729 | 0.8821 | +0.4092 |
| bN | 0.3066 | 0.2982 | -0.0084 |
| bB | 0.1025 | 0.3200 | +0.2175 |
| bQ | 0.2744 | 0.2443 | -0.0301 |
| bK | 0.1147 | 0.3609 | +0.2462 |
| empty | 0.9889 | 0.9979 | +0.0091 |

## Per-class deltas on aggregate held-out (vs. v1 zero-shot baseline if available)

| class | v1 baseline | stage3_323 | Δ |
|-------|------------:|----------:|---:|
| wP | n/a | 0.9636 | n/a |
| wR | n/a | 0.9762 | n/a |
| wN | n/a | 0.3609 | n/a |
| wB | n/a | 0.3161 | n/a |
| wQ | n/a | 0.1894 | n/a |
| wK | n/a | 0.4379 | n/a |
| bP | n/a | 0.9580 | n/a |
| bR | n/a | 0.8821 | n/a |
| bN | n/a | 0.2982 | n/a |
| bB | n/a | 0.3200 | n/a |
| bQ | n/a | 0.2443 | n/a |
| bK | n/a | 0.3609 | n/a |
| empty | n/a | 0.9979 | n/a |

## Per-class real_val trajectory analysis
- ALL of {wN, wB, wK, bN, bB, bK} crossed >5% on real_val — every knight/bishop/king class moved off zero, supporting the FT hypothesis.
- See plots/per_class_real_val.png for the 13-class trajectory.

## Artifacts
- `checkpoints/best_real.pt` (epoch 28, real_val_acc=0.9389) — headline ckpt
- `checkpoints/best_synth_monitor.pt` (epoch 1, synth_monitor_acc=0.8848) — monitor-only
- `checkpoints/latest.pt`
- `results/stage3_manual_manifest.csv` — the 30 manual-label frames
- `results/training_log.csv` — per-epoch log + 13 per-class real_val columns
- `results/synth_test_results.json` (catastrophic-forgetting probe)
- `results/game7_results.json`
- `results/game{2,6}_results.json`
- `results/held_out_aggregate.json`
- `results/stage2_reeval_on_games_2_6.json` (matched-partition bridge — Cell 22)
- `results/predictions/*.npy`
- `plots/aug_smoke_check.png`, `stage3_manual_samples.png`
- `plots/training_curves.png`, `per_class_real_val.png`
- `plots/synth_test_cm.png`, `game7_cm.png`, `game{2,6}_cm.png`, `aggregate_cm.png`
- `plots/game{2,6}_qualitative.png`
## Matched-partition bridge — stage 2 reevaluated on games 2/6

Stage 2's checkpoint evaluated on the EXACT same test set used by stage 3 (games 2/6 only). This is the cross-stage comparison to use, NOT the table above.

| metric | stage2_30 ON 2/6 | stage3_323 ON 2/6 | Δ |
|--------|----------------:|------------------:|---:|
| per-sq acc | 0.8578 | 0.9067 | +0.0489 |
| piece-only | 0.6317 | 0.7519 | +0.1202 |
| per-board  | 0.0000 (0/169) | 0.0178 (3/169) | +0.0178 |

### Per-class on games 2/6 (matched partition)

| class | stage2_30 ON 2/6 | stage3_323 ON 2/6 | Δ |
|-------|-----------------:|------------------:|---:|
| wP | 0.8684 | 0.9636 | +0.0952 |
| wR | 0.8143 | 0.9762 | +0.1619 |
| wN | 0.0237 | 0.3609 | +0.3373 |
| wB | 0.5172 | 0.3161 | -0.2011 |
| wQ | 0.1591 | 0.1894 | +0.0303 |
| wK | 0.1716 | 0.4379 | +0.2663 |
| bP | 0.8530 | 0.9580 | +0.1050 |
| bR | 0.5189 | 0.8821 | +0.3632 |
| bN | 0.4211 | 0.2982 | -0.1228 |
| bB | 0.0686 | 0.3200 | +0.2514 |
| bQ | 0.2366 | 0.2443 | +0.0076 |
| bK | 0.0888 | 0.3609 | +0.2722 |
| empty | 0.9910 | 0.9979 | +0.0069 |
