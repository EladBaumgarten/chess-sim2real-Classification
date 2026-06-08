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
- Ran **23** epochs in **47.2 min**.
- Stop reason: `early_stop_patience_8_no_improve_real_val (best=0.9347 at epoch 15)`.
- Best real_val_acc (game7):  **0.9347** at epoch 15.
- Best synth_monitor (5% v1): **0.9995** at epoch 1.

## Catastrophic-forgetting probe (5% slice of dataset_v1)
- Synth-monitor acc BEFORE FT (loaded baseline): **0.9997**
- Synth-monitor acc AFTER FT  (best_real ckpt):  **0.9536**
- **Catastrophic-forgetting Δ: -0.0461**

## Game7 monitor (NOT held-out — used for checkpoint selection)
- Per-square at best_real:  **0.9347**  (before FT: 0.5665; v1 ckpt-epoch real_val: 0.5670; v1 peak real_val: 0.5923)
- Improvement over loaded baseline: **+0.3682**
- Per-board acc: 0/55 = 0.0000
- Mean squares correct/board: 59.82/64

## Held-out real test (games 2/6 — subset of stage 2 partition; games 4/5 are training data in stage 3)

| game | n_frames | n_squares | per-sq acc | per-board acc | piece-only acc | mean correct |
|------|---------:|----------:|-----------:|--------------:|---------------:|-------------:|
| game2 | 77 | 4928 | 0.9304 | 0.0130 | 0.8085 | 59.55/64 |
| game6 | 92 | 5888 | 0.8668 | 0.0000 | 0.6563 | 55.48/64 |
| **agg** | 169 | 10816 | **0.8958** | **0.0059** | **0.7227** | 57.33/64 |

**Verdict:** stage3_323 per-square acc on held-out (games 2/6) is **0.8958**. v1 zero-shot held-out aggregate not available on disk — compare via `zero_shot/results/games_2_4_5_6_eval/` if present, or re-evaluate the baseline checkpoint on this exact partition.

## Comparison to stage2_30 (different test partition — see re-eval below)

⚠️ Stage 2 numbers below are on games 2/4/5/6 (full held-out). Stage 3 numbers are on games 2/6 only. NOT directly comparable. See 'Stage 2 reevaluated on games 2/6' section for the matched-partition bridge.

| metric | stage2_30 (2/4/5/6) | stage3_323 (2/6) | Δ (NOT VALID) |
|--------|--------------------:|-----------------:|--------------:|
| game7 real_val_acc | 0.8037 | 0.9347 | +0.1310 |
| held-out per-sq | 0.8582 | 0.8958 | +0.0376 |
| held-out piece-only | 0.5408 | 0.7227 | +0.1819 |
| forgetting Δ on 5% v1 | -0.0796 | -0.0461 | +0.0334 |

### Per-class on aggregate held-out (different test partition — caveat applies)

| class | stage2_30 (2/4/5/6) | stage3_323 (2/6) | Δ |
|-------|--------------------:|-----------------:|---:|
| wP | 0.8223 | 0.9255 | +0.1033 |
| wR | 0.5578 | 0.8762 | +0.3184 |
| wN | 0.0201 | 0.1893 | +0.1692 |
| wB | 0.5918 | 0.2759 | -0.3159 |
| wQ | 0.1797 | 0.1212 | -0.0585 |
| wK | 0.0866 | 0.4793 | +0.3927 |
| bP | 0.7163 | 0.9580 | +0.2417 |
| bR | 0.4729 | 0.7689 | +0.2960 |
| bN | 0.3066 | 0.1988 | -0.1078 |
| bB | 0.1025 | 0.4914 | +0.3889 |
| bQ | 0.2744 | 0.2519 | -0.0225 |
| bK | 0.1147 | 0.3373 | +0.2226 |
| empty | 0.9889 | 0.9978 | +0.0089 |

## Per-class deltas on aggregate held-out (vs. v1 zero-shot baseline if available)

| class | v1 baseline | stage3_323 | Δ |
|-------|------------:|----------:|---:|
| wP | n/a | 0.9255 | n/a |
| wR | n/a | 0.8762 | n/a |
| wN | n/a | 0.1893 | n/a |
| wB | n/a | 0.2759 | n/a |
| wQ | n/a | 0.1212 | n/a |
| wK | n/a | 0.4793 | n/a |
| bP | n/a | 0.9580 | n/a |
| bR | n/a | 0.7689 | n/a |
| bN | n/a | 0.1988 | n/a |
| bB | n/a | 0.4914 | n/a |
| bQ | n/a | 0.2519 | n/a |
| bK | n/a | 0.3373 | n/a |
| empty | n/a | 0.9978 | n/a |

## Per-class real_val trajectory analysis
- ALL of {wN, wB, wK, bN, bB, bK} crossed >5% on real_val — every knight/bishop/king class moved off zero, supporting the FT hypothesis.
- See plots/per_class_real_val.png for the 13-class trajectory.

## Artifacts
- `checkpoints/best_real.pt` (epoch 15, real_val_acc=0.9347) — headline ckpt
- `checkpoints/best_synth_monitor.pt` (epoch 1, synth_monitor_acc=0.9995) — monitor-only
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
| per-sq acc | 0.8576 | 0.8958 | +0.0382 |
| piece-only | 0.6314 | 0.7227 | +0.0913 |
| per-board  | 0.0000 (0/169) | 0.0059 (1/169) | +0.0059 |

### Per-class on games 2/6 (matched partition)

| class | stage2_30 ON 2/6 | stage3_323 ON 2/6 | Δ |
|-------|-----------------:|------------------:|---:|
| wP | 0.8675 | 0.9255 | +0.0580 |
| wR | 0.8143 | 0.8762 | +0.0619 |
| wN | 0.0237 | 0.1893 | +0.1657 |
| wB | 0.5172 | 0.2759 | -0.2414 |
| wQ | 0.1591 | 0.1212 | -0.0379 |
| wK | 0.1716 | 0.4793 | +0.3077 |
| bP | 0.8530 | 0.9580 | +0.1050 |
| bR | 0.5189 | 0.7689 | +0.2500 |
| bN | 0.4211 | 0.1988 | -0.2222 |
| bB | 0.0686 | 0.4914 | +0.4229 |
| bQ | 0.2366 | 0.2519 | +0.0153 |
| bK | 0.0888 | 0.3373 | +0.2485 |
| empty | 0.9909 | 0.9978 | +0.0069 |
