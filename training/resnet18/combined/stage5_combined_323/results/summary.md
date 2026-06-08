# Stage 5 — Joint synth+real training from ImageNet (30 manual + game4 + game5 PGN, balanced sampler)

## Recipe (vs. v1 zero-shot baseline)
- **Source weights:** ImageNet pretrained via torchvision (NOT v1 baseline, NOT stages 1/2/3/4 weights).
- **Training data:** dataset_v1 (full synth manifest, ~390k squares) + 30 manual labels + game4 PGN (184 frames) + game5 PGN (109 frames). Total real: ~353 frames / ~22,600 squares. Combined via two-level ConcatDataset (synth + (manual + game4 + game5)).
- **Single phase**: all params trainable from epoch 1. SGD(lr=0.0001, momentum=0.9, wd=0.0001). No scheduler. No freeze.
- **Aug:** color jitter @0.7 → shear @0.8 (±8°) → noise @0.5 (std=0.015), applied to BOTH synth and real.
- **Sampler:** WeightedRandomSampler at 50% synth / 50% real per batch, num_samples=100,000 per epoch (~1562 batches/epoch). Each real square cycles ~2.4× per epoch (vs. stage 4's ~26×).
- **Checkpoint by:** real_val_acc on game7 (NOT synth_val).
- **Early stop:** patience=8 on real_val_acc.

## Training
- Ran **30** epochs in **188.1 min**.
- Stop reason: `completed_all_epochs`.
- Best real_val_acc (game7):  **0.9517** at epoch 25.
- Best synth_monitor (5% v1): **0.9943** at epoch 30.

## Catastrophic-forgetting probe (5% slice of dataset_v1)

NOTE: stage 5 starts from ImageNet, so 'BEFORE training' is ~chance (not a meaningful baseline). The headline number here is `acc AFTER training` — joint training is expected to keep this HIGH (≥ 0.95) because synth is in the training mix, unlike stages 1/2/3 where it falls. With ~10× more real data than stage 4, watch whether the larger real pool erodes this number relative to stage 4.

- Synth-monitor acc BEFORE train (ImageNet init): **0.2075**
- Synth-monitor acc AFTER train  (best_real ckpt): **0.9921**
- **Δ from ImageNet init: +0.7846** (positive = the joint task learned synth, as expected; compare to stages 1/2/3 which START at ~0.999 and lose ground)

## Game7 monitor (NOT held-out — used for checkpoint selection)
- Per-square at best_real:  **0.9517**
- Per-board acc: 1/55 = 0.0182
- Mean squares correct/board: 60.91/64

## Held-out real test (games 2, 6 — subset of stages 1/2 partition; game4/5 are training data)

| game | n_frames | n_squares | per-sq acc | per-board acc | piece-only acc | mean correct |
|------|---------:|----------:|-----------:|--------------:|---------------:|-------------:|
| game2 | 77 | 4928 | 0.9616 | 0.1299 | 0.8937 | 61.55/64 |
| game6 | 92 | 5888 | 0.8777 | 0.0000 | 0.6829 | 56.17/64 |
| **agg** | 169 | 10816 | **0.9160** | **0.0592** | **0.7748** | 58.62/64 |

**Verdict:** stage5 (joint training) BEAT stage 3 (sequential FT) on the matched held-out partition (games 2/6): 0.9160 vs. 0.9083 per-square (Δ = +0.0077).

## Comparison to stage3_323 (same real data, same test partition — Comparison B)

Same real data (30 manual + game4 + game5 PGN), same test set (games 2/6), identical augmentation. The single axis varied is the **training procedure**: stage 3 fine-tunes the v1 synth-trained baseline on real data only (sequential), whereas stage 5 trains jointly on synth+real from ImageNet weights.

| metric | stage3_323 (sequential FT) | stage5_combined_323 (joint) | Δ |
|--------|---------------------------:|----------------------------:|---:|
| game7 real_val_acc | 0.9386 | 0.9517 | +0.0131 |
| held-out per-sq | 0.9083 | 0.9160 | +0.0077 |
| held-out piece-only | 0.7551 | 0.7748 | +0.0197 |
| forgetting Δ on 5% v1 | -0.1293 | +0.7846 | +0.9139 |

### Per-class delta vs. stage3_323 (same aggregate held-out)

| class | stage3_323 (sequential FT) | stage5_combined_323 (joint) | Δ |
|-------|---------------------------:|----------------------------:|---:|
| wP | 0.9671 | 0.9654 | -0.0017 |
| wR | 0.9762 | 0.9524 | -0.0238 |
| wN | 0.3373 | 0.3195 | -0.0178 |
| wB | 0.3333 | 0.3506 | +0.0172 |
| wQ | 0.1515 | 0.1364 | -0.0152 |
| wK | 0.4556 | 0.4852 | +0.0296 |
| bP | 0.9606 | 0.9738 | +0.0131 |
| bR | 0.8726 | 0.8585 | -0.0142 |
| bN | 0.3041 | 0.4211 | +0.1170 |
| bB | 0.3714 | 0.5600 | +0.1886 |
| bQ | 0.2366 | 0.1679 | -0.0687 |
| bK | 0.3728 | 0.5325 | +0.1598 |
| empty | 0.9985 | 0.9991 | +0.0006 |

## Per-class deltas on aggregate held-out (vs. v1 zero-shot baseline if available)

| class | v1 baseline | stage5_combined_323 | Δ |
|-------|------------:|-------------------:|---:|
| wP | n/a | 0.9654 | n/a |
| wR | n/a | 0.9524 | n/a |
| wN | n/a | 0.3195 | n/a |
| wB | n/a | 0.3506 | n/a |
| wQ | n/a | 0.1364 | n/a |
| wK | n/a | 0.4852 | n/a |
| bP | n/a | 0.9738 | n/a |
| bR | n/a | 0.8585 | n/a |
| bN | n/a | 0.4211 | n/a |
| bB | n/a | 0.5600 | n/a |
| bQ | n/a | 0.1679 | n/a |
| bK | n/a | 0.5325 | n/a |
| empty | n/a | 0.9991 | n/a |

## Per-class real_val trajectory analysis

Stage 5 starts from ImageNet (not v1 baseline), so the question for the dead classes (wN/wK/bK/wB/bB) is whether real_val per-class crosses zero at all — not whether it 'improves over baseline'.

- ALL of {wN, wB, wK, bN, bB, bK} crossed >5% on real_val — every knight/bishop/king class moved off zero, supporting the FT hypothesis.
- See plots/per_class_real_val.png for the 13-class trajectory.

## Artifacts
- `checkpoints/best_real.pt` (epoch 25, real_val_acc=0.9517) — headline ckpt
- `checkpoints/best_synth_monitor.pt` (epoch 30, synth_monitor_acc=0.9943) — monitor-only
- `checkpoints/latest.pt`
- `results/stage5_manual_manifest.csv` — the 30 manual-label frames (manual side of real)
- `results/training_log.csv` — per-epoch log + 13 per-class real_val columns
- `results/synth_test_results.json` (catastrophic-forgetting probe — AFTER training)
- `results/game7_results.json`
- `results/game{2,6}_results.json`
- `results/held_out_aggregate.json`
- `results/stage3_reeval_on_games_2_6.json` — same-partition bridge (Cell 22)
- `results/predictions/*.npy`
- `plots/aug_smoke_check.png`, `stage5_manual_samples.png`
- `plots/training_curves.png`, `per_class_real_val.png`
- `plots/synth_test_cm.png`, `game7_cm.png`, `game{2,6}_cm.png`, `aggregate_cm.png`
- `plots/game{2,6}_qualitative.png`
## Direct comparison — stage 3 reevaluated on games 2/6 (same partition)

Stage 3's checkpoint evaluated on the EXACT same test set used by stage 5. Because both stages already share the games 2/6 partition, these numbers should reproduce stage 3's published results — this section is both a sanity check that the test pipeline is identical and a single-pane comparison row for the report.

| metric | stage3_323 (sequential FT) | stage5_combined_323 (joint) | Δ |
|--------|---------------------------:|----------------------------:|---:|
| per-sq acc | 0.9084 | 0.9160 | +0.0076 |
| piece-only | 0.7554 | 0.7748 | +0.0195 |
| per-board  | 0.0355 (6/169) | 0.0592 (10/169) | +0.0237 |

### Per-class on games 2/6 (matched partition)

| class | stage3_323 (sequential FT) | stage5_combined_323 (joint) | Δ |
|-------|---------------------------:|----------------------------:|---:|
| wP | 0.9680 | 0.9654 | -0.0026 |
| wR | 0.9762 | 0.9524 | -0.0238 |
| wN | 0.3373 | 0.3195 | -0.0178 |
| wB | 0.3333 | 0.3506 | +0.0172 |
| wQ | 0.1515 | 0.1364 | -0.0152 |
| wK | 0.4556 | 0.4852 | +0.0296 |
| bP | 0.9606 | 0.9738 | +0.0131 |
| bR | 0.8726 | 0.8585 | -0.0142 |
| bN | 0.3041 | 0.4211 | +0.1170 |
| bB | 0.3714 | 0.5600 | +0.1886 |
| bQ | 0.2366 | 0.1679 | -0.0687 |
| bK | 0.3728 | 0.5325 | +0.1598 |
| empty | 0.9985 | 0.9991 | +0.0006 |
