# Stage 4 — Joint synth+real training from ImageNet (30 manual labels, balanced sampler)

## Recipe (vs. v1 zero-shot baseline)
- **Source weights:** ImageNet pretrained via torchvision (NOT v1 baseline, NOT stages 1/2/3 weights).
- **Training data:** dataset_v1 (full synth manifest, ~390k squares) + all 30 manual-label real frames (games 8-11). Combined via ConcatDataset.
- **Single phase**: all params trainable from epoch 1. SGD(lr=0.0001, momentum=0.9, wd=0.0001). No scheduler. No freeze.
- **Aug:** color jitter @0.7 → shear @0.8 (±8°) → noise @0.5 (std=0.015), applied to BOTH synth and real.
- **Sampler:** WeightedRandomSampler at 50% synth / 50% real per batch, num_samples=100,000 per epoch (~1562 batches/epoch).
- **Checkpoint by:** real_val_acc on game7 (NOT synth_val).
- **Early stop:** patience=8 on real_val_acc.

## Training
- Ran **6** epochs in **52.4 min**.
- Stop reason: `completed_all_epochs`.
- Best real_val_acc (game7):  **0.7986** at epoch 2.
- Best synth_monitor (5% v1): **0.9549** at epoch 6.

## Catastrophic-forgetting probe (5% slice of dataset_v1)

NOTE: stage 4 starts from ImageNet, so 'BEFORE training' is ~chance (not a meaningful baseline). The headline number here is `acc AFTER training` — joint training is expected to keep this HIGH (≥ 0.95) because synth is in the training mix, unlike stages 1/2/3 where it falls.

- Synth-monitor acc BEFORE train (ImageNet init): **0.2074**
- Synth-monitor acc AFTER train  (best_real ckpt): **0.9025**
- **Δ from ImageNet init: +0.6951** (positive = the joint task learned synth, as expected; compare to stages 1/2/3 which START at ~0.999 and lose ground)

## Game7 monitor (NOT held-out — used for checkpoint selection)
- Per-square at best_real:  **0.7986**
- Per-board acc: 0/55 = 0.0000
- Mean squares correct/board: 51.11/64

## Held-out real test (games 2, 4, 5, 6) — identical to stages 1/2 partition

| game | n_frames | n_squares | per-sq acc | per-board acc | piece-only acc | mean correct |
|------|---------:|----------:|-----------:|--------------:|---------------:|-------------:|
| game2 | 77 | 4928 | 0.8239 | 0.0000 | 0.5157 | 52.73/64 |
| game4 | 184 | 11776 | 0.8825 | 0.0000 | 0.4875 | 56.48/64 |
| game5 | 109 | 6976 | 0.8217 | 0.0000 | 0.4069 | 52.59/64 |
| game6 | 92 | 5888 | 0.8582 | 0.0000 | 0.6325 | 54.92/64 |
| **agg** | 462 | 29568 | **0.8535** | **0.0000** | **0.5124** | 54.63/64 |

**Verdict:** stage4 (joint training) did NOT beat stage 2 (sequential FT) on the matched held-out partition (games 2/4/5/6): 0.8535 vs. 0.8582 per-square (Δ = -0.0047).

## Comparison to stage2_30 (same test partition — Comparison A)

Same 30 real frames, same test set, identical augmentation. The single axis varied is the **training procedure**: stage 2 fine-tunes the v1 synth-trained baseline on real data only (sequential), whereas stage 4 trains jointly on synth+real from ImageNet weights.

| metric | stage2_30 (sequential FT) | stage4_combined_30 (joint) | Δ |
|--------|--------------------------:|---------------------------:|---:|
| game7 real_val_acc | 0.8037 | 0.7986 | -0.0051 |
| held-out per-sq | 0.8582 | 0.8535 | -0.0047 |
| held-out piece-only | 0.5408 | 0.5124 | -0.0284 |
| forgetting Δ on 5% v1 | -0.0796 | +0.6951 | +0.7747 |

### Per-class delta vs. stage2_30 (same aggregate held-out)

| class | stage2_30 (sequential FT) | stage4_combined_30 (joint) | Δ |
|-------|--------------------------:|---------------------------:|---:|
| wP | 0.8223 | 0.8669 | +0.0446 |
| wR | 0.5578 | 0.4590 | -0.0988 |
| wN | 0.0201 | 0.0940 | +0.0738 |
| wB | 0.5918 | 0.2904 | -0.3014 |
| wQ | 0.1797 | 0.0691 | -0.1106 |
| wK | 0.0866 | 0.0476 | -0.0390 |
| bP | 0.7163 | 0.6236 | -0.0927 |
| bR | 0.4729 | 0.3712 | -0.1017 |
| bN | 0.3066 | 0.1777 | -0.1289 |
| bB | 0.1025 | 0.0661 | -0.0364 |
| bQ | 0.2744 | 0.0698 | -0.2047 |
| bK | 0.1147 | 0.5216 | +0.4069 |
| empty | 0.9889 | 0.9939 | +0.0051 |

## Per-class deltas on aggregate held-out (vs. v1 zero-shot baseline if available)

| class | v1 baseline | stage4_combined_30 | Δ |
|-------|------------:|------------------:|---:|
| wP | n/a | 0.8669 | n/a |
| wR | n/a | 0.4590 | n/a |
| wN | n/a | 0.0940 | n/a |
| wB | n/a | 0.2904 | n/a |
| wQ | n/a | 0.0691 | n/a |
| wK | n/a | 0.0476 | n/a |
| bP | n/a | 0.6236 | n/a |
| bR | n/a | 0.3712 | n/a |
| bN | n/a | 0.1777 | n/a |
| bB | n/a | 0.0661 | n/a |
| bQ | n/a | 0.0698 | n/a |
| bK | n/a | 0.5216 | n/a |
| empty | n/a | 0.9939 | n/a |

## Per-class real_val trajectory analysis

Stage 4 starts from ImageNet (not v1 baseline), so the question for the dead classes (wN/wK/bK/wB/bB) is whether real_val per-class crosses zero at all — not whether it 'improves over baseline'.

- ALL of {wN, wB, wK, bN, bB, bK} crossed >5% on real_val — every knight/bishop/king class moved off zero, supporting the FT hypothesis.
- See plots/per_class_real_val.png for the 13-class trajectory.

## Artifacts
- `checkpoints/best_real.pt` (epoch 2, real_val_acc=0.7986) — headline ckpt
- `checkpoints/best_synth_monitor.pt` (epoch 6, synth_monitor_acc=0.9549) — monitor-only
- `checkpoints/latest.pt`
- `results/stage4_real_manifest.csv` — the 30 manual-label frames (real side)
- `results/training_log.csv` — per-epoch log + 13 per-class real_val columns
- `results/synth_test_results.json` (catastrophic-forgetting probe — AFTER training)
- `results/game7_results.json`
- `results/game{2,4,5,6}_results.json`
- `results/held_out_aggregate.json`
- `results/predictions/*.npy`
- `plots/aug_smoke_check.png`, `stage4_real_samples.png`
- `plots/training_curves.png`, `per_class_real_val.png`
- `plots/synth_test_cm.png`, `game7_cm.png`, `game{2,4,5,6}_cm.png`, `aggregate_cm.png`
- `plots/game{2,4,5,6}_qualitative.png`
## Direct comparison — stage 2 reevaluated on games 2/4/5/6 (same partition)

Stage 2's checkpoint evaluated on the EXACT same test set used by stage 4. Because both stages already share the games 2/4/5/6 partition, these numbers should reproduce stage 2's published results — this section is both a sanity check that the test pipeline is identical and a single-pane comparison row for the report.

| metric | stage2_30 (sequential FT) | stage4_combined_30 (joint) | Δ |
|--------|--------------------------:|---------------------------:|---:|
| per-sq acc | 0.8582 | 0.8535 | -0.0047 |
| piece-only | 0.5408 | 0.5124 | -0.0284 |
| per-board  | 0.0000 (0/462) | 0.0000 (0/462) | +0.0000 |

### Per-class on games 2/4/5/6 (matched partition)

| class | stage2_30 (sequential FT) | stage4_combined_30 (joint) | Δ |
|-------|--------------------------:|---------------------------:|---:|
| wP | 0.8223 | 0.8669 | +0.0446 |
| wR | 0.5578 | 0.4590 | -0.0988 |
| wN | 0.0201 | 0.0940 | +0.0738 |
| wB | 0.5918 | 0.2904 | -0.3014 |
| wQ | 0.1797 | 0.0691 | -0.1106 |
| wK | 0.0866 | 0.0476 | -0.0390 |
| bP | 0.7163 | 0.6236 | -0.0927 |
| bR | 0.4729 | 0.3712 | -0.1017 |
| bN | 0.3066 | 0.1777 | -0.1289 |
| bB | 0.1025 | 0.0661 | -0.0364 |
| bQ | 0.2744 | 0.0698 | -0.2047 |
| bK | 0.1147 | 0.5216 | +0.4069 |
| empty | 0.9889 | 0.9939 | +0.0051 |
