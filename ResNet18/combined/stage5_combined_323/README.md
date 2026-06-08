# Stage 5 — Joint synth+real training from ImageNet (30 manual + game4 + game5 PGN)

Stage 5 of five. **Joint synth+real combined training from ImageNet
pretrained weights** (NOT fine-tuning, NOT the v1 synth-trained baseline)
on the full ~353-frame real pool. Single experimental variable vs.
stage 3: **training procedure (joint synth+real from ImageNet vs.
sequential FT from the v1 baseline)**, holding real data and test
partition constant.

This is **Comparison B** in the report: joint vs sequential at the
larger real-data scale (~353 frames), the counterpart to stage 4's
Comparison A (joint vs sequential at 30 frames).

| stage | start | training data | size |
|------|-------|---------------|-----:|
| zero-shot (v1 baseline) | ImageNet | synth dataset_v1 only | 6,132 imgs |
| stage1_10 | v1 baseline | + 10 manual frames (FT) | 10 imgs |
| stage2_30 | v1 baseline | + 30 manual frames (FT) | 30 imgs |
| stage3_323 | v1 baseline | + 30 manual + game4 + game5 PGN (FT) | ~353 imgs |
| stage4_combined_30 | ImageNet | synth + 30 manual (joint, balanced) | ~390k + 30 |
| **stage5_combined_323 (this)** | **ImageNet** | synth + 30 manual + game4 + game5 PGN (joint, balanced) | ~390k + ~353 |

## Why ImageNet (not v1 baseline)

Same rationale as stage 4. Stage 5 answers: **does joint training match
or beat sequential FT (stage 3) on real-domain accuracy while preserving
synth knowledge — once we have enough real data?** Starting from the v1
baseline would conflate the comparison with "we already have a synth-good
init." ImageNet is the natural shared starting point for the joint stages
(4, 5) and the FT chain's grandparent.

## Hypothesis

Stage 4 found that joint training **underperformed** sequential FT at the
small real-data scale (30 frames). The likely cause is asymmetric novelty
under the 50/50 sampler: each of the 1,920 real squares was seen ~26× per
epoch while each of the ~390k synth squares was seen ~0.13× — the model
over-fit the tiny real set and/or under-learned real appearance relative
to the dominant synth gradient.

Stage 5 scales the real pool 10× (30 → ~353 frames, ~22,600 real squares).
At 50/50 with `num_samples=100,000`, each real square now cycles **~2.2×
per epoch** — much closer to balanced novelty. The hypothesis: at this
scale joint training should **match or beat** sequential FT (stage 3) on
games 2/6 **AND** preserve synth knowledge far better (stage 3's
sequential FT lost ~0.13 on the 5% v1 forgetting probe; joint training
keeps synth in the gradient every batch).

## Training recipe

- **Source weights:** ImageNet pretrained via torchvision
  (`ResNet18_Weights.IMAGENET1K_V1`). NO checkpoint loaded.
- **Training data:** two-level `ConcatDataset([synth_train_dataset,
  ConcatDataset([manual, game4_pgn, game5_pgn])])` — the full dataset_v1
  manifest (~390k squares) + 30 manual frames + game4 PGN (184 frames) +
  game5 PGN (109 frames). Total real ~353 frames / ~22,600 squares.
- **Sampler:** `WeightedRandomSampler` at 50% synth / 50% real per batch.
  `num_samples=100,000` per epoch ⇒ ~1,562 batches/epoch.
- **Single phase**: all params trainable from epoch 1.
  `SGD(lr=1e-4, momentum=0.9, weight_decay=1e-4)`, no scheduler, no
  freeze. No phase A / phase B.
- **Aug:** color jitter @0.7, shear @0.8 (±8°), noise @0.5 std=0.015 —
  identical to stages 1/2/3/4, applied to BOTH synth and real samples.
- **Checkpoint by:** real_val_acc on game7. **Early stop:** patience=8.

## Data splits

- **Train:** dataset_v1 (full) + 30 manual labels (`data/real_labels.csv`)
  + game4 PGN (`data/game4_per_frame/`) + game5 PGN
  (`data/game5_per_frame/`).
- **Monitor:** game7 (55 frames / 3,520 squares). Gates checkpoint
  selection.
- **Held-out test:** games **2 and 6** (169 frames / 10,816 squares).
  game4 and game5 are training data here, so they are excluded from the
  test set. **Identical to stage 3's partition** — direct, matched
  comparison (Comparison B).
- **Synth-forgetting probe:** 5% slice of dataset_v1 (SEED=42) — same
  slice as stages 1/2/3/4 for cross-stage Δ.

## What to look for

- **Held-out per-sq (games 2/6):** stage 3 reached 0.9083. Stage 5
  should match or beat it if the more-real-data hypothesis holds.
- **vs. stage 4:** stage 4 (30 real frames, joint) underperformed
  sequential FT. Stage 5 tests whether 10× more real data closes that
  gap.
- **Catastrophic forgetting (5% v1 slice):** stage 3 (sequential FT)
  lost ~0.13. Stage 5 should hold much higher because synth is in every
  batch — confirm the larger real pool does not erode this.
- **Dead classes (wN, wK, bK, bB):** with synth always present in the
  gradient, stage 5 should keep these visible.

## Expected runtime

Same `num_samples` (100k/epoch) as stage 4, so per-epoch time is similar:
- ~1,562 batches/epoch × ~0.5s ≈ ~780s train pass
- synth_monitor eval ≈ ~100s, game7 eval ≈ ~5s
- Per epoch ≈ ~9 min; 30 epochs worst case ≈ 4.5 h; realistic with early
  stop ≈ 2.5–3 h.
- Epoch 1 is slower (corner detection on 323 unique real frames + synth
  corner cache); epochs 2+ are faster (cached corners).

## Outputs

- `checkpoints/best_real.pt` — headline checkpoint, by real_val_acc.
- `checkpoints/best_synth_monitor.pt` — forgetting probe checkpoint.
- `checkpoints/latest.pt`.
- `results/stage5_manual_manifest.csv` — the 30 manual frames (manual
  side of the real data).
- `results/training_log.csv` — per-epoch log + 13 per-class real_val cols.
- `results/{synth_test,game7,game{2,6},held_out_aggregate}_results.json`.
- `results/stage3_reeval_on_games_2_6.json` — same-partition bridge
  (Cell 22; reproduces stage 3's published numbers on games 2/6).
- `results/predictions/*.npy`.
- `results/summary.md` — Comparison B table (stage 3 vs stage 5) +
  per-class deltas + catastrophic-forgetting Δ headline.
- `plots/{aug_smoke_check,stage5_manual_samples,training_curves,
  per_class_real_val,synth_test_cm,game{7,2,6}_cm,aggregate_cm,
  game{2,6}_qualitative}.png`.
