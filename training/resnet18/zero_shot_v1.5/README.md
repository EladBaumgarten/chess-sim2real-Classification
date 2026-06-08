# zero_shot_v1.5 — final zero-shot experiment

This is the LAST zero-shot training run for Project 2 before moving to
real-image fine-tuning. It tests two variables together vs. the v1 baseline:

  (a) Dataset: `data/dataset_v1.5/` — dataset_v1 (6,132 images) + the
      legacy 1.5K set (1,533 images) = **7,665 images total**, all FENs
      verified rot180 (see `data/dataset_v1.5/README.md`).
  (b) One additional augmentation: **mild shear** (±5° both axes via
      `RandomAffine`), added to the v1 baseline mild-color-jitter +
      mild-Gaussian-noise recipe.

Everything else — backbone (ResNet18 + ImageNet weights), optimizer
(SGD lr=0.001 momentum=0.9 wd=1e-4), scheduler (StepLR(step=7, γ=0.1)),
loss (CrossEntropyLoss, no class weights), sampler (sqrt-inverse-frequency
WeightedRandomSampler), batch size (64), epochs (10), seed (42) — is
identical to baseline.

## Baseline being tested against

| metric                          | v1 baseline (zero_shot/) |
|---------------------------------|--------------------------|
| peak real_val on game7          | **0.5923**               |

Stronger augmentations (7-stage stack, narrow-Wölflein) both underperformed
this baseline; mild shear is the one untested clean lever and v1.5 is the
one untested data lever.

## Hypothesis

The bishop / knight / king classes sit near 0% on real images in the v1
baseline. The mechanistic theory: synthetic crops are too perfectly axis-
aligned for the model to learn rotation-invariant piece features. Mild
shear should push those off zero. Per-class real_val accuracy is logged
each epoch to detect this (new column set in `training_log.csv`).

## Layout

```
zero_shot_v1.5/
├── training_scripts/
│   └── train.py            # cells-style script; run in VS Code Jupyter
├── checkpoints/
│   ├── best_synth.pt       # THE checkpoint — used for all reported numbers
│   ├── best_real_monitor.pt# monitor artifact only (NOT used for headline)
│   └── latest.pt           # for resume
├── results/
│   ├── training_log.csv    # per-epoch + per-class real_val accuracy
│   ├── synth_test_results.json
│   ├── game7_results.json
│   ├── game{2,4,5,6}_results.json
│   ├── held_out_aggregate.json
│   ├── predictions/        # raw (N,) pred/target tensors for re-plotting
│   └── summary.md
└── plots/
    ├── aug_smoke_check.png         # 4×4 augmented training crops
    ├── training_curves.png
    ├── per_class_real_val.png      # per-class game7 acc over epochs
    ├── synth_test_cm.png
    ├── game7_cm.png
    ├── game{2,4,5,6}_cm.png
    ├── aggregate_cm.png
    └── game{2,4,5,6}_qualitative.png
```

## Zero-shot hard rules (carried forward)

- No real images enter training or validation. Ever.
- `game7` is the per-epoch real monitor. It NEVER gates checkpoint selection.
- `game{2,4,5,6}` are held-out test only — evaluated ONCE at end on the
  best-by-synth-val checkpoint.
- Checkpoints are selected by `synth_val_acc`. `real_val_acc` is logged but
  never gated on. A second checkpoint best-by-real is also saved as a
  monitor artifact only; it never produces headline numbers.

## After this run

No more zero-shot variants. Next phase: real-image fine-tuning.
