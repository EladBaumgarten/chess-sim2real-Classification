# ConvNeXt-Tiny vs ResNet-18 — architecture comparison (games 2/6)

| model (games 2/6) | ResNet-18 per-sq | ConvNeXt per-sq | ResNet piece-only | ConvNeXt piece-only | forgetting Δ (ConvNeXt) |
|---|---|---|---|---|---|
| synth-only (zero-shot) | 0.5138 | 0.7960 | — | 0.4621 | 0.5922 |
| real fine-tune (Stage 3) | 0.9085 | 0.9468 | 0.7556 | 0.8589 | -0.0360 |
| combined (Stage 5) | 0.9160 | 0.9557 | 0.7748 | 0.8828 | 0.5910 |

## Per-class held-out (games 2/6) accuracy — ConvNeXt

| class | wP | wR | wN | wB | wQ | wK | bP | bR | bN | bB | bQ | bK | empty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real fine-tune (Stage 3) | 0.992 | 0.990 | 0.675 | 0.259 | 0.432 | 0.751 | 0.993 | 0.976 | 0.801 | 0.566 | 0.618 | 0.521 | 0.999 |
| combined (Stage 5) | 0.996 | 0.981 | 0.840 | 0.310 | 0.606 | 0.710 | 0.992 | 0.972 | 0.883 | 0.680 | 0.656 | 0.544 | 0.999 |

## Recipe (each architecture done right)

ResNet-18 used SGD + two-phase freeze. ConvNeXt-Tiny (~27.8M params, 2.49× ResNet's 11.2M) used AdamW + cosine + weight-decay + a ConvNeXt-stage-structured two-phase freeze. ConvNeXt uses LayerNorm (no BatchNorm running stats), so the BN-freeze lever does not apply; forgetting Δ is still logged above.

- **synth-only (zero-shot)** (`convnext_zeroshot`): AdamW, lr_head=0.0001, lr_backbone=1e-05, wd=0.05, epochs=10 (phaseA=1), select on synth_val @ epoch 7, source=ImageNet.

- **real fine-tune (Stage 3)** (`convnext_stage3`): AdamW, lr_head=0.0001, lr_backbone=3e-05, wd=0.05, epochs=20 (phaseA=2), select on game7_real_val @ epoch 17, source=/home/eladbaum/chess_project/convnext/checkpoints/convnext_zeroshot/best_synth.pt.

- **combined (Stage 5)** (`convnext_stage5`): AdamW, lr_head=0.0001, lr_backbone=3e-05, wd=0.05, epochs=20 (phaseA=2), select on game7_real_val @ epoch 9, source=ImageNet.
