# Option A (class-balanced weighted CE) — NEGATIVE RESULT

**Tested:** uniform inverse-sqrt weighted cross-entropy at `--balance_strength` 0.0 / 0.3 / 0.5,
fine-tuning the v1 synth baseline on 323 real frames (30 manual + game4 + game5 PGN). Single
variable vs. the frozen `stage3_323` baseline; everything else identical (same data, two-phase
recipe, augmentation, game7 checkpoint selection, early stop, seed=42, eval cells).

## Result: held-out test (games 2/6) degrades monotonically with strength

| strength | run | game7 per-sq (selection) | **2/6 per-sq** | **2/6 piece-only** | empty | over-corr? |
|---|---|---|---|---|---|---|
| baseline | stage3_323 | 0.9386 | 0.9083 | 0.7551 | 0.9985 | — |
| 0.0 | s00 (control) | 0.9386 | 0.9085 | 0.7556 | 0.9985 | no |
| 0.3 | s03 | 0.9389 | 0.9067 | 0.7519 | 0.9979 | no |
| 0.5 | s05 | 0.9409 | 0.9051 | 0.7471 | 0.9982 | no |

- s00 (strength 0.0) reproduces the frozen baseline within noise (per-sq |Δ| ≤ 0.0002) — pipeline
  validated, comparison fair.
- Both held-out metrics fall as strength rises: per-sq 0.9085 → 0.9067 → 0.9051; piece-only
  0.7556 → 0.7519 → 0.7471. No weighted run beats the unweighted baseline on the held-out test.
  (game7 per-sq rises with strength, but game7 is only the *selection* signal, not the test.)

## The per-class trade is structured, not random

Increasing weight strength shifts probability mass from black/pawn classes onto white-piece
classes (games 2/6, baseline → 0.3 → 0.5):

| | wN | wQ | wK | wP | bN | bB | bK |
|---|---|---|---|---|---|---|---|
| | .337→.361→.414 | .152→.189→.227 | .456→.438→.521 | .967→? | .304→.298→.281 | .371→.320→.314 | .373→.361→.349 |

White N/Q/K gain; black N/B/K and white pawns lose. New confusion at strength 0.5:
Black King → Black Bishop (69), absent from the baseline top-5.

## Why: the limiter is NOT the empty class and NOT class frequency

- The over-correction guard **never tripped** — `empty` stayed within 0.001 of baseline at every
  strength. So weighting was not "destroying the easy class"; it simply reweighted piece-vs-piece
  errors with no net gain.
- The real failure is **feature quality on dark pieces** and **feature drift during FT**, which a
  loss reweighting cannot touch:
  - On the synth-monitor (5% of dataset_v1), after FT the dark pieces collapse regardless of
    strength: wN ≈ 0.003, bK ≈ 0.000–0.003, bQ ≈ 0.01–0.09 (see `results/s*/synth_test_results.json`).
  - Catastrophic-forgetting Δ on synth ≈ **−0.13 to −0.14** (s00 −0.1295, s03 −0.1312, s05 −0.1400),
    i.e. the backbone features drift away from the synth init during Phase B, and more strongly with
    heavier weighting.

**Conclusion.** Loss reweighting cannot fix a feature problem. The bottleneck is feature
quality/drift on dark pieces, not class imbalance. We therefore drop Option A and pivot to
**weight-anchoring (L2-SP)**: penalize backbone drift from the synth init `theta_0` during
fine-tuning, to preserve the pretrained piece features instead of reshuffling the loss.

Artifacts: `results/{s00,s03,s05}/` (held_out_aggregate.json, game7_results.json,
synth_test_results.json, over_correction_guard.json), `results/sweep_report.{md,csv}`.
Diagnostic checkpoint-selection re-scan (test-tuning only): `results/checkpoint_selection_rescan.json`.
