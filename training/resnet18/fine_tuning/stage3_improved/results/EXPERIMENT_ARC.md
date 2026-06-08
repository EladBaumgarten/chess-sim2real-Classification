# Stage 3 Improvement Experiments — full arc

All runs live in `fine_tuning/stage3_improved/` (frozen `stage3_323/` baseline never touched;
write-guard enforced). Every run is one variable vs. the s00 control: same data (30 manual +
game4 + game5 PGN = 20,672 real train squares), same two-phase recipe (Phase A fc-only @1e-3 ep1-5,
Phase B all @1e-4 ep6-30), same augmentation, same game7 checkpoint selection, early stop p=8,
seed 42, eval cells reused verbatim. Held-out test = games 2/6 (clean, never trained/tuned on).

**Metric of record:** held-out games 2/6 per-square + piece-only. **Reference target:** Stage 5
(from-scratch joint synth+real) = **0.9160 / 0.7748**.

## Results table

| run | lever | held-out 2/6 per-sq | piece-only | forgetting Δ | game2 per-sq | game6 per-sq | verdict |
|-----|-------|--------------------:|-----------:|-------------:|-------------:|-------------:|---------|
| **s00** | plain CE FT (== baseline) | **0.9085** | **0.7556** | −0.1295 | 0.9562 | 0.8685 | control; reproduces frozen baseline within noise |
| s03 | weighted CE, strength 0.3 | 0.9067 | 0.7519 | — | — | — | ↓ slight |
| s05 | weighted CE, strength 0.5 | 0.9051 | 0.7471 | — | — | — | ↓ (monotonic with strength) |
| l2sp_5e-4 | L2-SP weight anchor | 0.9084 | 0.7554 | −0.1297 | — | — | NULL |
| bnfreeze | freeze BN running stats | 0.8958 | 0.7227 | **−0.0461** | 0.9304 | 0.8668 | forgetting win, real regressed |
| rehearsal_0.25 | 25% synth in each batch | 0.8983 | 0.7272 | **−0.0011** | 0.9493 | 0.8556 | forgetting≈0, real regressed |
| *Stage 5 (target)* | *from-scratch joint* | *0.9160* | *0.7748* | *n/a* | *0.9616* | *0.8777* | *reference* |

## Lever-by-lever findings

1. **Weighted CE (Option A).** Held-out degrades monotonically with weighting strength (0.0→0.3→0.5:
   0.9085→0.9067→0.9051). The `empty` class is never the casualty (over-correction guard never
   tripped); weighting just reshuffles probability mass between piece classes (white-piece gains
   paid for by black-piece + pawn losses). Conclusion: class frequency is not the bottleneck.

2. **L2-SP weight anchoring.** λ=5e-4 → exact null (forgetting −0.1297 vs −0.1295; held-out flat).
   **Key diagnosis:** the backbone's *learnable* drift is only ~1.06; catastrophic forgetting is
   **99.6% BatchNorm running-statistic drift** (252 of 253 ‖Δ‖² units) — buffers with no gradient,
   structurally outside any L2-SP-on-weights penalty. λ=1e-3/5e-3 predicted null (drift budget,
   no GPU) and skipped.

3. **BN-freeze** (freeze BN running stats at the synth init; verified BN-buffer drift = 0.000000).
   Confirmed the diagnosis: forgetting recovered −0.13 → **−0.046**. BUT held-out real **regressed**
   (0.9085→0.8958), because forcing synth-domain normalization onto real images hurts real features
   — the hit landed on the easier game2 (0.9562→0.9304). A forgetting↔real trade-off, not a win.

4. **Synthetic rehearsal** (25% synth per batch; pool = dataset_v1 minus the 5% monitor slice).
   Forgetting essentially eliminated (Δ = **−0.0011**, synth 0.9987) and game7 monitor best-in-class
   (0.944). YET held-out 2/6 **regressed** to 0.8983/0.7272 — both games down, dark pieces at/below
   s00, the game6 wall unmoved. A sharp **dissociation**: synth exposure flatters the game7 monitor
   and preserves synth, but pulls decision boundaries toward the synth distribution, misaligning
   with the real held-out set. Trend (s00→0.25 monotone down) implies more synth = worse real.

## Conclusions

- **s00 (plain-CE fine-tuning) remains the best FT-family result on held-out real (0.9085/0.7556).**
  None of the four levers beat it; each either reshuffled errors or traded real accuracy for
  forgetting/synth retention.
- **Two mechanisms learned (both genuine, report-worthy):**
  (i) catastrophic forgetting in this FT is driven by **BatchNorm running-stat drift**, not weight
  drift — a loss/weight regularizer cannot fix it (only BN-targeted methods move it);
  (ii) **game6 is a domain wall** — per-board accuracy is 0.0000 for *every* method including
  Stage 5, and even Stage 5 lifts game6 only 0.8685→0.8777. The aggregate gap to Stage 5 is almost
  entirely a game6 story; game2 is already strong everywhere.
- **The Stage-5 gap is regime-driven, not lever-driven.** Stage 5's advantage (esp. dark pieces
  bB .56 / bK .53 / bN .42) is not reproduced by adding synth exposure to fine-tuning — it appears
  to require the from-scratch *joint* synth+real training regime, which is qualitatively different
  from fine-tuning a synth-pretrained network.

## Next (one targeted attempt)
Model-side loss/regularizer levers are exhausted. The remaining promising directions target the
INPUT/inference side and the game6 wall: test-time augmentation, corner-detection/crop quality on
game6, or a better real-FT recipe — to be chosen after reviewing Wölflein (J. Imaging 2021) +
chesscog and current sim-to-real literature.
