# Stage 3 improved — class-balanced fine-tuning sweep

Baseline (frozen stage3_323): game7 per-sq 0.9386, 2/6 per-sq 0.9083, empty 0.9985.

| strength | run | game7 per-sq | 2/6 per-sq | piece-only | empty | wN | wB | wQ | wK | bN | bB | bQ | bK | over-corr? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | baseline | 0.9386 | 0.9083 | 0.7551 | 0.9985 | 0.3373 | 0.3333 | 0.1515 | 0.4556 | 0.3041 | 0.3714 | 0.2366 | 0.3728 | — |
| 0.0 | s00 | 0.9386 | 0.9085 | 0.7556 | 0.9985 | 0.3491 | 0.3333 | 0.1439 | 0.4497 | 0.3099 | 0.3714 | 0.2443 | 0.3728 | no |
| 0.3 | s03 | 0.9389 | 0.9067 | 0.7519 | 0.9979 | 0.3609 | 0.3161 | 0.1894 | 0.4379 | 0.2982 | 0.3200 | 0.2443 | 0.3609 | no |
| 0.5 | s05 | 0.9409 | 0.9051 | 0.7471 | 0.9982 | 0.4142 | 0.3276 | 0.2273 | 0.5207 | 0.2807 | 0.3143 | 0.2214 | 0.3491 | no |

**Recommended:** `s05` (strength=0.5) — best game7 per-square 0.9409 among runs that did NOT degrade `empty` (within 0.01 of baseline 0.9985).
