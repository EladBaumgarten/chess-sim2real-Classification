# Held-out real test — games 2, 4, 5, 6

## Setup note
- **Held-out real test set:** games 2, 4, 5, 6 — never seen during training, never used to gate checkpoints.
- **game7:** per-epoch real-image MONITOR during training; reported below for context only, NOT held-out.
- **Synthetic test (held-out FENs from dataset_v1):** reported for sim-to-real gap measurement.

## Held-out aggregate (games 2, 4, 5, 6 combined)
- **Per-square accuracy: 0.6193**
- **Per-board accuracy (all 64 correct): 0/462 = 0.0000**
- Mean squares correct / board: **39.63 / 64**
- Total: 462 frames, 29,568 squares

## Per-game
```
  game   n_frames  n_squares   per_sq  per_board   mean/64
  game2        77       4928   0.5181     0.0000     33.16
  game4       184      11776   0.7372     0.0000     47.18
  game5       109       6976   0.5839     0.0000     37.37
  game6        92       5888   0.5102     0.0000     32.65
  --------------------------------------------------------
  ALL         462      29568   0.6193     0.0000     39.63
```

## Combined per-class breakdown
```
class  name              n      acc
    0  White Pawn        2352  0.0680
    1  White Rook         597  0.1725
    2  White Knight       447  0.1208
    3  White Bishop       365  0.0986
    4  White Queen        217  0.6267
    5  White King         462  0.0000
    6  Black Pawn        2189  0.3632
    7  Black Rook         590  0.2373
    8  Black Knight       287  0.0662
    9  Black Bishop       439  0.0615
   10  Black Queen        215  0.3535
   11  Black King         462  0.0216
   12  Empty            20946  0.7999
```

## Top-5 confusion pairs (combined)
```
  Empty          → White Knight    1948
  Empty          → White Queen     1331
  White Pawn     → Black Pawn      794
  Empty          → White Bishop    708
  White Pawn     → White Queen     703
```

## Sim-to-real gap
- synth_test per-square acc: **0.9991**
- held-out real per-square acc (games 2,4,5,6): **0.6193**
- **Gap: 0.9991 − 0.6193 = +0.3798**

## Context — game7 real-monitor (NOT held-out)
- game7 per-square accuracy: 0.5670
- game7 per-board accuracy (all 64 correct): 0/55 = 0.0000
- game7 mean squares correct / board: 36.29 / 64
- (game7 was the in-training real monitor; treat as in-distribution-real for this experiment, not as a held-out result.)

### game7 per-class (for context)
```
class  name              n      acc
    0  White Pawn         390  0.0000
    1  White Rook          93  0.3978
    2  White Knight        82  0.0244
    3  White Bishop       110  0.1909
    4  White Queen         55  0.5091
    5  White King          55  0.0000
    6  Black Pawn         388  0.4046
    7  Black Rook         110  0.3636
    8  Black Knight        82  0.0000
    9  Black Bishop        99  0.0707
   10  Black Queen         55  0.6909
   11  Black King          55  0.0000
   12  Empty             1946  0.8561
```

## Artifacts
- `/home/eladbaum/chess_project/zero_shot/results/games_2_4_5_6_eval/real_test_combined.txt`
- `/home/eladbaum/chess_project/zero_shot/results/games_2_4_5_6_eval/real_test_per_game_summary.txt`
- `/home/eladbaum/chess_project/zero_shot/results/games_2_4_5_6_eval/real_test_game{N}.txt`  (per game)
- `/home/eladbaum/chess_project/zero_shot/plots/games_2_4_5_6_eval/real_test_combined_confusion.png`
- `/home/eladbaum/chess_project/zero_shot/plots/games_2_4_5_6_eval/real_test_game{N}_confusion.png`  (per game)