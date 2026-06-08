# Held-out real test — games 2, 4, 5, 6

## Setup note
- **Held-out real test set:** games 2, 4, 5, 6 — never seen during training, never used to gate checkpoints.
- **game7:** per-epoch real-image MONITOR during training; reported below for context only, NOT held-out.
- **Synthetic test (held-out FENs from dataset_v1):** reported for sim-to-real gap measurement.

## Held-out aggregate (games 2, 4, 5, 6 combined)
- **Per-square accuracy: 0.6466**
- **Per-board accuracy (all 64 correct): 0/462 = 0.0000**
- Mean squares correct / board: **41.39 / 64**
- Total: 462 frames, 29,568 squares

## Per-game
```
  game   n_frames  n_squares   per_sq  per_board   mean/64
  game2        77       4928   0.5800     0.0000     37.12
  game4       184      11776   0.7182     0.0000     45.96
  game5       109       6976   0.6724     0.0000     43.04
  game6        92       5888   0.5289     0.0000     33.85
  --------------------------------------------------------
  ALL         462      29568   0.6466     0.0000     41.39
```

## Combined per-class breakdown
```
class  name              n      acc
    0  White Pawn        2352  0.1756
    1  White Rook         597  0.0553
    2  White Knight       447  0.0067
    3  White Bishop       365  0.5781
    4  White Queen        217  0.3779
    5  White King         462  0.0022
    6  Black Pawn        2189  0.1444
    7  Black Rook         590  0.5034
    8  Black Knight       287  0.0000
    9  Black Bishop       439  0.3394
   10  Black Queen        215  0.1628
   11  Black King         462  0.0000
   12  Empty            20946  0.8393
```

## Top-5 confusion pairs (combined)
```
  Empty          → White Bishop    1266
  Empty          → White Queen     828
  White Pawn     → White Bishop    804
  Black Pawn     → Black Bishop    790
  Empty          → White Knight    748
```

## Sim-to-real gap
- synth_test per-square acc: **0.9993**
- held-out real per-square acc (games 2,4,5,6): **0.6466**
- **Gap: 0.9993 − 0.6466 = +0.3527**

## Context — game7 real-monitor (NOT held-out)
- game7 per-square accuracy: 0.5213
- game7 per-board accuracy (all 64 correct): 0/55 = 0.0000
- game7 mean squares correct / board: 33.36 / 64
- (game7 was the in-training real monitor; treat as in-distribution-real for this experiment, not as a held-out result.)

### game7 per-class (for context)
```
class  name              n      acc
    0  White Pawn         390  0.0821
    1  White Rook          93  0.0000
    2  White Knight        82  0.0122
    3  White Bishop       110  0.3455
    4  White Queen         55  0.4909
    5  White King          55  0.0000
    6  Black Pawn         388  0.1727
    7  Black Rook         110  0.6818
    8  Black Knight        82  0.0000
    9  Black Bishop        99  0.1616
   10  Black Queen         55  0.5091
   11  Black King          55  0.0000
   12  Empty             1946  0.7970
```

## Artifacts
- `/home/eladbaum/chess_project/zero_shot_augmentations/results/games_2_4_5_6_eval/real_test_combined.txt`
- `/home/eladbaum/chess_project/zero_shot_augmentations/results/games_2_4_5_6_eval/real_test_per_game_summary.txt`
- `/home/eladbaum/chess_project/zero_shot_augmentations/results/games_2_4_5_6_eval/real_test_game{N}.txt`  (per game)
- `/home/eladbaum/chess_project/zero_shot_augmentations/plots/games_2_4_5_6_eval/real_test_combined_confusion.png`
- `/home/eladbaum/chess_project/zero_shot_augmentations/plots/games_2_4_5_6_eval/real_test_game{N}_confusion.png`  (per game)