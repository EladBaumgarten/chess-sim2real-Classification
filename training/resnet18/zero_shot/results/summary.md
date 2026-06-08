# Zero-shot baseline — Step 6 results

## Training
- Total training time: **133.8 min** (8 epochs, 274,176 synth train samples).
- Best synth val acc: **0.9987** at epoch **7**.
- Final-epoch synth val acc: 0.9987
- Final-epoch real val acc (monitor): 0.5500

## Synthetic test (held-out FENs, 59,904 squares)
- **Overall accuracy: 0.9991**

### Per-class
```
class  name              n      acc
    0  White Pawn        5016  0.9992
    1  White Rook        1356  0.9971
    2  White Knight       912  0.9978
    3  White Bishop       852  0.9941
    4  White Queen        492  0.9919
    5  White King         936  0.9850
    6  Black Pawn        4656  0.9991
    7  Black Rook        1356  0.9971
    8  Black Knight       636  0.9984
    9  Black Bishop       984  0.9959
   10  Black Queen        480  0.9958
   11  Black King         936  0.9979
   12  Empty            41292  0.9999
```

### Per-view
```
view      n        acc
  overhead   19968   0.9986
  west       19968   0.9991
  east       19968   0.9995
```

## Real test — game7 (55 frames, 3520 squares)
- **Per-square accuracy: 0.5670**
- **Per-board accuracy (all 64 correct): 0/55 = 0.0000**
- Mean squares correct / board: **36.29 / 64**

### Per-class on real
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

## Sim-to-real gap
- synth_test − game7 (per-square): 0.9991 − 0.5670 = **+0.4320**

### Top-5 game7 confusion pairs
```
  White Pawn     → Black Pawn      149
  Black Pawn     → Black Queen     131
  Empty          → White Queen     127
  White Pawn     → White Queen     105
  White Pawn     → White Bishop    68
```

## Artifacts
- `results/training_log.csv`
- `results/best_synth.pt`  (epoch 7, synth_val_acc=0.9987)
- `results/latest.pt`
- `results/synth_test_per_class.txt`
- `results/synth_test_per_view.txt`
- `results/real_test_per_class.txt`
- `results/real_test_per_board_accuracy.txt`
- `plots/training_curves.png`
- `plots/synth_test_confusion.png`
- `plots/real_test_confusion.png`

## Matched-partition re-eval — games 2/6 only (added for the Stage 5 ablation)

Eval-only re-evaluation of the SAME `best_synth.pt` zero-shot weights on the **games 2/6**
partition (169 frames / 10,816 squares) — the exact held-out test Stage 3 and Stage 5 report
on — so all three ablation rows sit on one test set. Uses the identical games 2/6
loaders / warp + 100px crop / per-square metric as Stage 3 & Stage 5 (the same harness reproduces
Stage 3's held-out 0.9085 exactly). Reproduces this baseline's own per-game numbers exactly:
game2 0.5181, game6 0.5102 (cf. `games_2_4_5_6_eval/real_test_per_game_summary.txt`).

- **per-square acc: 0.5138**
- **piece-only acc: 0.2272**
- empty acc: 0.6826
- per-game: game2 0.5181 / game6 0.5102

### Per-class (games 2/6)
```
wP 0.070  wR 0.300  wN 0.053  wB 0.040  wQ 0.538  wK 0.000
bP 0.490  bR 0.142  bN 0.111  bB 0.091  bQ 0.351  bK 0.053  empty 0.683
```

### Matched ablation table (games 2/6)
| model (games 2/6)        | per-square | piece-only |
|--------------------------|-----------:|-----------:|
| synth-only (zero-shot)   |   0.5138   |   0.2272   |
| real fine-tune (Stage 3) |   0.9085   |   0.7556   |
| combined (Stage 5)       |   0.9160   |   0.7748   |

Artifact: `results/zeroshot_reeval_on_games_2_6.json` — produced eval-only (no training, no
checkpoint writes) by `fine_tuning/stage3_improved/zeroshot_reeval_on_games_2_6.py`.