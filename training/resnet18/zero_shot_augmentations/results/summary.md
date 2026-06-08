# Zero-shot baseline — Step 6 results

## Training
- Total training time: **156.5 min** (10 epochs, 274,176 synth train samples).
- Best synth val acc: **0.9992** at epoch **6**.
- Final-epoch synth val acc: 0.9992
- Final-epoch real val acc (monitor): 0.5372

## Synthetic test (held-out FENs, 59,904 squares)
- **Overall accuracy: 0.9993**

### Per-class
```
class  name              n      acc
    0  White Pawn        5016  0.9996
    1  White Rook        1356  0.9985
    2  White Knight       912  0.9945
    3  White Bishop       852  0.9894
    4  White Queen        492  0.9939
    5  White King         936  0.9925
    6  Black Pawn        4656  1.0000
    7  Black Rook        1356  0.9978
    8  Black Knight       636  1.0000
    9  Black Bishop       984  0.9959
   10  Black Queen        480  0.9958
   11  Black King         936  0.9968
   12  Empty            41292  1.0000
```

### Per-view
```
view      n        acc
  overhead   19968   0.9986
  west       19968   0.9996
  east       19968   0.9997
```

## Real test — game7 (55 frames, 3520 squares)
- **Per-square accuracy: 0.5213**
- **Per-board accuracy (all 64 correct): 0/55 = 0.0000**
- Mean squares correct / board: **33.36 / 64**

### Per-class on real
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

## Sim-to-real gap
- synth_test − game7 (per-square): 0.9993 − 0.5213 = **+0.4780**

### Top-5 game7 confusion pairs
```
  Empty          → White Queen     159
  Black Pawn     → Black Bishop    154
  Empty          → White Bishop    116
  White Pawn     → Black Pawn      108
  White Pawn     → White Queen     90
```

## Artifacts
- `results/training_log.csv`
- `results/best_synth.pt`  (epoch 6, synth_val_acc=0.9992)
- `results/latest.pt`
- `results/synth_test_per_class.txt`
- `results/synth_test_per_view.txt`
- `results/real_test_per_class.txt`
- `results/real_test_per_board_accuracy.txt`
- `plots/training_curves.png`
- `plots/synth_test_confusion.png`
- `plots/real_test_confusion.png`