# Real-only training — results

Training data: real images from games 2, 4, 5, 6 only — no synthetic data.
Validation (selection signal): 30 frames from real_labels.csv (games 8-11).
Test (held out, evaluated once): game7.

## Training
- Total training time: **14.3 min** (10 epochs, 29,568 real train squares).
- Best real_val_acc: **0.8297** at epoch **7**.
- Final-epoch train_acc: 0.9531
- Final-epoch real_val_acc: 0.8187

## Real test — game7 (55 frames, 3520 squares)
- **Per-square accuracy: 0.8926**
- **Per-board accuracy (all 64 correct): 0/55 = 0.0000**
- Mean squares correct / board: **57.13 / 64**

### Per-class on real test
```
class  name              n      acc
    0  White Pawn         390  0.9308
    1  White Rook          93  0.8602
    2  White Knight        82  0.6585
    3  White Bishop       110  0.5273
    4  White Queen         55  0.5636
    5  White King          55  0.7455
    6  Black Pawn         388  0.8943
    7  Black Rook         110  0.8636
    8  Black Knight        82  0.9512
    9  Black Bishop        99  0.6061
   10  Black Queen         55  0.1091
   11  Black King          55  0.9091
   12  Empty             1946  0.9656
```

### Top-5 game7 confusion pairs
```
  Black Queen    → Black Bishop    37
  Black Pawn     → Black Bishop    31
  White Knight   → White Pawn      23
  Empty          → Black Bishop    21
  Empty          → Black Pawn      18
```

## Comparison context (zero-shot reference, different model)
These numbers come from models trained only on synthetic data — not directly
comparable architecturally (same ResNet18 + ImageNet head, different training
data), but a useful reference point for game7 per-square accuracy.

| Model                        | per-square | per-board | mean correct/64 |
|------------------------------|-----------:|----------:|----------------:|
| Zero-shot baseline (synth)   |     0.5670 |      0/55 |           36.29 |
| Zero-shot Wölflein (synth)   |     0.5213 |      0/55 |           33.36 |
| **Real-only (this run)**     | **0.8926** | **0/55** | **57.13** |

## Artifacts
- `results/training_log.csv`
- `results/best_real.pt`  (epoch 7, real_val_acc=0.8297)
- `results/latest.pt`
- `results/real_test_per_class.txt`
- `results/real_test_per_board_accuracy.txt`
- `plots/training_curves.png`
- `plots/real_test_confusion.png`