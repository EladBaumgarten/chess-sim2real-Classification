"""EVAL-ONLY: re-evaluate the zero-shot (synth-only) v1 baseline on games 2/6, the exact
partition Stage 3 and Stage 5 use, so all three ablation rows sit on the same test set.

Reuses RealGameDataset + eval_loader from rescan_checkpoint_selection.py (a verbatim copy of
train.py Cell 5's crop/warp pipeline + the same ImageNet-normalize→argmax eval), so the number
is directly comparable to Stage 3 (0.9085/0.7556) and Stage 5 (0.9160/0.7748). The same harness
already reproduced s00's held-out 0.9085 exactly, validating the match.

No training, no checkpoint writing. Output: stage3_improved/results/zeroshot_reeval_on_games_2_6.json
"""
import sys, os, json
sys.path.insert(0, "/home/eladbaum/chess_project/training/resnet18/fine_tuning/stage3_improved")
import numpy as np
import torch
from torch.utils.data import DataLoader

from rescan_checkpoint_selection import (
    RealGameDataset, build_model, eval_loader, metrics,
    PROJECT_ROOT, EXP_DIR, HELD_OUT_GAMES, NUM_CLASSES, BATCH_SIZE, DEVICE,
)
CLASS_SHORT = ["wP","wR","wN","wB","wQ","wK","bP","bR","bN","bB","bQ","bK","empty"]

CKPT = f"{PROJECT_ROOT}/zero_shot/results/best_synth.pt"
OUT = f"{EXP_DIR}/results/zeroshot_reeval_on_games_2_6.json"

# write-guard: output must live under stage3_improved/, never the frozen/zero_shot dirs
_abs = os.path.realpath(OUT)
assert _abs.startswith(os.path.realpath(EXP_DIR) + os.sep), f"WRITE-GUARD: {_abs} not under {EXP_DIR}"
assert "zero_shot/results" not in _abs and "stage3_323" not in _abs, f"WRITE-GUARD: {_abs} hits a frozen dir"
assert "v1.5" not in CKPT, "must use the v1 baseline, not v1.5"

print(f"loading zero-shot checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = build_model().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
print(f"  source epoch={ckpt.get('epoch')}  synth_val_acc={ckpt.get('synth_val_acc', float('nan')):.4f}")

# Same games 2/6 loaders as Stage 3/5 (HELD_OUT_GAMES=[2,6], verbatim RealGameDataset, transform=None)
all_p, all_y = [], []
per_game = {}
for N in HELD_OUT_GAMES:
    ds = RealGameDataset(f"{PROJECT_ROOT}/data/game{N}_per_frame/gt.csv",
                         f"{PROJECT_ROOT}/data/game{N}_per_frame/images", f"game{N}", transform=None)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    p, y = eval_loader(model, ld)
    per_game[N] = metrics(p, y)
    all_p.append(p); all_y.append(y)
    print(f"  game{N}: {ds.manifest['image_name'].nunique()} frames, {len(ds)} squares  "
          f"per-sq={per_game[N][0]:.4f} piece={per_game[N][1]:.4f}")

preds = np.concatenate(all_p); labels = np.concatenate(all_y)
persq, piece, empty = metrics(preds, labels)
per_class = {CLASS_SHORT[c]: (float((preds[labels==c]==c).mean()) if (labels==c).any() else None)
             for c in range(NUM_CLASSES)}

payload = {
    "model": "zero-shot synth-only (v1 baseline, zero_shot/results/best_synth.pt)",
    "test_partition": ["game2", "game6"],
    "eval_path": "verbatim RealGameDataset (train.py Cell 5) + ImageNet-normalize -> argmax; "
                 "same loaders/metric as Stage 3 & Stage 5 held-out 2/6 (harness reproduced s00 0.9085 exactly)",
    "n_frames": int(np.sum([RealGameDataset(f'{PROJECT_ROOT}/data/game{N}_per_frame/gt.csv', f'{PROJECT_ROOT}/data/game{N}_per_frame/images', f'game{N}').manifest['image_name'].nunique() for N in HELD_OUT_GAMES])),
    "n_squares": int(len(preds)),
    "per_square_acc": persq,
    "piece_only_acc": piece,
    "empty_acc": empty,
    "per_class_acc": per_class,
    "per_game": {f"game{N}": {"per_square": per_game[N][0], "piece_only": per_game[N][1]} for N in HELD_OUT_GAMES},
}
with open(OUT, "w") as f:
    json.dump(payload, f, indent=2)

print("\n=== ZERO-SHOT (synth-only) on games 2/6 ===")
print(f"  per-square acc = {persq:.4f}")
print(f"  piece-only acc = {piece:.4f}")
print(f"  empty acc      = {empty:.4f}")
print("  per-class: " + "  ".join(f"{c}={per_class[c]:.3f}" if per_class[c] is not None else f"{c}=n/a" for c in CLASS_SHORT))
print(f"\nwrote {OUT}")

print("\n=== MATCHED ABLATION TABLE (games 2/6) ===")
print(f"| model (games 2/6)        | per-square | piece-only |")
print(f"| synth-only (zero-shot)   |   {persq:.4f}   |   {piece:.4f}   |")
print(f"| real fine-tune (Stage 3) |   0.9085   |   0.7556   |")
print(f"| combined (Stage 5)       |   0.9160   |   0.7748   |")
