"""
DIAGNOSTIC ONLY — test-tuning re-scan. NOT a change to the headline selection rule.

The headline rule stays: checkpoint selected by game7 real_val per-square accuracy,
with games 2/6 a clean held-out test for Comparison B. This script does NOT touch that.

It answers one question, eval-only (NO retraining): for the existing s00/s03/s05 runs,
if we had instead picked a DIFFERENT saved checkpoint, what would the held-out games-2/6
per-square / piece-only have been? We can only do this over the checkpoints that were
actually saved (best_real = game7-selected, latest, best_synth_monitor) — a true
per-epoch held-out scan would need per-epoch checkpoints we deliberately did not save
(saving them = the retraining we are avoiding). Treat the "best over saved ckpts" column
as an optimistic test-peeking bound, reported for diagnosis, not as a selection rule.

The RealGameDataset below is a VERBATIM copy of train.py Cell 5 (same corner-OOB
fallback, same warp/crop, same orientation) so crops — hence predictions — are identical
to the headline eval path. Metrics (per-square, piece-only) are the same trivial formulas.
"""
import sys
sys.path.insert(0, "/home/eladbaum/chess_project")
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from preprocessing.fen_to_grid import fen_to_label_grid
from preprocessing.verify_woelflein_crops import (
    warp_chessboard_image, crop_square, find_corners, ChessboardNotLocatedException,
)

PROJECT_ROOT = "/home/eladbaum/chess_project"
EXP_DIR = f"{PROJECT_ROOT}/fine_tuning/stage3_improved"
BASELINE_DIR = f"{PROJECT_ROOT}/fine_tuning/stage3_323/results"
HELD_OUT_GAMES = [2, 6]
NUM_CLASSES = 13
SEED = 42
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)


class RealGameDataset(Dataset):  # verbatim copy of train.py Cell 5 — keep in sync
    CORNER_OOB_TOLERANCE = 8

    def __init__(self, gt_csv_path, images_dir, game_name, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.game_name = game_name
        rows = []
        with open(gt_csv_path) as f:
            for r in csv.DictReader(f):
                grid = fen_to_label_grid(r["fen"], game_name)
                for br in range(8):
                    for bc in range(8):
                        rows.append({"image_name": r["image_name"], "board_row": br,
                                     "board_col": bc, "label": int(grid[br, bc])})
        import pandas as pd
        self.manifest = pd.DataFrame(rows).sort_values(
            ["image_name", "board_row", "board_col"]).reset_index(drop=True)
        self._corner_cache = {}

    def __len__(self):
        return len(self.manifest)

    def _get_corners(self, image_name, bgr):
        if image_name in self._corner_cache:
            return self._corner_cache[image_name]
        H, W = bgr.shape[:2]
        try:
            np.random.seed(SEED)
            corners = find_corners(bgr)
            lo, hi_x, hi_y = -self.CORNER_OOB_TOLERANCE, W + self.CORNER_OOB_TOLERANCE, H + self.CORNER_OOB_TOLERANCE
            if not bool(np.all((corners[:, 0] >= lo) & (corners[:, 0] <= hi_x)
                               & (corners[:, 1] >= lo) & (corners[:, 1] <= hi_y))):
                raise ChessboardNotLocatedException("corners OOB")
        except Exception:
            corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        self._corner_cache[image_name] = corners
        return corners

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        image_name = row["image_name"]
        bgr = cv2.imread(str(self.images_dir / image_name))
        corners = self._get_corners(image_name, bgr)
        warped = warp_chessboard_image(bgr, corners)
        crop_bgr = crop_square(warped, int(row["board_row"]), int(row["board_col"]))
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(crop_rgb)).permute(2, 0, 1).float() / 255.0
        return tensor, int(row["label"])


def build_model():
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


@torch.no_grad()
def eval_loader(model, loader):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        xb = (xb - IMAGENET_MEAN) / IMAGENET_STD
        preds.append(model(xb).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def metrics(preds, labels):
    persq = float((preds == labels).mean())
    pm = labels != 12
    piece = float((preds[pm] == labels[pm]).mean()) if pm.any() else float("nan")
    empty_m = labels == 12
    empty = float((preds[empty_m] == labels[empty_m]).mean()) if empty_m.any() else float("nan")
    return persq, piece, empty


def main():
    # Build held-out datasets/loaders ONCE (corner cache persists across all checkpoints).
    loaders = []
    for N in HELD_OUT_GAMES:
        ds = RealGameDataset(f"{PROJECT_ROOT}/data/game{N}_per_frame/gt.csv",
                             f"{PROJECT_ROOT}/data/game{N}_per_frame/images",
                             game_name=f"game{N}", transform=None)
        loaders.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=4, pin_memory=True))
    base = json.load(open(f"{BASELINE_DIR}/held_out_aggregate.json"))
    print(f"BASELINE (frozen stage3_323, game7-selected): "
          f"2/6 per-sq={base['per_square_acc']:.4f}  piece-only={base['piece_only_acc']:.4f}\n")

    CKPTS = ["best_real", "latest", "best_synth_monitor"]
    rows = []
    for run in ["s00", "s03", "s05"]:
        print(f"=== {run} ===")
        for ck in CKPTS:
            p = f"{EXP_DIR}/checkpoints/{run}/{ck}.pt"
            if not Path(p).exists():
                print(f"  {ck:20} MISSING"); continue
            d = torch.load(p, map_location=DEVICE, weights_only=False)
            model = build_model().to(DEVICE)
            model.load_state_dict(d["model_state_dict"])
            ap, al = [], []
            for ld in loaders:
                pr, la = eval_loader(model, ld)
                ap.append(pr); al.append(la)
            preds = np.concatenate(ap); labels = np.concatenate(al)
            persq, piece, empty = metrics(preds, labels)
            tag = "  <- headline (game7-selected)" if ck == "best_real" else ""
            print(f"  {ck:20} ep{d.get('epoch'):>2}  2/6 per-sq={persq:.4f}  "
                  f"piece-only={piece:.4f}  empty={empty:.4f}{tag}")
            rows.append({"run": run, "ckpt": ck, "epoch": d.get("epoch"),
                         "persq": persq, "piece_only": piece, "empty": empty})
        print()

    out = f"{EXP_DIR}/results/checkpoint_selection_rescan.json"
    json.dump({"note": "DIAGNOSTIC test-tuning only; headline selection unchanged (game7).",
               "baseline_2_6": {"per_square": base["per_square_acc"],
                                "piece_only": base["piece_only_acc"]},
               "rows": rows}, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
