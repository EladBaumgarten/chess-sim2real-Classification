"""
Project 2 — Real-only training (no synthetic data anywhere in the pipeline).

This is NOT fine-tuning of any zero-shot model. It is a fresh ResNet18
(ImageNet-pretrained backbone + random 13-class FC head) trained from
scratch on real chess frames only.

Splits:
  TRAIN: games 2, 4, 5, 6 (combined, ~33k real squares)
  VAL:   30 frames from real_labels.csv (games 8-11, ~1.9k squares)
         — used for checkpoint selection
  TEST:  game 7 (~3.5k squares) — held out, evaluated once at the end

Hyperparameters:
  ResNet18 + ImageNet weights, fresh 13-class FC head
  SGD (lr=1e-4, momentum=0.9, weight_decay=1e-4)
  StepLR(step=4, gamma=0.1)  → lr drops to 1e-5 at epoch 5
  10 epochs, batch size 64
  Sqrt-inverse-frequency class sampler
  Wölflein-style augmentation (color jitter + shear + gaussian noise)

Note on LR: 1e-4 (vs 1e-3 in the zero-shot runs) because (a) 462 frames is
small and aggressive LR would overfit fast, (b) ImageNet backbone features
are more useful for real photos than for synth renders so we want to
preserve them more carefully. Matches Wölflein 2021's fine-tuning LR.
"""

# %% [Cell 1 — Imports + GPU check + seeds]
import sys
sys.path.insert(0, "/home/eladbaum/chess_project")

import csv
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Project modules (no synthetic-data imports — ChessSquareDataset intentionally not imported)
from preprocessing.fen_to_grid import fen_to_label_grid
from preprocessing.verify_woelflein_crops import (
    warp_chessboard_image, crop_square,
    find_corners, ChessboardNotLocatedException,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")

print("\033[92m✓ Cell 1 — Imports + GPU check + seeds — OK\033[0m")




# %% [Cell 2 — Config + data-path verification]
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_WORKERS = 6
LR = 1e-4                       # 10× lower than zero-shot runs — see top-of-file
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 4
LR_GAMMA = 0.1

# === Augmentation strengths (Wölflein-style — KEPT verbatim from train_augmented.py)
COLOR_JITTER_BRIGHTNESS = 0.3
COLOR_JITTER_CONTRAST = 0.3
COLOR_JITTER_SATURATION = 0.3
COLOR_JITTER_HUE = 0.05
GAUSSIAN_NOISE_STD = 0.02
SHEAR_PROB = 0.80
SHEAR_DEG_RANGE = (-8, 8)
AFFINE_TRANSLATE_RANGE = (-0.03, 0.03)
AFFINE_SCALE_RANGE = (0.97, 1.03)
COLOR_JITTER_APPLY_PROB = 0.60
NOISE_APPLY_PROB = 0.30
# ==============================================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

NUM_CLASSES = 13
CLASS_NAMES = [
    "White Pawn",   "White Rook",   "White Knight", "White Bishop",
    "White Queen",  "White King",
    "Black Pawn",   "Black Rook",   "Black Knight", "Black Bishop",
    "Black Queen",  "Black King",
    "Empty",
]

PROJECT_ROOT = "/home/eladbaum/chess_project"
REAL_DATA_ROOT = f"{PROJECT_ROOT}/data"

TRAIN_GAMES = [2, 4, 5, 6]
TEST_GAME = 7

# Per-game train/test paths
def _game_paths(N):
    return (
        f"{REAL_DATA_ROOT}/game{N}_per_frame/gt.csv",
        f"{REAL_DATA_ROOT}/game{N}_per_frame/images",
    )

TRAIN_GAME_PATHS = {N: _game_paths(N) for N in TRAIN_GAMES}
TEST_GT_CSV, TEST_IMAGES_DIR = _game_paths(TEST_GAME)
REAL_VAL_CSV = f"{REAL_DATA_ROOT}/real_labels.csv"

# Output root — Real_Only, NOT zero_shot* (variable name kept for diff minimality)
ZERO_SHOT_DIR = f"{PROJECT_ROOT}/Real_Only"
RESULTS_DIR = f"{ZERO_SHOT_DIR}/results"
PLOTS_DIR = f"{ZERO_SHOT_DIR}/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Hard path verification: abort BEFORE any expensive work if anything is missing
required_paths = []
for N in TRAIN_GAMES:
    gt, img = TRAIN_GAME_PATHS[N]
    required_paths.append((f"game{N} gt.csv", gt))
    required_paths.append((f"game{N} images dir", img))
required_paths.append((f"game{TEST_GAME} gt.csv (test)", TEST_GT_CSV))
required_paths.append((f"game{TEST_GAME} images dir (test)", TEST_IMAGES_DIR))
required_paths.append(("real_labels.csv (val)", REAL_VAL_CSV))

missing = [(label, p) for label, p in required_paths if not Path(p).exists()]
if missing:
    msg = "ABORT: missing required data paths:\n" + "\n".join(
        f"  - {label}: {p}" for label, p in missing
    )
    raise FileNotFoundError(msg)

# Schema preview of real_labels.csv — verify column names before downstream code uses them
_val_df_preview = pd.read_csv(REAL_VAL_CSV)
print(f"real_labels.csv: {len(_val_df_preview)} rows, columns={list(_val_df_preview.columns)}")
print("first 3 rows (for schema verification):")
for i in range(min(3, len(_val_df_preview))):
    print(f"  row {i}: {dict(_val_df_preview.iloc[i])}")
expected_cols = {"image_path", "fen", "view"}
missing_cols = expected_cols - set(_val_df_preview.columns)
if missing_cols:
    raise ValueError(f"real_labels.csv missing required columns: {missing_cols}")
del _val_df_preview

print()
print(f"train games:    {TRAIN_GAMES}")
print(f"test game:      {TEST_GAME}")
print(f"val CSV:        {REAL_VAL_CSV}")
print(f"results dir:    {RESULTS_DIR}")
print(f"plots dir:      {PLOTS_DIR}")
print(f"LR:             {LR}  (10× lower than zero-shot runs)")
print(f"epochs:         {NUM_EPOCHS}")
print(f"batch size:     {BATCH_SIZE}")

print("\033[92m✓ Cell 2 — Config + path verification — OK\033[0m")




# %% [Cell 3 — Build combined TRAIN manifest from games 2/4/5/6]
# No train/val/test split here: train is the union of all four games' gt.csv,
# expanded to 64 rows per frame via fen_to_label_grid(fen, f"game{N}").
# Val (real_labels.csv) and test (game7) are built separately in Cells 5.
train_manifest_rows = []
per_game_frame_counts = {}
per_game_square_counts = {}

for N in TRAIN_GAMES:
    gt_csv, _ = TRAIN_GAME_PATHS[N]
    n_frames = 0
    with open(gt_csv) as f:
        for r in csv.DictReader(f):
            n_frames += 1
            fen = r["fen"]
            # fen_to_label_grid uses view_orientations.VIEW_ORIENTATIONS[f"game{N}"]
            # which is locked to "identity" for all train games.
            grid = fen_to_label_grid(fen, f"game{N}")
            for br in range(8):
                for bc in range(8):
                    train_manifest_rows.append({
                        "game_num": N,
                        "image_name": r["image_name"],
                        "board_row": br,
                        "board_col": bc,
                        "label": int(grid[br, bc]),
                        "fen": fen,
                    })
    per_game_frame_counts[N] = n_frames
    per_game_square_counts[N] = n_frames * 64

train_manifest = pd.DataFrame(train_manifest_rows)
# Sort by (game_num, image_name, board_row, board_col) so RealTrainDataset's
# manifest is in the same order as its __getitem__ routing.
train_manifest = train_manifest.sort_values(
    ["game_num", "image_name", "board_row", "board_col"]
).reset_index(drop=True)

print(f"Combined train manifest: {len(train_manifest):,} rows "
      f"({sum(per_game_frame_counts.values())} frames × 64)")
print()
print("Per-game frame and square counts:")
print(f"  {'game':>6s}  {'frames':>7s}  {'squares':>8s}  {'% squares':>9s}")
total_sq = sum(per_game_square_counts.values())
for N in TRAIN_GAMES:
    n_f = per_game_frame_counts[N]
    n_s = per_game_square_counts[N]
    print(f"  game{N:<2d}  {n_f:>7d}  {n_s:>8d}  {n_s/total_sq*100:>8.2f}%")
print(f"  {'TOTAL':>6s}  {sum(per_game_frame_counts.values()):>7d}  {total_sq:>8d}")
print()
print("Per-class square count across combined train set:")
print(f"  {'cls':>3s}  {'name':<14s}  {'count':>8s}  {'pct':>7s}")
for cls in range(NUM_CLASSES):
    n = int((train_manifest["label"] == cls).sum())
    pct = n / len(train_manifest) * 100
    print(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>8d}  {pct:>6.2f}%")

# Sanity check: every class must be present, or sqrt-inverse-freq sampler divides by 0.
present = set(train_manifest["label"].unique())
missing_classes = set(range(NUM_CLASSES)) - present
if missing_classes:
    raise ValueError(
        f"Combined train manifest is missing classes {missing_classes}. "
        "sqrt-inverse-frequency sampler would divide by zero. Check FEN content."
    )

print("\033[92m✓ Cell 3 — Build train manifest — OK\033[0m")




# %% [Cell 4 — Augmentation transforms + RealGameDataset + RealTrainDataset]
# Wölflein-style augmentation: color jitter (60%) + affine/shear (80%) + noise (30%).
# Kept verbatim from train_augmented.py.  Regularization is essential here: 33k
# training squares is small for an 11M-parameter ResNet18 and would overfit fast
# without augmentation.

def gaussian_noise(x_rgb_uint8):
    x = x_rgb_uint8.astype(np.float32)
    noise = np.random.normal(0, GAUSSIAN_NOISE_STD * 255, x.shape).astype(np.float32)
    return np.clip(x + noise, 0, 255).astype(np.uint8)


def color_jitter(x_rgb_uint8):
    x = x_rgb_uint8.astype(np.float32)
    b = 1.0 + np.random.uniform(-COLOR_JITTER_BRIGHTNESS, COLOR_JITTER_BRIGHTNESS)
    x = x * b
    c = 1.0 + np.random.uniform(-COLOR_JITTER_CONTRAST, COLOR_JITTER_CONTRAST)
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * c + mean
    s = 1.0 + np.random.uniform(-COLOR_JITTER_SATURATION, COLOR_JITTER_SATURATION)
    gray = x.mean(axis=2, keepdims=True)
    x = gray + s * (x - gray)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if COLOR_JITTER_HUE > 0:
        h_shift = np.random.uniform(-COLOR_JITTER_HUE, COLOR_JITTER_HUE) * 180.0
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(h_shift)) % 180
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return x


def affine_shear(x_rgb_uint8):
    h, w = x_rgb_uint8.shape[:2]
    shear_x_deg = np.random.uniform(*SHEAR_DEG_RANGE)
    shear_y_deg = np.random.uniform(*SHEAR_DEG_RANGE)
    tx = np.random.uniform(*AFFINE_TRANSLATE_RANGE) * w
    ty = np.random.uniform(*AFFINE_TRANSLATE_RANGE) * h
    scale = np.random.uniform(*AFFINE_SCALE_RANGE)
    sx = np.tan(np.deg2rad(shear_x_deg))
    sy = np.tan(np.deg2rad(shear_y_deg))
    cx, cy = w / 2.0, h / 2.0
    M = np.array([
        [scale,         scale * sx, cx - cx * scale - cy * scale * sx + tx],
        [scale * sy,    scale,      cy - cy * scale - cx * scale * sy + ty],
    ], dtype=np.float32)
    return cv2.warpAffine(
        x_rgb_uint8, M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def train_transform(crop_rgb_uint8):
    """Wölflein-style augmentation: color jitter + shear + noise.
    HWC uint8 RGB → HWC uint8 RGB. Tensorization happens after in the model loop.
    Order (each gated by its own probability):
      (a) color jitter — 60%
      (b) affine + shear — 80% — Wölflein's key sim-to-real lever
      (c) gaussian noise — 30%
    """
    x = crop_rgb_uint8
    if random.random() < COLOR_JITTER_APPLY_PROB:
        x = color_jitter(x)
    if random.random() < SHEAR_PROB:
        x = affine_shear(x)
    if random.random() < NOISE_APPLY_PROB:
        x = gaussian_noise(x)
    return x


# --- Dataset classes -----------------------------------------------------------
# RealGameDataset: one sample per (frame × board square) for a single game.
# Per-image find_corners with OOB rejection + image-corner fallback (chesscog
# hallucinates board extensions on tight-cropped real photos — Step 6a finding).
# In-memory corner cache so 64 squares from one frame don't re-run detection.

class RealGameDataset(Dataset):
    CORNER_OOB_TOLERANCE = 8

    def __init__(self, gt_csv_path, images_dir, game_name, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.game_name = game_name
        rows = []
        with open(gt_csv_path) as f:
            for r in csv.DictReader(f):
                fen = r["fen"]
                grid = fen_to_label_grid(fen, game_name)
                for br in range(8):
                    for bc in range(8):
                        rows.append({
                            "image_name": r["image_name"],
                            "board_row": br,
                            "board_col": bc,
                            "label": int(grid[br, bc]),
                            "fen": fen,
                        })
        self.manifest = pd.DataFrame(rows)
        self.manifest = self.manifest.sort_values(
            ["image_name", "board_row", "board_col"]
        ).reset_index(drop=True)
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
            lo, hi_x, hi_y = (
                -self.CORNER_OOB_TOLERANCE,
                W + self.CORNER_OOB_TOLERANCE,
                H + self.CORNER_OOB_TOLERANCE,
            )
            in_bounds = bool(np.all(
                (corners[:, 0] >= lo) & (corners[:, 0] <= hi_x)
                & (corners[:, 1] >= lo) & (corners[:, 1] <= hi_y)
            ))
            if not in_bounds:
                raise ChessboardNotLocatedException("corners OOB")
        except Exception:
            corners = np.array(
                [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                dtype=np.float32,
            )
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
        if self.transform is not None:
            crop_rgb = self.transform(crop_rgb)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(crop_rgb))
                 .permute(2, 0, 1).float() / 255.0
        )
        return tensor, int(row["label"])


# RealTrainDataset: concatenates RealGameDataset across train games (2/4/5/6).
# Exposes a unified .manifest in the same row order as __getitem__ routing,
# so train_dataset.manifest["label"].values matches index i of __getitem__.

class RealTrainDataset(Dataset):
    def __init__(self, train_games, train_game_paths, transform=None):
        self.subsets = []
        manifests = []
        for N in train_games:
            gt_csv, images_dir = train_game_paths[N]
            sub = RealGameDataset(
                gt_csv_path=gt_csv,
                images_dir=images_dir,
                game_name=f"game{N}",
                transform=transform,
            )
            self.subsets.append(sub)
            mani = sub.manifest.copy()
            mani["game_num"] = N
            manifests.append(mani)
        self.manifest = pd.concat(manifests, ignore_index=True)
        # cumulative offsets so __getitem__ can route flat index → (subset, local_idx)
        self._offsets = np.cumsum([0] + [len(s) for s in self.subsets])

    def __len__(self):
        return int(self._offsets[-1])

    def __getitem__(self, idx):
        sub_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local_idx = idx - int(self._offsets[sub_idx])
        return self.subsets[sub_idx][local_idx]


train_dataset = RealTrainDataset(
    train_games=TRAIN_GAMES,
    train_game_paths=TRAIN_GAME_PATHS,
    transform=train_transform,
)
print(f"train_dataset: {len(train_dataset):,} samples "
      f"(games {TRAIN_GAMES}, Wölflein-style augmentation)")

# Cross-check: manifest length must equal __len__ — if not, sampler weights misalign
assert len(train_dataset.manifest) == len(train_dataset), (
    f"manifest length {len(train_dataset.manifest)} != "
    f"dataset length {len(train_dataset)} — sampler weights would misalign"
)

print("\033[92m✓ Cell 4 — Augmentation + RealTrainDataset — OK\033[0m")




# %% [Cell 5 — RealValDataset + build real_val_dataset + real_test_dataset]
# Two datasets are built here:
#   - real_val_dataset:  30 frames from real_labels.csv (games 8-11)
#                        SELECTION SIGNAL (gates checkpoint saves in Cell 12)
#   - real_test_dataset: game7 — HELD-OUT test, evaluated once at the end (Cell 15)

class RealValDataset(Dataset):
    """30-frame val set from real_labels.csv.

    Schema: ply, image_path, fen, view, game.
    image_path values are relative (e.g. 'c06/game8/images/frame_X.jpg' or
    'c17/game11/images/frame_X.jpg').  Resolution: f'{REAL_DATA_ROOT}/{image_path}'.
    Missing images warn-and-skip rather than abort.

    Orientation: 'view' column drives FEN→grid mapping locally (no dependency on
    VIEW_ORIENTATIONS having entries for games 8-11):
      - 'white' → identity (FEN-native row 0 = rank 8 = image top)
      - 'black' → rot180 (camera behind black; rank 1 at image top)
    Identity is obtained by calling fen_to_label_grid(fen, 'game7'), whose
    orientation is locked to 'identity' in view_orientations.py.

    Same corner-detection + OOB-fallback + in-memory corner cache as
    RealGameDataset.
    """

    CORNER_OOB_TOLERANCE = 8

    def __init__(self, csv_path, real_data_root, transform=None):
        self.transform = transform
        df = pd.read_csv(csv_path)

        # Schema verification — print first 3 rows for human eyeballing
        required = {"image_path", "fen", "view"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"real_labels.csv missing required columns: {missing_cols}. "
                f"Got: {list(df.columns)}"
            )
        print(f"  RealValDataset reading {csv_path}: {len(df)} rows, "
              f"columns={list(df.columns)}")
        for i in range(min(3, len(df))):
            print(f"    row {i}: {dict(df.iloc[i])}")

        rows = []
        n_resolved = 0
        n_skipped = 0
        for ri, row in df.iterrows():
            abs_path = f"{real_data_root}/{row['image_path']}"
            if not os.path.exists(abs_path):
                print(f"  [warn] row {ri}: image not found, skipping: {abs_path}")
                n_skipped += 1
                continue
            n_resolved += 1
            # Build the (8,8) image-space label grid.
            # game7 is locked to 'identity' in VIEW_ORIENTATIONS, so this gives
            # FEN-native grid (row 0 = rank 8 = image top for white-view).
            raw = fen_to_label_grid(row["fen"], "game7")
            if row["view"] == "white":
                grid = raw
            elif row["view"] == "black":
                grid = np.ascontiguousarray(np.rot90(raw, 2))
            else:
                raise ValueError(
                    f"row {ri}: unknown view {row['view']!r} (expected 'white' or 'black')"
                )
            for br in range(8):
                for bc in range(8):
                    rows.append({
                        "image_path_abs": abs_path,
                        "image_key": row["image_path"],   # relative — cache key
                        "board_row": br,
                        "board_col": bc,
                        "label": int(grid[br, bc]),
                        "fen": row["fen"],
                        "view": row["view"],
                        "game": row.get("game", ""),
                    })

        print(f"  RealValDataset: resolved {n_resolved} frames, "
              f"skipped {n_skipped} (missing on disk)")
        if n_resolved == 0:
            raise RuntimeError(
                "RealValDataset resolved 0 frames — val set is empty. "
                "Verify c06.zip / c17.zip were extracted under data/."
            )

        self.manifest = pd.DataFrame(rows)
        self.manifest = self.manifest.sort_values(
            ["image_key", "board_row", "board_col"]
        ).reset_index(drop=True)
        self._corner_cache = {}

    def __len__(self):
        return len(self.manifest)

    def _get_corners(self, image_key, bgr):
        if image_key in self._corner_cache:
            return self._corner_cache[image_key]
        H, W = bgr.shape[:2]
        try:
            np.random.seed(SEED)
            corners = find_corners(bgr)
            lo, hi_x, hi_y = (
                -self.CORNER_OOB_TOLERANCE,
                W + self.CORNER_OOB_TOLERANCE,
                H + self.CORNER_OOB_TOLERANCE,
            )
            in_bounds = bool(np.all(
                (corners[:, 0] >= lo) & (corners[:, 0] <= hi_x)
                & (corners[:, 1] >= lo) & (corners[:, 1] <= hi_y)
            ))
            if not in_bounds:
                raise ChessboardNotLocatedException("corners OOB")
        except Exception:
            corners = np.array(
                [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                dtype=np.float32,
            )
        self._corner_cache[image_key] = corners
        return corners

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        bgr = cv2.imread(row["image_path_abs"])
        corners = self._get_corners(row["image_key"], bgr)
        warped = warp_chessboard_image(bgr, corners)
        crop_bgr = crop_square(warped, int(row["board_row"]), int(row["board_col"]))
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            crop_rgb = self.transform(crop_rgb)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(crop_rgb))
                 .permute(2, 0, 1).float() / 255.0
        )
        return tensor, int(row["label"])


# Build real_val_dataset — checkpoint-selection signal
real_val_dataset = RealValDataset(
    csv_path=REAL_VAL_CSV,
    real_data_root=REAL_DATA_ROOT,
    transform=None,
)
n_val_frames = real_val_dataset.manifest["image_key"].nunique()
print(f"real_val_dataset: {len(real_val_dataset):,} samples "
      f"({n_val_frames} frames × 64 squares)")
print("  class distribution:")
for cls in range(NUM_CLASSES):
    n = int((real_val_dataset.manifest["label"] == cls).sum())
    print(f"    {cls:>2d} {CLASS_NAMES[cls]:<14s}: {n}")

# Build real_test_dataset — game7, held-out test set
real_test_dataset = RealGameDataset(
    gt_csv_path=TEST_GT_CSV,
    images_dir=TEST_IMAGES_DIR,
    game_name=f"game{TEST_GAME}",
    transform=None,
)
n_test_frames = real_test_dataset.manifest["image_name"].nunique()
print(f"\nreal_test_dataset (game{TEST_GAME}): {len(real_test_dataset):,} samples "
      f"({n_test_frames} frames × 64 squares)")
print("  class distribution:")
for cls in range(NUM_CLASSES):
    n = int((real_test_dataset.manifest["label"] == cls).sum())
    print(f"    {cls:>2d} {CLASS_NAMES[cls]:<14s}: {n}")

print("\033[92m✓ Cell 5 — RealValDataset + real_test_dataset — OK\033[0m")




# %% [Cell 6 — Weighted sampler (sqrt-inverse-frequency) on TRAIN labels only]
train_labels = train_dataset.manifest["label"].values
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
print("Train-set class counts and sqrt-inverse-frequency weights:")
print(f"  {'cls':>3s}  {'name':<14s}  {'count':>8s}  {'weight':>10s}  "
      f"{'eff_prob':>9s}")
class_weights = np.zeros(NUM_CLASSES, dtype=np.float64)
for cls in range(NUM_CLASSES):
    if class_counts[cls] > 0:
        class_weights[cls] = 1.0 / np.sqrt(class_counts[cls])
sample_weights = class_weights[train_labels]
sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
total_w = sample_weights.sum()
for cls in range(NUM_CLASSES):
    mask = (train_labels == cls)
    cls_w = sample_weights[mask].sum()
    print(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  "
          f"{class_counts[cls]:>8d}  {class_weights[cls]:>10.6f}  "
          f"{cls_w/total_w:>8.3%}")

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

print("\033[92m✓ Cell 6 — Weighted sampler — OK\033[0m")




# %% [Cell 7 — DataLoaders]
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
)
real_val_loader = DataLoader(
    real_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True,
)
real_test_loader = DataLoader(
    real_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True,
)

for name, loader in [("train", train_loader),
                     ("real_val", real_val_loader),
                     ("real_test", real_test_loader)]:
    xb, yb = next(iter(loader))
    label_dist = np.bincount(yb.numpy(), minlength=NUM_CLASSES)
    nonzero = [(c, int(n)) for c, n in enumerate(label_dist) if n > 0]
    print(f"{name:>10s}: x={tuple(xb.shape)} {xb.dtype}  y={tuple(yb.shape)} {yb.dtype}  "
          f"labels in batch: {nonzero}")

print("\033[92m✓ Cell 7 — DataLoaders — OK\033[0m")




# %% [Cell 8 — Build model]
def build_model():
    """Fresh ResNet18 with ImageNet weights, FC swapped to 13 classes."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m.to(DEVICE)


model = build_model()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: ResNet18 (ImageNet pretrained, FC swapped to {NUM_CLASSES} outputs)")
print(f"Trainable params: {n_params/1e6:.2f}M")

print("\033[92m✓ Cell 8 — Build model — OK\033[0m")




# %% [Cell 9 — Loss, optimizer, scheduler]
def build_optimizer(model):
    opt = torch.optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    return opt, sched


criterion = nn.CrossEntropyLoss()
optimizer, scheduler = build_optimizer(model)
print(f"Loss: CrossEntropyLoss")
print(f"Optim: SGD(lr={LR}, momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY})")
print(f"Sched: StepLR(step_size={LR_STEP_SIZE}, gamma={LR_GAMMA})")

print("\033[92m✓ Cell 9 — Loss, optimizer, scheduler — OK\033[0m")




# %% [Cell 10 — Helper functions]
IMAGENET_MEAN_DEV = IMAGENET_MEAN.to(DEVICE)
IMAGENET_STD_DEV = IMAGENET_STD.to(DEVICE)


def imagenet_normalize(x):
    return (x - IMAGENET_MEAN_DEV) / IMAGENET_STD_DEV


def train_one_epoch(model, loader, criterion, optimizer, device, print_every=50):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    t0 = time.perf_counter()
    for i, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb = imagenet_normalize(xb)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_count += bs
        if i % print_every == 0:
            dt = time.perf_counter() - t0
            print(f"    batch {i:5d}/{len(loader)}  "
                  f"loss={total_loss/total_count:.4f}  "
                  f"acc={total_correct/total_count:.4f}  "
                  f"({dt:.0f}s, {dt/i*1000:.0f}ms/batch)")
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device, name="val"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds = []
    all_labels = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb = imagenet_normalize(xb)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = yb.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(1)
        total_correct += (preds == yb).sum().item()
        total_count += bs
        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())
    return (
        total_loss / total_count,
        total_correct / total_count,
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


print("\033[92m✓ Cell 10 — Helper functions — OK\033[0m")




# %% [Cell 11 — Smoke test (MUST PASS before Cell 12)]
print("=" * 64)
print("SMOKE TEST — surfacing bugs before any training time is spent.")
print("=" * 64)

# 1. Shape/dtype assertions on each DataLoader (one batch each)
print("\n[1] DataLoader shape/dtype/range checks:")
for name, loader in [("train", train_loader),
                     ("real_val", real_val_loader),
                     ("real_test", real_test_loader)]:
    xb, yb = next(iter(loader))
    label_dist = np.bincount(yb.numpy(), minlength=NUM_CLASSES)
    print(f"  {name:>10s}: x={tuple(xb.shape)} {xb.dtype} "
          f"range=[{xb.min():.3f},{xb.max():.3f}]  "
          f"y={tuple(yb.shape)} {yb.dtype} "
          f"range=[{yb.min().item()},{yb.max().item()}]")
    assert xb.shape[1:] == (3, 100, 100), f"{name} bad image shape {xb.shape}"
    assert xb.dtype == torch.float32, f"{name} bad image dtype {xb.dtype}"
    assert yb.dtype == torch.int64, f"{name} bad label dtype {yb.dtype}"
    assert int(yb.min()) >= 0 and int(yb.max()) <= 12, f"{name} labels out of range"
    assert torch.isfinite(xb).all(), f"{name} non-finite values in images"

# 1b. Augmentation actually fires: same sample read twice should differ.
print("\n[1b] Augmentation-firing check (same sample, two reads):")
idx = 0
s1, _ = train_dataset[idx]
s2, _ = train_dataset[idx]
if hasattr(s1, "numpy"):
    arr1 = s1.float().numpy()
    arr2 = s2.float().numpy()
else:
    arr1 = s1.astype(np.float32)
    arr2 = s2.astype(np.float32)
same_sample_diff = float(np.abs(arr1 - arr2).mean())
print(f"  mean |sample_1 - sample_2| reading idx={idx} twice: {same_sample_diff:.4f}")
assert same_sample_diff > 0.01, (
    f"reading the same training sample twice produced near-identical pixels "
    f"(mean abs diff {same_sample_diff:.4f}) — augmentations appear not to be "
    f"firing in train_transform"
)

# 2. One forward pass on a real batch
print("\n[2] One forward pass on a train batch:")
xb, yb = next(iter(train_loader))
xb = xb.to(DEVICE)
yb = yb.to(DEVICE)
logits = model(imagenet_normalize(xb))
assert logits.shape == (xb.size(0), NUM_CLASSES), f"bad logit shape {logits.shape}"
assert torch.isfinite(logits).all(), "non-finite logits"
print(f"  logits shape: {tuple(logits.shape)}  "
      f"mean={logits.mean().item():+.3f}  std={logits.std().item():.3f}")

# 3. One training step
print("\n[3] One backward+step on the same batch:")
loss = criterion(logits, yb)
loss.backward()
optimizer.step()
optimizer.zero_grad()
assert torch.isfinite(loss).item() and loss.item() > 0, f"bad loss {loss.item()}"
print(f"  loss = {loss.item():.4f}")

# 4. Eval on 3 batches of real_val (untrained — noise not quality)
print("\n[4] Eval on 3 batches of real_val:")
model.eval()
correct = 0; total = 0
all_preds_smoke = []
with torch.no_grad():
    for i, (xb, yb) in enumerate(real_val_loader):
        if i >= 3:
            break
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        logits = model(imagenet_normalize(xb))
        assert torch.isfinite(logits).all(), "non-finite logits on real_val"
        preds = logits.argmax(1)
        assert preds.min().item() >= 0 and preds.max().item() < NUM_CLASSES, \
            f"preds out of range: [{preds.min().item()}, {preds.max().item()}]"
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        all_preds_smoke.append(preds.cpu())
unique_preds = int(torch.cat(all_preds_smoke).unique().numel())
print(f"  acc = {correct/total:.4f}  (random baseline ≈ {1/NUM_CLASSES:.4f}; "
      f"untrained — accuracy here is noise, not a quality signal)")
print(f"  distinct predicted classes across {total} samples: {unique_preds}")

# 4b. Loss sanity check on real_val: should be near ln(NUM_CLASSES) at random init.
print("\n[4b] Eval on 3 batches of real_val (loss sanity check):")
model.eval()
total_loss = 0.0; total = 0
with torch.no_grad():
    for i, (xb, yb) in enumerate(real_val_loader):
        if i >= 3:
            break
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        logits = model(imagenet_normalize(xb))
        total_loss += criterion(logits, yb).item() * yb.size(0)
        total += yb.size(0)
avg_loss = total_loss / total
print(f"  avg loss = {avg_loss:.4f}  (expected ≈ ln({NUM_CLASSES}) = {math.log(NUM_CLASSES):.4f})")
assert 1.5 < avg_loss < 5.0, f"loss {avg_loss:.4f} outside sane range for random init"

# 5. Eval on 3 batches of real_test (game7 — RealGameDataset path).
print("\n[5] Eval on 3 batches of real_test (game7):")
model.eval()
correct = 0; total = 0
with torch.no_grad():
    for i, (xb, yb) in enumerate(real_test_loader):
        if i >= 3:
            break
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        assert xb.shape[1:] == (3, 100, 100), f"real_test bad shape {xb.shape}"
        logits = model(imagenet_normalize(xb))
        assert torch.isfinite(logits).all(), "non-finite logits on real_test"
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
print(f"  acc = {correct/total:.4f}")

# 6. Reset model + optimizer for clean training start
print("\n[6] Resetting model and optimizer for the real training run.")
model = build_model()
optimizer, scheduler = build_optimizer(model)

print("\n" + "=" * 64)
print("Smoke test passed. Ready for training.")
print("=" * 64)

print("\033[92m✓ Cell 11 — Smoke test — OK\033[0m")




# %% [Cell 12 — Training loop]
# REAL-ONLY TRAINING: no synthetic data anywhere in the pipeline.
# real_val (30 frames from games 8-11) is the selection signal.
# game7 is the held-out test set, evaluated once at the end (Cell 15).
training_log = []
best_real_val_acc = -1.0
best_epoch = -1

CKPT_BEST = f"{RESULTS_DIR}/best_real.pt"
CKPT_LATEST = f"{RESULTS_DIR}/latest.pt"
LOG_CSV = f"{RESULTS_DIR}/training_log.csv"

t_total = time.perf_counter()
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n{'='*64}")
    print(f"Epoch {epoch}/{NUM_EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.6f})")
    print(f"{'='*64}")
    t_ep = time.perf_counter()

    # Train
    print("  [train]")
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, print_every=200)

    # Eval real val — SELECTION SIGNAL
    print("  [eval real_val (selection signal)]")
    rv_loss, rv_acc, _, _ = evaluate(model, real_val_loader, criterion, DEVICE, "real_val")

    scheduler.step()
    dt = time.perf_counter() - t_ep

    print(f"\n  Epoch {epoch:2d}: "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  |  "
          f"real_val_loss={rv_loss:.4f} real_val_acc={rv_acc:.4f}  "
          f"|  {dt/60:.1f}min")

    training_log.append({
        "epoch": epoch,
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": train_loss, "train_acc": train_acc,
        "real_val_loss": rv_loss, "real_val_acc": rv_acc,
        "epoch_time_s": dt,
    })

    # Save log every epoch
    pd.DataFrame(training_log).to_csv(LOG_CSV, index=False)

    # Save latest
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "real_val_acc": rv_acc,
    }, CKPT_LATEST)

    # Save best by REAL val acc (the selection signal)
    if rv_acc > best_real_val_acc:
        best_real_val_acc = rv_acc
        best_epoch = epoch
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "real_val_acc": rv_acc,
        }, CKPT_BEST)
        print(f"  → NEW BEST real_val_acc={rv_acc:.4f} → saved {CKPT_BEST}")

total_train_time = time.perf_counter() - t_total
print(f"\nTraining done. Total time: {total_train_time/60:.1f} min")
print(f"Best real_val_acc: {best_real_val_acc:.4f} at epoch {best_epoch}")

print("\033[92m✓ Cell 12 — Training loop — OK\033[0m")




# %% [Cell 13 — Training curves]
LOG_CSV = f"{RESULTS_DIR}/training_log.csv"

if not Path(LOG_CSV).exists():
    raise FileNotFoundError(
        f"No training log at {LOG_CSV}. "
        "Run Cell 12 (Training loop) at least one epoch first."
    )

log_df = pd.read_csv(LOG_CSV)
if len(log_df) == 0:
    raise RuntimeError(
        f"{LOG_CSV} exists but is empty. Cell 12 must have crashed before "
        "the first epoch finished — re-run it."
    )

best_epoch = int(log_df.loc[log_df["real_val_acc"].idxmax(), "epoch"])
best_real_val_acc = float(log_df["real_val_acc"].max())
print(f"Loaded {len(log_df)} epochs from {LOG_CSV}")
print(f"Best real_val_acc = {best_real_val_acc:.4f} at epoch {best_epoch}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(log_df["epoch"], log_df["train_acc"], label="train")
ax1.plot(log_df["epoch"], log_df["real_val_acc"], label="real val (selection)")
ax1.axvline(best_epoch, color="k", linestyle=":", alpha=0.4,
            label=f"best epoch ({best_epoch})")
ax1.set_xlabel("epoch")
ax1.set_ylabel("accuracy")
ax1.set_title("Accuracy")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(log_df["epoch"], log_df["train_loss"], label="train")
ax2.plot(log_df["epoch"], log_df["real_val_loss"], label="real val")
ax2.set_xlabel("epoch")
ax2.set_ylabel("loss")
ax2.set_title("Loss")
ax2.legend()
ax2.grid(alpha=0.3)

curves_path = f"{PLOTS_DIR}/training_curves.png"
plt.tight_layout()
plt.savefig(curves_path, dpi=120)
plt.close()
print(f"wrote {curves_path}")

print("\033[92m✓ Cell 13 — Training curves — OK\033[0m")




# %% [Cell 14 — Load best checkpoint]
CKPT_BEST = f"{RESULTS_DIR}/best_real.pt"

if not Path(CKPT_BEST).exists():
    raise FileNotFoundError(
        f"No best checkpoint at {CKPT_BEST}. "
        "Run Cell 12 (Training loop) for at least one epoch first."
    )

ckpt = torch.load(CKPT_BEST, map_location=DEVICE, weights_only=False)
print(f"Best checkpoint: epoch {ckpt['epoch']}, "
      f"real_val_acc={ckpt['real_val_acc']:.4f}")
model = build_model()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("\033[92m✓ Cell 14 — Load best checkpoint — OK\033[0m")




# %% [Cell 15 — Evaluate on game7 (held-out test set)]
print("Evaluating on real_test_loader (game7) ...")
rt_loss, rt_acc, rt_preds, rt_labels = evaluate(
    model, real_test_loader, criterion, DEVICE, "real_test")
print(f"  per-square acc = {rt_acc:.4f}  (loss {rt_loss:.4f})")

real_test_manifest = real_test_dataset.manifest.copy()
real_test_manifest["pred"] = rt_preds
assert len(real_test_manifest) == len(rt_preds), \
    f"prediction length mismatch: {len(rt_preds)} vs {len(real_test_manifest)}"

# Per-board accuracy
per_board = (
    real_test_manifest.assign(correct=lambda d: d["pred"] == d["label"])
    .groupby("image_name")["correct"]
    .agg(["sum", "count"])
)
per_board["all_correct"] = per_board["sum"] == per_board["count"]
n_boards = len(per_board)
n_all_correct = int(per_board["all_correct"].sum())
mean_squares_correct = per_board["sum"].mean()
print(f"  per-board acc (all 64 correct): {n_all_correct}/{n_boards} = "
      f"{n_all_correct/n_boards:.4f}")
print(f"  mean squares correct per board: {mean_squares_correct:.1f}/64")

real_board_lines = [
    f"n_boards: {n_boards}",
    f"boards with all 64 correct: {n_all_correct}/{n_boards} = {n_all_correct/n_boards:.4f}",
    f"mean squares correct / board: {mean_squares_correct:.2f} / 64",
    "",
    "per-board breakdown (sorted by accuracy):",
    f"  {'image':<30s}  correct/total",
]
sorted_boards = per_board.sort_values("sum", ascending=False)
for img, row in sorted_boards.iterrows():
    real_board_lines.append(f"  {img:<30s}  {int(row['sum'])}/{int(row['count'])}")
Path(f"{RESULTS_DIR}/real_test_per_board_accuracy.txt").write_text(
    "\n".join(real_board_lines))

# Per-class accuracy on real test
real_class_lines = ["class  name              n      acc"]
for cls in range(NUM_CLASSES):
    mask = (rt_labels == cls)
    n = int(mask.sum())
    if n:
        acc = float((rt_preds[mask] == cls).mean())
        real_class_lines.append(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}  {acc:.4f}")
    else:
        real_class_lines.append(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}    n/a")
real_class_report = "\n".join(real_class_lines)
print(real_class_report)
Path(f"{RESULTS_DIR}/real_test_per_class.txt").write_text(real_class_report)

# Confusion matrix on real test
cm_r = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
for t, p in zip(rt_labels, rt_preds):
    cm_r[t, p] += 1
cm_r_norm = cm_r / np.maximum(cm_r.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm_r_norm, cmap="Reds", vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)], rotation=45, ha="right")
ax.set_yticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)])
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Game7 (real test) confusion — per-square acc={rt_acc:.4f}")
plt.colorbar(im)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if cm_r[i, j] > 0:
            ax.text(j, i, str(cm_r[i, j]), ha="center", va="center",
                    fontsize=7, color="black" if cm_r_norm[i, j] < 0.5 else "white")
plt.tight_layout()
real_cm_path = f"{PLOTS_DIR}/real_test_confusion.png"
plt.savefig(real_cm_path, dpi=120)
plt.close()
print(f"wrote {real_cm_path}")

# Top-5 confusion pairs
pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm_r[i, j] > 0:
            pairs.append((int(cm_r[i, j]), i, j))
pairs.sort(reverse=True)
print("\nTop-5 real-image confusion pairs (true → pred, count):")
top5_lines = []
for n, t, p in pairs[:5]:
    line = f"  {CLASS_NAMES[t]:<14s} → {CLASS_NAMES[p]:<14s}  {n}"
    print(line)
    top5_lines.append(line)

print("\033[92m✓ Cell 15 — Evaluate on game7 — OK\033[0m")




# %% [Cell 16 — Summary.md]
try:
    total_train_time
except NameError:
    total_train_time = float(log_df["epoch_time_s"].sum())
    print(f"(reconstructed total_train_time from log: {total_train_time/60:.1f} min "
          f"over {len(log_df)} epochs)")

n_epochs_ran = len(log_df)
final_train_acc = float(log_df["train_acc"].iloc[-1])
final_real_val_acc = float(log_df["real_val_acc"].iloc[-1])

summary_lines = [
    "# Real-only training — results",
    "",
    "Training data: real images from games 2, 4, 5, 6 only — no synthetic data.",
    f"Validation (selection signal): {n_val_frames} frames from real_labels.csv (games 8-11).",
    f"Test (held out, evaluated once): game{TEST_GAME}.",
    "",
    "## Training",
    f"- Total training time: **{total_train_time/60:.1f} min** ({n_epochs_ran} epochs, "
    f"{len(train_dataset):,} real train squares).",
    f"- Best real_val_acc: **{best_real_val_acc:.4f}** at epoch **{best_epoch}**.",
    f"- Final-epoch train_acc: {final_train_acc:.4f}",
    f"- Final-epoch real_val_acc: {final_real_val_acc:.4f}",
    "",
    f"## Real test — game{TEST_GAME} ({n_boards} frames, {len(real_test_dataset)} squares)",
    f"- **Per-square accuracy: {rt_acc:.4f}**",
    f"- **Per-board accuracy (all 64 correct): {n_all_correct}/{n_boards} = "
    f"{n_all_correct/n_boards:.4f}**",
    f"- Mean squares correct / board: **{mean_squares_correct:.2f} / 64**",
    "",
    "### Per-class on real test",
    "```",
    real_class_report,
    "```",
    "",
    "### Top-5 game7 confusion pairs",
    "```",
    "\n".join(top5_lines) if top5_lines else "(none)",
    "```",
    "",
    "## Comparison context (zero-shot reference, different model)",
    "These numbers come from models trained only on synthetic data — not directly",
    "comparable architecturally (same ResNet18 + ImageNet head, different training",
    f"data), but a useful reference point for game{TEST_GAME} per-square accuracy.",
    "",
    "| Model                        | per-square | per-board | mean correct/64 |",
    "|------------------------------|-----------:|----------:|----------------:|",
    "| Zero-shot baseline (synth)   |     0.5670 |      0/55 |           36.29 |",
    "| Zero-shot Wölflein (synth)   |     0.5213 |      0/55 |           33.36 |",
    f"| **Real-only (this run)**     | **{rt_acc:.4f}** | "
    f"**{n_all_correct}/{n_boards}** | **{mean_squares_correct:.2f}** |",
    "",
    "## Artifacts",
    f"- `results/training_log.csv`",
    f"- `results/best_real.pt`  (epoch {best_epoch}, real_val_acc={best_real_val_acc:.4f})",
    f"- `results/latest.pt`",
    f"- `results/real_test_per_class.txt`",
    f"- `results/real_test_per_board_accuracy.txt`",
    f"- `plots/training_curves.png`",
    f"- `plots/real_test_confusion.png`",
]
summary_text = "\n".join(summary_lines)
Path(f"{RESULTS_DIR}/summary.md").write_text(summary_text)
print(summary_text)
print(f"\nwrote {RESULTS_DIR}/summary.md")

print("\033[92m✓ Cell 16 — Summary.md — OK\033[0m")

# %%

# %% [Cell 23 — check corner detection + cropping on a real test sample]
ds = real_test_dataset
crop_tensor, label = ds[0]  # some non-empty square
import matplotlib.pyplot as plt
crop_np = crop_tensor.permute(1,2,0).numpy()
plt.figure(figsize=(6, 6))
plt.imshow(crop_np)
plt.title(f"label={label} ({CLASS_NAMES[label]}), shape={crop_np.shape}")
plt.axhline(50, color='r')
plt.axvline(50, color='r')
plt.savefig("/home/eladbaum/chess_project/training/resnet18/Real_Only/plots/crop_geometry_check.png", dpi=120)
plt.close()
print("saved to /home/eladbaum/chess_project/training/resnet18/Real_Only/plots/crop_geometry_check.png")
# %%

# %% [Cell 24 — check corner detection + cropping on a real test sample]
import matplotlib.pyplot as plt
import numpy as np

ds = real_test_dataset

# Find one example of each class we have on the board
indices_to_show = []
for cls in [0, 1, 2, 3, 4, 5, 12]:  # WP, WR, WN, WB, WQ, WK, Empty
    cls_indices = ds.manifest[ds.manifest["label"] == cls].index.tolist()
    if cls_indices:
        indices_to_show.append((cls, cls_indices[0]))

fig, axes = plt.subplots(1, len(indices_to_show), figsize=(3*len(indices_to_show), 4))
for ax, (cls, idx) in zip(axes, indices_to_show):
    crop_tensor, label = ds[idx]
    crop_np = crop_tensor.permute(1, 2, 0).numpy()
    ax.imshow(crop_np)
    ax.set_title(f"{CLASS_NAMES[label]}\n(row={ds.manifest.iloc[idx]['board_row']}, col={ds.manifest.iloc[idx]['board_col']})")
    # crosshair at (50, 50) — image center
    ax.axhline(50, color='r', alpha=0.7, linewidth=1)
    ax.axvline(50, color='r', alpha=0.7, linewidth=1)
    # crosshair at (75, 75) — possible bottom-left anchor
    ax.axhline(75, color='cyan', alpha=0.5, linewidth=1, linestyle='--')
    ax.axvline(75, color='cyan', alpha=0.5, linewidth=1, linestyle='--')
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)  # flip y so origin is top-left like image coords

plt.suptitle("Crop geometry check — red = (50,50), cyan dashed = (75,75)", y=1.02)
plt.tight_layout()
out_path = "/home/eladbaum/chess_project/training/resnet18/Real_Only/plots/crop_geometry_check.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"saved to {out_path}")
# %%
