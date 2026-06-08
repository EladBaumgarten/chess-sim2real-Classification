"""
Project 2 — Stage 4: FROM-SCRATCH COMBINED training (ImageNet → synth + 30 real).

# NOT FINE-TUNING. Stage 4 trains ResNet18 from ImageNet pretrained
# weights on a COMBINED dataset: dataset_v1 synth squares + 30
# manual-label real frames (games 8-11). 50/50 batch balancing via
# WeightedRandomSampler. game7 is the monitor and gates checkpoint
# selection. Test: games 2/4/5/6 (same partition as stages 1, 2).

Single-phase recipe (no Wölflein phase A/B):
  All params trainable from epoch 1. SGD(lr=1e-4, momentum=0.9, wd=1e-4).
  No scheduler.
Augmentation: same as stages 1/2/3 (color jitter @0.7, shear @0.8 ±8°,
noise @0.5 std=0.015), applied to BOTH synth and real samples.
Sampler: WeightedRandomSampler targeting 50% synth / 50% real per batch.
  num_samples per epoch = 100,000.
  One epoch covers ~25% of synth + ~50 cycles of 30 real frames.
Checkpoint selection: real_val_acc on game7. Early stop patience=8.

Stage 4 / 5 (Comparison A alongside stages 1, 2). ImageNet pretrained
start (NOT v1 baseline). Single experimental variable vs. stage 2:
training procedure (joint-combined vs sequential FT).

Test partition (identical to stages 1, 2):
   monitor: game7 (55 frames);
   held-out test: games 2/4/5/6 (462 frames / 29,568 squares).
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
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import ColorJitter, RandomAffine, InterpolationMode
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scripts.chess_dataset import ChessSquareDataset
from scripts.fen_to_grid import fen_to_label_grid
from scripts.verify_woelflein_crops import (
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




# %% [Cell 2 — Config]
PROJECT_ROOT = "/home/eladbaum/chess_project"

# NO PRETRAINED_CKPT. Stage 4 starts from ImageNet weights via torchvision
# (loaded in Cell 7). There is no on-disk checkpoint to load.

# === Real data ===
REAL_LABELS_CSV = f"{PROJECT_ROOT}/data/real_labels.csv"
REAL_IMAGES_ROOT = f"{PROJECT_ROOT}/data"
GAME7_DIR = f"{PROJECT_ROOT}/data/game7_per_frame/images"
GAME7_GT_CSV = f"{PROJECT_ROOT}/data/game7_per_frame/gt.csv"
HELD_OUT_GAMES = [2, 4, 5, 6]     # SAME as stages 1/2 — direct comparison

# === Synth training data (v1; full manifest for joint training) ===
SYNTH_DATASET_DIR = f"{PROJECT_ROOT}/data/dataset_v1/images"
SYNTH_MANIFEST_PATH = f"{PROJECT_ROOT}/scripts/manifest.csv"
SYNTH_CORNERS_PATH = f"{PROJECT_ROOT}/scripts/corners.json"

# === Experiment dirs ===
EXP_DIR = f"{PROJECT_ROOT}/stage4_combined_30"
CHECKPOINTS_DIR = f"{EXP_DIR}/checkpoints"
RESULTS_DIR = f"{EXP_DIR}/results"
PLOTS_DIR = f"{EXP_DIR}/plots"
PREDS_DIR = f"{RESULTS_DIR}/predictions"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

# === Hyperparameters (single-phase) ===
BATCH_SIZE = 64
NUM_EPOCHS = 30
EARLY_STOP_PATIENCE = 8           # on real_val_acc
LR = 1e-4                         # single LR for all params
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
# NO scheduler. NO phase split. NO LR_STEP_SIZE / LR_GAMMA / PHASE_A_EPOCHS.

# === Joint-training balance ===
SYNTH_TRAINING_FRAC = 1.0         # use the FULL synth manifest for training
SYNTH_BATCH_FRAC = 0.5            # target fraction of synth samples per batch
NUM_SAMPLES_PER_EPOCH = 100_000   # WeightedRandomSampler draws per epoch

# === Catastrophic-forgetting probe ===
SYNTH_MONITOR_FRAC = 0.05         # 5% slice of dataset_v1, same as stages 1/2/3

NUM_WORKERS_SYNTH = 4
NUM_WORKERS_REAL = 4

# === Augmentation (same as stages 1/2/3) ===
COLOR_JITTER_APPLY_PROB = 0.7
SHEAR_APPLY_PROB = 0.8
NOISE_APPLY_PROB = 0.5
NOISE_STD = 0.015

COLOR_JITTER = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)
AFFINE_SHEAR = RandomAffine(
    degrees=0,
    translate=(0.04, 0.04),
    scale=(0.95, 1.05),
    shear=(-8.0, 8.0, -8.0, 8.0),
    interpolation=InterpolationMode.BILINEAR,
    fill=0,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

NUM_CLASSES = 13
CLASS_NAMES = [
    "White Pawn", "White Rook", "White Knight", "White Bishop",
    "White Queen", "White King",
    "Black Pawn", "Black Rook", "Black Knight", "Black Bishop",
    "Black Queen", "Black King",
    "Empty",
]
CLASS_SHORT = [
    "wP", "wR", "wN", "wB", "wQ", "wK",
    "bP", "bR", "bN", "bB", "bQ", "bK",
    "empty",
]

print(f"real labels:        {REAL_LABELS_CSV}")
print(f"real images root:   {REAL_IMAGES_ROOT}")
print(f"game7 monitor:      {GAME7_DIR}")
print(f"held-out games:     {HELD_OUT_GAMES}")
print(f"synth dataset:      {SYNTH_DATASET_DIR}")
print(f"synth manifest:     {SYNTH_MANIFEST_PATH}")
print(f"synth corners:      {SYNTH_CORNERS_PATH}")
print(f"checkpoints:        {CHECKPOINTS_DIR}")
print(f"results:            {RESULTS_DIR}")
print(f"plots:              {PLOTS_DIR}")
print(f"NUM_SAMPLES/epoch:  {NUM_SAMPLES_PER_EPOCH:,}  "
      f"(WeightedRandomSampler @ {SYNTH_BATCH_FRAC:.0%} synth / "
      f"{1-SYNTH_BATCH_FRAC:.0%} real per batch)")
print(f"LR:                 {LR}  (single phase, no scheduler)")

print("\033[92m✓ Cell 2 — Config — OK\033[0m")




# %% [Cell 3 — Load manual labels (all 30 — real side of joint training)]
manual_df = pd.read_csv(REAL_LABELS_CSV)
print(f"Loaded {len(manual_df)} manual-label rows from {REAL_LABELS_CSV}")
print(f"\nPer-game count of manual labels:")
per_game_counts = manual_df["game"].value_counts().sort_index()
for game, cnt in per_game_counts.items():
    print(f"  {game}: {cnt}")

# Stage 4 uses ALL 30 manual labels (no subsetting) as the REAL side of
# the joint-training dataset. The SYNTH side is the full dataset_v1 manifest
# (built in Cell 4).
STAGE4_REAL_N = 30
picked_rows = []
for _, row in manual_df.sort_values(["game", "ply"]).reset_index(
        drop=True).iterrows():
    picked_rows.append({
        "game_num": int(row["game"].replace("game", "")),
        "game": row["game"],
        "image_name": row["image_path"],
        "fen": row["fen"],
        "view": row["view"],
        "label_source": "manual",
        "ply": int(row["ply"]),
    })

stage4_real_df = pd.DataFrame(picked_rows).reset_index(drop=True)
assert len(stage4_real_df) == STAGE4_REAL_N, (
    f"selected {len(stage4_real_df)} real images, expected {STAGE4_REAL_N} "
    f"(all manual labels). Check data/real_labels.csv row count."
)
games_covered = set(stage4_real_df["game"].unique())
assert games_covered == {"game8", "game9", "game10", "game11"}, (
    f"all 4 games must be represented; got {games_covered}"
)

stage4_real_path = f"{RESULTS_DIR}/stage4_real_manifest.csv"
stage4_real_df.to_csv(stage4_real_path, index=False)
print(f"\nSelected stage-4 REAL training manifest (n={len(stage4_real_df)}):")
print(stage4_real_df[["game", "ply", "image_name", "fen"]].to_string(index=False))
print(f"\nwrote {stage4_real_path}")
print(f"per-game picks: {stage4_real_df['game'].value_counts().sort_index().to_dict()}")

print("\033[92m✓ Cell 3 — Use all 30 manual labels — OK\033[0m")




# %% [Cell 4 — Datasets: ManualLabelsDataset + synth-monitor slice]
class ManualLabelsDataset(Dataset):
    """
    Real fine-tuning training set. Sourced from data/real_labels.csv —
    `image_name` is a path RELATIVE to data/ (e.g. c06/game8/images/frame_*.jpg).

    Per-image find_corners with OOB fallback + corner caching, mirroring
    RealGameDataset (Cell 5). One sample per (frame × board square).
    Game name (game8/game9/...) drives the orientation in fen_to_label_grid.
    """

    CORNER_OOB_TOLERANCE = 8  # px

    def __init__(self, manifest_df, images_root, transform=None):
        self.images_root = Path(images_root)
        self.transform = transform
        rows = []
        for _, r in manifest_df.iterrows():
            game_key = r["game"]  # "game8" / "game9" / "game10" / "game11"
            grid = fen_to_label_grid(r["fen"], game_key)
            for br in range(8):
                for bc in range(8):
                    rows.append({
                        "image_name": r["image_name"],
                        "game": game_key,
                        "board_row": br,
                        "board_col": bc,
                        "label": int(grid[br, bc]),
                        "fen": r["fen"],
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
        bgr = cv2.imread(str(self.images_root / image_name))
        if bgr is None:
            raise FileNotFoundError(f"cv2 could not read {self.images_root / image_name}")
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


def train_transform(crop_rgb_uint8):
    """HWC uint8 RGB → HWC uint8 RGB. Stronger than zero-shot:
    color jitter @0.7 → shear @0.8 (±8°) → noise @0.5 (std=0.015).
    """
    img = Image.fromarray(crop_rgb_uint8)
    if random.random() < COLOR_JITTER_APPLY_PROB:
        img = COLOR_JITTER(img)
    if random.random() < SHEAR_APPLY_PROB:
        img = AFFINE_SHEAR(img)
    x = np.array(img)
    if random.random() < NOISE_APPLY_PROB:
        noise = np.random.normal(0, NOISE_STD * 255.0, x.shape).astype(np.float32)
        x = np.clip(x.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return x


# --- Manual-labels portion of the joint training set (30 real frames).
manual_train_dataset = ManualLabelsDataset(
    stage4_real_df, REAL_IMAGES_ROOT, transform=train_transform,
)
print(f"manual_train_dataset (30 frames × 64): "
      f"{len(manual_train_dataset):,} samples (with train_transform aug)")

# --- 5% slice of dataset_v1 for catastrophic-forgetting probe (post-training).
synth_manifest = pd.read_csv(SYNTH_MANIFEST_PATH)
print(f"\nLoaded dataset_v1 manifest: {len(synth_manifest):,} rows, "
      f"{synth_manifest['source_image'].nunique():,} unique images")

unique_synth_imgs = sorted(synth_manifest["source_image"].unique())
slice_rng = random.Random(SEED)
slice_rng.shuffle(unique_synth_imgs)
n_slice = max(1, int(len(unique_synth_imgs) * SYNTH_MONITOR_FRAC))
slice_imgs = set(unique_synth_imgs[:n_slice])
synth_monitor_df = synth_manifest[
    synth_manifest["source_image"].isin(slice_imgs)
].reset_index(drop=True)
print(f"  5% slice (SEED={SEED}): {n_slice} images, {len(synth_monitor_df):,} squares")

synth_monitor_dataset = ChessSquareDataset(
    synth_monitor_df, SYNTH_CORNERS_PATH, dataset_dir=SYNTH_DATASET_DIR, transform=None,
)
print(f"synth_monitor_dataset: {len(synth_monitor_dataset):,} samples (no aug)")

# --- Stage 4: synth TRAINING dataset (FULL manifest, with aug).
print(f"\nBuilding FULL synth training dataset from manifest "
      f"({len(synth_manifest):,} rows)...")
synth_train_dataset = ChessSquareDataset(
    synth_manifest,        # FULL manifest, NOT the 5% slice
    SYNTH_CORNERS_PATH,
    dataset_dir=SYNTH_DATASET_DIR,
    transform=train_transform,
)
print(f"synth_train_dataset: {len(synth_train_dataset):,} samples "
      f"(with train_transform aug)")

print("\033[92m✓ Cell 4 — Datasets — OK\033[0m")




# %% [Cell 5 — RealGameDataset (game7 monitor + games 2/4/5/6 test) + train ConcatDataset]
class RealGameDataset(Dataset):
    """Per-frame × per-square dataset for one full game's gt.csv.
    Same logic as in zero_shot scripts; one corner_cache per game.
    """

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


real_val_dataset = RealGameDataset(
    GAME7_GT_CSV, GAME7_DIR, game_name="game7", transform=None,
)
print(f"real_val_dataset (game7): {len(real_val_dataset):,} samples "
      f"({real_val_dataset.manifest['image_name'].nunique()} frames × 64 squares)")

# --- Stage 4: train_dataset = synth + real (manual labels).
# We will use a WeightedRandomSampler in Cell 6 to balance these
# at 50/50 per batch despite their wildly different sizes.
train_dataset = ConcatDataset([synth_train_dataset, manual_train_dataset])
print(f"\ntrain_dataset (synth + manual concat): "
      f"{len(train_dataset):,} samples total "
      f"({len(synth_train_dataset):,} synth + "
      f"{len(manual_train_dataset):,} manual)")
print(f"  synth fraction: {len(synth_train_dataset)/len(train_dataset):.4f}")
print(f"  real  fraction: {len(manual_train_dataset)/len(train_dataset):.4f}")
print(f"  ⚠️  natural ratio is ~{len(synth_train_dataset)/len(manual_train_dataset):.0f}:1"
      f" synth:real. Sampler will compensate.")

print("\033[92m✓ Cell 5 — RealGameDataset (game7) + synth+manual concat — OK\033[0m")




# %% [Cell 6 — DataLoaders (WeightedRandomSampler — 50/50 synth/real per batch,
#     100k samples/epoch)]
def _worker_init_fn(worker_id):
    import random as _r
    worker_seed = SEED + worker_id
    _r.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# WeightedRandomSampler at 50/50 synth/real per batch.
# Weights are PER SAMPLE in the concat dataset's indexing.
# Synth occupies indices [0, len(synth_train_dataset))
# Real  occupies indices [len(synth_train_dataset), len(train_dataset))
n_synth = len(synth_train_dataset)
n_real = len(manual_train_dataset)
# If we want exactly synth_batch_frac of probability mass on synth:
w_per_synth = SYNTH_BATCH_FRAC / n_synth
w_per_real  = (1.0 - SYNTH_BATCH_FRAC) / n_real
sample_weights = (
    [w_per_synth] * n_synth + [w_per_real] * n_real
)
sample_weights = torch.tensor(sample_weights, dtype=torch.double)
assert abs(sample_weights.sum().item() - 1.0) < 1e-6, (
    f"sample weights don't sum to 1: {sample_weights.sum().item()}"
)
print(f"WeightedRandomSampler weights: "
      f"per-synth={w_per_synth:.3e}  per-real={w_per_real:.3e}")
print(f"  target batch composition: "
      f"~{SYNTH_BATCH_FRAC*BATCH_SIZE:.0f} synth + "
      f"~{(1-SYNTH_BATCH_FRAC)*BATCH_SIZE:.0f} real per batch of {BATCH_SIZE}")
print(f"  num_samples per epoch = {NUM_SAMPLES_PER_EPOCH:,}")
print(f"  effective batches per epoch ≈ "
      f"{NUM_SAMPLES_PER_EPOCH // BATCH_SIZE:,}")

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=NUM_SAMPLES_PER_EPOCH,
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,   # NOT shuffle=True; sampler replaces it
    num_workers=NUM_WORKERS_REAL,
    pin_memory=True,
    persistent_workers=True,
    worker_init_fn=_worker_init_fn,
    drop_last=False,
)
synth_monitor_loader = DataLoader(
    synth_monitor_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS_SYNTH, pin_memory=True, persistent_workers=True,
    worker_init_fn=_worker_init_fn,
)
real_val_loader = DataLoader(
    real_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS_REAL, pin_memory=True, persistent_workers=True,
    worker_init_fn=_worker_init_fn,
)

for name, loader in [("train", train_loader),
                     ("synth_monitor", synth_monitor_loader),
                     ("real_val", real_val_loader)]:
    xb, yb = next(iter(loader))
    label_dist = np.bincount(yb.numpy(), minlength=NUM_CLASSES)
    nonzero = [(c, int(n)) for c, n in enumerate(label_dist) if n > 0]
    print(f"{name:>14s}: x={tuple(xb.shape)} {xb.dtype}  y={tuple(yb.shape)} {yb.dtype}  "
          f"labels in batch: {nonzero}")

print("\033[92m✓ Cell 6 — DataLoaders — OK\033[0m")




# %% [Cell 7 — Build model from ImageNet pretrained weights]
def build_model():
    """ResNet18 with ImageNet pretrained weights, FC swapped to 13 classes.
    Stage 4 starts from ImageNet — NOT v1 baseline.
    """
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


model = build_model().to(DEVICE)

# Sanity: report num params, but no state_dict comparison since we are
# NOT loading a checkpoint.
n_total = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Built ResNet18 from ImageNet pretrained weights (FC → {NUM_CLASSES})")
print(f"  total params:     {n_total:,}")
print(f"  trainable params: {n_trainable:,} (all trainable from epoch 1)")

print("\033[92m✓ Cell 7 — Build model from ImageNet — OK\033[0m")




# %% [Cell 8 — Single-phase optimizer (all params trainable)]
# NOTE: stage 4 has NO phase A / phase B split. ImageNet weights are
# a good enough starting point for the body, and joint synth+real
# training is supposed to be a single coherent task.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
)
scheduler = None  # no scheduler
print(f"Optimizer: SGD(lr={LR}, momentum={MOMENTUM}, wd={WEIGHT_DECAY})")
print(f"All {sum(p.numel() for p in model.parameters()):,} params trainable.")

print("\033[92m✓ Cell 8 — Single-phase optimizer + loss — OK\033[0m")




# %% [Cell 9 — Helper functions]
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
            print(f"    batch {i:4d}/{len(loader)}  "
                  f"loss={total_loss/total_count:.4f}  "
                  f"acc={total_correct/total_count:.4f}  "
                  f"({dt:.0f}s)")
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


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
        total_loss / max(total_count, 1),
        total_correct / max(total_count, 1),
        np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64),
        np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64),
    )


def per_class_accuracy(preds, labels, num_classes=NUM_CLASSES):
    out = []
    for cls in range(num_classes):
        mask = (labels == cls)
        n = int(mask.sum())
        out.append(float((preds[mask] == cls).mean()) if n else float("nan"))
    return out


print("\033[92m✓ Cell 9 — Helper functions — OK\033[0m")




# %% [Cell 10 — Smoke test (MUST PASS before training)]
print("=" * 64)
print("SMOKE TEST — surfacing bugs before any GPU time is spent.")
print("=" * 64)

# [1] DataLoader shape/dtype/range
print("\n[1] DataLoader shape/dtype/range checks:")
for name, loader in [("train", train_loader),
                     ("synth_monitor", synth_monitor_loader),
                     ("real_val", real_val_loader)]:
    xb, yb = next(iter(loader))
    print(f"  {name:>14s}: x={tuple(xb.shape)} {xb.dtype} "
          f"range=[{xb.min():.3f},{xb.max():.3f}]  "
          f"y={tuple(yb.shape)} {yb.dtype} "
          f"range=[{yb.min().item()},{yb.max().item()}]")
    assert xb.shape[1:] == (3, 100, 100), f"{name} bad image shape {xb.shape}"
    assert xb.dtype == torch.float32, f"{name} bad image dtype {xb.dtype}"
    assert yb.dtype == torch.int64, f"{name} bad label dtype {yb.dtype}"
    assert int(yb.min()) >= 0 and int(yb.max()) <= 12, f"{name} labels OOR"
    assert torch.isfinite(xb).all(), f"{name} non-finite values"

# [1b] Augmentation fires (same sample, two reads from train_dataset).
# Index 0 lands in the synth half of the concat dataset (synth is first).
print("\n[1b] Augmentation-firing check (synth sample):")
idx = 0
s1, _ = train_dataset[idx]
s2, _ = train_dataset[idx]
arr1 = s1.float().numpy()
arr2 = s2.float().numpy()
same_sample_diff = float(np.abs(arr1 - arr2).mean())
print(f"  mean |s1 - s2| reading train_dataset[{idx}] twice: {same_sample_diff:.4f}")
assert same_sample_diff > 0.01, (
    f"synth augmentations not firing — mean abs diff {same_sample_diff:.4f}"
)

# [1b-extra] Confirm real (manual-label) samples are ALSO augmented.
# Sample a real index from the concat dataset.
real_start_idx = len(synth_train_dataset)
real_check_idx = real_start_idx + 5  # 6th real sample
s1_real, _ = train_dataset[real_check_idx]
s2_real, _ = train_dataset[real_check_idx]
real_diff = float(np.abs(s1_real.numpy() - s2_real.numpy()).mean())
print(f"  mean |s1 - s2| reading train_dataset[{real_check_idx}] (real sample) "
      f"twice: {real_diff:.4f}")
assert real_diff > 0.01, (
    f"real samples not augmented — mean abs diff {real_diff:.4f}. "
    f"Did ManualLabelsDataset receive transform=train_transform?"
)

# [1b-sampler] Confirm sampler produces ~50/50 synth/real per batch.
# We can't tell from xb/yb alone whether a sample is synth or real, so we
# sample 1000 indices directly from a fresh sampler with the same weights.
print("\n[1b-sampler] Verify WeightedRandomSampler balances batches:")
sample_indices = list(WeightedRandomSampler(
    weights=sample_weights, num_samples=1000, replacement=True
))
n_synth_drawn = sum(1 for i in sample_indices if i < len(synth_train_dataset))
n_real_drawn = 1000 - n_synth_drawn
synth_frac_drawn = n_synth_drawn / 1000.0
print(f"  drew 1000 indices: {n_synth_drawn} synth ({synth_frac_drawn:.2%}), "
      f"{n_real_drawn} real ({1-synth_frac_drawn:.2%})")
print(f"  target synth fraction: {SYNTH_BATCH_FRAC:.2%}")
assert abs(synth_frac_drawn - SYNTH_BATCH_FRAC) < 0.05, (
    f"sampler balance off: drew {synth_frac_drawn:.2%} synth, "
    f"expected {SYNTH_BATCH_FRAC:.2%} ± 5%"
)

# [1c] Aug visualization on 16 random concat samples (mix of synth + real).
print("\n[1c] Augmentation visualization (16 random crops → aug_smoke_check.png):")
np.random.seed(SEED + 1)
# With 30 frames there's no need to over-sample per-frame to fill 16 cells.
# Just pick 16 random sample indices across the full train_dataset.
viz_indices = list(np.random.choice(len(train_dataset), size=16, replace=False))

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, vi in zip(axes.ravel(), viz_indices):
    aug_tensor, lab = train_dataset[int(vi)]
    aug_hwc = (aug_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    ax.imshow(aug_hwc)
    ax.set_title(f"idx={int(vi)}  y={lab} {CLASS_SHORT[lab]}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle(
    f"Stage-4 augmented training crops (synth + real mix) "
    f"(jitter @{COLOR_JITTER_APPLY_PROB} → shear @{SHEAR_APPLY_PROB} → noise @{NOISE_APPLY_PROB})",
    fontsize=11,
)
plt.tight_layout()
aug_smoke_path = f"{PLOTS_DIR}/aug_smoke_check.png"
plt.savefig(aug_smoke_path, dpi=120)
plt.close()
print(f"  wrote {aug_smoke_path}")

# [1c-b] Sample plot of the 30 manual training boards with labels overlaid.
# This is the "user can sanity-check before training" plot. Synth boards
# are skipped here — there are ~6,100 of them.
print("\n[1c-b] Stage4 manual-frames sample plot (warped boards + labels):")
fig, axes = plt.subplots(5, 6, figsize=(24, 20))
for ax, (_, row) in zip(axes.ravel(), stage4_real_df.iterrows()):
    img_path = Path(REAL_IMAGES_ROOT) / row["image_name"]
    bgr = cv2.imread(str(img_path))
    H, W = bgr.shape[:2]
    try:
        np.random.seed(SEED)
        corners = find_corners(bgr)
        lo, hi_x, hi_y = -8, W + 8, H + 8
        if not bool(np.all(
            (corners[:, 0] >= lo) & (corners[:, 0] <= hi_x)
            & (corners[:, 1] >= lo) & (corners[:, 1] <= hi_y)
        )):
            raise ChessboardNotLocatedException("OOB")
        corner_status = "detected"
    except Exception:
        corners = np.array(
            [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        corner_status = "FALLBACK"
    warped = warp_chessboard_image(bgr, corners)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    ax.imshow(warped_rgb)
    # Overlay class short codes for non-empty squares.
    grid = fen_to_label_grid(row["fen"], row["game"])
    BOARD_OFFSET = 50
    Hw = warped_rgb.shape[0]
    SQ = (Hw - 2 * BOARD_OFFSET) // 8
    for br in range(8):
        for bc in range(8):
            lab = int(grid[br, bc])
            if lab == 12:
                continue
            x0 = BOARD_OFFSET + bc * SQ
            y0 = BOARD_OFFSET + br * SQ
            ax.text(
                x0 + SQ / 2, y0 + SQ / 2, CLASS_SHORT[lab],
                color="lime" if lab < 6 else "yellow",
                fontsize=8, ha="center", va="center", weight="bold",
            )
    ax.set_title(
        f"{row['game']} ply={row['ply']}  [{corner_status}]\n{Path(row['image_name']).name}",
        fontsize=8,
    )
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Stage4 training set (real side) — 30 manual-label frames with FEN labels overlaid", fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.96))
train_samples_path = f"{PLOTS_DIR}/stage4_real_samples.png"
plt.savefig(train_samples_path, dpi=120)
plt.close()
print(f"  wrote {train_samples_path}")

# Stage 4 has no [1d] pretrained-weights check — we start from ImageNet
# via torchvision (no on-disk checkpoint to compare against).

# [2] Pre-training synth-monitor + real_val check on the IMAGENET-initialised
# model. ImageNet weights are not trained on chess; pre-training acc should
# be roughly chance (~13% for 13-class). The forgetting Δ vs. stage 2 still
# benefits from tracking these numbers, but they're not meaningful headline
# numbers in stage 4.
print("\n[2] Pre-training eval (ImageNet init — expect ~chance accuracy):")
sm_loss_pre, sm_acc_pre, _, _ = evaluate(
    model, synth_monitor_loader, criterion, DEVICE, "synth_monitor")
rv_loss_pre, rv_acc_pre, _, _ = evaluate(
    model, real_val_loader, criterion, DEVICE, "real_val")
print(f"  synth_monitor (5% v1 slice, BEFORE train): acc={sm_acc_pre:.4f}  loss={sm_loss_pre:.4f}")
print(f"  real_val (game7,             BEFORE train): acc={rv_acc_pre:.4f}  loss={rv_loss_pre:.4f}")
# ImageNet weights are not trained on chess; pre-training acc should be
# roughly chance (~13% for 13-class). We don't assert tight bounds, just
# sanity-check it's finite and not 1.0.
assert 0.0 < sm_acc_pre < 0.5, (
    f"synth_monitor pre-training acc = {sm_acc_pre:.4f}; "
    f"expected ~0.13 (random chance for 13-class). "
    f"Is the model state correct?"
)
PRE_FT_SYNTH_MONITOR_ACC = sm_acc_pre
PRE_FT_REAL_VAL_ACC = rv_acc_pre

# [3] Forward pass on a train batch
print("\n[3] One forward pass on a train batch:")
xb, yb = next(iter(train_loader))
xb = xb.to(DEVICE); yb = yb.to(DEVICE)
logits = model(imagenet_normalize(xb))
assert logits.shape == (xb.size(0), NUM_CLASSES), f"bad logits {logits.shape}"
assert torch.isfinite(logits).all(), "non-finite logits"
print(f"  logits shape: {tuple(logits.shape)}  "
      f"mean={logits.mean().item():+.3f}  std={logits.std().item():.3f}")
init_loss = criterion(logits, yb)
print(f"  init train-batch loss: {init_loss.item():.4f}")

# [4] One backward+step on the SAME batch (warm up the optimizer state)
print("\n[4] One backward+step:")
init_loss.backward()
optimizer.step()
optimizer.zero_grad()

# [4c] Single-phase parameter check (all params trainable in stage 4).
all_trainable = all(p.requires_grad for p in model.parameters())
print(f"\n[4c] All params trainable: {all_trainable}")
assert all_trainable, "stage 4 has no freeze phase; all params must be trainable"

# [5] real_val loads 3 batches
print("\n[5] real_val (game7) loads 3 batches:")
model.eval()
n_loaded = 0
with torch.no_grad():
    for i, (xb, yb) in enumerate(real_val_loader):
        if i >= 3: break
        assert xb.shape[1:] == (3, 100, 100)
        assert torch.isfinite(xb).all()
        n_loaded += xb.size(0)
print(f"  loaded {n_loaded} samples ✓")

# [6] Real_val → model → finite logits
print("\n[6] real_val → model → finite logits:")
correct = 0; total = 0
with torch.no_grad():
    for i, (xb, yb) in enumerate(real_val_loader):
        if i >= 3: break
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        logits = model(imagenet_normalize(xb))
        assert torch.isfinite(logits).all()
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
print(f"  3-batch acc on real_val: {correct/total:.4f}")

# Loss sanity
print(f"\n[loss sanity] pre-train batch loss = {init_loss.item():.4f}  "
      f"(expected ≈ ln({NUM_CLASSES}) ≈ {math.log(NUM_CLASSES):.4f} — "
      f"ImageNet backbone but freshly-random FC layer)")

print("\n" + "=" * 64)
print("Smoke test passed. Ready for training.")
print("=" * 64)

print("\033[92m✓ Cell 10 — Smoke test — OK\033[0m")




# %% [Cell 11 — Training loop (single phase — joint synth+real)]
# Stage 4 starts from ImageNet pretrained weights built in Cell 7.
# No checkpoint reload needed; if the kernel restarts, re-run Cell 7
# to rebuild the ImageNet-pretrained model before Cell 11.
# (The smoke test's stage [4] did one optimizer.step() on the loaded
# model, so we rebuild here for a clean training entry.)
model = build_model().to(DEVICE)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
)
scheduler = None  # single-phase, no scheduler
print(f"[Cell 11 reset] rebuilt ImageNet model + fresh SGD(lr={LR})")

training_log = []
best_real_val_acc = -1.0
best_real_epoch = -1
best_synth_monitor_acc = -1.0
best_synth_monitor_epoch = -1
epochs_since_best_real = 0
stop_reason = "completed_all_epochs"

CKPT_BEST_REAL = f"{CHECKPOINTS_DIR}/best_real.pt"
CKPT_BEST_SYNTH_MONITOR = f"{CHECKPOINTS_DIR}/best_synth_monitor.pt"
CKPT_LATEST = f"{CHECKPOINTS_DIR}/latest.pt"
LOG_CSV = f"{RESULTS_DIR}/training_log.csv"

t_total = time.perf_counter()

for epoch in range(1, NUM_EPOCHS + 1):
    # Stage 4: single phase, all params trainable from epoch 1.
    # `phase="combined"` is logged in the CSV for cross-stage schema parity.
    phase = "combined"

    print(f"\n{'='*64}")
    print(f"Epoch {epoch}/{NUM_EPOCHS}  lr={optimizer.param_groups[0]['lr']:.6f}")
    print(f"{'='*64}")
    t_ep = time.perf_counter()

    print("  [train]")
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, print_every=200)

    print("  [eval synth_monitor (5% v1 slice — forgetting probe)]")
    sm_loss, sm_acc, _, _ = evaluate(
        model, synth_monitor_loader, criterion, DEVICE, "synth_monitor")

    print("  [eval real_val (game7 — checkpoint-selection signal)]")
    rv_loss, rv_acc, rv_preds_ep, rv_labels_ep = evaluate(
        model, real_val_loader, criterion, DEVICE, "real_val")
    rv_per_class = per_class_accuracy(rv_preds_ep, rv_labels_ep, NUM_CLASSES)

    if scheduler is not None:
        scheduler.step()
    dt = time.perf_counter() - t_ep

    print(f"\n  Epoch {epoch:2d}: "
          f"train={train_acc:.4f}  synth_mon={sm_acc:.4f}  "
          f"real_val={rv_acc:.4f}  ({dt:.1f}s)")
    knight_acc = rv_per_class[2]; bishop_acc = rv_per_class[3]; king_w = rv_per_class[5]
    bknight = rv_per_class[8]; bbishop = rv_per_class[9]; bking = rv_per_class[11]
    def _fmt(v): return f"{v:.3f}" if not math.isnan(v) else "n/a"
    print(f"    per-class real_val knights/bishops/kings: "
          f"wN={_fmt(knight_acc)} wB={_fmt(bishop_acc)} wK={_fmt(king_w)}  "
          f"bN={_fmt(bknight)} bB={_fmt(bbishop)} bK={_fmt(bking)}")

    log_row = {
        "epoch": epoch, "phase": phase,
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": train_loss, "train_acc": train_acc,
        "synth_val_loss": sm_loss, "synth_val_acc": sm_acc,   # synth_monitor logged under these column names for cross-stage compatibility
        "real_val_loss": rv_loss, "real_val_acc": rv_acc,
        "epoch_time_s": dt,
    }
    for short, acc in zip(CLASS_SHORT, rv_per_class):
        log_row[f"real_val_acc_{short}"] = acc
    training_log.append(log_row)
    pd.DataFrame(training_log).to_csv(LOG_CSV, index=False)

    # Save latest
    torch.save({
        "epoch": epoch, "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "synth_monitor_acc": sm_acc, "real_val_acc": rv_acc,
    }, CKPT_LATEST)

    # Save best by REAL_VAL_ACC (game7) — THE headline checkpoint
    if rv_acc > best_real_val_acc:
        best_real_val_acc = rv_acc
        best_real_epoch = epoch
        epochs_since_best_real = 0
        torch.save({
            "epoch": epoch, "phase": phase,
            "model_state_dict": model.state_dict(),
            "synth_monitor_acc": sm_acc, "real_val_acc": rv_acc,
        }, CKPT_BEST_REAL)
        print(f"  → NEW BEST real_val_acc={rv_acc:.4f} → saved {CKPT_BEST_REAL}")
    else:
        epochs_since_best_real += 1

    # Save best by synth_monitor_acc — monitor-only artifact
    if sm_acc > best_synth_monitor_acc:
        best_synth_monitor_acc = sm_acc
        best_synth_monitor_epoch = epoch
        torch.save({
            "epoch": epoch, "phase": phase,
            "model_state_dict": model.state_dict(),
            "synth_monitor_acc": sm_acc, "real_val_acc": rv_acc,
            "monitor_only": True,
        }, CKPT_BEST_SYNTH_MONITOR)

    # Early stopping
    if epochs_since_best_real >= EARLY_STOP_PATIENCE:
        stop_reason = (
            f"early_stop_patience_{EARLY_STOP_PATIENCE}_no_improve_real_val "
            f"(best={best_real_val_acc:.4f} at epoch {best_real_epoch})"
        )
        print(f"\n[early stop] real_val_acc has not improved in "
              f"{EARLY_STOP_PATIENCE} epochs. Stopping at epoch {epoch}.")
        break

total_train_time = time.perf_counter() - t_total
n_epochs_ran = len(training_log)
print(f"\nTraining done. {n_epochs_ran} epochs in {total_train_time/60:.1f} min")
print(f"Stop reason: {stop_reason}")
print(f"Best real_val_acc (game7):           {best_real_val_acc:.4f}  at epoch {best_real_epoch}")
print(f"Best synth_monitor_acc (5% v1 slice): {best_synth_monitor_acc:.4f}  at epoch {best_synth_monitor_epoch}")

print("\033[92m✓ Cell 11 — Training loop — OK\033[0m")




# %% [Cell 12 — Training curves + per-class real_val plot]
log_df = pd.read_csv(LOG_CSV)
n_epochs_ran = len(log_df)
print(f"Loaded {n_epochs_ran} epochs from {LOG_CSV}")

best_real_epoch = int(log_df.loc[log_df["real_val_acc"].idxmax(), "epoch"])
best_real_val_acc = float(log_df["real_val_acc"].max())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(log_df["epoch"], log_df["train_acc"], label="train", marker="o", ms=4)
ax1.plot(log_df["epoch"], log_df["synth_val_acc"], label="synth_monitor (5% v1)",
         marker="s", ms=4)
ax1.plot(log_df["epoch"], log_df["real_val_acc"], label="real_val (game7)",
         marker="^", ms=4, linestyle="--", linewidth=2)
ax1.axvline(best_real_epoch, color="k", linestyle=":", alpha=0.4,
            label=f"best real_val (ep{best_real_epoch})")
ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy")
ax1.set_title("Accuracy — Stage 4 joint synth+real, single phase")
ax1.set_ylim(-0.02, 1.02); ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

ax2.plot(log_df["epoch"], log_df["train_loss"], label="train", marker="o", ms=4)
ax2.plot(log_df["epoch"], log_df["synth_val_loss"], label="synth_monitor",
         marker="s", ms=4)
ax2.plot(log_df["epoch"], log_df["real_val_loss"], label="real_val (game7)",
         marker="^", ms=4, linestyle="--", linewidth=2)
ax2.set_xlabel("epoch"); ax2.set_ylabel("loss")
ax2.set_title("Loss — Stage 4 joint synth+real, single phase")
ax2.legend(); ax2.grid(alpha=0.3)

curves_path = f"{PLOTS_DIR}/training_curves.png"
plt.tight_layout()
plt.savefig(curves_path, dpi=120)
plt.close()
print(f"wrote {curves_path}")

# --- per_class_real_val.png ----------------------------------------------------
per_class_cols = [f"real_val_acc_{s}" for s in CLASS_SHORT]
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
flat_axes = axes.ravel()
for i, (col, name) in enumerate(zip(per_class_cols, CLASS_NAMES)):
    ax = flat_axes[i]
    if col in log_df.columns:
        ax.plot(log_df["epoch"], log_df[col], marker="o", linewidth=1.5)
    ax.set_title(f"{CLASS_SHORT[i]} — {name}", fontsize=10)
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    ax.axhline(0.0, color="r", linestyle=":", alpha=0.4)
for j in range(NUM_CLASSES, 16):
    flat_axes[j].axis("off")
fig.suptitle(
    "Per-class real_val (game7) accuracy over epochs "
    "(red dotted horiz = 0%; stage 4 is single-phase)",
    fontsize=12,
)
fig.text(0.5, 0.04, "epoch", ha="center")
fig.text(0.06, 0.5, "real_val accuracy", va="center", rotation="vertical")
plt.tight_layout(rect=(0.07, 0.05, 1.0, 0.97))
per_class_path = f"{PLOTS_DIR}/per_class_real_val.png"
plt.savefig(per_class_path, dpi=120)
plt.close()
print(f"wrote {per_class_path}")

print("\033[92m✓ Cell 12 — Training curves — OK\033[0m")




# %% [Cell 13 — Load best-real checkpoint for end-of-run evaluation]
if not Path(CKPT_BEST_REAL).exists():
    raise FileNotFoundError(f"missing checkpoint: {CKPT_BEST_REAL}")

best_ckpt = torch.load(CKPT_BEST_REAL, map_location=DEVICE, weights_only=False)
print(f"Best-real checkpoint: epoch {best_ckpt['epoch']} (phase {best_ckpt['phase']}), "
      f"real_val_acc={best_ckpt['real_val_acc']:.4f}, "
      f"synth_monitor_acc={best_ckpt['synth_monitor_acc']:.4f}")

model = build_model().to(DEVICE)
model.load_state_dict(best_ckpt["model_state_dict"])
model.eval()
print("\033[92m✓ Cell 13 — Load best-real checkpoint — OK\033[0m")




# %% [Cell 14 — CM helpers]
def plot_confusion_matrix(cm, title, save_path, cmap="Blues"):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)],
                       rotation=45, ha="right")
    ax.set_yticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    plt.colorbar(im)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="black" if cm_norm[i, j] < 0.5 else "white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def confusion_matrix_np(preds, labels, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def per_class_table_text(preds, labels, num_classes=NUM_CLASSES):
    lines = ["class  name              n      acc"]
    for cls in range(num_classes):
        mask = (labels == cls)
        n = int(mask.sum())
        if n:
            acc = float((preds[mask] == cls).mean())
            lines.append(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}  {acc:.4f}")
        else:
            lines.append(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}    n/a")
    return "\n".join(lines)


print("\033[92m✓ Cell 14 — CM helpers — OK\033[0m")




# %% [Cell 15 — Synth-monitor eval (catastrophic-forgetting probe)]
# Rebuild synth_monitor_loader before the post-training eval. The
# persistent workers built in Cell 6 stay alive across the whole training
# run, and on this cluster the worker pool occasionally goes stale after
# many hours (queue.Empty / "worker exited unexpectedly"). Replacing the
# loader with num_workers=0 is a one-time ~30s slowdown and avoids the
# stale-pool issue. Same pattern applied to real_val_loader for Cell 16.
print("[Cell 15] Rebuilding eval loaders with num_workers=0 to avoid "
      "stale worker pools after long training run...")
del synth_monitor_loader, real_val_loader
synth_monitor_loader = DataLoader(
    synth_monitor_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True,
)
real_val_loader = DataLoader(
    real_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True,
)

print("Evaluating BEST_REAL checkpoint on synth_monitor (5% slice of dataset_v1) ...")
sm_loss, sm_acc, sm_preds, sm_labels = evaluate(
    model, synth_monitor_loader, criterion, DEVICE, "synth_monitor")
print(f"  synth_monitor acc AFTER FT:  {sm_acc:.4f}  (loss {sm_loss:.4f})")
print(f"  synth_monitor acc BEFORE FT: {PRE_FT_SYNTH_MONITOR_ACC:.4f}")
print(f"  catastrophic-forgetting Δ:   {sm_acc - PRE_FT_SYNTH_MONITOR_ACC:+.4f}")

sm_per_class = per_class_accuracy(sm_preds, sm_labels, NUM_CLASSES)
piece_mask = (sm_labels != 12)
sm_piece_acc = float((sm_preds[piece_mask] == sm_labels[piece_mask]).mean()) \
    if piece_mask.any() else float("nan")

sm_cm = confusion_matrix_np(sm_preds, sm_labels)
plot_confusion_matrix(sm_cm,
                      f"Synth-monitor (5% v1 slice) — after FT acc={sm_acc:.4f}",
                      f"{PLOTS_DIR}/synth_test_cm.png", cmap="Blues")

synth_results = {
    "n_samples": int(len(sm_preds)),
    "slice_frac": SYNTH_MONITOR_FRAC,
    "overall_acc_after_ft": sm_acc,
    "overall_acc_before_ft": PRE_FT_SYNTH_MONITOR_ACC,
    "catastrophic_forgetting_delta": sm_acc - PRE_FT_SYNTH_MONITOR_ACC,
    "piece_only_acc": sm_piece_acc,
    "loss": sm_loss,
    "per_class_acc": {CLASS_SHORT[c]: sm_per_class[c] for c in range(NUM_CLASSES)},
}
Path(f"{RESULTS_DIR}/synth_test_results.json").write_text(
    json.dumps(synth_results, indent=2))
np.save(f"{PREDS_DIR}/synth_test_preds.npy", sm_preds.astype(np.int64))
np.save(f"{PREDS_DIR}/synth_test_labels.npy", sm_labels.astype(np.int64))
print(f"wrote {RESULTS_DIR}/synth_test_results.json + {PLOTS_DIR}/synth_test_cm.png")

print("\033[92m✓ Cell 15 — Synth-monitor eval — OK\033[0m")




# %% [Cell 16 — Game7 eval at best_real checkpoint]
print("Evaluating on real_val_loader (game7) ...")
rv_loss, rv_acc, rv_preds, rv_labels = evaluate(
    model, real_val_loader, criterion, DEVICE, "real_val")
print(f"  game7 per-square acc: {rv_acc:.4f}  (loss {rv_loss:.4f})")

real_manifest = real_val_dataset.manifest.copy()
real_manifest["pred"] = rv_preds
per_board = (
    real_manifest.assign(correct=lambda d: d["pred"] == d["label"])
    .groupby("image_name")["correct"]
    .agg(["sum", "count"])
)
per_board["all_correct"] = per_board["sum"] == per_board["count"]
n_boards = len(per_board)
n_all_correct = int(per_board["all_correct"].sum())
mean_squares_correct = float(per_board["sum"].mean())

g7_per_class = per_class_accuracy(rv_preds, rv_labels, NUM_CLASSES)
piece_mask = (rv_labels != 12)
g7_piece_acc = float((rv_preds[piece_mask] == rv_labels[piece_mask]).mean()) \
    if piece_mask.any() else float("nan")

g7_cm = confusion_matrix_np(rv_preds, rv_labels)
plot_confusion_matrix(g7_cm,
                      f"Game7 (monitor) — best_real acc={rv_acc:.4f}",
                      f"{PLOTS_DIR}/game7_cm.png", cmap="Reds")

game7_results = {
    "n_frames": int(n_boards),
    "n_squares": int(len(rv_preds)),
    "per_square_acc": rv_acc,
    "per_square_acc_before_ft": PRE_FT_REAL_VAL_ACC,
    "improvement_over_baseline": rv_acc - PRE_FT_REAL_VAL_ACC,
    "per_board_acc": n_all_correct / n_boards if n_boards else 0.0,
    "n_all_correct": int(n_all_correct),
    "mean_squares_correct": mean_squares_correct,
    "piece_only_acc": g7_piece_acc,
    "loss": rv_loss,
    "per_class_acc": {CLASS_SHORT[c]: g7_per_class[c] for c in range(NUM_CLASSES)},
}
Path(f"{RESULTS_DIR}/game7_results.json").write_text(
    json.dumps(game7_results, indent=2))
np.save(f"{PREDS_DIR}/game7_preds.npy", rv_preds.astype(np.int64))
np.save(f"{PREDS_DIR}/game7_labels.npy", rv_labels.astype(np.int64))
print(f"  per-board acc: {n_all_correct}/{n_boards} = "
      f"{n_all_correct/max(n_boards,1):.4f}")
print(f"  piece-only acc: {g7_piece_acc:.4f}")
print(f"wrote {RESULTS_DIR}/game7_results.json + {PLOTS_DIR}/game7_cm.png")

print("\033[92m✓ Cell 16 — Game7 eval — OK\033[0m")




# %% [Cell 17 — Held-out games: dataset + loader for games 2/4/5/6 (stage 4 partition)]
real_test_datasets = {}
real_test_loaders = {}
for N in HELD_OUT_GAMES:
    gt_csv = f"{PROJECT_ROOT}/data/game{N}_per_frame/gt.csv"
    images_dir = f"{PROJECT_ROOT}/data/game{N}_per_frame/images"
    if not (Path(gt_csv).exists() and Path(images_dir).exists()):
        print(f"  [skip] game{N}: missing gt.csv or images/")
        continue
    ds = RealGameDataset(gt_csv, images_dir, game_name=f"game{N}", transform=None)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS_REAL, pin_memory=True,
                    persistent_workers=True, worker_init_fn=_worker_init_fn)
    real_test_datasets[N] = ds
    real_test_loaders[N] = ld
    n_frames = ds.manifest["image_name"].nunique()
    print(f"  game{N}: {n_frames} frames, {len(ds)} squares")

print("\033[92m✓ Cell 17 — Held-out loaders — OK\033[0m")




# %% [Cell 18 — Evaluate each held-out game]
per_game_stats = {}
per_game_preds = {}
per_game_labels = {}
per_game_manifest = {}

for N, loader in real_test_loaders.items():
    print(f"\nEvaluating on game{N} ...")
    g_loss, g_acc, g_preds, g_labels = evaluate(
        model, loader, criterion, DEVICE, f"game{N}")

    ds = real_test_datasets[N]
    manifest_g = ds.manifest.copy()
    manifest_g["pred"] = g_preds

    per_board_g = (
        manifest_g.assign(correct=lambda d: d["pred"] == d["label"])
        .groupby("image_name")["correct"]
        .agg(["sum", "count"])
    )
    per_board_g["all_correct"] = per_board_g["sum"] == per_board_g["count"]
    n_boards_g = len(per_board_g)
    n_all_correct_g = int(per_board_g["all_correct"].sum())
    mean_correct_g = float(per_board_g["sum"].mean())

    g_per_class = per_class_accuracy(g_preds, g_labels, NUM_CLASSES)
    piece_mask_g = (g_labels != 12)
    g_piece_acc = float((g_preds[piece_mask_g] == g_labels[piece_mask_g]).mean()) \
        if piece_mask_g.any() else float("nan")

    cm_g = confusion_matrix_np(g_preds, g_labels)
    plot_confusion_matrix(
        cm_g,
        f"game{N} — best_real acc={g_acc:.4f}",
        f"{PLOTS_DIR}/game{N}_cm.png", cmap="Reds",
    )

    results_g = {
        "game": f"game{N}",
        "n_frames": int(n_boards_g),
        "n_squares": int(len(g_preds)),
        "per_square_acc": g_acc,
        "per_board_acc": n_all_correct_g / n_boards_g if n_boards_g else 0.0,
        "n_all_correct": int(n_all_correct_g),
        "mean_squares_correct": mean_correct_g,
        "piece_only_acc": g_piece_acc,
        "loss": g_loss,
        "per_class_acc": {CLASS_SHORT[c]: g_per_class[c] for c in range(NUM_CLASSES)},
    }
    Path(f"{RESULTS_DIR}/game{N}_results.json").write_text(
        json.dumps(results_g, indent=2))
    np.save(f"{PREDS_DIR}/game{N}_preds.npy", g_preds.astype(np.int64))
    np.save(f"{PREDS_DIR}/game{N}_labels.npy", g_labels.astype(np.int64))

    per_game_stats[N] = results_g
    per_game_preds[N] = g_preds
    per_game_labels[N] = g_labels
    per_game_manifest[N] = manifest_g

    print(f"  game{N}: per-square={g_acc:.4f}  per-board={n_all_correct_g}/{n_boards_g}"
          f"={n_all_correct_g/max(n_boards_g,1):.4f}  mean-correct={mean_correct_g:.1f}/64  "
          f"piece-only={g_piece_acc:.4f}")

print("\033[92m✓ Cell 18 — Held-out games eval — OK\033[0m")




# %% [Cell 19 — Aggregate across held-out games (games 2/4/5/6 — stage 4 partition, same as stages 1/2)]
combined_preds = np.concatenate([per_game_preds[N] for N in per_game_preds])
combined_labels = np.concatenate([per_game_labels[N] for N in per_game_labels])

combined_manifest = pd.concat(
    [per_game_manifest[N].assign(game_num=N) for N in per_game_manifest],
    ignore_index=True,
)
per_board_combined = (
    combined_manifest.assign(correct=lambda d: d["pred"] == d["label"])
    .groupby(["game_num", "image_name"])["correct"]
    .agg(["sum", "count"])
)
per_board_combined["all_correct"] = (
    per_board_combined["sum"] == per_board_combined["count"]
)
combined_n_boards = len(per_board_combined)
combined_n_all_correct = int(per_board_combined["all_correct"].sum())
combined_mean_correct = float(per_board_combined["sum"].mean()) \
    if combined_n_boards else 0.0
combined_per_square_acc = float((combined_preds == combined_labels).mean())
combined_per_board_acc = (
    combined_n_all_correct / combined_n_boards if combined_n_boards else 0.0
)
piece_mask_c = (combined_labels != 12)
combined_piece_acc = float(
    (combined_preds[piece_mask_c] == combined_labels[piece_mask_c]).mean()
) if piece_mask_c.any() else float("nan")
combined_per_class = per_class_accuracy(combined_preds, combined_labels, NUM_CLASSES)

cm_c = confusion_matrix_np(combined_preds, combined_labels)
plot_confusion_matrix(
    cm_c,
    f"Held-out aggregate (games 2/4/5/6 — stage 4 partition) — per-square acc={combined_per_square_acc:.4f}",
    f"{PLOTS_DIR}/aggregate_cm.png", cmap="Reds",
)

pairs_c = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm_c[i, j] > 0:
            pairs_c.append((int(cm_c[i, j]), i, j))
pairs_c.sort(reverse=True)
top5_pairs = [
    {"true": CLASS_NAMES[t], "pred": CLASS_NAMES[p], "count": int(n_pair)}
    for n_pair, t, p in pairs_c[:5]
]

aggregate_results = {
    "games": [f"game{N}" for N in sorted(per_game_stats.keys())],
    "n_frames": int(combined_n_boards),
    "n_squares": int(len(combined_preds)),
    "per_square_acc": combined_per_square_acc,
    "per_board_acc": combined_per_board_acc,
    "n_all_correct": int(combined_n_all_correct),
    "mean_squares_correct": combined_mean_correct,
    "piece_only_acc": combined_piece_acc,
    "per_class_acc": {CLASS_SHORT[c]: combined_per_class[c] for c in range(NUM_CLASSES)},
    "top5_confusion_pairs": top5_pairs,
}
Path(f"{RESULTS_DIR}/held_out_aggregate.json").write_text(
    json.dumps(aggregate_results, indent=2))
np.save(f"{PREDS_DIR}/held_out_preds.npy", combined_preds.astype(np.int64))
np.save(f"{PREDS_DIR}/held_out_labels.npy", combined_labels.astype(np.int64))

print(f"\nAggregate (games {sorted(per_game_stats.keys())}):")
print(f"  per-square acc:  {combined_per_square_acc:.4f}")
print(f"  per-board  acc:  {combined_n_all_correct}/{combined_n_boards}="
      f"{combined_per_board_acc:.4f}")
print(f"  piece-only acc:  {combined_piece_acc:.4f}")
print(f"  top-5 confusion: {top5_pairs}")
print(f"wrote {RESULTS_DIR}/held_out_aggregate.json + {PLOTS_DIR}/aggregate_cm.png")

print("\033[92m✓ Cell 19 — Aggregate eval — OK\033[0m")




# %% [Cell 20 — Qualitative plots: 8 boards per held-out game]
def _short_for(label):
    return CLASS_SHORT[int(label)] if 0 <= int(label) < NUM_CLASSES else "?"


for N in sorted(per_game_manifest.keys()):
    manifest_g = per_game_manifest[N]
    images_order = sorted(manifest_g["image_name"].unique())
    n_frames_g = len(images_order)
    if n_frames_g == 0:
        continue
    k = min(8, n_frames_g)
    idxs = np.linspace(0, n_frames_g - 1, k).round().astype(int)
    selected_imgs = [images_order[i] for i in idxs]
    images_dir = Path(f"{PROJECT_ROOT}/data/game{N}_per_frame/images")
    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    flat = axes.ravel()
    for ax, img_name in zip(flat, selected_imgs):
        sub = manifest_g[manifest_g["image_name"] == img_name].sort_values(
            ["board_row", "board_col"])
        bgr = cv2.imread(str(images_dir / img_name))
        ds_g = real_test_datasets[N]
        corners = ds_g._get_corners(img_name, bgr)
        warped = warp_chessboard_image(bgr, corners)
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        H, W = warped_rgb.shape[:2]
        ax.imshow(warped_rgb)
        BOARD_OFFSET = 50
        SQ = (W - 2 * BOARD_OFFSET) // 8
        n_correct = 0; n_total = 0
        for _, r in sub.iterrows():
            br, bc = int(r["board_row"]), int(r["board_col"])
            true_lab = int(r["label"]); pred_lab = int(r["pred"])
            ok = (true_lab == pred_lab)
            n_correct += int(ok); n_total += 1
            x0 = BOARD_OFFSET + bc * SQ
            y0 = BOARD_OFFSET + br * SQ
            color = "lime" if ok else "red"
            rect = plt.Rectangle((x0, y0), SQ, SQ, linewidth=1.6,
                                 edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            if pred_lab != 12 or not ok:
                ax.text(x0 + SQ / 2, y0 + SQ / 2, _short_for(pred_lab),
                        color=color, fontsize=8, ha="center", va="center", weight="bold")
        ax.set_title(f"{img_name}  {n_correct}/{n_total}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"game{N} qualitative — best_real predictions "
                 f"(green=correct, red=wrong)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    qual_path = f"{PLOTS_DIR}/game{N}_qualitative.png"
    plt.savefig(qual_path, dpi=110)
    plt.close()
    print(f"  wrote {qual_path}")

print("\033[92m✓ Cell 20 — Qualitative plots — OK\033[0m")




# %% [Cell 21 — Summary.md]
# Stage 4 starts from ImageNet — there are no SRC_* baseline-checkpoint
# numbers to display in this run. The v1 zero-shot per-class held-out
# aggregate is still loaded below (if present) for the per-class delta
# table that's kept across stages.

# Guard: if training was interrupted (KeyboardInterrupt), Cell 11's post-loop
# assignment of total_train_time / stop_reason never ran. Reconstruct from the
# training log so the summary still writes.
try:
    total_train_time
except NameError:
    total_train_time = float(log_df["epoch_time_s"].sum())
try:
    stop_reason
except NameError:
    stop_reason = "interrupted_before_loop_completed"

# Read v1 baseline summary's held-out aggregate numbers if available.
v1_baseline_held_out = {}
v1_summary_path = Path(f"{PROJECT_ROOT}/zero_shot/results/games_2_4_5_6_eval")
if v1_summary_path.exists():
    # Look for an aggregate file written by build_correct_vs_wrong_plots etc.
    for cand in ["held_out_aggregate.json", "aggregate.json"]:
        if (v1_summary_path / cand).exists():
            v1_baseline_held_out = json.loads((v1_summary_path / cand).read_text())
            break

# Fallback search: scan zero_shot/results for any aggregate-shaped JSON.
# The v1 baseline may have written its held-out aggregate under a
# different filename. We accept any JSON with a "per_square_acc" key
# and a "games" key including game2 OR game{2,4,5,6}.
if not v1_baseline_held_out:
    zs_results_root = Path(f"{PROJECT_ROOT}/zero_shot/results")
    if zs_results_root.exists():
        for json_path in sorted(zs_results_root.rglob("*.json")):
            try:
                data = json.loads(json_path.read_text())
            except Exception:
                continue
            if (
                isinstance(data, dict)
                and "per_square_acc" in data
                and "games" in data
                and isinstance(data["games"], list)
                and any(g in data["games"] for g in
                        ("game2", "game4", "game5", "game6"))
            ):
                v1_baseline_held_out = data
                print(f"[Cell 21] fallback: loaded v1 held-out aggregate "
                      f"from {json_path}")
                break
    if not v1_baseline_held_out:
        print(f"[Cell 21] WARNING: could not locate v1 baseline held-out "
              f"aggregate under {PROJECT_ROOT}/zero_shot/results/. "
              f"Per-class delta table will show n/a; verdict sentence "
              f"will not include direct comparison.")

def _moved_off_zero(class_short, thresh=0.05):
    col = f"real_val_acc_{class_short}"
    if col not in log_df.columns:
        return False
    return bool((log_df[col].dropna() > thresh).any())


moved = {k: _moved_off_zero(k) for k in ("wN", "wB", "wK", "bN", "bB", "bK")}
moved_names = [k for k, v in moved.items() if v]
not_moved = [k for k, v in moved.items() if not v]
if moved_names and not not_moved:
    moved_sentence = (
        "ALL of {wN, wB, wK, bN, bB, bK} crossed >5% on real_val — every "
        "knight/bishop/king class moved off zero, supporting the FT hypothesis."
    )
elif moved_names:
    moved_sentence = (
        f"{', '.join(moved_names)} crossed >5% on real_val; "
        f"{', '.join(not_moved)} stayed effectively at zero. Mixed evidence."
    )
else:
    moved_sentence = (
        "NONE of {wN, wB, wK, bN, bB, bK} crossed >5% on real_val. "
        "Joint synth+real training did not measurably move the failing classes "
        "above chance for stage 4."
    )

def _fmt_pct(x): return f"{x:.4f}" if x == x else "n/a"

game_rows = []
for N in sorted(per_game_stats.keys()):
    s = per_game_stats[N]
    game_rows.append(
        f"| game{N} | {s['n_frames']} | {s['n_squares']} | "
        f"{_fmt_pct(s['per_square_acc'])} | {_fmt_pct(s['per_board_acc'])} | "
        f"{_fmt_pct(s['piece_only_acc'])} | {s['mean_squares_correct']:.2f}/64 |"
    )
game_rows.append(
    f"| **agg** | {combined_n_boards} | {len(combined_preds)} | "
    f"**{_fmt_pct(combined_per_square_acc)}** | "
    f"**{_fmt_pct(combined_per_board_acc)}** | "
    f"**{_fmt_pct(combined_piece_acc)}** | "
    f"{combined_mean_correct:.2f}/64 |"
)
game_table = (
    "| game | n_frames | n_squares | per-sq acc | per-board acc | piece-only acc | mean correct |\n"
    "|------|---------:|----------:|-----------:|--------------:|---------------:|-------------:|\n"
    + "\n".join(game_rows)
)

# Per-class delta on aggregate held-out (after FT vs. v1 baseline if available).
per_class_delta_lines = []
v1_per_class = v1_baseline_held_out.get("per_class_acc", {}) if v1_baseline_held_out else {}
for cls in range(NUM_CLASSES):
    short = CLASS_SHORT[cls]
    after = combined_per_class[cls]
    if v1_per_class and short in v1_per_class and v1_per_class[short] == v1_per_class[short]:
        before = v1_per_class[short]
        delta = after - before if (after == after) else float("nan")
        per_class_delta_lines.append(
            f"| {short} | {before:.4f} | {after if after == after else float('nan'):.4f} | "
            f"{delta:+.4f} |"
        )
    else:
        per_class_delta_lines.append(
            f"| {short} | n/a | {after if after == after else float('nan'):.4f} | n/a |"
        )
per_class_table = (
    "| class | v1 baseline | stage4_combined_30 | Δ |\n"
    "|-------|------------:|------------------:|---:|\n"
    + "\n".join(per_class_delta_lines)
)

# Headline verdict for stage 4 — direct comparison to stage 2 (same partition,
# same 30 real frames). Built after we load stage 2 numbers below.

# --- Stage 2 comparison block (inserted into summary BEFORE the v1 per-class
# delta block). Stage 4 and stage 2 use the SAME test partition (games
# 2/4/5/6) and the SAME 30 real frames, so this comparison is DIRECT and
# valid — no caveat. Variable name `stage1_compare_lines` / `s1_paths` is
# kept from template lineage; the *contents* now refer to stage 2 paths
# and the headline comparison is stage 2 (sequential FT) vs.
# stage 4 (joint).
stage1_compare_lines = [
    "## Comparison to stage2_30 (same test partition — Comparison A)",
    "",
    "Same 30 real frames, same test set, identical augmentation. The single "
    "axis varied is the **training procedure**: stage 2 fine-tunes the v1 "
    "synth-trained baseline on real data only (sequential), whereas stage 4 "
    "trains jointly on synth+real from ImageNet weights.",
    "",
    "| metric | stage2_30 (sequential FT) | stage4_combined_30 (joint) | Δ |",
    "|--------|--------------------------:|---------------------------:|---:|",
]
s1_paths = {
    "game7":     f"{PROJECT_ROOT}/fine_tuning/stage2_30/results/game7_results.json",
    "agg":       f"{PROJECT_ROOT}/fine_tuning/stage2_30/results/held_out_aggregate.json",
    "synth":     f"{PROJECT_ROOT}/fine_tuning/stage2_30/results/synth_test_results.json",
}
s1 = {}
for _k, _p in s1_paths.items():
    if Path(_p).exists():
        s1[_k] = json.loads(Path(_p).read_text())


def _fmt_delta(after, before):
    if after is None or before is None:
        return "n/a"
    if after != after or before != before:  # NaN check
        return "n/a"
    return f"{after - before:+.4f}"


g7_s1 = s1.get("game7", {}).get("per_square_acc")
g7_s2 = rv_acc
stage1_compare_lines.append(
    f"| game7 real_val_acc | "
    f"{f'{g7_s1:.4f}' if g7_s1 is not None else 'n/a'} | "
    f"{g7_s2:.4f} | {_fmt_delta(g7_s2, g7_s1)} |"
)

agg_s1 = s1.get("agg", {}).get("per_square_acc")
stage1_compare_lines.append(
    f"| held-out per-sq | "
    f"{f'{agg_s1:.4f}' if agg_s1 is not None else 'n/a'} | "
    f"{combined_per_square_acc:.4f} | "
    f"{_fmt_delta(combined_per_square_acc, agg_s1)} |"
)

piece_s1 = s1.get("agg", {}).get("piece_only_acc")
stage1_compare_lines.append(
    f"| held-out piece-only | "
    f"{f'{piece_s1:.4f}' if piece_s1 is not None else 'n/a'} | "
    f"{combined_piece_acc:.4f} | "
    f"{_fmt_delta(combined_piece_acc, piece_s1)} |"
)

forget_s1 = s1.get("synth", {}).get("catastrophic_forgetting_delta")
forget_s2 = sm_acc - PRE_FT_SYNTH_MONITOR_ACC
stage1_compare_lines.append(
    f"| forgetting Δ on 5% v1 | "
    f"{f'{forget_s1:+.4f}' if forget_s1 is not None else 'n/a'} | "
    f"{forget_s2:+.4f} | "
    f"{_fmt_delta(forget_s2, forget_s1)} |"
)

stage1_compare_lines.append("")

# Per-class deltas vs. stage 2 on the same aggregate held-out partition.
if "agg" in s1 and "per_class_acc" in s1["agg"]:
    stage1_compare_lines.extend([
        "### Per-class delta vs. stage2_30 (same aggregate held-out)",
        "",
        "| class | stage2_30 (sequential FT) | stage4_combined_30 (joint) | Δ |",
        "|-------|--------------------------:|---------------------------:|---:|",
    ])
    for cls in range(NUM_CLASSES):
        short = CLASS_SHORT[cls]
        s1_acc = s1["agg"]["per_class_acc"].get(short)
        s2_acc = combined_per_class[cls]
        if s1_acc is None or s1_acc != s1_acc:
            stage1_compare_lines.append(
                f"| {short} | n/a | "
                f"{s2_acc if s2_acc == s2_acc else float('nan'):.4f} | n/a |"
            )
        else:
            delta = s2_acc - s1_acc if s2_acc == s2_acc else float("nan")
            stage1_compare_lines.append(
                f"| {short} | {s1_acc:.4f} | "
                f"{s2_acc if s2_acc == s2_acc else float('nan'):.4f} | "
                f"{delta:+.4f} |"
            )
    stage1_compare_lines.append("")

# Headline verdict — now that stage 2 numbers are loaded.
agg_s2_ref = s1.get("agg", {}).get("per_square_acc")
if agg_s2_ref is not None:
    beat = combined_per_square_acc > agg_s2_ref
    verdict = (
        f"**Verdict:** stage4 (joint training) {'BEAT' if beat else 'did NOT beat'} "
        f"stage 2 (sequential FT) on the matched held-out partition (games "
        f"2/4/5/6): {combined_per_square_acc:.4f} vs. {agg_s2_ref:.4f} "
        f"per-square (Δ = {combined_per_square_acc - agg_s2_ref:+.4f})."
    )
else:
    verdict = (
        f"**Verdict:** stage4 per-square acc on held-out (games 2/4/5/6) is "
        f"**{combined_per_square_acc:.4f}**. stage 2 held-out aggregate not "
        f"available on disk — re-run stage 2 to populate "
        f"`fine_tuning/stage2_30/results/held_out_aggregate.json` for the "
        f"matched comparison."
    )

summary_lines = [
    "# Stage 4 — Joint synth+real training from ImageNet (30 manual labels, balanced sampler)",
    "",
    "## Recipe (vs. v1 zero-shot baseline)",
    "- **Source weights:** ImageNet pretrained via torchvision (NOT v1 baseline, NOT stages 1/2/3 weights).",
    "- **Training data:** dataset_v1 (full synth manifest, ~390k squares) + "
    "all 30 manual-label real frames (games 8-11). Combined via ConcatDataset.",
    f"- **Single phase**: all params trainable from epoch 1. SGD(lr={LR}, "
    f"momentum={MOMENTUM}, wd={WEIGHT_DECAY}). No scheduler. No freeze.",
    f"- **Aug:** color jitter @{COLOR_JITTER_APPLY_PROB} → shear @{SHEAR_APPLY_PROB} "
    f"(±8°) → noise @{NOISE_APPLY_PROB} (std={NOISE_STD}), applied to BOTH synth and real.",
    f"- **Sampler:** WeightedRandomSampler at {SYNTH_BATCH_FRAC:.0%} synth / "
    f"{1-SYNTH_BATCH_FRAC:.0%} real per batch, "
    f"num_samples={NUM_SAMPLES_PER_EPOCH:,} per epoch "
    f"(~{NUM_SAMPLES_PER_EPOCH // BATCH_SIZE} batches/epoch).",
    f"- **Checkpoint by:** real_val_acc on game7 (NOT synth_val).",
    f"- **Early stop:** patience={EARLY_STOP_PATIENCE} on real_val_acc.",
    "",
    "## Training",
    f"- Ran **{n_epochs_ran}** epochs in **{total_train_time/60:.1f} min**.",
    f"- Stop reason: `{stop_reason}`.",
    f"- Best real_val_acc (game7):  **{best_real_val_acc:.4f}** at epoch {best_real_epoch}.",
    f"- Best synth_monitor (5% v1): **{best_synth_monitor_acc:.4f}** at epoch {best_synth_monitor_epoch}.",
    "",
    "## Catastrophic-forgetting probe (5% slice of dataset_v1)",
    "",
    "NOTE: stage 4 starts from ImageNet, so 'BEFORE training' is ~chance "
    "(not a meaningful baseline). The headline number here is "
    "`acc AFTER training` — joint training is expected to keep this HIGH "
    "(≥ 0.95) because synth is in the training mix, unlike stages 1/2/3 "
    "where it falls.",
    "",
    f"- Synth-monitor acc BEFORE train (ImageNet init): **{PRE_FT_SYNTH_MONITOR_ACC:.4f}**",
    f"- Synth-monitor acc AFTER train  (best_real ckpt): **{sm_acc:.4f}**",
    f"- **Δ from ImageNet init: {sm_acc - PRE_FT_SYNTH_MONITOR_ACC:+.4f}** "
    f"(positive = the joint task learned synth, as expected; compare to "
    f"stages 1/2/3 which START at ~0.999 and lose ground)",
    "",
    "## Game7 monitor (NOT held-out — used for checkpoint selection)",
    f"- Per-square at best_real:  **{rv_acc:.4f}**",
    f"- Per-board acc: {n_all_correct}/{n_boards} = {n_all_correct/max(n_boards,1):.4f}",
    f"- Mean squares correct/board: {mean_squares_correct:.2f}/64",
    "",
    "## Held-out real test (games 2, 4, 5, 6) — identical to stages 1/2 partition",
    "",
    game_table,
    "",
    verdict,
    "",
    *stage1_compare_lines,
    "## Per-class deltas on aggregate held-out (vs. v1 zero-shot baseline if available)",
    "",
    per_class_table,
    "",
    "## Per-class real_val trajectory analysis",
    "",
    "Stage 4 starts from ImageNet (not v1 baseline), so the question for "
    "the dead classes (wN/wK/bK/wB/bB) is whether real_val per-class "
    "crosses zero at all — not whether it 'improves over baseline'.",
    "",
    f"- {moved_sentence}",
    "- See plots/per_class_real_val.png for the 13-class trajectory.",
    "",
    "## Artifacts",
    "- `checkpoints/best_real.pt` "
    f"(epoch {best_real_epoch}, real_val_acc={best_real_val_acc:.4f}) — headline ckpt",
    "- `checkpoints/best_synth_monitor.pt` "
    f"(epoch {best_synth_monitor_epoch}, synth_monitor_acc={best_synth_monitor_acc:.4f}) — monitor-only",
    "- `checkpoints/latest.pt`",
    "- `results/stage4_real_manifest.csv` — the 30 manual-label frames (real side)",
    "- `results/training_log.csv` — per-epoch log + 13 per-class real_val columns",
    "- `results/synth_test_results.json` (catastrophic-forgetting probe — AFTER training)",
    "- `results/game7_results.json`",
    "- `results/game{2,4,5,6}_results.json`",
    "- `results/held_out_aggregate.json`",
    "- `results/predictions/*.npy`",
    "- `plots/aug_smoke_check.png`, `stage4_real_samples.png`",
    "- `plots/training_curves.png`, `per_class_real_val.png`",
    "- `plots/synth_test_cm.png`, `game7_cm.png`, `game{2,4,5,6}_cm.png`, `aggregate_cm.png`",
    "- `plots/game{2,4,5,6}_qualitative.png`",
]
summary_text = "\n".join(summary_lines)
Path(f"{RESULTS_DIR}/summary.md").write_text(summary_text)
print(summary_text)
print(f"\nwrote {RESULTS_DIR}/summary.md")

print("\033[92m✓ Cell 21 — Summary.md — OK\033[0m")




# %% [Cell 22 — Stage 2 re-eval on games 2/4/5/6 (same-partition direct comparison)]
# NOTE: stage 4 and stage 2 already share the SAME test partition, so this
# re-eval reproduces stage 2's published numbers exactly. It's partly
# redundant, but acts as a sanity check that the test pipeline is identical
# between scripts (and writes a JSON the report tooling can read uniformly).
print("=" * 64)
print("Loading stage_2 best_real checkpoint and re-evaluating on the")
print("stage 4 test partition (games 2/4/5/6) for direct comparison.")
print("=" * 64)

stage2_ckpt_path = f"{PROJECT_ROOT}/fine_tuning/stage2_30/checkpoints/best_real.pt"
if not Path(stage2_ckpt_path).exists():
    print(f"[Cell 22] stage 2 checkpoint not found: {stage2_ckpt_path}")
    print(f"[Cell 22] skipping re-eval. Stage 4 summary.md will not have"
          f" the bridge row.")
else:
    stage2_ckpt = torch.load(stage2_ckpt_path, map_location=DEVICE,
                             weights_only=False)
    # build_model() here returns a freshly ImageNet-initialised ResNet18,
    # but we overwrite its weights with stage 2's checkpoint immediately
    # — so the ImageNet init is just scaffold.
    stage2_model = build_model().to(DEVICE)
    stage2_model.load_state_dict(stage2_ckpt["model_state_dict"])
    stage2_model.eval()
    print(f"loaded stage_2 best_real ckpt (epoch {stage2_ckpt['epoch']}, "
          f"real_val_acc={stage2_ckpt['real_val_acc']:.4f})")

    # Re-evaluate on the SAME loaders we just built (games 2/4/5/6).
    s2_per_game_acc = {}
    s2_combined_preds = []
    s2_combined_labels = []
    s2_combined_manifest = []
    for N, loader in real_test_loaders.items():
        _, g_acc, g_preds, g_labels = evaluate(
            stage2_model, loader, criterion, DEVICE, f"stage2_on_game{N}")
        s2_per_game_acc[N] = g_acc
        s2_combined_preds.append(g_preds)
        s2_combined_labels.append(g_labels)
        mf = real_test_datasets[N].manifest.copy()
        mf["pred"] = g_preds; mf["game_num"] = N
        s2_combined_manifest.append(mf)
        print(f"  stage2 on game{N}: per-sq={g_acc:.4f}")

    s2_combined_preds = np.concatenate(s2_combined_preds)
    s2_combined_labels = np.concatenate(s2_combined_labels)
    s2_combined_manifest = pd.concat(s2_combined_manifest, ignore_index=True)

    s2_combined_per_sq = float((s2_combined_preds == s2_combined_labels).mean())
    s2_piece_mask = (s2_combined_labels != 12)
    s2_combined_piece = float(
        (s2_combined_preds[s2_piece_mask] == s2_combined_labels[s2_piece_mask]).mean()
    ) if s2_piece_mask.any() else float("nan")
    s2_per_class = per_class_accuracy(
        s2_combined_preds, s2_combined_labels, NUM_CLASSES)

    # Per-board accuracy for stage 2 on games 2/4/5/6.
    s2_per_board = (
        s2_combined_manifest.assign(correct=lambda d: d["pred"] == d["label"])
        .groupby(["game_num", "image_name"])["correct"]
        .agg(["sum", "count"])
    )
    s2_per_board["all_correct"] = s2_per_board["sum"] == s2_per_board["count"]
    s2_n_boards = len(s2_per_board)
    s2_n_all_correct = int(s2_per_board["all_correct"].sum())

    stage2_reeval = {
        "test_partition": ["game2", "game4", "game5", "game6"],
        "n_frames": int(s2_n_boards),
        "n_squares": int(len(s2_combined_preds)),
        "per_square_acc": s2_combined_per_sq,
        "per_board_acc": s2_n_all_correct / s2_n_boards if s2_n_boards else 0.0,
        "n_all_correct": int(s2_n_all_correct),
        "piece_only_acc": s2_combined_piece,
        "per_class_acc": {
            CLASS_SHORT[c]: s2_per_class[c] for c in range(NUM_CLASSES)
        },
    }
    Path(f"{RESULTS_DIR}/stage2_reeval_on_games_2_4_5_6.json").write_text(
        json.dumps(stage2_reeval, indent=2))
    print(f"\nstage 2 on games 2/4/5/6 partition:")
    print(f"  per-sq: {s2_combined_per_sq:.4f}")
    print(f"  piece-only: {s2_combined_piece:.4f}")
    print(f"  per-board: {s2_n_all_correct}/{s2_n_boards}="
          f"{s2_n_all_correct/max(s2_n_boards,1):.4f}")
    print(f"wrote {RESULTS_DIR}/stage2_reeval_on_games_2_4_5_6.json")

    # Append to summary.md: a same-partition direct-comparison section.
    bridge_lines = [
        "",
        "## Direct comparison — stage 2 reevaluated on games 2/4/5/6 (same partition)",
        "",
        "Stage 2's checkpoint evaluated on the EXACT same test set used by "
        "stage 4. Because both stages already share the games 2/4/5/6 "
        "partition, these numbers should reproduce stage 2's published "
        "results — this section is both a sanity check that the test "
        "pipeline is identical and a single-pane comparison row for the "
        "report.",
        "",
        "| metric | stage2_30 (sequential FT) | stage4_combined_30 (joint) | Δ |",
        "|--------|--------------------------:|---------------------------:|---:|",
        f"| per-sq acc | {s2_combined_per_sq:.4f} | "
        f"{combined_per_square_acc:.4f} | "
        f"{combined_per_square_acc - s2_combined_per_sq:+.4f} |",
        f"| piece-only | {s2_combined_piece:.4f} | "
        f"{combined_piece_acc:.4f} | "
        f"{combined_piece_acc - s2_combined_piece:+.4f} |",
        f"| per-board  | "
        f"{s2_n_all_correct/max(s2_n_boards,1):.4f} "
        f"({s2_n_all_correct}/{s2_n_boards}) | "
        f"{combined_per_board_acc:.4f} "
        f"({combined_n_all_correct}/{combined_n_boards}) | "
        f"{combined_per_board_acc - s2_n_all_correct/max(s2_n_boards,1):+.4f} |",
        "",
        "### Per-class on games 2/4/5/6 (matched partition)",
        "",
        "| class | stage2_30 (sequential FT) | stage4_combined_30 (joint) | Δ |",
        "|-------|--------------------------:|---------------------------:|---:|",
    ]
    for cls in range(NUM_CLASSES):
        s2 = s2_per_class[cls]
        s4 = combined_per_class[cls]
        if s2 == s2 and s4 == s4:
            bridge_lines.append(
                f"| {CLASS_SHORT[cls]} | {s2:.4f} | {s4:.4f} | "
                f"{s4 - s2:+.4f} |"
            )
        else:
            bridge_lines.append(
                f"| {CLASS_SHORT[cls]} | "
                f"{s2 if s2 == s2 else float('nan'):.4f} | "
                f"{s4 if s4 == s4 else float('nan'):.4f} | n/a |"
            )
    bridge_lines.append("")

    # Append to existing summary.md
    with open(f"{RESULTS_DIR}/summary.md", "a") as f:
        f.write("\n".join(bridge_lines))
    print(f"appended bridge section to {RESULTS_DIR}/summary.md")

print("\033[92m✓ Cell 22 — Stage 2 re-eval bridge — OK\033[0m")
