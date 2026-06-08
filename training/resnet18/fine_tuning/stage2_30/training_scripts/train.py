"""
Project 2 — Fine-tuning STAGE 2 (30 real images — all manual labels).

# FINE-TUNING (NOT ZERO-SHOT). All 30 manual-labelled real images from
# games 8-11 are in training. game7 is the monitor and gates checkpoint
# selection. Pretrained weights: zero_shot/results/best_synth.pt (v1
# baseline, peak game7 real_val 0.5923) — NOT zero_shot_v1.5, NOT
# stage1_10 weights.

Two-phase recipe (Wölflein 2021, §4.3):
  Phase A (epochs 1-5)   — freeze backbone, train fc only @ lr=1e-3.
  Phase B (epochs 6-30)  — unfreeze all, lr=1e-4. NO scheduler (stage 1's
                            LR step at epoch 21 produced no improvement).
Augmentation: stronger than zero-shot baseline (color jitter @0.7, shear
@0.8 ±8°, noise @0.5 std=0.015). Sampler: shuffle=True (no class-balanced
sampling — 1,920 samples is still small). Checkpoint selection:
real_val_acc on game7. Early stop patience=8.

Stage 2 / 3. Cold-start from v1 baseline weights (NOT stage 1 weights).
Single experimental variable vs. stage 1: training data 10 → 30 images.
Everything else (LR, aug, recipe, partition) is identical.

Test partition (identical across all three stages):
   monitor: game7 (55 frames),
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
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.transforms import ColorJitter, RandomAffine, InterpolationMode
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from preprocessing.chess_dataset import ChessSquareDataset
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




# %% [Cell 2 — Config]
PROJECT_ROOT = "/home/eladbaum/chess_project"

# === Pretrained-weights source: v1 zero-shot baseline (NOT v1.5). ===
# The folder name in the project is `zero_shot/`. Checkpoint actually lives
# under results/ (not checkpoints/) because the baseline script wrote it
# there. Verified before launch via the smoke test [1d] stage.
PRETRAINED_CKPT = f"{PROJECT_ROOT}/zero_shot/results/best_synth.pt"

# === Real data ===
REAL_LABELS_CSV = f"{PROJECT_ROOT}/data/real_labels.csv"
REAL_IMAGES_ROOT = f"{PROJECT_ROOT}/data"
GAME7_DIR = f"{PROJECT_ROOT}/data/game7_per_frame/images"
GAME7_GT_CSV = f"{PROJECT_ROOT}/data/game7_per_frame/gt.csv"
HELD_OUT_GAMES = [2, 4, 5, 6]

# === Synth data (v1) for catastrophic-forgetting probe ===
SYNTH_DATASET_DIR = f"{PROJECT_ROOT}/data/dataset_v1/images"
SYNTH_MANIFEST_PATH = f"{PROJECT_ROOT}/scripts/manifest.csv"
SYNTH_CORNERS_PATH = f"{PROJECT_ROOT}/scripts/corners.json"

# === Experiment dirs ===
EXP_DIR = f"{PROJECT_ROOT}/fine_tuning/stage2_30"
CHECKPOINTS_DIR = f"{EXP_DIR}/checkpoints"
RESULTS_DIR = f"{EXP_DIR}/results"
PLOTS_DIR = f"{EXP_DIR}/plots"
PREDS_DIR = f"{RESULTS_DIR}/predictions"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

# === Hyperparameters ===
BATCH_SIZE = 64
NUM_EPOCHS = 30
EARLY_STOP_PATIENCE = 8           # on real_val_acc
PHASE_A_EPOCHS = 5
LR_HEAD_WARMUP = 1e-3             # phase A
LR_FULL_FT = 1e-4                 # phase B uniform
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 15                 # in phase-B local epoch count → drops at absolute epoch 21
LR_GAMMA = 0.1
# LR_STEP_SIZE/LR_GAMMA kept for documentation only. Stage 2 does
# NOT use a scheduler in phase B (stage 1's LR step at epoch 21
# produced no improvement; we skip the perturbation).
SYNTH_MONITOR_FRAC = 0.05         # 5% slice of dataset_v1

NUM_WORKERS_SYNTH = 4
NUM_WORKERS_REAL = 4

# === Augmentation (stronger than zero-shot) ===
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

assert "v1.5" not in PRETRAINED_CKPT, (
    f"refusing to fine-tune from v1.5 weights — spec mandates the v1 baseline. "
    f"got PRETRAINED_CKPT={PRETRAINED_CKPT}"
)
assert "dataset_v1.5" not in SYNTH_DATASET_DIR, (
    f"refusing to use dataset_v1.5 for the synth-monitor slice — the source "
    f"checkpoint trained on dataset_v1; comparing against v1.5 measures "
    f"transfer, not forgetting. got SYNTH_DATASET_DIR={SYNTH_DATASET_DIR}"
)

print(f"pretrained:        {PRETRAINED_CKPT}")
print(f"real labels:       {REAL_LABELS_CSV}")
print(f"real images root:  {REAL_IMAGES_ROOT}")
print(f"game7 monitor:     {GAME7_DIR}")
print(f"synth dataset:     {SYNTH_DATASET_DIR}")
print(f"synth manifest:    {SYNTH_MANIFEST_PATH}")
print(f"synth corners:     {SYNTH_CORNERS_PATH}")
print(f"checkpoints:       {CHECKPOINTS_DIR}")
print(f"results:           {RESULTS_DIR}")
print(f"plots:             {PLOTS_DIR}")

print("\033[92m✓ Cell 2 — Config — OK\033[0m")




# %% [Cell 3 — Load manual labels + use ALL 30 as training frames]
manual_df = pd.read_csv(REAL_LABELS_CSV)
print(f"Loaded {len(manual_df)} manual-label rows from {REAL_LABELS_CSV}")
print(f"\nPer-game count of manual labels:")
per_game_counts = manual_df["game"].value_counts().sort_index()
for game, cnt in per_game_counts.items():
    print(f"  {game}: {cnt}")

# Stage 2 uses ALL 30 manual labels (no subsetting).
STAGE2_N = 30
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

stage2_train_df = pd.DataFrame(picked_rows).reset_index(drop=True)
assert len(stage2_train_df) == STAGE2_N, (
    f"selected {len(stage2_train_df)} images, expected {STAGE2_N} "
    f"(all manual labels). Check data/real_labels.csv row count."
)
games_covered = set(stage2_train_df["game"].unique())
assert games_covered == {"game8", "game9", "game10", "game11"}, (
    f"all 4 games must be represented; got {games_covered}"
)

stage2_train_path = f"{RESULTS_DIR}/stage2_train_manifest.csv"
stage2_train_df.to_csv(stage2_train_path, index=False)
print(f"\nSelected stage-2 training manifest (n={len(stage2_train_df)}):")
print(stage2_train_df[["game", "ply", "image_name", "fen"]].to_string(index=False))
print(f"\nwrote {stage2_train_path}")
print(f"per-game picks: {stage2_train_df['game'].value_counts().sort_index().to_dict()}")

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


train_dataset = ManualLabelsDataset(
    stage2_train_df, REAL_IMAGES_ROOT, transform=train_transform,
)
print(f"train_dataset (30 frames × 64): {len(train_dataset):,} samples "
      f"({stage2_train_df['game'].nunique()} games)")

# --- 5% slice of dataset_v1 for catastrophic-forgetting probe.
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

print("\033[92m✓ Cell 4 — Datasets — OK\033[0m")




# %% [Cell 5 — RealGameDataset (game7 monitor + games 2/4/5/6 test)]
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

print("\033[92m✓ Cell 5 — RealGameDataset (game7) — OK\033[0m")




# %% [Cell 6 — DataLoaders (NO weighted sampler — 1,920 samples is still small)]
def _worker_init_fn(worker_id):
    import random as _r
    worker_seed = SEED + worker_id
    _r.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS_REAL, pin_memory=True, persistent_workers=True,
    worker_init_fn=_worker_init_fn, drop_last=False,
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




# %% [Cell 7 — Build model + load v1 baseline weights]
def build_model():
    """ResNet18 with FC swapped to 13 classes. Random init — caller loads weights."""
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


model = build_model().to(DEVICE)

assert os.path.exists(PRETRAINED_CKPT), (
    f"pretrained checkpoint not found: {PRETRAINED_CKPT}. "
    f"The v1 baseline lives at zero_shot/results/best_synth.pt; do NOT "
    f"substitute zero_shot_v1.5/checkpoints/best_synth.pt."
)
src_ckpt = torch.load(PRETRAINED_CKPT, map_location=DEVICE, weights_only=False)
missing, unexpected = model.load_state_dict(src_ckpt["model_state_dict"], strict=True)
SRC_EPOCH = int(src_ckpt.get("epoch", -1))
SRC_SYNTH_VAL_ACC = float(src_ckpt.get("synth_val_acc", float("nan")))
SRC_REAL_VAL_ACC = float(src_ckpt.get("real_val_acc", float("nan")))
print(f"Loaded v1 baseline weights from {PRETRAINED_CKPT}")
print(f"  source epoch              = {SRC_EPOCH}")
print(f"  source synth_val_acc      = {SRC_SYNTH_VAL_ACC:.4f}")
print(f"  source real_val_acc       = {SRC_REAL_VAL_ACC:.4f}  (game7, at-that-epoch)")
print(f"  missing keys in load      = {len(missing)}")
print(f"  unexpected keys in load   = {len(unexpected)}")
assert len(missing) == 0 and len(unexpected) == 0, (
    "state_dict mismatch — architecture differs from the v1 baseline. "
    "Verify ResNet18+fc(13) is unchanged."
)

print("\033[92m✓ Cell 7 — Build model + load v1 baseline weights — OK\033[0m")




# %% [Cell 8 — Phase-A optimizer (head only) + loss]
def freeze_backbone(model):
    """Freeze conv1, bn1, layer1-4. Leave model.fc trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


freeze_backbone(model)
trainable_params = [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable_params)
n_total = sum(p.numel() for p in model.parameters())
print(f"Phase A: trainable {n_trainable:,} / {n_total:,} params  "
      f"({100 * n_trainable / n_total:.2f}%)  — fc only")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    trainable_params, lr=LR_HEAD_WARMUP, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
)
scheduler = None  # no scheduler in phase A
print(f"Phase A optim: SGD(lr={LR_HEAD_WARMUP}, momentum={MOMENTUM}, wd={WEIGHT_DECAY})")

print("\033[92m✓ Cell 8 — Phase-A optimizer + loss — OK\033[0m")




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

# [1b] Augmentation fires (same sample, two reads from train_dataset)
print("\n[1b] Augmentation-firing check:")
idx = 0
s1, _ = train_dataset[idx]
s2, _ = train_dataset[idx]
arr1 = s1.float().numpy()
arr2 = s2.float().numpy()
same_sample_diff = float(np.abs(arr1 - arr2).mean())
print(f"  mean |s1 - s2| reading train_dataset[{idx}] twice: {same_sample_diff:.4f}")
assert same_sample_diff > 0.01, (
    f"augmentations not firing — mean abs diff {same_sample_diff:.4f}"
)

# [1c] Aug visualization on the 30 chosen training frames (4×4 grid)
print("\n[1c] Augmentation visualization (30 chosen frames → aug_smoke_check.png):")
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
    f"Stage-2 augmented training crops "
    f"(jitter @{COLOR_JITTER_APPLY_PROB} → shear @{SHEAR_APPLY_PROB} → noise @{NOISE_APPLY_PROB})",
    fontsize=11,
)
plt.tight_layout()
aug_smoke_path = f"{PLOTS_DIR}/aug_smoke_check.png"
plt.savefig(aug_smoke_path, dpi=120)
plt.close()
print(f"  wrote {aug_smoke_path}")

# [1c-b] Sample plot of the 30 chosen training boards with labels overlaid.
# This is the "user can sanity-check before training" plot.
print("\n[1c-b] Stage2 train-sample plot (warped boards + labels):")
fig, axes = plt.subplots(5, 6, figsize=(24, 20))
for ax, (_, row) in zip(axes.ravel(), stage2_train_df.iterrows()):
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
plt.suptitle("Stage2 training set — all 30 manual-label frames with FEN labels overlaid", fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.96))
train_samples_path = f"{PLOTS_DIR}/stage2_train_samples.png"
plt.savefig(train_samples_path, dpi=120)
plt.close()
print(f"  wrote {train_samples_path}")

# [1d] Verify pretrained weights actually loaded (compare fc to a random fresh build)
print("\n[1d] Pretrained-weight load verification:")
print(f"  source ckpt:        {PRETRAINED_CKPT}")
print(f"  source epoch:       {SRC_EPOCH}")
print(f"  source synth_val:   {SRC_SYNTH_VAL_ACC:.4f}")
print(f"  source real_val:    {SRC_REAL_VAL_ACC:.4f}  (game7)")
torch.manual_seed(SEED + 99)
torch.cuda.manual_seed_all(SEED + 99)
fresh = build_model().to(DEVICE)
fc_diff = (model.fc.weight - fresh.fc.weight).abs().mean().item()
print(f"  mean |fc - fresh_fc| = {fc_diff:.4f}  (expect > 0.01 — proves weights loaded)")
assert fc_diff > 0.005, (
    f"model.fc.weight is too close to a random init (diff={fc_diff:.4f}). "
    f"v1 baseline weights may not have loaded."
)
del fresh
# Re-seed back for deterministic continuation.
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# [2] Pre-training synth-monitor + real_val check on the LOADED v1 baseline.
# These are the "before-FT" numbers used by the catastrophic-forgetting comparison.
print("\n[2] Pre-FT eval (these are the baseline numbers for forgetting):")
sm_loss_pre, sm_acc_pre, _, _ = evaluate(
    model, synth_monitor_loader, criterion, DEVICE, "synth_monitor")
rv_loss_pre, rv_acc_pre, _, _ = evaluate(
    model, real_val_loader, criterion, DEVICE, "real_val")
print(f"  synth_monitor (5% v1 slice, BEFORE FT): acc={sm_acc_pre:.4f}  loss={sm_loss_pre:.4f}")
print(f"  real_val (game7,             BEFORE FT): acc={rv_acc_pre:.4f}  loss={rv_loss_pre:.4f}")
assert sm_acc_pre > 0.95, (
    f"v1 baseline scored only {sm_acc_pre:.4f} on its own dataset (5% slice). "
    f"Expected > 0.95 since the baseline reported 0.9991 on its disjoint synth_test. "
    f"Something's wrong with the loaded weights or the synth-monitor pipeline."
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

# [4c] Phase-A vs Phase-B parameter check
print("\n[4c] Phase-A parameter check (only fc has requires_grad=True):")
fc_trainable = all(p.requires_grad for p in model.fc.parameters())
non_fc_frozen = all(
    (not p.requires_grad)
    for name, p in model.named_parameters() if not name.startswith("fc.")
)
print(f"  model.fc all trainable:   {fc_trainable}")
print(f"  non-fc all frozen:        {non_fc_frozen}")
assert fc_trainable and non_fc_frozen, "phase-A freeze logic is broken"

# Now hypothetically flip to phase B and check all params trainable
unfreeze_all(model)
all_trainable = all(p.requires_grad for p in model.parameters())
print(f"  after unfreeze_all(): all trainable = {all_trainable}")
assert all_trainable, "phase-B unfreeze logic is broken"
# Restore phase-A state for the training loop entry.
freeze_backbone(model)
fc_only = all(p.requires_grad for p in model.fc.parameters()) and all(
    (not p.requires_grad) for name, p in model.named_parameters() if not name.startswith("fc.")
)
assert fc_only, "could not restore phase-A freeze after unfreeze test"

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
      f"(expected < ln({NUM_CLASSES}) ≈ {math.log(NUM_CLASSES):.4f} since weights are pretrained)")

print("\n" + "=" * 64)
print("Smoke test passed. Ready for training.")
print("=" * 64)

print("\033[92m✓ Cell 10 — Smoke test — OK\033[0m")




# %% [Cell 11 — Training loop (Phase A → Phase B → early stop)]
# Re-load pretrained weights cleanly from disk (not from in-memory
# `src_ckpt`). This makes Cell 11 robust to kernel restart: a user can
# rerun from this cell onward without depending on Cell 7 having executed
# in the current session. Also resets any optimizer state from the smoke
# test's stage [4] optimizer.step().
model = build_model().to(DEVICE)
_src_ckpt_reload = torch.load(
    PRETRAINED_CKPT, map_location=DEVICE, weights_only=False,
)
assert "model_state_dict" in _src_ckpt_reload, (
    f"reload of {PRETRAINED_CKPT} missing 'model_state_dict' key — "
    f"checkpoint format unexpected."
)
missing, unexpected = model.load_state_dict(
    _src_ckpt_reload["model_state_dict"], strict=True,
)
assert len(missing) == 0 and len(unexpected) == 0, (
    f"state_dict mismatch on Cell 11 reload: "
    f"missing={len(missing)}, unexpected={len(unexpected)}"
)
print(f"[Cell 11 reload] loaded v1 baseline from {PRETRAINED_CKPT} "
      f"(epoch {_src_ckpt_reload.get('epoch', '?')}, "
      f"synth_val={_src_ckpt_reload.get('synth_val_acc', float('nan')):.4f})")
del _src_ckpt_reload

freeze_backbone(model)
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR_HEAD_WARMUP, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
)
scheduler = None  # phase-A has no scheduler

# Snapshot a backbone weight to assert it does not move during phase A.
_layer1_snapshot_phaseA = model.layer1[0].conv1.weight.detach().cpu().clone()

training_log = []
best_real_val_acc = -1.0
best_real_epoch = -1
best_synth_monitor_acc = -1.0
best_synth_monitor_epoch = -1
epochs_since_best_real = 0
stop_reason = "completed_all_epochs"
phase_b_started = False

CKPT_BEST_REAL = f"{CHECKPOINTS_DIR}/best_real.pt"
CKPT_BEST_SYNTH_MONITOR = f"{CHECKPOINTS_DIR}/best_synth_monitor.pt"
CKPT_LATEST = f"{CHECKPOINTS_DIR}/latest.pt"
LOG_CSV = f"{RESULTS_DIR}/training_log.csv"

t_total = time.perf_counter()

for epoch in range(1, NUM_EPOCHS + 1):
    phase = "A" if epoch <= PHASE_A_EPOCHS else "B"

    # --- Phase A → B transition at the START of epoch PHASE_A_EPOCHS + 1
    if phase == "B" and not phase_b_started:
        # Sanity: backbone has not moved during phase A.
        _layer1_now = model.layer1[0].conv1.weight.detach().cpu()
        _layer1_delta = (_layer1_now - _layer1_snapshot_phaseA).abs().max().item()
        print(f"\n[phase A→B] backbone layer1.0.conv1 max |Δ| during phase A: "
              f"{_layer1_delta:.2e}  (must be 0 — backbone was frozen)")
        assert _layer1_delta < 1e-7, (
            f"backbone moved during phase A (max |Δ|={_layer1_delta:.2e}). "
            f"freeze_backbone() is broken."
        )

        unfreeze_all(model)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=LR_FULL_FT, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
        )
        scheduler = None  # stage 2 skips the LR step — see Cell 2 note
        phase_b_started = True
        print(f"[phase A→B] all params unfrozen; "
              f"new SGD(lr={LR_FULL_FT}, momentum={MOMENTUM}, wd={WEIGHT_DECAY}); "
              f"no scheduler (stage 2 design choice)")

    print(f"\n{'='*64}")
    print(f"Epoch {epoch}/{NUM_EPOCHS}  phase={phase}  "
          f"lr={optimizer.param_groups[0]['lr']:.6f}")
    print(f"{'='*64}")
    t_ep = time.perf_counter()

    print("  [train]")
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, print_every=50)

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

    print(f"\n  Epoch {epoch:2d} (phase {phase}): "
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
phase_boundary_epoch = PHASE_A_EPOCHS + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(log_df["epoch"], log_df["train_acc"], label="train", marker="o", ms=4)
ax1.plot(log_df["epoch"], log_df["synth_val_acc"], label="synth_monitor (5% v1)",
         marker="s", ms=4)
ax1.plot(log_df["epoch"], log_df["real_val_acc"], label="real_val (game7)",
         marker="^", ms=4, linestyle="--", linewidth=2)
ax1.axvline(phase_boundary_epoch, color="r", linestyle=":", alpha=0.6,
            label=f"phase A→B (after ep{PHASE_A_EPOCHS})")
ax1.axvline(best_real_epoch, color="k", linestyle=":", alpha=0.4,
            label=f"best real_val (ep{best_real_epoch})")
ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy")
ax1.set_title("Accuracy")
ax1.set_ylim(-0.02, 1.02); ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

ax2.plot(log_df["epoch"], log_df["train_loss"], label="train", marker="o", ms=4)
ax2.plot(log_df["epoch"], log_df["synth_val_loss"], label="synth_monitor",
         marker="s", ms=4)
ax2.plot(log_df["epoch"], log_df["real_val_loss"], label="real_val (game7)",
         marker="^", ms=4, linestyle="--", linewidth=2)
ax2.axvline(phase_boundary_epoch, color="r", linestyle=":", alpha=0.6)
ax2.set_xlabel("epoch"); ax2.set_ylabel("loss")
ax2.set_title("Loss")
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
    ax.axvline(phase_boundary_epoch, color="r", linestyle=":", alpha=0.5)
    ax.set_title(f"{CLASS_SHORT[i]} — {name}", fontsize=10)
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    ax.axhline(0.0, color="r", linestyle=":", alpha=0.4)
for j in range(NUM_CLASSES, 16):
    flat_axes[j].axis("off")
fig.suptitle(
    "Per-class real_val (game7) accuracy over epochs "
    "(red dotted vert = phase A→B; red dotted horiz = 0%)",
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




# %% [Cell 17 — Held-out games: dataset + loader for games 2/4/5/6]
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




# %% [Cell 19 — Aggregate across held-out games (games 2/4/5/6)]
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
    f"Held-out aggregate (games 2/4/5/6) — per-square acc={combined_per_square_acc:.4f}",
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
# v1 zero-shot baseline reference numbers (read from disk).
BASELINE_PEAK_REAL_VAL_GAME7 = 0.5923   # epoch 5 of v1 (best obs.)
BASELINE_CKPT_REAL_VAL_GAME7 = SRC_REAL_VAL_ACC  # at-saved-epoch from the loaded ckpt

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
        "10 real frames + heavy aug did not measurably move the failing classes."
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
    "| class | v1 baseline | stage2_30 | Δ |\n"
    "|-------|------------:|----------:|---:|\n"
    + "\n".join(per_class_delta_lines)
)

# One-sentence verdict.
if v1_baseline_held_out and "per_square_acc" in v1_baseline_held_out:
    v1_held_out = float(v1_baseline_held_out["per_square_acc"])
    beat = combined_per_square_acc > v1_held_out
    verdict = (
        f"**Verdict:** stage2_30 {'BEAT' if beat else 'did NOT beat'} v1 zero-shot "
        f"on held-out (games 2/4/5/6) — "
        f"{combined_per_square_acc:.4f} vs. {v1_held_out:.4f} "
        f"(Δ = {combined_per_square_acc - v1_held_out:+.4f})."
    )
else:
    verdict = (
        f"**Verdict:** stage2_30 per-square acc on held-out (games 2/4/5/6) is "
        f"**{combined_per_square_acc:.4f}**. v1 zero-shot held-out aggregate not "
        f"available on disk — compare via `zero_shot/results/games_2_4_5_6_eval/` "
        f"if present, or re-evaluate the baseline checkpoint on this exact partition."
    )

# --- Stage 1 comparison block (inserted into summary BEFORE the v1 per-class
# delta block). Reads stage 1 results from disk if present; falls back to
# "n/a" rows otherwise. Per-class deltas vs. stage 1 are more directly useful
# than per-class deltas vs. zero-shot v1 for this run.
stage1_compare_lines = [
    "## Comparison to stage1_10 (10 real images)",
    "",
    "| metric | stage1_10 | stage2_30 | Δ |",
    "|--------|----------:|----------:|---:|",
]
s1_paths = {
    "game7":     f"{PROJECT_ROOT}/fine_tuning/stage1_10/results/game7_results.json",
    "agg":       f"{PROJECT_ROOT}/fine_tuning/stage1_10/results/held_out_aggregate.json",
    "synth":     f"{PROJECT_ROOT}/fine_tuning/stage1_10/results/synth_test_results.json",
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

# Per-class deltas vs. stage 1 (which we have on disk and is more
# directly useful than the v1 zero-shot deltas the existing block
# tries to compute).
if "agg" in s1 and "per_class_acc" in s1["agg"]:
    stage1_compare_lines.extend([
        "### Per-class delta vs. stage1_10 (aggregate held-out)",
        "",
        "| class | stage1_10 | stage2_30 | Δ |",
        "|-------|----------:|----------:|---:|",
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

summary_lines = [
    "# Fine-tuning stage 2 — 30 real images from games 8-11 (all manual labels)",
    "",
    "## Recipe (vs. v1 zero-shot baseline)",
    "- **Source weights:** v1 baseline (zero_shot/results/best_synth.pt). Cold-start "
    "(NOT stage 1 weights).",
    "- **Training data:** all 30 manual-label frames from game8/9/10/11.",
    f"- **Phase A** (epochs 1-{PHASE_A_EPOCHS}): freeze conv1/bn1/layer1-4; train fc only "
    f"@ lr={LR_HEAD_WARMUP}.",
    f"- **Phase B** (epochs {PHASE_A_EPOCHS+1}-{NUM_EPOCHS}): unfreeze all; "
    f"lr={LR_FULL_FT}; no scheduler (stage 2 design choice — stage 1's LR step "
    f"at epoch 21 produced no improvement).",
    f"- **Aug:** color jitter @{COLOR_JITTER_APPLY_PROB} → shear @{SHEAR_APPLY_PROB} "
    f"(±8°) → noise @{NOISE_APPLY_PROB} (std={NOISE_STD}).",
    f"- **Sampler:** shuffle=True (NO weighted sampler, 1,920 samples is still small).",
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
    f"- Synth-monitor acc BEFORE FT (loaded baseline): **{PRE_FT_SYNTH_MONITOR_ACC:.4f}**",
    f"- Synth-monitor acc AFTER FT  (best_real ckpt):  **{sm_acc:.4f}**",
    f"- **Catastrophic-forgetting Δ: {sm_acc - PRE_FT_SYNTH_MONITOR_ACC:+.4f}**",
    "",
    "## Game7 monitor (NOT held-out — used for checkpoint selection)",
    f"- Per-square at best_real:  **{rv_acc:.4f}**  "
    f"(before FT: {PRE_FT_REAL_VAL_ACC:.4f}; "
    f"v1 ckpt-epoch real_val: {BASELINE_CKPT_REAL_VAL_GAME7:.4f}; "
    f"v1 peak real_val: {BASELINE_PEAK_REAL_VAL_GAME7:.4f})",
    f"- Improvement over loaded baseline: "
    f"**{rv_acc - PRE_FT_REAL_VAL_ACC:+.4f}**",
    f"- Per-board acc: {n_all_correct}/{n_boards} = {n_all_correct/max(n_boards,1):.4f}",
    f"- Mean squares correct/board: {mean_squares_correct:.2f}/64",
    "",
    "## Held-out real test (games 2, 4, 5, 6) — same partition as zero-shot",
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
    f"- {moved_sentence}",
    "- See plots/per_class_real_val.png for the 13-class trajectory.",
    "",
    "## Artifacts",
    "- `checkpoints/best_real.pt` "
    f"(epoch {best_real_epoch}, real_val_acc={best_real_val_acc:.4f}) — headline ckpt",
    "- `checkpoints/best_synth_monitor.pt` "
    f"(epoch {best_synth_monitor_epoch}, synth_monitor_acc={best_synth_monitor_acc:.4f}) — monitor-only",
    "- `checkpoints/latest.pt`",
    "- `results/stage2_train_manifest.csv` — the 30 manual-label frames",
    "- `results/training_log.csv` — per-epoch log + 13 per-class real_val columns",
    "- `results/synth_test_results.json` (catastrophic-forgetting probe)",
    "- `results/game7_results.json`",
    "- `results/game{2,4,5,6}_results.json`",
    "- `results/held_out_aggregate.json`",
    "- `results/predictions/*.npy`",
    "- `plots/aug_smoke_check.png`, `stage2_train_samples.png`",
    "- `plots/training_curves.png`, `per_class_real_val.png`",
    "- `plots/synth_test_cm.png`, `game7_cm.png`, `game{2,4,5,6}_cm.png`, `aggregate_cm.png`",
    "- `plots/game{2,4,5,6}_qualitative.png`",
]
summary_text = "\n".join(summary_lines)
Path(f"{RESULTS_DIR}/summary.md").write_text(summary_text)
print(summary_text)
print(f"\nwrote {RESULTS_DIR}/summary.md")

print("\033[92m✓ Cell 21 — Summary.md — OK\033[0m")

# %%
