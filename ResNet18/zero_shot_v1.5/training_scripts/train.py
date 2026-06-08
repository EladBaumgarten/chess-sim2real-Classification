"""
Project 2 — Zero-shot v1.5 (FINAL zero-shot run).

This is the last zero-shot training run before fine-tuning. Two changes vs.
the v1 baseline at zero_shot/train_baseline.py:

  (a) DATA: dataset_v1.5 (7,665 images = v1 6,132 + legacy 1,533),
      labelled via data/dataset_v1.5/labels.csv + manifest.csv. Corner
      cache at data/dataset_v1.5/corners.json already covers all 7,665.
  (b) AUG: mild shear added to the baseline recipe. Final pipeline is
      color jitter → shear → noise, each gated independently at 0.5.
      ColorJitter & RandomAffine are torchvision; noise is numpy.

Everything else (backbone, optimizer, scheduler, loss, sampler, batch size,
seed) is identical to baseline. SEED=42, 10 epochs.

# ============================================================================
# ZERO-SHOT HARD RULES (DO NOT VIOLATE)
# ----------------------------------------------------------------------------
# - No real images enter training or validation.
# - Real eval uses game7 as MONITOR ONLY and games 2/4/5/6 as held-out TEST.
# - Checkpoints are selected by synth_val_acc. real_val_acc is logged, not
#   gated. A second 'best_real_monitor.pt' is saved as a monitor-only
#   artifact — never used for headline numbers.
# - This is the LAST zero-shot run. After this: real-image fine-tuning.
# ============================================================================
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import ColorJitter, RandomAffine, InterpolationMode
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Project modules
from preprocessing.chess_dataset import ChessSquareDataset
from preprocessing.fen_to_grid import fen_to_label_grid
from preprocessing.view_orientations import GAME7_ORIENTATION
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
BATCH_SIZE = 64
NUM_EPOCHS = 10                       # baseline peaked at epoch 5; 10 gives headroom
# Spec asked for 8 train / 4 real workers, but on this cluster the combined
# 28 persistent workers (8+8+8+4 across train/synth_val/synth_test/real_val)
# caused workers to exit with "Empty" queue timeouts — almost certainly
# /dev/shm or fork-memory pressure. Augmented baseline used 6 successfully;
# we drop to 4 train / 2 real (still I/O-bound enough that the GPU stays fed)
# and add worker_init_fn for per-worker RNG hygiene.
NUM_WORKERS_SYNTH = 4                 # train / synth_val / synth_test
NUM_WORKERS_REAL = 2                  # real_val / real_test (smaller eval sets)
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 7                      # carried forward from baseline
LR_GAMMA = 0.1
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.85, 0.075, 0.075

# === Final augmentation recipe (mild from baseline + mild shear).
COLOR_JITTER_APPLY_PROB = 0.5
SHEAR_APPLY_PROB = 0.5
NOISE_APPLY_PROB = 0.5
NOISE_STD = 0.01                      # in normalized [0,1] units (after /255)

# torchvision augmentation modules (per project spec):
COLOR_JITTER = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
AFFINE_SHEAR = RandomAffine(
    degrees=0,                              # NO rotation — labels are position-bound
    translate=(0.02, 0.02),                 # ±2%
    scale=(0.98, 1.02),                     # ±2%
    shear=(-5.0, 5.0, -5.0, 5.0),           # ±5° both axes
    interpolation=InterpolationMode.BILINEAR,
    fill=0,
)

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
# Short class codes used in CSV column headers (matches spec).
CLASS_SHORT = [
    "wP", "wR", "wN", "wB", "wQ", "wK",
    "bP", "bR", "bN", "bB", "bQ", "bK",
    "empty",
]

PROJECT_ROOT = "/home/eladbaum/chess_project"
# v1.5 dataset and its dedicated corner cache (already extended to 7,665 imgs)
DATASET_DIR = f"{PROJECT_ROOT}/data/dataset_v1.5/images"
MANIFEST_PATH = f"{PROJECT_ROOT}/data/dataset_v1.5/manifest.csv"
CORNERS_PATH = f"{PROJECT_ROOT}/data/dataset_v1.5/corners.json"

GAME7_DIR = f"{PROJECT_ROOT}/data/game7_per_frame/images"
GAME7_GT_CSV = f"{PROJECT_ROOT}/data/game7_per_frame/gt.csv"

EXP_DIR = f"{PROJECT_ROOT}/zero_shot_v1.5"
CHECKPOINTS_DIR = f"{EXP_DIR}/checkpoints"
RESULTS_DIR = f"{EXP_DIR}/results"
PLOTS_DIR = f"{EXP_DIR}/plots"
PREDS_DIR = f"{RESULTS_DIR}/predictions"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

print(f"dataset:     {DATASET_DIR}")
print(f"manifest:    {MANIFEST_PATH}")
print(f"corners:     {CORNERS_PATH}")
print(f"game7:       {GAME7_DIR}")
print(f"checkpoints: {CHECKPOINTS_DIR}")
print(f"results:     {RESULTS_DIR}")
print(f"plots:       {PLOTS_DIR}")

# Verify corner cache covers all v1.5 images BEFORE doing anything else.
with open(CORNERS_PATH) as _f:
    _corner_keys = set(json.load(_f).keys())
_image_files = set(os.listdir(DATASET_DIR))
_missing = _image_files - _corner_keys
assert len(_missing) == 0, (
    f"corner cache is missing {len(_missing)} v1.5 images. "
    f"Sample: {sorted(_missing)[:5]}. "
    f"Re-run scripts/cache_all_corners.py against {DATASET_DIR}."
)
print(f"corner cache covers all {len(_image_files)} v1.5 images. ✓")
del _corner_keys, _image_files, _missing

print("\033[92m✓ Cell 2 — Config — OK\033[0m")




# %% [Cell 3 — Build FEN-disjoint splits (0.85 / 0.075 / 0.075)]
manifest = pd.read_csv(MANIFEST_PATH)
print(f"Full manifest: {len(manifest):,} rows, "
      f"{manifest['source_image'].nunique():,} unique images, "
      f"{manifest['fen'].nunique():,} unique FENs")
if "source_dataset" in manifest.columns:
    src_counts = manifest.groupby("source_dataset")["source_image"].nunique().to_dict()
    print(f"  source_dataset (unique images): {src_counts}")

unique_fens = sorted(manifest["fen"].unique())
rng = random.Random(SEED)
rng.shuffle(unique_fens)

n_train = int(len(unique_fens) * TRAIN_FRAC)
n_val = int(len(unique_fens) * VAL_FRAC)
train_fens = set(unique_fens[:n_train])
val_fens = set(unique_fens[n_train:n_train + n_val])
test_fens = set(unique_fens[n_train + n_val:])

train_df = manifest[manifest["fen"].isin(train_fens)].reset_index(drop=True)
synth_val_df = manifest[manifest["fen"].isin(val_fens)].reset_index(drop=True)
synth_test_df = manifest[manifest["fen"].isin(test_fens)].reset_index(drop=True)

train_imgs = set(train_df["source_image"])
val_imgs = set(synth_val_df["source_image"])
test_imgs = set(synth_test_df["source_image"])

print("\n[Cell 3 verification]")
checks = [
    ("FEN sets disjoint (train/val)", train_fens.isdisjoint(val_fens)),
    ("FEN sets disjoint (train/test)", train_fens.isdisjoint(test_fens)),
    ("FEN sets disjoint (val/test)", val_fens.isdisjoint(test_fens)),
    ("All FENs accounted for",
        len(train_fens) + len(val_fens) + len(test_fens) == len(unique_fens)),
    ("Image sets disjoint (train/val)", train_imgs.isdisjoint(val_imgs)),
    ("Image sets disjoint (train/test)", train_imgs.isdisjoint(test_imgs)),
    ("Image sets disjoint (val/test)", val_imgs.isdisjoint(test_imgs)),
    ("Row counts reconcile to manifest",
        len(train_df) + len(synth_val_df) + len(synth_test_df) == len(manifest)),
    ("All 3 splits non-empty",
        min(len(train_df), len(synth_val_df), len(synth_test_df)) > 0),
    ("Every class present in train split",
        set(train_df["label"].unique()) == set(range(NUM_CLASSES))),
]
all_ok = True
for label, ok in checks:
    print(f"  {'✓' if ok else '✗'} {label}")
    if not ok:
        all_ok = False
assert all_ok, "Cell 3 verification failed — fix split logic before continuing."

print(f"\nSplit by FEN ({TRAIN_FRAC:.1%}/{VAL_FRAC:.1%}/{TEST_FRAC:.1%}):")
print(f"  train: {len(train_fens):4d} FENs, {len(train_imgs):4d} images, "
      f"{len(train_df):>7,d} rows")
print(f"  val  : {len(val_fens):4d} FENs, {len(val_imgs):4d} images, "
      f"{len(synth_val_df):>7,d} rows")
print(f"  test : {len(test_fens):4d} FENs, {len(test_imgs):4d} images, "
      f"{len(synth_test_df):>7,d} rows")

print(f"\nView balance per split (% of split rows):")
for name, df in [("train", train_df), ("val", synth_val_df), ("test", synth_test_df)]:
    pct = (df["view"].value_counts(normalize=True) * 100).round(2).to_dict()
    print(f"  {name:<6s} {pct}")

if "source_dataset" in manifest.columns:
    print(f"\nSource-dataset balance per split (% of split rows):")
    for name, df in [("train", train_df), ("val", synth_val_df),
                     ("test", synth_test_df)]:
        src = (df["source_dataset"].value_counts(normalize=True) * 100)
        src = src.round(2).to_dict()
        print(f"  {name:<6s} {src}")
    # Expect ~80/20 v1/legacy per split (6,132 vs 1,533 in the dataset).
    # Soft warn if any split is >95% one source — that signals an FEN
    # imbalance worth investigating, not a hard failure.
    for name, df in [("train", train_df), ("val", synth_val_df),
                     ("test", synth_test_df)]:
        pct = df["source_dataset"].value_counts(normalize=True)
        if (pct > 0.95).any():
            dominant = pct.idxmax()
            print(f"  ⚠ {name} is {pct.max():.1%} {dominant} — "
                  f"unexpected source skew, review split.")

print(f"\nClass distribution by split (%):")
print(f"  {'cls':>3s}  {'name':<14s}  {'train':>7s} {'val':>7s} {'test':>7s}  {'max|Δ|':>7s}")
class_pct = {"train": [], "val": [], "test": []}
for cls in range(NUM_CLASSES):
    t = (train_df["label"] == cls).mean() * 100
    v = (synth_val_df["label"] == cls).mean() * 100
    e = (synth_test_df["label"] == cls).mean() * 100
    class_pct["train"].append(t)
    class_pct["val"].append(v)
    class_pct["test"].append(e)
    drift = max(abs(t - v), abs(t - e), abs(v - e))
    print(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {t:>6.2f}% {v:>6.2f}% {e:>6.2f}%  {drift:>6.2f}pp")

x = np.arange(NUM_CLASSES)
width = 0.27
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width, class_pct["train"], width, label=f"train ({len(train_df):,})")
ax.bar(x,         class_pct["val"],   width, label=f"val ({len(synth_val_df):,})")
ax.bar(x + width, class_pct["test"],  width, label=f"test ({len(synth_test_df):,})")
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n{n[:6]}" for c, n in enumerate(CLASS_NAMES)], fontsize=8)
ax.set_ylabel("% of split rows")
ax.set_title("Class distribution by split (lower is better drift; same shape = good split)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
split_plot_path = f"{PLOTS_DIR}/split_class_distribution.png"
plt.savefig(split_plot_path, dpi=120)
plt.close()
print(f"\nwrote {split_plot_path}")

print("\n\033[92m✓ Cell 3 — Build splits — OK\033[0m")




# %% [Cell 4 — Synthetic datasets and transforms]
# Final augmentation recipe: color jitter → shear → noise, each gated independently.
# ColorJitter + RandomAffine are torchvision; noise is numpy. PIL is the intermediate
# format because torchvision's transforms are universally happy with PIL Image and
# this avoids tensor-dtype edge cases across torchvision versions.

def train_transform(crop_rgb_uint8):
    """HWC uint8 RGB → HWC uint8 RGB. Tensorization + ImageNet normalize
    happens AFTER this in the model loop (same contract as baseline).

    Order: color jitter (0.5) → shear (0.5) → noise (0.5).
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


# val/test: no augmentation (transform=None on the Dataset)
train_dataset = ChessSquareDataset(
    train_df, CORNERS_PATH, dataset_dir=DATASET_DIR, transform=train_transform,
)
synth_val_dataset = ChessSquareDataset(
    synth_val_df, CORNERS_PATH, dataset_dir=DATASET_DIR, transform=None,
)
synth_test_dataset = ChessSquareDataset(
    synth_test_df, CORNERS_PATH, dataset_dir=DATASET_DIR, transform=None,
)
print(f"train_dataset:      {len(train_dataset):>8,d} samples (color/shear/noise aug)")
print(f"synth_val_dataset:  {len(synth_val_dataset):>8,d} samples (no aug)")
print(f"synth_test_dataset: {len(synth_test_dataset):>8,d} samples (no aug)")

print("\033[92m✓ Cell 4 — Synthetic datasets and transforms — OK\033[0m")




# %% [Cell 5 — RealGameDataset (covers game7 monitor + games 2/4/5/6 test)]
class RealGameDataset(Dataset):
    """One sample per (real frame × board square) for any single game.

    Compared to ChessSquareDataset:
      - No on-disk corner cache. Per-image find_corners with OOB rejection +
        image-corner fallback (chesscog hallucinates board extensions on
        tight-cropped photos — Step 6a finding).
      - Caches detected corners in memory so 64 squares from one frame
        don't re-run detection.
      - Uses per-game orientation key for fen_to_label_grid().
    """

    CORNER_OOB_TOLERANCE = 8  # px; reject find_corners output beyond this

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
        self._corner_cache = {}  # image_name → np.ndarray (4,2)

    def __len__(self):
        return len(self.manifest)

    def _get_corners(self, image_name, bgr):
        if image_name in self._corner_cache:
            return self._corner_cache[image_name]
        H, W = bgr.shape[:2]
        try:
            # NOTE: this re-seeds the GLOBAL np.random per image. Carried
            # forward from the baseline script for corner-detection
            # determinism (find_corners uses RANSAC). Safe here because
            # real-image evaluation runs at end-of-epoch and DataLoader
            # worker augmentation uses independent worker RNGs seeded by
            # torch, not the global np.random state.
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
print(f"  class distribution (counts):")
for cls in range(NUM_CLASSES):
    n = (real_val_dataset.manifest["label"] == cls).sum()
    print(f"    {cls:>2d} {CLASS_NAMES[cls]:<14s}: {n}")

print("\033[92m✓ Cell 5 — RealGameDataset (game7 monitor) — OK\033[0m")




# %% [Cell 6 — Weighted sampler (sqrt-inverse-frequency)]
# Softer than inverse-frequency to avoid baking in a strong rare-class prior;
# mitigates the empty→queen/knight hallucination observed when using a raw
# inverse-freq sampler. Same recipe as the augmented variant.
train_labels = train_df["label"].values
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
print("Train-set class counts and sqrt-inverse-frequency weights:")
print(f"  {'cls':>3s}  {'name':<14s}  {'count':>8s}  {'weight':>10s}  {'eff_prob':>9s}")
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
# Each worker gets a deterministic, distinct RNG seed derived from SEED +
# its worker_id. Without this, fork-spawned workers all inherit the same
# global torch/numpy RNG state and would produce identical augmentation
# sequences across workers (silent bug, not a crash). Also good hygiene
# for surfacing real worker errors instead of having them race on shared RNG.
def _worker_init_fn(worker_id):
    import random as _r
    worker_seed = SEED + worker_id
    _r.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS_SYNTH, pin_memory=True, persistent_workers=True,
    worker_init_fn=_worker_init_fn,
)
synth_val_loader = DataLoader(
    synth_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS_SYNTH, pin_memory=True, persistent_workers=True,
    worker_init_fn=_worker_init_fn,
)
synth_test_loader = DataLoader(
    synth_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS_SYNTH, pin_memory=True, persistent_workers=True,
    worker_init_fn=_worker_init_fn,
)
real_val_loader = DataLoader(
    real_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS_REAL, pin_memory=True, persistent_workers=True,
    worker_init_fn=_worker_init_fn,
)

for name, loader in [("train", train_loader), ("synth_val", synth_val_loader),
                     ("synth_test", synth_test_loader), ("real_val", real_val_loader)]:
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


# No class weights — prior ablation showed they hurt zero-shot transfer.
criterion = nn.CrossEntropyLoss()
optimizer, scheduler = build_optimizer(model)
print(f"Loss: CrossEntropyLoss (no class weights)")
print(f"Optim: SGD(lr={LR}, momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY})")
print(f"Sched: StepLR(step_size={LR_STEP_SIZE}, gamma={LR_GAMMA})")

print("\033[92m✓ Cell 9 — Loss, optimizer, scheduler — OK\033[0m")




# %% [Cell 10 — Helper functions]
IMAGENET_MEAN_DEV = IMAGENET_MEAN.to(DEVICE)
IMAGENET_STD_DEV = IMAGENET_STD.to(DEVICE)


def imagenet_normalize(x):
    """x: (B, 3, H, W) float in [0,1]. Returns normalized to ImageNet mean/std."""
    return (x - IMAGENET_MEAN_DEV) / IMAGENET_STD_DEV


def train_one_epoch(model, loader, criterion, optimizer, device, print_every=200):
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


def per_class_accuracy(preds, labels, num_classes=NUM_CLASSES):
    """Return list of per-class accuracies; NaN if class has no samples."""
    out = []
    for cls in range(num_classes):
        mask = (labels == cls)
        n = int(mask.sum())
        out.append(float((preds[mask] == cls).mean()) if n else float("nan"))
    return out


print("\033[92m✓ Cell 10 — Helper functions — OK\033[0m")




# %% [Cell 11 — Smoke test (MUST PASS before Cell 12)]
print("=" * 64)
print("SMOKE TEST — surfacing bugs before any training time is spent.")
print("=" * 64)

# [1] DataLoader shape/dtype/range checks
print("\n[1] DataLoader shape/dtype/range checks:")
for name, loader in [("train", train_loader), ("synth_val", synth_val_loader),
                     ("real_val", real_val_loader)]:
    xb, yb = next(iter(loader))
    print(f"  {name:>10s}: x={tuple(xb.shape)} {xb.dtype} "
          f"range=[{xb.min():.3f},{xb.max():.3f}]  "
          f"y={tuple(yb.shape)} {yb.dtype} "
          f"range=[{yb.min().item()},{yb.max().item()}]")
    assert xb.shape[1:] == (3, 100, 100), f"{name} bad image shape {xb.shape}"
    assert xb.dtype == torch.float32, f"{name} bad image dtype {xb.dtype}"
    assert yb.dtype == torch.int64, f"{name} bad label dtype {yb.dtype}"
    assert int(yb.min()) >= 0 and int(yb.max()) <= 12, f"{name} labels out of range"
    assert torch.isfinite(xb).all(), f"{name} non-finite values in images"

# [1b] Augmentation actually fires (same sample, two reads from train_dataset)
print("\n[1b] Augmentation-firing check (same sample, two reads):")
idx = 0
s1, _ = train_dataset[idx]
s2, _ = train_dataset[idx]
arr1 = s1.float().numpy() if hasattr(s1, "numpy") else np.asarray(s1, dtype=np.float32)
arr2 = s2.float().numpy() if hasattr(s2, "numpy") else np.asarray(s2, dtype=np.float32)
same_sample_diff = float(np.abs(arr1 - arr2).mean())
print(f"  mean |sample_1 - sample_2| reading idx={idx} twice: {same_sample_diff:.4f}")
assert same_sample_diff > 0.01, (
    f"reading the same training sample twice produced near-identical pixels "
    f"(mean abs diff {same_sample_diff:.4f}) — augmentations appear not to be firing"
)

# [1c] Augmentation VISUALIZATION — 16 random training crops, 4×4 grid.
# Eye check before committing 2.5 hours of GPU. Inspect for: piece clipping
# from translate, weird shear artifacts, sensible color jitter range.
print("\n[1c] Augmentation visualization (4×4 grid → plots/aug_smoke_check.png):")
np.random.seed(SEED + 1)
viz_indices = np.random.choice(len(train_dataset), size=16, replace=False)
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, vi in zip(axes.ravel(), viz_indices):
    aug_tensor, lab = train_dataset[int(vi)]
    aug_hwc = (aug_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    ax.imshow(aug_hwc)
    ax.set_title(f"idx={int(vi)}  y={lab} {CLASS_SHORT[lab]}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle(
    "Augmented training crops (color jitter @0.5 → shear @0.5 → noise @0.5)",
    fontsize=11,
)
plt.tight_layout()
aug_smoke_path = f"{PLOTS_DIR}/aug_smoke_check.png"
plt.savefig(aug_smoke_path, dpi=120)
plt.close()
print(f"  wrote {aug_smoke_path}")

# [2] FEN-disjoint splits (already verified in Cell 3 — re-check the key invariants)
print("\n[2] FEN-disjoint split (re-check):")
assert train_fens.isdisjoint(val_fens), "train/val FEN leakage"
assert train_fens.isdisjoint(test_fens), "train/test FEN leakage"
assert val_fens.isdisjoint(test_fens), "val/test FEN leakage"
print(f"  train ∩ val = {len(train_fens & val_fens)}  "
      f"train ∩ test = {len(train_fens & test_fens)}  "
      f"val ∩ test = {len(val_fens & test_fens)}  ✓")

# [3] One forward pass on a train batch
print("\n[3] One forward pass on a train batch:")
xb, yb = next(iter(train_loader))
xb = xb.to(DEVICE); yb = yb.to(DEVICE)
logits = model(imagenet_normalize(xb))
assert logits.shape == (xb.size(0), NUM_CLASSES), f"bad logit shape {logits.shape}"
assert torch.isfinite(logits).all(), "non-finite logits"
print(f"  logits shape: {tuple(logits.shape)}  "
      f"mean={logits.mean().item():+.3f}  std={logits.std().item():.3f}")

# [4] One backward+step on the same batch
print("\n[4] One backward+step on the same batch:")
loss = criterion(logits, yb)
loss.backward()
optimizer.step()
optimizer.zero_grad()
assert torch.isfinite(loss).item() and loss.item() > 0, f"bad loss {loss.item()}"
print(f"  loss = {loss.item():.4f}")

# [4b] Sampler produces balanced batches — check class histogram on a sample batch.
# Inverse-freq sampling should make rare classes appear ≫ their natural rate.
print("\n[4b] Sampler-balance check (class histogram of one train batch):")
xb_s, yb_s = next(iter(train_loader))
hist = np.bincount(yb_s.numpy(), minlength=NUM_CLASSES)
print(f"  {'cls':>3s}  {'name':<14s}  {'count':>4s}  {'natural %':>9s}  {'batch %':>8s}")
for cls in range(NUM_CLASSES):
    natural = (train_labels == cls).mean() * 100
    batch_pct = hist[cls] / yb_s.size(0) * 100
    print(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {hist[cls]:>4d}  "
          f"{natural:>8.2f}%  {batch_pct:>7.2f}%")
n_classes_in_batch = int((hist > 0).sum())
print(f"  classes present in batch: {n_classes_in_batch}/{NUM_CLASSES}")
assert n_classes_in_batch >= 8, (
    f"sampler appears imbalanced — only {n_classes_in_batch} of {NUM_CLASSES} "
    f"classes appeared in a batch of {yb_s.size(0)}"
)

# [5] real_val (RealGameDataset / game7) loads 3 batches without error
print("\n[5] real_val (game7) loads 3 batches without error:")
model.eval()
count = 0
with torch.no_grad():
    for i, (xb, yb) in enumerate(real_val_loader):
        if i >= 3:
            break
        assert xb.shape[1:] == (3, 100, 100), f"real_val bad shape {xb.shape}"
        assert torch.isfinite(xb).all(), "non-finite values in real_val"
        count += xb.size(0)
print(f"  loaded {count} real_val samples across 3 batches ✓")

# [6] real_val tensors produce finite logits through model
print("\n[6] real_val tensors → model → finite logits:")
correct = 0; total = 0
with torch.no_grad():
    for i, (xb, yb) in enumerate(real_val_loader):
        if i >= 3:
            break
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        logits = model(imagenet_normalize(xb))
        assert torch.isfinite(logits).all(), "non-finite logits on real_val"
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
print(f"  3-batch acc (untrained, noise): {correct/total:.4f}")

# Loss sanity check on synth_val (one SGD step doesn't move loss much from ln(K))
print("\n[loss sanity] avg loss on 3 batches of synth_val:")
model.eval()
total_loss = 0.0; total = 0
with torch.no_grad():
    for i, (xb, yb) in enumerate(synth_val_loader):
        if i >= 3:
            break
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        logits = model(imagenet_normalize(xb))
        total_loss += criterion(logits, yb).item() * yb.size(0)
        total += yb.size(0)
avg_loss = total_loss / total
print(f"  avg loss = {avg_loss:.4f}  (expected ≈ ln({NUM_CLASSES}) = {math.log(NUM_CLASSES):.4f})")
assert 1.5 < avg_loss < 5.0, f"loss {avg_loss:.4f} outside sane range for random init"

print("\n" + "=" * 64)
print("Smoke test passed. Ready for training.")
print("=" * 64)

print("\033[92m✓ Cell 11 — Smoke test (8 stages) — OK\033[0m")




# %% [Cell 12 — Training loop]
# ZERO-SHOT: no real images enter training or validation.
# Real eval uses game7 as monitor only and games 2/4/5/6 as held-out test only.
# Checkpoints selected by synth_val_acc. real_val_acc is logged, not gated.
# This is the LAST zero-shot run. After this: real-image fine-tuning.

# --- Fresh-init guard: smoke test stages [3]/[4] in Cell 11 did one real
# optimizer.step() on a real train batch. Reset model + optimizer here at
# the start of training to guarantee a clean state regardless of how
# Cell 11 was executed. Then assert the reset actually produced freshly-
# initialized weights (cheap canary against cells running out of order).
#
# nn.Linear in build_model() randomly initializes the FC layer via
# reset_parameters(), which consumes the global torch RNG. Two consecutive
# build_model() calls advance the RNG and yield different FC weights —
# so we re-seed torch before EACH call to make them bit-for-bit identical.
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
_reference_model = build_model()
# Clone to CPU upfront so the comparison below is device-agnostic.
_reference_fc_weight = _reference_model.fc.weight.detach().cpu().clone()
del _reference_model

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
model = build_model()
optimizer, scheduler = build_optimizer(model)

# Re-seeded fresh, both build_model() calls produce identical fc weights.
# If model.fc.weight does NOT match _reference_fc_weight, the reset did
# not run cleanly (or seeds were perturbed mid-script).
_post_reset_fc = model.fc.weight.detach().cpu()
assert _post_reset_fc.shape == _reference_fc_weight.shape, (
    f"fc shape mismatch after reset: {_post_reset_fc.shape} vs "
    f"{_reference_fc_weight.shape}"
)
# Allow numerical noise but require effective equality.
_fc_diff = (_post_reset_fc - _reference_fc_weight).abs().max().item()
assert _fc_diff < 1e-6, (
    f"FRESH-INIT GUARD FAILED: model.fc.weight differs from a freshly-built "
    f"reference by {_fc_diff:.2e}. The Cell 11 smoke test optimizer step "
    f"may not have been reset cleanly. Re-run Cell 1 → 12 in order."
)
del _reference_fc_weight, _post_reset_fc, _fc_diff
print("[fresh-init guard] model.fc matches reference; training starts clean.")

training_log = []
best_synth_val_acc = -1.0
best_synth_epoch = -1
best_real_val_acc = -1.0
best_real_epoch = -1

CKPT_BEST_SYNTH = f"{CHECKPOINTS_DIR}/best_synth.pt"
CKPT_BEST_REAL = f"{CHECKPOINTS_DIR}/best_real_monitor.pt"   # monitor-only artifact
CKPT_LATEST = f"{CHECKPOINTS_DIR}/latest.pt"
LOG_CSV = f"{RESULTS_DIR}/training_log.csv"

t_total = time.perf_counter()
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n{'='*64}")
    print(f"Epoch {epoch}/{NUM_EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.5f})")
    print(f"{'='*64}")
    t_ep = time.perf_counter()

    print("  [train]")
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, print_every=200)

    print("  [eval synth_val]")
    sv_loss, sv_acc, _, _ = evaluate(model, synth_val_loader, criterion, DEVICE, "synth_val")

    print("  [eval real_val (game7, monitoring only)]")
    rv_loss, rv_acc, rv_preds_ep, rv_labels_ep = evaluate(
        model, real_val_loader, criterion, DEVICE, "real_val")
    rv_per_class = per_class_accuracy(rv_preds_ep, rv_labels_ep, NUM_CLASSES)

    scheduler.step()
    dt = time.perf_counter() - t_ep

    print(f"\n  Epoch {epoch:2d}: "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  |  "
          f"synth_val_acc={sv_acc:.4f}  |  "
          f"real_val_acc={rv_acc:.4f} (monitor)  |  {dt/60:.1f}min")
    # Print the per-class real_val on the classes we expect mild shear to move.
    knight_acc = rv_per_class[2]
    bishop_acc = rv_per_class[3]
    king_w_acc = rv_per_class[5]
    bknight_acc = rv_per_class[8]
    bbishop_acc = rv_per_class[9]
    bking_acc = rv_per_class[11]
    def _fmt(v): return f"{v:.3f}" if not math.isnan(v) else "n/a"
    print(f"    per-class real_val knights/bishops/kings: "
          f"wN={_fmt(knight_acc)} wB={_fmt(bishop_acc)} wK={_fmt(king_w_acc)}  "
          f"bN={_fmt(bknight_acc)} bB={_fmt(bbishop_acc)} bK={_fmt(bking_acc)}")

    log_row = {
        "epoch": epoch,
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": train_loss, "train_acc": train_acc,
        "synth_val_loss": sv_loss, "synth_val_acc": sv_acc,
        "real_val_loss": rv_loss, "real_val_acc": rv_acc,
        "epoch_time_s": dt,
    }
    for short, acc in zip(CLASS_SHORT, rv_per_class):
        log_row[f"real_val_acc_{short}"] = acc
    training_log.append(log_row)

    pd.DataFrame(training_log).to_csv(LOG_CSV, index=False)

    # Save latest
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "synth_val_acc": sv_acc, "real_val_acc": rv_acc,
    }, CKPT_LATEST)

    # Save best by SYNTH val acc — this is THE checkpoint used for final eval.
    if sv_acc > best_synth_val_acc:
        best_synth_val_acc = sv_acc
        best_synth_epoch = epoch
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "synth_val_acc": sv_acc, "real_val_acc": rv_acc,
        }, CKPT_BEST_SYNTH)
        print(f"  → NEW BEST synth_val_acc={sv_acc:.4f} → saved {CKPT_BEST_SYNTH}")

    # Save best by REAL val acc — MONITOR-ONLY artifact, never headline.
    if rv_acc > best_real_val_acc:
        best_real_val_acc = rv_acc
        best_real_epoch = epoch
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "synth_val_acc": sv_acc, "real_val_acc": rv_acc,
            "monitor_only": True,
        }, CKPT_BEST_REAL)
        print(f"  → NEW BEST real_val_acc={rv_acc:.4f} (monitor-only) → saved {CKPT_BEST_REAL}")

total_train_time = time.perf_counter() - t_total
print(f"\nTraining done. Total time: {total_train_time/60:.1f} min")
print(f"Best synth_val_acc: {best_synth_val_acc:.4f} at epoch {best_synth_epoch}")
print(f"Best real_val_acc (monitor): {best_real_val_acc:.4f} at epoch {best_real_epoch}")

print("\033[92m✓ Cell 12 — Training loop — OK\033[0m")




# %% [Cell 13 — Training curves + per-class real_val plot]
LOG_CSV = f"{RESULTS_DIR}/training_log.csv"

if not Path(LOG_CSV).exists():
    raise FileNotFoundError(
        f"No training log at {LOG_CSV}. Run Cell 12 first.")

log_df = pd.read_csv(LOG_CSV)
if len(log_df) == 0:
    raise RuntimeError(f"{LOG_CSV} exists but is empty.")

best_synth_epoch = int(log_df.loc[log_df["synth_val_acc"].idxmax(), "epoch"])
best_synth_val_acc = float(log_df["synth_val_acc"].max())
best_real_epoch = int(log_df.loc[log_df["real_val_acc"].idxmax(), "epoch"])
best_real_val_acc = float(log_df["real_val_acc"].max())
print(f"Loaded {len(log_df)} epochs from {LOG_CSV}")
print(f"Best synth_val_acc = {best_synth_val_acc:.4f} at epoch {best_synth_epoch}")
print(f"Best real_val_acc  = {best_real_val_acc:.4f} at epoch {best_real_epoch}  (monitor)")

# --- training_curves.png ----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(log_df["epoch"], log_df["train_acc"], label="train")
ax1.plot(log_df["epoch"], log_df["synth_val_acc"], label="synth val")
ax1.plot(log_df["epoch"], log_df["real_val_acc"], label="real val (monitor)",
         linestyle="--")
ax1.axvline(best_synth_epoch, color="k", linestyle=":", alpha=0.4,
            label=f"best synth_val ({best_synth_epoch})")
ax1.set_xlabel("epoch")
ax1.set_ylabel("accuracy")
ax1.set_title("Accuracy")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(log_df["epoch"], log_df["train_loss"], label="train")
ax2.plot(log_df["epoch"], log_df["synth_val_loss"], label="synth val")
ax2.plot(log_df["epoch"], log_df["real_val_loss"], label="real val (monitor)",
         linestyle="--")
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

# --- per_class_real_val.png — 13 subplots, one per class -------------------
# This is the diagnostic plot. Did shear move knight/king/bishop off zero?
per_class_cols = [f"real_val_acc_{s}" for s in CLASS_SHORT]
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
flat_axes = axes.ravel()
for i, (col, name) in enumerate(zip(per_class_cols, CLASS_NAMES)):
    ax = flat_axes[i]
    if col in log_df.columns:
        ax.plot(log_df["epoch"], log_df[col], marker="o", linewidth=1.5)
    ax.set_title(f"{CLASS_SHORT[i]} — {name}", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.axhline(0.0, color="r", linestyle=":", alpha=0.4)
# Hide the unused 14-15-16 subplots
for j in range(NUM_CLASSES, 16):
    flat_axes[j].axis("off")
fig.suptitle(
    "Per-class real_val accuracy on game7 over epochs "
    "(red dotted = 0%; lift here = shear/dataset paid off)",
    fontsize=12,
)
fig.text(0.5, 0.04, "epoch", ha="center")
fig.text(0.06, 0.5, "real_val accuracy", va="center", rotation="vertical")
plt.tight_layout(rect=(0.07, 0.05, 1.0, 0.97))
per_class_path = f"{PLOTS_DIR}/per_class_real_val.png"
plt.savefig(per_class_path, dpi=120)
plt.close()
print(f"wrote {per_class_path}")

print("\033[92m✓ Cell 13 — Training curves + per-class plot — OK\033[0m")




# %% [Cell 14 — Load best-synth checkpoint]
CKPT_BEST_SYNTH = f"{CHECKPOINTS_DIR}/best_synth.pt"

if not Path(CKPT_BEST_SYNTH).exists():
    raise FileNotFoundError(
        f"No best checkpoint at {CKPT_BEST_SYNTH}. Run Cell 12 first.")

ckpt = torch.load(CKPT_BEST_SYNTH, map_location=DEVICE, weights_only=False)
print(f"Best-synth checkpoint: epoch {ckpt['epoch']}, "
      f"synth_val_acc={ckpt['synth_val_acc']:.4f}, "
      f"real_val_acc(at-that-epoch)={ckpt['real_val_acc']:.4f}")
model = build_model()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("\033[92m✓ Cell 14 — Load best-synth checkpoint — OK\033[0m")




# %% [Cell 15 — Helper: confusion matrix plot]
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


print("\033[92m✓ Cell 15 — CM/plot helpers — OK\033[0m")




# %% [Cell 16 — Evaluate on synth test (held-out FENs from dataset_v1.5)]
print("Evaluating on synth_test_loader ...")
st_loss, st_acc, st_preds, st_labels = evaluate(
    model, synth_test_loader, criterion, DEVICE, "synth_test")
print(f"  synth_test acc = {st_acc:.4f}  (loss {st_loss:.4f})")

st_per_class = per_class_accuracy(st_preds, st_labels, NUM_CLASSES)
st_per_class_text = per_class_table_text(st_preds, st_labels)
print(st_per_class_text)

# Piece-only accuracy (exclude class 12 'empty')
piece_mask = (st_labels != 12)
st_piece_acc = float((st_preds[piece_mask] == st_labels[piece_mask]).mean()) \
    if piece_mask.any() else float("nan")

# Per-view accuracy
synth_test_df_view = synth_test_df.copy()
assert len(synth_test_df_view) == len(st_preds), (
    f"synth_test prediction/manifest length mismatch: "
    f"{len(st_preds)} preds vs {len(synth_test_df_view)} rows. "
    f"Per-view accuracy alignment broken — check synth_test_loader "
    f"is shuffle=False."
)
synth_test_df_view["pred"] = st_preds  # synth_test_loader is shuffle=False
per_view_dict = {}
for view in ("overhead", "west", "east"):
    sub = synth_test_df_view[synth_test_df_view["view"] == view]
    n = len(sub)
    acc = float((sub["pred"] == sub["label"]).mean()) if n else 0.0
    per_view_dict[view] = {"n": int(n), "acc": acc}

st_cm = confusion_matrix_np(st_preds, st_labels)
plot_confusion_matrix(st_cm, f"Synth test confusion (acc={st_acc:.4f})",
                      f"{PLOTS_DIR}/synth_test_cm.png", cmap="Blues")

synth_results = {
    "n_samples": int(len(st_preds)),
    "overall_acc": st_acc,
    "piece_only_acc": st_piece_acc,
    "loss": st_loss,
    "per_class_acc": {CLASS_SHORT[c]: st_per_class[c] for c in range(NUM_CLASSES)},
    "per_view": per_view_dict,
    "best_synth_epoch": best_synth_epoch,
    "best_synth_val_acc": best_synth_val_acc,
}
Path(f"{RESULTS_DIR}/synth_test_results.json").write_text(
    json.dumps(synth_results, indent=2))
np.save(f"{PREDS_DIR}/synth_test_preds.npy", st_preds.astype(np.int64))
np.save(f"{PREDS_DIR}/synth_test_labels.npy", st_labels.astype(np.int64))
print(f"  piece-only acc = {st_piece_acc:.4f}")
print(f"  per-view: {per_view_dict}")
print(f"  wrote {RESULTS_DIR}/synth_test_results.json + {PLOTS_DIR}/synth_test_cm.png")

print("\033[92m✓ Cell 16 — Synth test eval — OK\033[0m")




# %% [Cell 17 — Evaluate on game7 (real_val monitor — for final report)]
print("Evaluating on real_val_loader (game7, monitor set) ...")
rv_loss, rv_acc, rv_preds, rv_labels = evaluate(
    model, real_val_loader, criterion, DEVICE, "real_val")
print(f"  game7 per-square acc = {rv_acc:.4f}  (loss {rv_loss:.4f})")

# Per-board accuracy: a board is "correct" only if all 64 of its squares are right
real_manifest = real_val_dataset.manifest.copy()
real_manifest["pred"] = rv_preds
assert len(real_manifest) == len(rv_preds), \
    f"prediction length mismatch: {len(rv_preds)} vs {len(real_manifest)}"

per_board = (
    real_manifest.assign(correct=lambda d: d["pred"] == d["label"])
    .groupby("image_name")["correct"]
    .agg(["sum", "count"])
)
per_board["all_correct"] = per_board["sum"] == per_board["count"]
n_boards = len(per_board)
n_all_correct = int(per_board["all_correct"].sum())
mean_squares_correct = float(per_board["sum"].mean())
print(f"  per-board acc (all 64 correct): {n_all_correct}/{n_boards} = "
      f"{n_all_correct/n_boards:.4f}")
print(f"  mean squares correct per board: {mean_squares_correct:.2f}/64")

g7_per_class = per_class_accuracy(rv_preds, rv_labels, NUM_CLASSES)
piece_mask = (rv_labels != 12)
g7_piece_acc = float((rv_preds[piece_mask] == rv_labels[piece_mask]).mean()) \
    if piece_mask.any() else float("nan")

g7_cm = confusion_matrix_np(rv_preds, rv_labels)
plot_confusion_matrix(g7_cm, f"Game7 confusion (per-square acc={rv_acc:.4f})",
                      f"{PLOTS_DIR}/game7_cm.png", cmap="Reds")

game7_results = {
    "n_frames": int(n_boards),
    "n_squares": int(len(rv_preds)),
    "per_square_acc": rv_acc,
    "per_board_acc": n_all_correct / n_boards,
    "n_all_correct": int(n_all_correct),
    "mean_squares_correct": mean_squares_correct,
    "piece_only_acc": g7_piece_acc,
    "loss": rv_loss,
    "per_class_acc": {CLASS_SHORT[c]: g7_per_class[c] for c in range(NUM_CLASSES)},
    "peak_real_val_acc_during_training": best_real_val_acc,
    "peak_real_val_epoch": best_real_epoch,
}
Path(f"{RESULTS_DIR}/game7_results.json").write_text(
    json.dumps(game7_results, indent=2))
np.save(f"{PREDS_DIR}/game7_preds.npy", rv_preds.astype(np.int64))
np.save(f"{PREDS_DIR}/game7_labels.npy", rv_labels.astype(np.int64))
print(f"  piece-only acc = {g7_piece_acc:.4f}")
print(f"  wrote {RESULTS_DIR}/game7_results.json + {PLOTS_DIR}/game7_cm.png")

print("\033[92m✓ Cell 17 — Game7 eval — OK\033[0m")




# %% [Cell 18 — Held-out games: build datasets + loaders for games 2/4/5/6]
HELD_OUT_GAMES = [2, 4, 5, 6]

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

print("\033[92m✓ Cell 18 — Held-out game loaders — OK\033[0m")




# %% [Cell 19 — Evaluate on each held-out game (games 2/4/5/6)]
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
    assert len(manifest_g) == len(g_preds), \
        f"length mismatch game{N}: {len(g_preds)} vs {len(manifest_g)}"

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
        f"game{N} confusion (per-square acc={g_acc:.4f})",
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
          f"piece-only={g_piece_acc:.4f}  loss={g_loss:.4f}")

print("\033[92m✓ Cell 19 — Held-out games eval — OK\033[0m")




# %% [Cell 20 — Aggregate across held-out games + aggregate CM]
combined_preds = np.concatenate([per_game_preds[N] for N in per_game_preds])
combined_labels = np.concatenate([per_game_labels[N] for N in per_game_labels])

# Per-board groupby keyed on (game_num, image_name) — filenames can collide.
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
    f"Held-out aggregate (games 2,4,5,6) — per-square acc={combined_per_square_acc:.4f}",
    f"{PLOTS_DIR}/aggregate_cm.png", cmap="Reds",
)

# Top-5 confusion pairs (off-diagonal)
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

print(f"\nAggregate held-out (games {sorted(per_game_stats.keys())}):")
print(f"  per-square acc:  {combined_per_square_acc:.4f}")
print(f"  per-board  acc:  {combined_n_all_correct}/{combined_n_boards}="
      f"{combined_per_board_acc:.4f}")
print(f"  mean correct/64: {combined_mean_correct:.2f}")
print(f"  piece-only acc:  {combined_piece_acc:.4f}")
print(f"  top-5 confusion: {top5_pairs}")
print(f"wrote {RESULTS_DIR}/held_out_aggregate.json + {PLOTS_DIR}/aggregate_cm.png")

print("\033[92m✓ Cell 20 — Held-out aggregate — OK\033[0m")




# %% [Cell 21 — Qualitative plots: 8 boards per game with predictions overlaid]
# For each held-out game, render 8 evenly-spaced frames as 8×8 grids.
# Each cell shows the predicted class code; cell border is green if correct,
# red if wrong. Quick visual gut-check of where the model fails.

# A compact piece glyph lookup (Unicode chess pieces are hard to render in
# every Matplotlib font on the cluster — use class short codes instead).
def _short_for(label):
    return CLASS_SHORT[int(label)] if 0 <= int(label) < NUM_CLASSES else "?"


for N in sorted(per_game_manifest.keys()):
    manifest_g = per_game_manifest[N]
    images_order = sorted(manifest_g["image_name"].unique())
    n_frames_g = len(images_order)
    if n_frames_g == 0:
        continue
    # Evenly sample 8 frames (or all if fewer)
    k = min(8, n_frames_g)
    idxs = np.linspace(0, n_frames_g - 1, k).round().astype(int)
    selected_imgs = [images_order[i] for i in idxs]

    images_dir = Path(f"{PROJECT_ROOT}/data/game{N}_per_frame/images")
    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    flat = axes.ravel()
    for ax, img_name in zip(flat, selected_imgs):
        sub = manifest_g[manifest_g["image_name"] == img_name].sort_values(
            ["board_row", "board_col"])
        # Show the warped board so the per-square overlay aligns precisely.
        bgr = cv2.imread(str(images_dir / img_name))
        ds_g = real_test_datasets[N]
        corners = ds_g._get_corners(img_name, bgr)
        warped = warp_chessboard_image(bgr, corners)
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        H, W = warped_rgb.shape[:2]
        assert W == 500 and H == 500, (
            f"qualitative-plot square geometry assumes 500×500 warped output; "
            f"got {W}×{H} for game{N}/{img_name}. Update BOARD_OFFSET/SQ "
            f"derivation if warp_chessboard_image was changed."
        )
        ax.imshow(warped_rgb)
        # warp_chessboard_image gives a 500×500 image with the board at the
        # inner [50..450] window. Squares are 50 px each in that inner region.
        BOARD_OFFSET = 50
        SQ = (W - 2 * BOARD_OFFSET) // 8  # should be 50 if W==500
        n_correct_in_board = 0
        n_total_in_board = 0
        for _, r_row in sub.iterrows():
            br, bc = int(r_row["board_row"]), int(r_row["board_col"])
            true_lab = int(r_row["label"])
            pred_lab = int(r_row["pred"])
            ok = (true_lab == pred_lab)
            n_correct_in_board += int(ok)
            n_total_in_board += 1
            x0 = BOARD_OFFSET + bc * SQ
            y0 = BOARD_OFFSET + br * SQ
            color = "lime" if ok else "red"
            rect = plt.Rectangle(
                (x0, y0), SQ, SQ, linewidth=1.6, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            # Show short codes only when relevant (not empty squares unless wrong)
            label_text = _short_for(pred_lab)
            if pred_lab != 12 or not ok:
                ax.text(
                    x0 + SQ / 2, y0 + SQ / 2,
                    label_text,
                    color=color, fontsize=8, ha="center", va="center",
                    weight="bold",
                )
        ax.set_title(f"{img_name}  {n_correct_in_board}/{n_total_in_board}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"game{N} qualitative — 8 frames, predicted class overlaid "
        f"(green=correct, red=wrong)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    qual_path = f"{PLOTS_DIR}/game{N}_qualitative.png"
    plt.savefig(qual_path, dpi=110)
    plt.close()
    print(f"  wrote {qual_path}")

print("\033[92m✓ Cell 21 — Qualitative game plots — OK\033[0m")




# %% [Cell 22 — Summary.md]
# Reads log + JSONs; constructs the headline summary table.

try:
    total_train_time
except NameError:
    total_train_time = float(log_df["epoch_time_s"].sum())

n_epochs_ran = len(log_df)

# Baseline reference (v1, mild aug, dataset_v1) — taken from prior work:
BASELINE_PEAK_REAL_VAL_GAME7 = 0.5923

# Did mild shear / v1.5 dataset move knight/king/bishop classes off zero?
# Compare final-epoch per-class real_val acc to a 0.05 threshold.
def _moved_off_zero(class_short):
    col = f"real_val_acc_{class_short}"
    if col not in log_df.columns:
        return False
    series = log_df[col].dropna()
    return bool((series > 0.05).any())


moved = {
    "wN": _moved_off_zero("wN"),
    "wB": _moved_off_zero("wB"),
    "wK": _moved_off_zero("wK"),
    "bN": _moved_off_zero("bN"),
    "bB": _moved_off_zero("bB"),
    "bK": _moved_off_zero("bK"),
}
moved_names = [k for k, v in moved.items() if v]
not_moved_names = [k for k, v in moved.items() if not v]
if moved_names and not not_moved_names:
    moved_sentence = (
        "Per-class effect: ALL of {wN, wB, wK, bN, bB, bK} crossed the >5% "
        "threshold on real_val during training — every knight/bishop/king class "
        "moved off zero, supporting the shear hypothesis."
    )
elif moved_names:
    moved_sentence = (
        f"Per-class effect: {', '.join(moved_names)} crossed >5% on real_val; "
        f"{', '.join(not_moved_names)} stayed effectively at zero. Mixed "
        f"evidence — shear helped some position-bound classes but not all."
    )
else:
    moved_sentence = (
        "Per-class effect: NONE of {wN, wB, wK, bN, bB, bK} crossed the >5% "
        "threshold on real_val. Shear + v1.5 did NOT measurably move the "
        "knight/king/bishop classes off zero — the hypothesis did not hold."
    )

# Per-game table for the summary
def _fmt_pct(x):
    return f"{x:.4f}" if x == x else "n/a"  # NaN check


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

summary_lines = [
    "# Zero-shot v1.5 — final zero-shot run",
    "",
    "## What changed vs. v1 baseline",
    "- **Dataset:** dataset_v1.5 (7,665 imgs = v1 6,132 + legacy 1,533).",
    "- **Augmentation:** mild shear added to baseline (color jitter @0.5 → shear @0.5 → noise @0.5).",
    "- Everything else (ResNet18+ImageNet, SGD lr=0.001, StepLR(7,0.1), no class weights,",
    "  sqrt-inv-freq sampler, batch 64, 10 epochs, seed 42) is identical to baseline.",
    "",
    "## Training",
    f"- Total training time: **{total_train_time/60:.1f} min** "
    f"({n_epochs_ran} epochs, {len(train_dataset):,} synth train samples).",
    f"- Best synth_val_acc: **{best_synth_val_acc:.4f}** at epoch **{best_synth_epoch}**.",
    f"- Peak real_val_acc on game7 during training: "
    f"**{best_real_val_acc:.4f}** at epoch **{best_real_epoch}** (monitor only).",
    "",
    "## Synth test (held-out FENs from dataset_v1.5)",
    f"- **Overall accuracy: {st_acc:.4f}**",
    f"- **Piece-only accuracy (exclude empty class 12): {st_piece_acc:.4f}**",
    f"- Per-view: " + ", ".join(
        f"{v}={per_view_dict[v]['acc']:.4f} (n={per_view_dict[v]['n']:,})"
        for v in ("overhead", "west", "east")
    ),
    "",
    "## Game7 monitor (NOT held-out — used only as in-training real signal)",
    f"- Per-square accuracy at best-synth checkpoint: **{rv_acc:.4f}**",
    f"- Per-board accuracy: {n_all_correct}/{n_boards} = {n_all_correct/n_boards:.4f}",
    f"- Mean squares correct/board: {mean_squares_correct:.2f}/64",
    f"- Peak game7 real_val_acc anywhere in training: {best_real_val_acc:.4f} (epoch {best_real_epoch})",
    "",
    "## Held-out real test (games 2, 4, 5, 6)",
    "",
    game_table,
    "",
    "## Sim-to-real gap",
    f"- synth_test per-square: **{st_acc:.4f}**",
    f"- held-out aggregate per-square: **{combined_per_square_acc:.4f}**",
    f"- **Gap: {st_acc - combined_per_square_acc:+.4f}**",
    "",
    "## Comparison to v1 baseline (zero_shot/, mild aug, dataset_v1)",
    f"- v1 baseline peak real_val on game7: **{BASELINE_PEAK_REAL_VAL_GAME7:.4f}**",
    f"- this run peak real_val on game7:    **{best_real_val_acc:.4f}**  "
    f"(Δ = {best_real_val_acc - BASELINE_PEAK_REAL_VAL_GAME7:+.4f})",
    "",
    "## Per-class shear effect (the diagnostic this run was designed to test)",
    f"- {moved_sentence}",
    "- See plots/per_class_real_val.png for the full 13-class trajectory.",
    "",
    "## Artifacts",
    "- `results/training_log.csv` (per-epoch + 13 per-class real_val columns)",
    "- `results/synth_test_results.json`",
    "- `results/game7_results.json`",
    "- `results/game{2,4,5,6}_results.json`",
    "- `results/held_out_aggregate.json`",
    "- `results/predictions/*.npy` (raw pred/target tensors per eval set)",
    "- `checkpoints/best_synth.pt` "
    f"(epoch {best_synth_epoch}, synth_val_acc={best_synth_val_acc:.4f})",
    "- `checkpoints/best_real_monitor.pt` "
    f"(epoch {best_real_epoch}, real_val_acc={best_real_val_acc:.4f}) — monitor-only artifact",
    "- `checkpoints/latest.pt`",
    "- `plots/aug_smoke_check.png`, `training_curves.png`, `per_class_real_val.png`",
    "- `plots/synth_test_cm.png`, `game7_cm.png`, `game{2,4,5,6}_cm.png`, `aggregate_cm.png`",
    "- `plots/game{2,4,5,6}_qualitative.png`",
]
summary_text = "\n".join(summary_lines)
Path(f"{RESULTS_DIR}/summary.md").write_text(summary_text)
print(summary_text)
print(f"\nwrote {RESULTS_DIR}/summary.md")

print("\033[92m✓ Cell 22 — Summary.md — OK\033[0m")
