"""
Project 2 — Zero-shot baseline (Step 6b).

Trains ResNet18 on synthetic dataset_v1 squares (100×100 RGB crops from
chesscog warp + crop_square), monitors on game7 real frames during training,
and evaluates on held-out synth-test + game7 at the end.

Pipeline summary:
  cv2.imread → warp_chessboard_image → crop_square → BGR→RGB → augment
  → tensor [3,100,100] float32 ∈ [0,1] → ImageNet normalize → ResNet18 → 13-class softmax.

Run cells with Shift+Enter in VS Code (Python + Jupyter extensions installed).
Each `# %%` block is a separate cell; state persists between runs.

Hard rules (per Step 6 brief):
  - real_val_acc is for MONITORING ONLY. Never used to pick checkpoints.
  - All outputs go to {RESULTS_DIR} or {PLOTS_DIR}.  Nothing in project root.
  - Smoke test (Cell 11) must pass before training (Cell 12).
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
matplotlib.use("Agg", force=True)  # non-interactive — save to PNG, never show().
                                   # force=True is needed because Jupyter kernels pre-set
                                   # the inline backend; without it we get a UserWarning.
import matplotlib.pyplot as plt

# Project modules
from scripts.chess_dataset import ChessSquareDataset
from scripts.fen_to_grid import fen_to_label_grid
from scripts.view_orientations import GAME7_ORIENTATION
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
# All hyperparameters and paths as named constants. No magic numbers below.
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 6   # matches system's suggested max; silences the torch warning
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 7
LR_GAMMA = 0.1
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Augmentation strengths (baseline: mild). Tune later if needed.
COLOR_JITTER_BRIGHTNESS = 0.15
COLOR_JITTER_CONTRAST = 0.15
COLOR_JITTER_SATURATION = 0.15
GAUSSIAN_NOISE_STD = 0.02  # in normalized [0,1] units

# ImageNet stats for ResNet18 pretrained weights
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
MANIFEST_PATH = f"{PROJECT_ROOT}/scripts/manifest.csv"   # moved into scripts/
CORNERS_PATH = f"{PROJECT_ROOT}/scripts/corners.json"    # moved into scripts/
DATASET_DIR = f"{PROJECT_ROOT}/data/dataset_v1/images"   # moved out of Project2_3/
GAME7_DIR = f"{PROJECT_ROOT}/data/game7_per_frame/images"
GAME7_GT_CSV = f"{PROJECT_ROOT}/data/game7_per_frame/gt.csv"

ZERO_SHOT_DIR = f"{PROJECT_ROOT}/zero_shot"
RESULTS_DIR = f"{ZERO_SHOT_DIR}/results"
PLOTS_DIR = f"{ZERO_SHOT_DIR}/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"manifest:    {MANIFEST_PATH}")
print(f"corners:     {CORNERS_PATH}")
print(f"game7:       {GAME7_DIR}")
print(f"results dir: {RESULTS_DIR}")
print(f"plots dir:   {PLOTS_DIR}")

print("\033[92m✓ Cell 2 — Config — OK\033[0m")







# %% [Cell 3 — Build splits (group by FEN so views don't leak across splits)]
manifest = pd.read_csv(MANIFEST_PATH)
print(f"Full manifest: {len(manifest):,} rows, "
      f"{manifest['source_image'].nunique():,} unique images, "
      f"{manifest['fen'].nunique():,} unique FENs")

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

# --- Verification block: explicit PASS/FAIL on every invariant ----------------
print("\n[Cell 3 verification]")
checks = [
    ("FEN sets disjoint (train/val)",
        train_fens.isdisjoint(val_fens)),
    ("FEN sets disjoint (train/test)",
        train_fens.isdisjoint(test_fens)),
    ("FEN sets disjoint (val/test)",
        val_fens.isdisjoint(test_fens)),
    ("All FENs accounted for",
        len(train_fens) + len(val_fens) + len(test_fens) == len(unique_fens)),
    ("Image sets disjoint (train/val)",
        train_imgs.isdisjoint(val_imgs)),
    ("Image sets disjoint (train/test)",
        train_imgs.isdisjoint(test_imgs)),
    ("Image sets disjoint (val/test)",
        val_imgs.isdisjoint(test_imgs)),
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

print(f"\nSplit by FEN ({TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{TEST_FRAC:.0%}):")
print(f"  train: {len(train_fens):4d} FENs, {len(train_imgs):4d} images, "
      f"{len(train_df):>7,d} rows")
print(f"  val  : {len(val_fens):4d} FENs, {len(val_imgs):4d} images, "
      f"{len(synth_val_df):>7,d} rows")
print(f"  test : {len(test_fens):4d} FENs, {len(test_imgs):4d} images, "
      f"{len(synth_test_df):>7,d} rows")

# View balance per split — should be ~33/33/33 because each FEN renders into all 3 views.
print(f"\nView balance per split (% of split rows):")
for name, df in [("train", train_df), ("val", synth_val_df), ("test", synth_test_df)]:
    pct = (df["view"].value_counts(normalize=True) * 100).round(2).to_dict()
    print(f"  {name:<6s} {pct}")

# Class distribution per split — printed table.
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

# Plot: class distribution across splits (so drift is visible at a glance).
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
def gaussian_noise(x_rgb_uint8):
    """Add small Gaussian noise. Input/output: HWC uint8 RGB."""
    x = x_rgb_uint8.astype(np.float32)
    noise = np.random.normal(0, GAUSSIAN_NOISE_STD * 255, x.shape).astype(np.float32)
    x = np.clip(x + noise, 0, 255).astype(np.uint8)
    return x

def color_jitter(x_rgb_uint8):
    """Brightness / contrast / saturation jitter. Input/output: HWC uint8 RGB."""
    x = x_rgb_uint8.astype(np.float32)
    # Brightness: scalar multiply
    b = 1.0 + np.random.uniform(-COLOR_JITTER_BRIGHTNESS, COLOR_JITTER_BRIGHTNESS)
    x = x * b
    # Contrast: scale around mean
    c = 1.0 + np.random.uniform(-COLOR_JITTER_CONTRAST, COLOR_JITTER_CONTRAST)
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * c + mean
    # Saturation: blend with grayscale
    s = 1.0 + np.random.uniform(-COLOR_JITTER_SATURATION, COLOR_JITTER_SATURATION)
    gray = x.mean(axis=2, keepdims=True)
    x = gray + s * (x - gray)
    return np.clip(x, 0, 255).astype(np.uint8)


def train_transform(crop_rgb_uint8):
    """Augmentation pipeline used during training. HWC uint8 RGB → HWC uint8 RGB.
    Tensorization + ImageNet normalize happens AFTER this in the model loop."""
    if random.random() < 0.5:
        crop_rgb_uint8 = color_jitter(crop_rgb_uint8)
    if random.random() < 0.5:
        crop_rgb_uint8 = gaussian_noise(crop_rgb_uint8)
    return crop_rgb_uint8


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
print(f"train_dataset:      {len(train_dataset):>8,d} samples (augmented)")
print(f"synth_val_dataset:  {len(synth_val_dataset):>8,d} samples (no aug)")
print(f"synth_test_dataset: {len(synth_test_dataset):>8,d} samples (no aug)")

print("\033[92m✓ Cell 4 — Synthetic datasets and transforms — OK\033[0m")












# %% [Cell 5 — Game7Dataset]
class Game7Dataset(Dataset):
    """One sample per (game7 frame × board square).

    Differences from ChessSquareDataset:
      - No corner cache on disk. Per-image find_corners with OOB rejection +
        image-corner fallback (chesscog hallucinates board extensions on
        game7's tight-cropped photos — Step 6a finding).
      - Caches detected corners in memory so 64 squares from one frame don't
        re-run detection.
      - Uses GAME7_ORIENTATION ('identity') for FEN→grid mapping.
    """

    CORNER_OOB_TOLERANCE = 8  # px; reject find_corners output beyond this

    def __init__(self, gt_csv_path, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        # Load gt.csv: image_name → fen
        rows = []
        with open(gt_csv_path) as f:
            for r in csv.DictReader(f):
                fen = r["fen"]
                grid = fen_to_label_grid(fen, "game7")
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
        # Sort so consecutive 64 rows belong to the same image — lets evaluation
        # code group predictions per-frame just by walking the sequence.
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
            np.random.seed(SEED)  # deterministic RANSAC
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


real_val_dataset = Game7Dataset(GAME7_GT_CSV, GAME7_DIR, transform=None)
print(f"real_val_dataset (game7): {len(real_val_dataset):,} samples "
      f"({real_val_dataset.manifest['image_name'].nunique()} frames × 64 squares)")
print(f"  class distribution (counts):")
for cls in range(NUM_CLASSES):
    n = (real_val_dataset.manifest["label"] == cls).sum()
    print(f"    {cls:>2d} {CLASS_NAMES[cls]:<14s}: {n}")

print("\033[92m✓ Cell 5 — Game7Dataset — OK\033[0m")











# %% [Cell 6 — Weighted sampler]
# Inverse-frequency weights — minority classes (queens, kings) sampled
# proportionally more so the training loss isn't dominated by 'empty'.
train_labels = train_df["label"].values
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
print("Train-set class counts and inverse-frequency weights:")
print(f"  {'cls':>3s}  {'name':<14s}  {'count':>8s}  {'weight':>10s}  "
      f"{'eff_prob':>9s}")
class_weights = np.zeros(NUM_CLASSES, dtype=np.float64)
for cls in range(NUM_CLASSES):
    if class_counts[cls] > 0:
        class_weights[cls] = 1.0 / class_counts[cls]
sample_weights = class_weights[train_labels]
sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
# Effective per-class sampling probability (sum of weights for that class
# divided by total weight)
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
synth_val_loader = DataLoader(
    synth_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
)
synth_test_loader = DataLoader(
    synth_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
)
real_val_loader = DataLoader(
    real_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True,
)

# Print one batch from each — shape, dtype, label distribution.
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
    """SGD + StepLR. Returns (optimizer, scheduler)."""
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
    """x: (B, 3, H, W) float in [0,1]. Returns normalized to ImageNet mean/std."""
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
for name, loader in [("train", train_loader), ("synth_val", synth_val_loader),
                     ("real_val", real_val_loader)]:
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

# 4. Eval on 3 batches of synth_val
#    NOTE: we don't assert acc > random here. The model is untrained at this point —
#    one SGD step is not enough to learn anything, and a freshly-initialized
#    13-class FC head can latch onto a minority class purely by random init, which
#    drives accuracy well below 1/13. That isn't a pipeline failure; it's just noise.
#    What we DO check: the eval loop completes, shapes are right, predictions are
#    finite ints in [0, 12], and at least some prediction diversity exists.
print("\n[4] Eval on 3 batches of synth_val:")
model.eval()
correct = 0; total = 0
all_preds_smoke = []
with torch.no_grad():
    for i, (xb, yb) in enumerate(synth_val_loader):
        if i >= 3:
            break
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        logits = model(imagenet_normalize(xb))
        assert torch.isfinite(logits).all(), "non-finite logits on synth_val"
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

# 5. Eval on 3 batches of real_val (highest-risk DataLoader — Game7Dataset)
print("\n[4] Eval on 3 batches of synth_val (loss sanity check):")
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


# 6. Reset model + optimizer so the real training run starts clean
print("\n[6] Resetting model and optimizer for the real training run.")
model = build_model()
optimizer, scheduler = build_optimizer(model)

print("\n" + "=" * 64)
print("Smoke test passed. Ready for training.")
print("=" * 64)

print("\033[92m✓ Cell 11 — Smoke test — OK\033[0m")






# %% [Cell 12 — Training loop]
training_log = []
best_synth_val_acc = -1.0
best_epoch = -1

CKPT_BEST = f"{RESULTS_DIR}/best_synth.pt"
CKPT_LATEST = f"{RESULTS_DIR}/latest.pt"
LOG_CSV = f"{RESULTS_DIR}/training_log.csv"

t_total = time.perf_counter()
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n{'='*64}")
    print(f"Epoch {epoch}/{NUM_EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.5f})")
    print(f"{'='*64}")
    t_ep = time.perf_counter()

    # Train
    print("  [train]")
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, print_every=200)

    # Eval synth val
    print("  [eval synth_val]")
    sv_loss, sv_acc, _, _ = evaluate(model, synth_val_loader, criterion, DEVICE, "synth_val")

    # Eval real val (MONITORING ONLY — never used to pick checkpoints)
    print("  [eval real_val (monitoring only)]")
    rv_loss, rv_acc, _, _ = evaluate(model, real_val_loader, criterion, DEVICE, "real_val")

    scheduler.step()
    dt = time.perf_counter() - t_ep

    print(f"\n  Epoch {epoch:2d}: "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  |  "
          f"synth_val_acc={sv_acc:.4f}  |  "
          f"real_val_acc={rv_acc:.4f} (monitor)  "
          f"|  {dt/60:.1f}min")

    training_log.append({
        "epoch": epoch,
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": train_loss, "train_acc": train_acc,
        "synth_val_loss": sv_loss, "synth_val_acc": sv_acc,
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
        "synth_val_acc": sv_acc, "real_val_acc": rv_acc,
    }, CKPT_LATEST)

    # Save best by SYNTH val acc (real_val_acc never gates checkpoint)
    if sv_acc > best_synth_val_acc:
        best_synth_val_acc = sv_acc
        best_epoch = epoch
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "synth_val_acc": sv_acc, "real_val_acc": rv_acc,
        }, CKPT_BEST)
        print(f"  → NEW BEST synth_val_acc={sv_acc:.4f} → saved {CKPT_BEST}")

total_train_time = time.perf_counter() - t_total
print(f"\nTraining done. Total time: {total_train_time/60:.1f} min")
print(f"Best synth_val_acc: {best_synth_val_acc:.4f} at epoch {best_epoch}")

print("\033[92m✓ Cell 12 — Training loop — OK\033[0m")










# %% [Cell 13 — Training curves]
# Self-contained: reads the log CSV from disk, derives best_epoch from the data.
# No dependency on Cell 12's in-memory state, so this cell can be run on its own
# any time after at least one epoch has completed and written to the log.
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

best_epoch = int(log_df.loc[log_df["synth_val_acc"].idxmax(), "epoch"])
best_synth_val_acc = float(log_df["synth_val_acc"].max())
print(f"Loaded {len(log_df)} epochs from {LOG_CSV}")
print(f"Best synth_val_acc = {best_synth_val_acc:.4f} at epoch {best_epoch}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(log_df["epoch"], log_df["train_acc"], label="train")
ax1.plot(log_df["epoch"], log_df["synth_val_acc"], label="synth val")
ax1.plot(log_df["epoch"], log_df["real_val_acc"], label="real val (monitor)",
         linestyle="--")
ax1.axvline(best_epoch, color="k", linestyle=":", alpha=0.4,
            label=f"best epoch ({best_epoch})")
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

print("\033[92m✓ Cell 13 — Training curves — OK\033[0m")










# %% [Cell 14 — Load best checkpoint]
# Self-contained: reconstruct the checkpoint path from RESULTS_DIR so this cell
# works after a kernel restart (without needing Cell 12's CKPT_BEST in memory).
CKPT_BEST = f"{RESULTS_DIR}/best_synth.pt"

if not Path(CKPT_BEST).exists():
    raise FileNotFoundError(
        f"No best checkpoint at {CKPT_BEST}. "
        "Run Cell 12 (Training loop) for at least one epoch first."
    )

ckpt = torch.load(CKPT_BEST, map_location=DEVICE, weights_only=False)
print(f"Best checkpoint: epoch {ckpt['epoch']}, "
      f"synth_val_acc={ckpt['synth_val_acc']:.4f}, "
      f"real_val_acc(monitor)={ckpt['real_val_acc']:.4f}")
model = build_model()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("\033[92m✓ Cell 14 — Load best checkpoint — OK\033[0m")










# %% [Cell 15 — Evaluate on synth test]
print("Evaluating on synth_test_loader ...")
st_loss, st_acc, st_preds, st_labels = evaluate(
    model, synth_test_loader, criterion, DEVICE, "synth_test")
print(f"  synth test acc = {st_acc:.4f}  (loss {st_loss:.4f})")

# Per-class accuracy
per_class_lines = ["class  name              n      acc"]
for cls in range(NUM_CLASSES):
    mask = (st_labels == cls)
    n = int(mask.sum())
    if n:
        acc = float((st_preds[mask] == cls).mean())
        per_class_lines.append(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}  {acc:.4f}")
    else:
        per_class_lines.append(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}    n/a")
per_class_report = "\n".join(per_class_lines)
print(per_class_report)
Path(f"{RESULTS_DIR}/synth_test_per_class.txt").write_text(per_class_report)

# Per-view accuracy
synth_test_df["pred"] = st_preds  # aligned because synth_test_loader is shuffle=False
per_view_lines = ["view      n        acc"]
for view in ("overhead", "west", "east"):
    sub = synth_test_df[synth_test_df["view"] == view]
    n = len(sub)
    acc = (sub["pred"] == sub["label"]).mean() if n else 0.0
    per_view_lines.append(f"  {view:<8s}  {n:>6d}   {acc:.4f}")
per_view_report = "\n".join(per_view_lines)
print(per_view_report)
Path(f"{RESULTS_DIR}/synth_test_per_view.txt").write_text(per_view_report)

# Confusion matrix
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
for t, p in zip(st_labels, st_preds):
    cm[t, p] += 1
cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)], rotation=45, ha="right")
ax.set_yticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)])
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Synth test confusion (acc={st_acc:.4f})")
plt.colorbar(im)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if cm[i, j] > 0:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="black" if cm_norm[i, j] < 0.5 else "white")
plt.tight_layout()
synth_cm_path = f"{PLOTS_DIR}/synth_test_confusion.png"
plt.savefig(synth_cm_path, dpi=120)
plt.close()
print(f"wrote {synth_cm_path}")

print("\033[92m✓ Cell 15 — Evaluate on synth test — OK\033[0m")










# %% [Cell 16 — Evaluate on game7]
print("Evaluating on real_val_loader (game7) ...")
rv_loss, rv_acc, rv_preds, rv_labels = evaluate(
    model, real_val_loader, criterion, DEVICE, "real_val")
print(f"  per-square acc = {rv_acc:.4f}  (loss {rv_loss:.4f})")

# Recover image_name for each prediction. real_val_loader is shuffle=False,
# so the order matches real_val_dataset.manifest exactly.
real_manifest = real_val_dataset.manifest.copy()
real_manifest["pred"] = rv_preds  # length must match — assert before relying on it
assert len(real_manifest) == len(rv_preds), \
    f"prediction length mismatch: {len(rv_preds)} vs {len(real_manifest)}"

# Per-board accuracy: a board is "correct" only if all 64 of its squares are right
per_board = (
    real_manifest.assign(correct=lambda d: d["pred"] == d["label"])
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

# Per-class accuracy on real
real_class_lines = ["class  name              n      acc"]
for cls in range(NUM_CLASSES):
    mask = (rv_labels == cls)
    n = int(mask.sum())
    if n:
        acc = float((rv_preds[mask] == cls).mean())
        real_class_lines.append(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}  {acc:.4f}")
    else:
        real_class_lines.append(f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}    n/a")
real_class_report = "\n".join(real_class_lines)
print(real_class_report)
Path(f"{RESULTS_DIR}/real_test_per_class.txt").write_text(real_class_report)

# Confusion matrix on real
cm_r = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
for t, p in zip(rv_labels, rv_preds):
    cm_r[t, p] += 1
cm_r_norm = cm_r / np.maximum(cm_r.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm_r_norm, cmap="Reds", vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)], rotation=45, ha="right")
ax.set_yticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)])
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Game7 confusion (per-square acc={rv_acc:.4f})")
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

# Top-5 confusion pairs (off-diagonal)
pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm_r[i, j] > 0:
            pairs.append((cm_r[i, j], i, j))
pairs.sort(reverse=True)
print("\nTop-5 real-image confusion pairs (true → pred, count):")
top5_lines = []
for n, t, p in pairs[:5]:
    line = f"  {CLASS_NAMES[t]:<14s} → {CLASS_NAMES[p]:<14s}  {n}"
    print(line)
    top5_lines.append(line)

print("\033[92m✓ Cell 16 — Evaluate on game7 — OK\033[0m")









# %% [Cell 17 — Summary.md]
# game 7 test case summary saved
# total_train_time may not exist in the kernel if Cell 12 was interrupted or the
# kernel was restarted between Cell 12 and here. Reconstruct it from the per-epoch
# times stored in the log CSV (epoch_time_s column), which is the same number Cell 12
# would have summed up anyway.
try:
    total_train_time
except NameError:
    total_train_time = float(log_df["epoch_time_s"].sum())
    print(f"(reconstructed total_train_time from log: {total_train_time/60:.1f} min "
          f"over {len(log_df)} epochs)")

n_epochs_ran = len(log_df)
summary_lines = [
    "# Zero-shot baseline — Step 6 results",
    "",
    f"## Training",
    f"- Total training time: **{total_train_time/60:.1f} min** ({n_epochs_ran} epochs, "
    f"{len(train_dataset):,} synth train samples).",
    f"- Best synth val acc: **{best_synth_val_acc:.4f}** at epoch **{best_epoch}**.",
    f"- Final-epoch synth val acc: {log_df['synth_val_acc'].iloc[-1]:.4f}",
    f"- Final-epoch real val acc (monitor): {log_df['real_val_acc'].iloc[-1]:.4f}",
    "",
    f"## Synthetic test (held-out FENs, {len(synth_test_dataset):,} squares)",
    f"- **Overall accuracy: {st_acc:.4f}**",
    "",
    "### Per-class",
    "```",
    per_class_report,
    "```",
    "",
    "### Per-view",
    "```",
    per_view_report,
    "```",
    "",
    f"## Real test — game7 ({n_boards} frames, {len(real_val_dataset)} squares)",
    f"- **Per-square accuracy: {rv_acc:.4f}**",
    f"- **Per-board accuracy (all 64 correct): {n_all_correct}/{n_boards} = "
    f"{n_all_correct/n_boards:.4f}**",
    f"- Mean squares correct / board: **{mean_squares_correct:.2f} / 64**",
    "",
    "### Per-class on real",
    "```",
    real_class_report,
    "```",
    "",
    f"## Sim-to-real gap",
    f"- synth_test − game7 (per-square): "
    f"{st_acc:.4f} − {rv_acc:.4f} = **{st_acc-rv_acc:+.4f}**",
    "",
    "### Top-5 game7 confusion pairs",
    "```",
    "\n".join(top5_lines) if top5_lines else "(none)",
    "```",
    "",
    "## Artifacts",
    f"- `results/training_log.csv`",
    f"- `results/best_synth.pt`  (epoch {best_epoch}, synth_val_acc={best_synth_val_acc:.4f})",
    f"- `results/latest.pt`",
    f"- `results/synth_test_per_class.txt`",
    f"- `results/synth_test_per_view.txt`",
    f"- `results/real_test_per_class.txt`",
    f"- `results/real_test_per_board_accuracy.txt`",
    f"- `plots/training_curves.png`",
    f"- `plots/synth_test_confusion.png`",
    f"- `plots/real_test_confusion.png`",
]
summary_text = "\n".join(summary_lines)
Path(f"{RESULTS_DIR}/summary.md").write_text(summary_text)
print(summary_text)
print(f"\nwrote {RESULTS_DIR}/summary.md")

print("\033[92m✓ Cell 17 — Summary.md — OK\033[0m")







# %% [Cell 18 — RealGameDataset + build datasets/loaders for games 2,4,5,6]
# RealGameDataset inherits all of Game7Dataset's corner-detection, OOB-fallback,
# in-memory corner-cache, and __getitem__ pipeline. We only override __init__ to
# accept a game_name parameter so fen_to_label_grid uses the right view key
# (looked up from VIEW_ORIENTATIONS).
HELD_OUT_GAMES = [2, 4, 5, 6]
GAMES_EVAL_DIR = f"{RESULTS_DIR}/games_2_4_5_6_eval"
GAMES_EVAL_PLOTS_DIR = f"{PLOTS_DIR}/games_2_4_5_6_eval"
os.makedirs(GAMES_EVAL_DIR, exist_ok=True)
os.makedirs(GAMES_EVAL_PLOTS_DIR, exist_ok=True)


class RealGameDataset(Game7Dataset):
    """Held-out real test dataset for game{N} where N in {2,4,5,6}.

    Reuses Game7Dataset's _get_corners + __getitem__ unchanged. Only difference:
    a game_name parameter drives fen_to_label_grid(fen, game_name) so each game's
    orientation is looked up per-game from VIEW_ORIENTATIONS.
    """

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
        # Sort so consecutive 64 rows belong to the same image — same convention
        # as Game7Dataset; lets eval code group predictions per-frame by walking
        # the sequence.
        self.manifest = self.manifest.sort_values(
            ["image_name", "board_row", "board_col"]
        ).reset_index(drop=True)
        self._corner_cache = {}


real_test_datasets = {}
real_test_loaders = {}
for N in HELD_OUT_GAMES:
    gt_csv = f"/home/eladbaum/chess_project/data/game{N}_per_frame/gt.csv"
    images_dir = f"/home/eladbaum/chess_project/data/game{N}_per_frame/images"
    if not (Path(gt_csv).exists() and Path(images_dir).exists()):
        print(f"  [skip] game{N}: missing gt.csv or images/ — skipping this game")
        continue
    ds = RealGameDataset(gt_csv, images_dir, game_name=f"game{N}", transform=None)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True, persistent_workers=True)
    real_test_datasets[N] = ds
    real_test_loaders[N] = ld
    n_frames = ds.manifest["image_name"].nunique()
    print(f"  game{N}: {n_frames} frames, {len(ds)} squares")

print("\033[92m✓ Cell 18 — RealGameDataset + held-out loaders — OK\033[0m")


# %% [Cell 19 — Evaluate model on each held-out real game]
# Per-game: per-square acc, per-board acc (all 64 correct), mean correct/64,
# per-class table, confusion matrix PNG, and a combined per-game .txt report.
per_game_stats = {}
per_game_preds = {}
per_game_labels = {}
per_game_manifest = {}

for N, loader in real_test_loaders.items():
    print(f"\nEvaluating on game{N} ...")
    g_loss, g_acc, g_preds, g_labels = evaluate(
        model, loader, criterion, DEVICE, f"game{N}")

    ds = real_test_datasets[N]
    manifest = ds.manifest.copy()
    manifest["pred"] = g_preds
    assert len(manifest) == len(g_preds), \
        f"length mismatch game{N}: {len(g_preds)} vs {len(manifest)}"

    per_board = (
        manifest.assign(correct=lambda d: d["pred"] == d["label"])
        .groupby("image_name")["correct"]
        .agg(["sum", "count"])
    )
    per_board["all_correct"] = per_board["sum"] == per_board["count"]
    n_boards_g = len(per_board)
    n_all_correct_g = int(per_board["all_correct"].sum())
    mean_correct_g = float(per_board["sum"].mean())

    # Per-class table
    per_class_lines = ["class  name              n      acc"]
    for cls in range(NUM_CLASSES):
        mask = (g_labels == cls)
        n = int(mask.sum())
        if n:
            acc = float((g_preds[mask] == cls).mean())
            per_class_lines.append(
                f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}  {acc:.4f}")
        else:
            per_class_lines.append(
                f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}    n/a")
    per_class_report_g = "\n".join(per_class_lines)

    # Per-board breakdown sorted by accuracy desc
    board_breakdown = []
    for img, row in per_board.sort_values("sum", ascending=False).iterrows():
        board_breakdown.append(
            f"  {img:<30s}  {int(row['sum'])}/{int(row['count'])}")

    report = "\n".join([
        f"# game{N} held-out real test",
        f"n_frames: {n_boards_g}",
        f"n_squares: {len(g_preds)}",
        f"per-square accuracy: {g_acc:.4f}  (loss {g_loss:.4f})",
        f"per-board accuracy (all 64 correct): {n_all_correct_g}/{n_boards_g} = "
            f"{n_all_correct_g/n_boards_g:.4f}",
        f"mean squares correct / board: {mean_correct_g:.2f} / 64",
        "",
        "## per-class",
        per_class_report_g,
        "",
        "## per-board breakdown (sorted by correct count, desc)",
        f"  {'image':<30s}  correct/total",
        *board_breakdown,
    ])
    out_txt = f"{GAMES_EVAL_DIR}/real_test_game{N}.txt"
    Path(out_txt).write_text(report)

    # Per-game confusion matrix (same style as Cell 16)
    cm_g = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(g_labels, g_preds):
        cm_g[t, p] += 1
    cm_g_norm = cm_g / np.maximum(cm_g.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_g_norm, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)],
                       rotation=45, ha="right")
    ax.set_yticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"game{N} confusion (per-square acc={g_acc:.4f})")
    plt.colorbar(im)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm_g[i, j] > 0:
                ax.text(j, i, str(cm_g[i, j]), ha="center", va="center",
                        fontsize=7,
                        color="black" if cm_g_norm[i, j] < 0.5 else "white")
    plt.tight_layout()
    cm_path = f"{GAMES_EVAL_PLOTS_DIR}/real_test_game{N}_confusion.png"
    plt.savefig(cm_path, dpi=120); plt.close()

    per_game_stats[N] = {
        "n_frames": n_boards_g, "n_squares": len(g_preds),
        "per_square_acc": g_acc, "loss": g_loss,
        "per_board_acc": n_all_correct_g / n_boards_g,
        "n_all_correct": n_all_correct_g,
        "mean_correct": mean_correct_g,
        "per_class_report": per_class_report_g,
    }
    per_game_preds[N] = g_preds
    per_game_labels[N] = g_labels
    per_game_manifest[N] = manifest

    print(f"  game{N}: per-square={g_acc:.4f}  per-board={n_all_correct_g}/{n_boards_g}="
          f"{n_all_correct_g/n_boards_g:.4f}  mean-correct={mean_correct_g:.1f}/64  "
          f"loss={g_loss:.4f}")
    print(f"    wrote {out_txt}")
    print(f"    wrote {cm_path}")

print("\033[92m✓ Cell 19 — Evaluate held-out games — OK\033[0m")


# %% [Cell 20 — Aggregate evaluation across held-out games]
combined_preds = np.concatenate([per_game_preds[N] for N in per_game_preds])
combined_labels = np.concatenate([per_game_labels[N] for N in per_game_labels])

# Per-board groupby keyed on (game_num, image_name) — image filenames like
# frame_000044.jpg can collide across games and must be disambiguated.
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
combined_mean_correct = float(per_board_combined["sum"].mean())
combined_per_square_acc = float((combined_preds == combined_labels).mean())
combined_per_board_acc = (
    combined_n_all_correct / combined_n_boards if combined_n_boards else 0.0
)

# Combined per-class
combined_class_lines = ["class  name              n      acc"]
for cls in range(NUM_CLASSES):
    mask = (combined_labels == cls)
    n = int(mask.sum())
    if n:
        acc = float((combined_preds[mask] == cls).mean())
        combined_class_lines.append(
            f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}  {acc:.4f}")
    else:
        combined_class_lines.append(
            f"  {cls:>3d}  {CLASS_NAMES[cls]:<14s}  {n:>6d}    n/a")
combined_per_class_report = "\n".join(combined_class_lines)

# Combined confusion matrix (Reds for real)
cm_c = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
for t, p in zip(combined_labels, combined_preds):
    cm_c[t, p] += 1
cm_c_norm = cm_c / np.maximum(cm_c.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm_c_norm, cmap="Reds", vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)],
                   rotation=45, ha="right")
ax.set_yticklabels([f"{c} {n[:6]}" for c, n in enumerate(CLASS_NAMES)])
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(
    f"Held-out real (games 2,4,5,6) — per-square acc={combined_per_square_acc:.4f}"
)
plt.colorbar(im)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if cm_c[i, j] > 0:
            ax.text(j, i, str(cm_c[i, j]), ha="center", va="center",
                    fontsize=7,
                    color="black" if cm_c_norm[i, j] < 0.5 else "white")
plt.tight_layout()
combined_cm_path = f"{GAMES_EVAL_PLOTS_DIR}/real_test_combined_confusion.png"
plt.savefig(combined_cm_path, dpi=120); plt.close()

# Top-5 confusion pairs across all four games
pairs_c = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm_c[i, j] > 0:
            pairs_c.append((int(cm_c[i, j]), i, j))
pairs_c.sort(reverse=True)
top5_combined_lines = [
    f"  {CLASS_NAMES[t]:<14s} → {CLASS_NAMES[p]:<14s}  {n_pair}"
    for n_pair, t, p in pairs_c[:5]
]
top5_combined_report = "\n".join(top5_combined_lines) if top5_combined_lines else "(none)"

combined_report = "\n".join([
    "# Combined held-out real test (games 2, 4, 5, 6)",
    f"games evaluated: {sorted(per_game_stats.keys())}",
    f"total frames: {combined_n_boards}",
    f"total squares: {len(combined_preds)}",
    f"combined per-square accuracy: {combined_per_square_acc:.4f}",
    f"combined per-board accuracy (all 64 correct): "
        f"{combined_n_all_correct}/{combined_n_boards} = {combined_per_board_acc:.4f}",
    f"mean squares correct / board: {combined_mean_correct:.2f} / 64",
    "",
    "## combined per-class",
    combined_per_class_report,
    "",
    "## top-5 confusion pairs",
    top5_combined_report,
])
combined_txt = f"{GAMES_EVAL_DIR}/real_test_combined.txt"
Path(combined_txt).write_text(combined_report)

print(f"\ncombined per-square acc: {combined_per_square_acc:.4f}")
print(f"combined per-board acc:  {combined_n_all_correct}/{combined_n_boards}="
      f"{combined_per_board_acc:.4f}")
print(f"combined mean-correct:   {combined_mean_correct:.1f}/64")
print(f"wrote {combined_txt}")
print(f"wrote {combined_cm_path}")
print()
print("top-5 combined confusion pairs:")
print(top5_combined_report)

print("\033[92m✓ Cell 20 — Aggregate held-out evaluation — OK\033[0m")








# %% [Cell 21 — Per-game summary table]
table_header = (
    f"  {'game':<6s} {'n_frames':>8s} {'n_squares':>10s} {'per_sq':>8s} "
    f"{'per_board':>10s} {'mean/64':>9s}"
)
table_rows = [table_header]
for N in sorted(per_game_stats):
    s = per_game_stats[N]
    table_rows.append(
        f"  game{N:<2d} {s['n_frames']:>8d} {s['n_squares']:>10d} "
        f"{s['per_square_acc']:>8.4f} {s['per_board_acc']:>10.4f} "
        f"{s['mean_correct']:>9.2f}"
    )
table_rows.append("  " + "-" * (len(table_header) - 2))
table_rows.append(
    f"  {'ALL':<6s} {combined_n_boards:>8d} {len(combined_preds):>10d} "
    f"{combined_per_square_acc:>8.4f} {combined_per_board_acc:>10.4f} "
    f"{combined_mean_correct:>9.2f}"
)
table_text = "\n".join(table_rows)
print(table_text)
per_game_summary_path = f"{GAMES_EVAL_DIR}/real_test_per_game_summary.txt"
Path(per_game_summary_path).write_text(table_text)
print(f"\nwrote {per_game_summary_path}")

print("\033[92m✓ Cell 21 — Per-game summary table — OK\033[0m")








# %% [Cell 22 — Real-test summary report]
# Pulls in: st_acc from Cell 15, rv_acc/n_boards/n_all_correct/mean_squares_correct
# /real_class_report from Cell 16, per_game_stats from Cell 19, combined_* from Cell 20.
real_summary_lines = [
    "# Held-out real test — games 2, 4, 5, 6",
    "",
    "## Setup note",
    "- **Held-out real test set:** games 2, 4, 5, 6 — never seen during training, "
    "never used to gate checkpoints.",
    "- **game7:** per-epoch real-image MONITOR during training; reported below for "
    "context only, NOT held-out.",
    "- **Synthetic test (held-out FENs from dataset_v1):** reported for sim-to-real "
    "gap measurement.",
    "",
    "## Held-out aggregate (games 2, 4, 5, 6 combined)",
    f"- **Per-square accuracy: {combined_per_square_acc:.4f}**",
    f"- **Per-board accuracy (all 64 correct): "
        f"{combined_n_all_correct}/{combined_n_boards} = {combined_per_board_acc:.4f}**",
    f"- Mean squares correct / board: **{combined_mean_correct:.2f} / 64**",
    f"- Total: {combined_n_boards} frames, {len(combined_preds):,} squares",
    "",
    "## Per-game",
    "```",
    table_text,
    "```",
    "",
    "## Combined per-class breakdown",
    "```",
    combined_per_class_report,
    "```",
    "",
    "## Top-5 confusion pairs (combined)",
    "```",
    top5_combined_report,
    "```",
    "",
    "## Sim-to-real gap",
    f"- synth_test per-square acc: **{st_acc:.4f}**",
    f"- held-out real per-square acc (games 2,4,5,6): "
        f"**{combined_per_square_acc:.4f}**",
    f"- **Gap: {st_acc:.4f} − {combined_per_square_acc:.4f} = "
        f"{st_acc - combined_per_square_acc:+.4f}**",
    "",
    "## Context — game7 real-monitor (NOT held-out)",
    f"- game7 per-square accuracy: {rv_acc:.4f}",
    f"- game7 per-board accuracy (all 64 correct): "
        f"{n_all_correct}/{n_boards} = {n_all_correct/n_boards:.4f}",
    f"- game7 mean squares correct / board: {mean_squares_correct:.2f} / 64",
    "- (game7 was the in-training real monitor; treat as in-distribution-real for "
    "this experiment, not as a held-out result.)",
    "",
    "### game7 per-class (for context)",
    "```",
    real_class_report,
    "```",
    "",
    "## Artifacts",
    f"- `{GAMES_EVAL_DIR}/real_test_combined.txt`",
    f"- `{GAMES_EVAL_DIR}/real_test_per_game_summary.txt`",
    f"- `{GAMES_EVAL_DIR}/real_test_game{{N}}.txt`  (per game)",
    f"- `{GAMES_EVAL_PLOTS_DIR}/real_test_combined_confusion.png`",
    f"- `{GAMES_EVAL_PLOTS_DIR}/real_test_game{{N}}_confusion.png`  (per game)",
]
real_summary_text = "\n".join(real_summary_lines)
real_summary_path = f"{GAMES_EVAL_DIR}/real_test_summary.md"
Path(real_summary_path).write_text(real_summary_text)
print(real_summary_text)
print(f"\nwrote {real_summary_path}")

print("\033[92m✓ Cell 22 — Real-test summary report — OK\033[0m")

