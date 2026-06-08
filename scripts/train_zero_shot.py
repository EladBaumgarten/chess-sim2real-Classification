"""
Project 2 — Synthetic-to-Real Generalization for Chessboard Square Classification.

Trains a ResNet18 per-square classifier on the synthetic dataset (dataset_v2),
then evaluates zero-shot on real games 5, 6, 7.

Class encoding (per project spec, Project 2):
    0..5   = white P, R, N, B, Q, K
    6..11  = black p, r, n, b, q, k
    12     = empty
    (13 = OOD; project 1 only — not used here)

Run cells with Shift+Enter in VS Code (with the Jupyter extension installed).
"""

# %% [Imports + GPU check]
import os
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm

# Enable autoreload so edits to imported modules take effect without restart
try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except Exception:
    pass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__}")
print(f"Device:  {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"CUDA:    {torch.version.cuda}")







# %% [Paths + class encoding]
# Adjust these to wherever you unzipped things on the cluster.
DATA_ROOT  = Path("~/chess_project/data").expanduser()
SYNTH_ROOT = DATA_ROOT / "dataset_v2"          # synthetic training data
REAL_ROOTS = {                                  # real games for zero-shot eval
    "game5": DATA_ROOT / "game5_per_frame",
    "game6": DATA_ROOT / "game6_per_frame",
    "game7": DATA_ROOT / "game7_per_frame",
}

RESULTS_DIR = Path("./results")
CKPT_DIR    = Path("./checkpoints")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "wP", "wR", "wN", "wB", "wQ", "wK",   # 0-5
    "bP", "bR", "bN", "bB", "bQ", "bK",   # 6-11
    "empty",                               # 12
]
NUM_CLASSES = 13   # Project 2: no OOD class

FEN_TO_LABEL = {
    'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5,
    'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11,
}
EMPTY_LABEL = 12

SQUARE_SIZE = 64           # each square resized to this px
BOARD_SIZE  = 8 * SQUARE_SIZE
print(f"Board input: {BOARD_SIZE}x{BOARD_SIZE} → 64 squares of {SQUARE_SIZE}x{SQUARE_SIZE}")
EXPERIMENT_NAME = "no_class_weights"
EVAL_DIR = RESULTS_DIR / EXPERIMENT_NAME
EVAL_DIR.mkdir(parents=True, exist_ok=True)
print(f"Experiment: {EXPERIMENT_NAME}")







# %% [FEN parsing + view handling]
def fen_to_grid(fen):
    """FEN piece-placement string → 8x8 int grid.
    Output is in standard FEN orientation: grid[0] = rank 8 (top), grid[0,0] = a8."""
    placement = fen.split()[0]
    grid = np.full((8, 8), EMPTY_LABEL, dtype=np.int64)
    for r, rank_str in enumerate(placement.split('/')):
        c = 0
        for ch in rank_str:
            if ch.isdigit():
                c += int(ch)
            else:
                grid[r, c] = FEN_TO_LABEL[ch]
                c += 1
    return grid

import re

VIEW_TRANSFORMS = {
    "white":    lambda g: g.copy(),                    # no rotation
    "black":    lambda g: np.rot90(g, k=2).copy(),     # 180°
    "east":     lambda g: np.fliplr(g).copy(),         # horizontal flip
    "west":     lambda g: np.fliplr(g).copy(),         # horizontal flip
    "overhead": lambda g: np.fliplr(g).copy(),         # horizontal flip
}

def grid_for_view(fen, view):
    g = fen_to_grid(fen)
    v = str(view).lower().strip()
    if v not in VIEW_TRANSFORMS:
        print(f"⚠ Unknown view '{view}', using FEN-native")
        return g.copy()
    return VIEW_TRANSFORMS[v](g)


_VIEW_RE = re.compile(r"_(white|black|east|west|overhead)\.\w+$", re.IGNORECASE)

def view_from_filename(name):
    m = _VIEW_RE.search(str(name))
    return m.group(1).lower() if m else None

# Sanity test
_test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
print("White view starting position (top-left should be 7 = bR):")
print(grid_for_view(_test_fen, "white"))








# %% [Dataset]
class ChessBoardDataset(Dataset):
    """One item = one full board image, returned as (64, 3, S, S) squares + (64,) labels.

    Expects:
        gt.csv with columns: image_name, fen (and optionally view).
        images/ folder OR images at the root next to gt.csv (auto-detected).
    """
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD  = [0.229, 0.224, 0.225]

    def __init__(self, root, square_size=SQUARE_SIZE, augment=False):
        self.root = Path(root)
        self.square_size = square_size
        self.augment = augment

        # Locate CSV
        for name in ("gt.csv", "game.csv", "labels.csv"):
            csv_path = self.root / name
            if csv_path.exists():
                self.csv_path = csv_path
                break
        else:
            raise FileNotFoundError(f"No gt.csv / game.csv / labels.csv in {self.root}")

        # Locate images dir
        for sub in ("images", "imgs", "."):
            cand = self.root / sub
            if cand.exists() and any(p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                                     for p in cand.iterdir() if p.is_file()):
                self.images_dir = cand
                break
        else:
            raise FileNotFoundError(f"No images directory found in {self.root}")

        self.df = pd.read_csv(self.csv_path)
        # Normalize column names
        cmap = {c.lower(): c for c in self.df.columns}
        self.img_col  = cmap.get("image_name", cmap.get("image", list(self.df.columns)[0]))
        self.fen_col  = cmap.get("fen")
        # cam_view is the actual rendering camera (white/black/east/west/overhead).
        # The `view` column collapses white+black both to "white" and is wrong for labeling.
        self.view_col = cmap.get("cam_view", cmap.get("view"))
        if self.fen_col is None:
            raise ValueError(f"No 'fen' column in {self.csv_path}; got {list(self.df.columns)}")

        # Drop rows with missing files (sometimes csv is bigger than images/)
        before = len(self.df)
        self.df = self.df[self.df[self.img_col].apply(
            lambda n: (self.images_dir / str(n)).exists()
        )].reset_index(drop=True)
        if len(self.df) < before:
            print(f"  [{self.root.name}] dropped {before - len(self.df)} rows with missing images")

        # Augmentations only at the BOARD level (we don't geometric-augment
        # because that would invert square positions).
        if augment:
            self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2,
                                              saturation=0.2, hue=0.05)
        else:
            self.color_jitter = None

        self.normalize = T.Normalize(mean=self.NORM_MEAN, std=self.NORM_STD)

        # The CSV 'view' column is sometimes mislabeled. The filename suffix
        # (e.g. fen_00150_18_white.png) is the source of truth. Override.
        extracted = self.df[self.img_col].apply(view_from_filename)
        n_matched = extracted.notna().sum()
        if n_matched >= 0.5 * len(self.df):
            if self.view_col is None:
                self.view_col = "view"
                self.df[self.view_col] = extracted
            else:
                mismatches = ((extracted != self.df[self.view_col]) & extracted.notna()).sum()
                if mismatches > 0:
                    print(f"  [{self.root.name}] overriding view from filename for "
                          f"{mismatches}/{len(self.df)} mismatched rows")
                self.df[self.view_col] = extracted.fillna(self.df[self.view_col])


        print(f"  [{self.root.name}] {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / str(row[self.img_col])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((BOARD_SIZE, BOARD_SIZE), Image.BILINEAR)

        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL → (3, BOARD_SIZE, BOARD_SIZE), [0,1]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t   = torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)

        # Slice into 64 squares: (64, 3, S, S)
        s = self.square_size
        squares = t.unfold(1, s, s).unfold(2, s, s)            # (3, 8, 8, S, S)
        squares = squares.permute(1, 2, 0, 3, 4).contiguous()  # (8, 8, 3, S, S)
        squares = squares.view(64, 3, s, s)

        # Normalize each square (ImageNet stats — using ResNet18 pretrained)
        squares = (squares - torch.tensor(self.NORM_MEAN).view(1, 3, 1, 1)) \
                  / torch.tensor(self.NORM_STD).view(1, 3, 1, 1)

        # Labels
        fen  = row[self.fen_col]
        view = row[self.view_col] if self.view_col else "white"
        labels = grid_for_view(fen, view).flatten()           # (64,)
        labels = torch.from_numpy(labels).long()

        return squares, labels



# %% [Sanity check — one sample per view]
synth_ds_preview = ChessBoardDataset(SYNTH_ROOT, augment=False)
print(f"Synthetic dataset: {len(synth_ds_preview)} boards")

def _unnorm(t):
    m = torch.tensor(ChessBoardDataset.NORM_MEAN).view(1, 3, 1, 1)
    s = torch.tensor(ChessBoardDataset.NORM_STD).view(1, 3, 1, 1)
    return (t * s + m).clamp(0, 1)

# Pick one random sample per unique view
unique_views = (synth_ds_preview.df[synth_ds_preview.view_col].unique().tolist()
                if synth_ds_preview.view_col else ["(no view col)"])
rng = np.random.default_rng(SEED)
chosen = []
for v in unique_views:
    if synth_ds_preview.view_col:
        rows = synth_ds_preview.df[synth_ds_preview.df[synth_ds_preview.view_col] == v]
        idx = int(rng.choice(rows.index))
    else:
        idx = int(rng.integers(len(synth_ds_preview)))
    chosen.append((v, idx))

fig, axes = plt.subplots(len(chosen), 2, figsize=(11, 4.5 * len(chosen)))
if len(chosen) == 1:
    axes = axes[None, :]

for row_i, (view, idx) in enumerate(chosen):
    squares, labels = synth_ds_preview[idx]
    info = synth_ds_preview.df.iloc[idx]
    print(f"view={view:9s}  idx={idx}  file={info[synth_ds_preview.img_col]}")

    sq_un = _unnorm(squares)
    bd = sq_un.view(8, 8, 3, SQUARE_SIZE, SQUARE_SIZE).permute(2, 0, 3, 1, 4)
    bd = bd.reshape(3, BOARD_SIZE, BOARD_SIZE).permute(1, 2, 0).numpy()

    ax_img, ax_lbl = axes[row_i]
    ax_img.imshow(bd); ax_img.axis("off")
    ax_img.set_title(f"view = {view!r}  ({info[synth_ds_preview.img_col]})")

    ax_lbl.imshow(np.ones((8, 8, 3)) * 0.97)
    for i in range(8):
        for j in range(8):
            lbl = labels.view(8, 8)[i, j].item()
            ax_lbl.text(j, i, CLASS_NAMES[lbl], ha="center", va="center",
                        fontsize=8, color="gray" if lbl == 12 else "darkred",
                        fontweight="normal" if lbl == 12 else "bold")
    ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
    ax_lbl.set_title("labels (image-oriented)")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "sanity_check_all_views.png", dpi=120)
plt.show()






# %% [Train / val split of the synthetic dataset]
BATCH_SIZE = 32        # boards per batch → 32 * 64 = 2048 squares/batch
NUM_WORKERS = 4

# Two underlying datasets: one with augment, one without (for val).
val_ds_full   = ChessBoardDataset(SYNTH_ROOT, augment=False)
train_ds_full = ChessBoardDataset(SYNTH_ROOT, augment=True)

# Deterministic split on indices so train/val don't overlap.
full_idxs = np.arange(len(val_ds_full))
rng = np.random.default_rng(SEED)
rng.shuffle(full_idxs)
n_val   = int(0.1 * len(full_idxs))
val_idx = set(full_idxs[:n_val].tolist())
train_idx = [i for i in full_idxs if i not in val_idx]
val_idx   = sorted(val_idx)

print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

train_subset = torch.utils.data.Subset(train_ds_full, train_idx)
val_subset   = torch.utils.data.Subset(val_ds_full,   val_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)






# %% [Model: ResNet18 head replaced for 13 classes]
def build_model(num_classes=NUM_CLASSES, pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

model = build_model().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"ResNet18 — trainable params: {n_params/1e6:.2f} M")






# %% [Loss + optimizer]
# Compute class weights from a small sample to counter heavy class imbalance
# (empty squares dominate). Cheap: scan ~200 boards.
print("Estimating class frequencies for class weighting…")
counter = Counter()
for i in tqdm(train_idx[:200], desc="freq"):
    _, lbl = train_ds_full[i]
    counter.update(lbl.tolist())
freqs = np.array([counter.get(c, 1) for c in range(NUM_CLASSES)], dtype=np.float64)
weights = freqs.sum() / (NUM_CLASSES * freqs)
weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
print("class weights:", {CLASS_NAMES[i]: round(w, 3) for i, w in enumerate(weights.tolist())})

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)






# %% [Train / eval helpers]
def run_epoch(model, loader, criterion, optimizer=None, desc=""):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []
    grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with grad_ctx:
        for boards, labels in tqdm(loader, desc=desc, leave=False):
            B = boards.size(0)
            squares = boards.view(B * 64, 3, SQUARE_SIZE, SQUARE_SIZE).to(DEVICE, non_blocking=True)
            targets = labels.view(B * 64).to(DEVICE, non_blocking=True)
            logits = model(squares)
            loss = criterion(logits, targets)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = logits.argmax(1)
            total_loss += loss.item() * targets.size(0)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            if not train_mode:
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
    avg_loss = total_loss / total
    acc = correct / total
    if not train_mode:
        return avg_loss, acc, torch.cat(all_preds), torch.cat(all_targets)
    return avg_loss, acc


def board_accuracy(preds, targets):
    """Fraction of boards where ALL 64 squares are correct."""
    p = preds.view(-1, 64)
    t = targets.view(-1, 64)
    return (p == t).all(dim=1).float().mean().item()



# %% [Overfit sanity check: can the model memorize a tiny batch?]
# If yes → labels, model, loss, optimizer are all wired up correctly.
# If no  → something is broken; don't bother running full training.

import copy

# Take 4 boards from the train set
overfit_indices = train_idx[:4]
overfit_subset  = torch.utils.data.Subset(train_ds_full, overfit_indices)
overfit_loader  = DataLoader(overfit_subset, batch_size=4, shuffle=False, num_workers=0)

# Fresh copy of the model + a fast optimizer (no scheduler, no weight decay)
overfit_model = build_model().to(DEVICE)
overfit_opt   = optim.AdamW(overfit_model.parameters(), lr=1e-3)
overfit_crit  = nn.CrossEntropyLoss()   # no class weighting — pure memorization

print(f"Overfitting on {len(overfit_subset)} boards × 64 squares = {len(overfit_subset)*64} square examples")

losses, accs = [], []
N_STEPS = 100
overfit_model.train()
pbar = tqdm(range(N_STEPS), desc="overfit")
for step in pbar:
    for boards, labels in overfit_loader:
        B = boards.size(0)
        squares = boards.view(B*64, 3, SQUARE_SIZE, SQUARE_SIZE).to(DEVICE)
        targets = labels.view(B*64).to(DEVICE)
        logits = overfit_model(squares)
        loss = overfit_crit(logits, targets)

        overfit_opt.zero_grad()
        loss.backward()
        overfit_opt.step()

        with torch.no_grad():
            acc = (logits.argmax(1) == targets).float().mean().item()
        losses.append(loss.item()); accs.append(acc)
    pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses); axes[0].set_title("Overfit loss (should → 0)")
axes[0].set_xlabel("step"); axes[0].grid(alpha=0.3); axes[0].set_yscale("log")
axes[1].plot(accs);   axes[1].set_title("Overfit accuracy (should → 1.0)")
axes[1].set_xlabel("step"); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "overfit_check.png", dpi=120)
plt.show()

print(f"\nFinal loss: {losses[-1]:.6f}")
print(f"Final acc : {accs[-1]:.4f}")
print()
if accs[-1] > 0.98:
    print("✓ Model can memorize 4 boards. Pipeline is healthy — proceed to full training.")
elif accs[-1] > 0.85:
    print("⚠ Almost overfit but not quite. Could be class imbalance dominating loss.")
    print("  If accuracy keeps climbing past step 100, run more steps.")
else:
    print("✗ Model cannot overfit 4 boards. STOP and debug:")
    print("  - check labels are valid (0..12)")
    print("  - check images aren't all zeros after normalization")
    print("  - check optimizer is actually updating params")




# %% [Training loop]
EPOCHS = 10
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_board_acc": []}
best_val = 0.0

for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, desc=f"ep{ep} train")
    val_loss, val_acc, vp, vt = run_epoch(model, val_loader, criterion, None, desc=f"ep{ep} val")
    val_board = board_accuracy(vp, vt)
    scheduler.step()

    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["val_loss"].append(val_loss);  history["val_acc"].append(val_acc)
    history["val_board_acc"].append(val_board)

    print(f"epoch {ep:2d} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"val loss {val_loss:.4f} sq-acc {val_acc:.4f} board-acc {val_board:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save({"model": model.state_dict(), "epoch": ep, "val_acc": val_acc},
                   CKPT_DIR / f"resnet18_{EXPERIMENT_NAME}_best.pt")
        print(f"  ✓ saved best (val sq-acc {val_acc:.4f})")

torch.save({"model": model.state_dict(), "epoch": EPOCHS, "val_acc": val_acc},
           CKPT_DIR / f"resnet18_{EXPERIMENT_NAME}_last.pt")
with open(EVAL_DIR / "history.json", "w") as f:
    json.dump(history, f, indent=2)






# %% [Training curves]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history["train_acc"], label="train square-acc")
axes[1].plot(history["val_acc"], label="val square-acc")
axes[1].plot(history["val_board_acc"], label="val board-acc", linestyle="--")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_curves.png", dpi=120)
plt.show()








# %% [Confusion matrix helper]
def plot_confusion(preds, targets, title, save_name=None):
    cm = confusion_matrix(targets.numpy(), preds.numpy(), labels=list(range(NUM_CLASSES)))
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sum
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={"label": "row-normalized"}, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_name:
        plt.savefig(RESULTS_DIR / save_name, dpi=120)
    plt.show()

    sq_acc = (preds == targets).float().mean().item()
    board_acc = board_accuracy(preds, targets)
    print(f"{title}")
    print(f"  square accuracy : {sq_acc:.4f}")
    print(f"  board  accuracy : {board_acc:.4f}")
    print("classification report:")
    print(classification_report(targets.numpy(), preds.numpy(),
                                labels=list(range(NUM_CLASSES)),
                                target_names=CLASS_NAMES, zero_division=0))





# %% [Load best checkpoint + eval on synthetic val]
ckpt = torch.load(CKPT_DIR / f"resnet18_{EXPERIMENT_NAME}_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model"])
print(f"Loaded best checkpoint from epoch {ckpt['epoch']} (val sq-acc={ckpt['val_acc']:.4f})")

_, _, preds_synth, targets_synth = run_epoch(model, val_loader, criterion, None, desc="synth val")
plot_confusion(preds_synth, targets_synth,
               "Synthetic val — confusion matrix",
               "cm_synth_val.png")






# %% [Build real-data loaders for games 5, 6, 7]
import zipfile
import shutil

REAL_GAME_CSV_NAMES = {
    "game5": "game5.csv",
    "game6": "game6.csv",
    "game7": "game7.csv",
}

def ensure_unzipped(target_dir):
    target_dir = Path(target_dir)
    if not (target_dir.exists() and any(target_dir.iterdir())):
        zip_path = target_dir.with_suffix(".zip")
        if not zip_path.exists():
            return False
        tmp_dir = target_dir.parent / f".tmp_{target_dir.name}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        print(f"  extracting {zip_path.name} → {target_dir} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        entries = [p for p in tmp_dir.iterdir()]
        if len(entries) == 1 and entries[0].is_dir():
            shutil.move(str(entries[0]), str(target_dir))
            shutil.rmtree(tmp_dir)
        else:
            shutil.move(str(tmp_dir), str(target_dir))
    # Real-game zips contain tagged_images/, but our dataset/build code reads images/.
    # Create a symlink images -> tagged_images so everything else works unmodified.
    tagged = target_dir / "tagged_images"
    images = target_dir / "images"
    if tagged.exists() and not images.exists():
        images.symlink_to(tagged.resolve(), target_is_directory=True)
    return target_dir.exists()


def build_realgame_gt(game_root, game_name):
    game_root = Path(game_root)
    images_dir = game_root / "images"          # was tagged_images; now symlinked
    csv_path = game_root / REAL_GAME_CSV_NAMES.get(game_name, f"{game_name}.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"No {csv_path.name} in {game_root}")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.sort_values("from_frame").reset_index(drop=True)
    img_files = sorted(images_dir.glob("frame_*.*"))
    rows = []
    for img_path in img_files:
        try:
            frame_num = int(img_path.stem.split("_")[-1])
        except ValueError:
            continue
        matching = df[df["from_frame"] <= frame_num]
        if len(matching) == 0:
            continue
        match = matching.iloc[-1]
        rows.append({
            "image_name": img_path.name,
            "fen": match["fen"],
            "view": match.get("view", "white"),
        })
    out_df = pd.DataFrame(rows)
    out_path = game_root / "gt.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  [{game_name}] wrote {out_path} ({len(out_df)} rows)")
    return out_path


real_loaders = {}
for name, root in REAL_ROOTS.items():
    if not ensure_unzipped(root):
        print(f"⚠ {root} not found — skipping.")
        continue
    if not (root / "gt.csv").exists():
        build_realgame_gt(root, name)
    ds = ChessBoardDataset(root, augment=False)
    real_loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  {name}: {len(ds)} samples")




# %% Verify real-game label orientation
ds = ChessBoardDataset(REAL_ROOTS["game5"], augment=False)
# Print what columns its CSV has
print("game5 columns in gt.csv:", list(ds.df.columns))
print("unique views:", ds.df[ds.view_col].unique())
print("first row:", ds.df.iloc[0].to_dict())

# Visualize one sample
squares, labels = ds[0]
sq_un = _unnorm(squares)
bd = sq_un.view(8, 8, 3, SQUARE_SIZE, SQUARE_SIZE).permute(2, 0, 3, 1, 4)
bd = bd.reshape(3, BOARD_SIZE, BOARD_SIZE).permute(1, 2, 0).numpy()
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(bd); axes[0].set_title("real game5 sample 0"); axes[0].axis("off")
axes[1].imshow(np.ones((8, 8, 3)) * 0.97)
for i in range(8):
    for j in range(8):
        lbl = labels.view(8, 8)[i, j].item()
        axes[1].text(j, i, CLASS_NAMES[lbl], ha="center", va="center",
                     fontsize=8, color="gray" if lbl == 12 else "darkred")
axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].set_title("labels")
plt.tight_layout(); plt.show()





# %% Verify on a mid-game sample (asymmetric position)
ds = ChessBoardDataset(REAL_ROOTS["game5"], augment=False)
mid_idx = len(ds) // 2     # middle of the game
squares, labels = ds[mid_idx]
info = ds.df.iloc[mid_idx]
print(f"frame: {info['image_name']}  fen: {info['fen']}  view: {info['view']}")

sq_un = _unnorm(squares)
bd = sq_un.view(8, 8, 3, SQUARE_SIZE, SQUARE_SIZE).permute(2, 0, 3, 1, 4)
bd = bd.reshape(3, BOARD_SIZE, BOARD_SIZE).permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(bd); axes[0].set_title(f"game5 mid-game ({info['image_name']})"); axes[0].axis("off")
axes[1].imshow(np.ones((8, 8, 3)) * 0.97)
for i in range(8):
    for j in range(8):
        lbl = labels.view(8, 8)[i, j].item()
        axes[1].text(j, i, CLASS_NAMES[lbl], ha="center", va="center",
                     fontsize=8, color="gray" if lbl == 12 else "darkred",
                     fontweight="normal" if lbl == 12 else "bold")
axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].set_title("labels")
plt.tight_layout(); plt.show()







# %% [Zero-shot evaluation on real games]
all_preds = []
all_targets = []
per_game_results = {}

for name, loader in real_loaders.items():
    _, acc, p, t = run_epoch(model, loader, criterion, None, desc=f"zero-shot {name}")
    bacc = board_accuracy(p, t)
    per_game_results[name] = {"square_acc": acc, "board_acc": bacc, "n_boards": len(loader.dataset)}
    print(f"  {name}: square_acc={acc:.4f}  board_acc={bacc:.4f}  (n={len(loader.dataset)})")
    plot_confusion(p, t, f"Zero-shot on {name}", f"cm_zeroshot_{name}.png")
    all_preds.append(p); all_targets.append(t)

# Combined across all real games
if all_preds:
    combined_p = torch.cat(all_preds)
    combined_t = torch.cat(all_targets)
    plot_confusion(combined_p, combined_t,
                   "Zero-shot — all real games combined",
                   "cm_zeroshot_combined.png")

with open(RESULTS_DIR / "zero_shot_results.json", "w") as f:
    json.dump(per_game_results, f, indent=2)
print("Per-game results saved to results/zero_shot_results.json")





# %% [Visualize a few real predictions side-by-side]
def visualize_predictions(loader, n=4, title_prefix=""):
    model.eval()
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = axes[None, :]
    seen = 0
    for boards, labels in loader:
        B = boards.size(0)
        with torch.no_grad():
            squares = boards.view(B * 64, 3, SQUARE_SIZE, SQUARE_SIZE).to(DEVICE)
            preds = model(squares).argmax(1).view(B, 8, 8).cpu()
        targets = labels.view(B, 8, 8)
        for b in range(B):
            if seen >= n: break
            ax_img, ax_grid = axes[seen]
            sq_un = _unnorm(boards[b])
            bd = sq_un.view(8, 8, 3, SQUARE_SIZE, SQUARE_SIZE).permute(2, 0, 3, 1, 4)
            bd = bd.reshape(3, BOARD_SIZE, BOARD_SIZE).permute(1, 2, 0).numpy()
            ax_img.imshow(bd); ax_img.axis("off")
            ax_img.set_title(f"{title_prefix} input #{seen}")

            grid_view = np.ones((8, 8, 3))
            for i in range(8):
                for j in range(8):
                    p = preds[b, i, j].item()
                    g = targets[b, i, j].item()
                    color = "green" if p == g else "red"
                    grid_view[i, j] = (0.85, 0.95, 0.85) if p == g else (1.0, 0.85, 0.85)
                    ax_grid.text(j, i, f"{CLASS_NAMES[p]}\n({CLASS_NAMES[g]})",
                                 ha="center", va="center", fontsize=7, color=color)
            ax_grid.imshow(grid_view); ax_grid.set_xticks([]); ax_grid.set_yticks([])
            ax_grid.set_title("pred (gt)")
            seen += 1
        if seen >= n: break
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"qualitative_{title_prefix.strip().replace(' ', '_') or 'preds'}.png", dpi=120)
    plt.show()

for name, loader in real_loaders.items():
    visualize_predictions(loader, n=3, title_prefix=name)


# %% [Summary]
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Synthetic val sq-acc: {ckpt['val_acc']:.4f}")
print()
print("Zero-shot on real games:")
for name, r in per_game_results.items():
    print(f"  {name}: square-acc={r['square_acc']:.4f}  board-acc={r['board_acc']:.4f}")
print()
print("Artifacts:")
print(f"  best model:  {CKPT_DIR/'resnet18_synth_best.pt'}")
print(f"  curves:      {RESULTS_DIR/'training_curves.png'}")
print(f"  CMs:         {RESULTS_DIR}/cm_*.png")
print(f"  qualitative: {RESULTS_DIR}/qualitative_*.png")




# %% [Persist all eval artifacts] save results, predictions, classification reports, confusion matrices, and a summary JSON with all numbers.
import pickle

EVAL_DIR = RESULTS_DIR / "baseline_synth_only"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Helper: capture classification report as a string (not just printed)
def save_classification_report(preds, targets, name):
    rep = classification_report(targets.numpy(), preds.numpy(),
                                labels=list(range(NUM_CLASSES)),
                                target_names=CLASS_NAMES, zero_division=0)
    with open(EVAL_DIR / f"report_{name}.txt", "w") as f:
        f.write(rep)
    return rep

# Save raw predictions + targets per game
torch.save({"preds": preds_synth, "targets": targets_synth},
           EVAL_DIR / "predictions_synth_val.pt")
save_classification_report(preds_synth, targets_synth, "synth_val")

per_game_preds = {}
for name, loader in real_loaders.items():
    # Re-run to capture preds/targets (cheap, same forward pass)
    _, _, p, t = run_epoch(model, loader, criterion, None, desc=f"capture {name}")
    per_game_preds[name] = {"preds": p, "targets": t}
    torch.save({"preds": p, "targets": t},
               EVAL_DIR / f"predictions_{name}.pt")
    save_classification_report(p, t, name)

# Combined
combined_p = torch.cat([per_game_preds[n]["preds"]   for n in per_game_preds])
combined_t = torch.cat([per_game_preds[n]["targets"] for n in per_game_preds])
torch.save({"preds": combined_p, "targets": combined_t},
           EVAL_DIR / "predictions_zeroshot_combined.pt")
save_classification_report(combined_p, combined_t, "zeroshot_combined")

# A single summary JSON with all numbers
import datetime as dt
summary = {
    "experiment": "baseline_synth_only",
    "timestamp": dt.datetime.now().isoformat(),
    "model": "resnet18 (ImageNet pretrained, 13-class head)",
    "training": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "class_weighting": "inverse-frequency",
        "augmentation": "color jitter only",
        "square_size": SQUARE_SIZE,
        "best_val_sq_acc": ckpt["val_acc"],
        "best_epoch": ckpt["epoch"],
    },
    "results": {
        "synth_val_sq_acc": float((preds_synth == targets_synth).float().mean()),
        "synth_val_board_acc": board_accuracy(preds_synth, targets_synth),
        "zero_shot_per_game": per_game_results,
        "zero_shot_combined_sq_acc": float((combined_p == combined_t).float().mean()),
        "zero_shot_combined_board_acc": board_accuracy(combined_p, combined_t),
    },
}
with open(EVAL_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"All artifacts saved to {EVAL_DIR}/")
print("\nFiles:")
for p in sorted(EVAL_DIR.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size:,} bytes)")
# %%
