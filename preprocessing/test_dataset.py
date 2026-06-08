"""Smoke test for ChessSquareDataset: asserts len == 392448, pulls 3 seeded
samples (checking shape/dtype/range/label and latency), saves each crop through
the full __getitem__ path to ./results/dataset_smoke_test/, and prints an
epoch-time estimate.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/eladbaum/chess_project")
from preprocessing.chess_dataset import ChessSquareDataset


MANIFEST = "/home/eladbaum/chess_project/manifest.csv"
CORNERS = "/home/eladbaum/chess_project/corners.json"
OUT_DIR = Path("/home/eladbaum/chess_project/results/dataset_smoke_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EXPECTED_LEN = 392448

CLASS_NAMES = {
    0: "White Pawn",   1: "White Rook",   2: "White Knight",
    3: "White Bishop", 4: "White Queen",  5: "White King",
    6: "Black Pawn",   7: "Black Rook",   8: "Black Knight",
    9: "Black Bishop", 10: "Black Queen", 11: "Black King",
    12: "Empty",
}


def _font(size):
    for p in [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def safe_name(s):
    return s.replace(" ", "_").replace("/", "_")


def save_crop_png(tensor, label, idx, manifest_row, out_path):
    """Denormalize tensor → uint8 RGB → upscale → annotate with label."""
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    big = np.array(Image.fromarray(arr).resize((300, 300), Image.NEAREST))
    name = CLASS_NAMES[label]
    title = f"{name} ({label})"
    subtitle = (
        f"manifest idx={idx}   {manifest_row['view']}  "
        f"({manifest_row['board_row']}, {manifest_row['board_col']})\n"
        f"{manifest_row['source_image']}"
    )
    canvas = Image.new("RGB", (big.shape[1], big.shape[0] + 70), (240, 240, 240))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), title, fill=(0, 0, 0), font=_font(22))
    d.text((10, 36), subtitle, fill=(60, 60, 60), font=_font(11))
    canvas.paste(Image.fromarray(big), (0, 70))
    canvas.save(out_path)


def main():
    print("Instantiating ChessSquareDataset...")
    t0 = time.perf_counter()
    ds = ChessSquareDataset(MANIFEST, CORNERS, transform=None)
    init_dt = time.perf_counter() - t0
    print(f"  init time: {init_dt*1000:.0f}ms")

    n = len(ds)
    print(f"\nlen(dataset) = {n}")
    assert n == EXPECTED_LEN, f"expected {EXPECTED_LEN}, got {n}"
    print(f"  OK (matches expected {EXPECTED_LEN})")

    _ = ds[0]  # warm-up: first cv2/torch call is slower

    rng = random.Random(SEED)
    indices = [rng.randrange(n) for _ in range(3)]
    print(f"\nPulling 3 random samples (seed={SEED}): {indices}")

    times = []
    for i, idx in enumerate(indices, 1):
        t0 = time.perf_counter()
        tensor, label = ds[idx]
        dt = time.perf_counter() - t0
        times.append(dt)
        row = ds.manifest.iloc[idx]
        print(f"\n  sample {i}: manifest idx={idx}")
        print(f"    tensor.shape:  {tuple(tensor.shape)}")
        print(f"    tensor.dtype:  {tensor.dtype}")
        print(f"    tensor range:  [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
        print(f"    label:         {label}  ({CLASS_NAMES[label]})")
        print(f"    view:          {row['view']}")
        print(f"    (board_row, board_col):  ({row['board_row']}, {row['board_col']})")
        print(f"    source_image:  {row['source_image']}")
        print(f"    __getitem__ time:        {dt*1000:.1f}ms")

        out_path = OUT_DIR / f"sample_{i}_idx{idx}_{safe_name(CLASS_NAMES[label])}.png"
        save_crop_png(tensor, label, idx, row, out_path)
        print(f"    wrote {out_path.name}")

    avg = sum(times) / len(times)
    print(f"\n=== Throughput estimate ===")
    print(f"avg __getitem__ time: {avg*1000:.1f}ms")
    print(f"single-threaded epoch ({n} samples): "
          f"{avg * n:.0f}s = {avg * n / 60:.1f}min")
    # Rough multi-worker estimate (assumes near-linear scaling up to ~8 workers).
    for w in (4, 8, 16):
        secs = avg * n / w
        print(f"  with {w:2d} DataLoader workers:  ~{secs/60:.1f}min")

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
