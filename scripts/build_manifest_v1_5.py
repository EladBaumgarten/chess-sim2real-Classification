"""
build_manifest_v1_5.py — produce data/dataset_v1.5/manifest.csv with one row
per (image, board square). Adapted from build_manifest.py for the v1.5
labels.csv schema (image_name / fen / view / source_dataset / ...).

Source of truth: data/dataset_v1.5/labels.csv (one row per image, 7,665 rows).
Output schema (matches what ChessSquareDataset and train_baseline expect):
    source_image, view, board_row, board_col, label, fen

Expected total: 7,665 × 64 = 490,560 rows.

Sanity output: per-class distribution, per-view balance, per-source-dataset
breakdown, and warnings if anything looks off.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/eladbaum/chess_project")
from scripts.fen_to_grid import fen_to_label_grid


V1_5_DIR = Path("/home/eladbaum/chess_project/data/dataset_v1.5")
LABELS_CSV = V1_5_DIR / "labels.csv"
OUT_CSV = V1_5_DIR / "manifest.csv"

VALID_VIEWS = ("overhead", "west", "east")
VALID_SOURCES = ("v1", "legacy")

CLASS_NAMES = {
    0: "P (white pawn)",   1: "R (white rook)",   2: "N (white knight)",
    3: "B (white bishop)", 4: "Q (white queen)",  5: "K (white king)",
    6: "p (black pawn)",   7: "r (black rook)",   8: "n (black knight)",
    9: "b (black bishop)", 10: "q (black queen)", 11: "k (black king)",
    12: ".  (empty)",
}


def main():
    print(f"Reading {LABELS_CSV} ...")
    with LABELS_CSV.open() as f:
        label_rows = list(csv.DictReader(f))
    print(f"  {len(label_rows)} label entries")

    # Build manifest
    print(f"\nBuilding manifest ...")
    manifest_rows = []
    n_processed = 0
    n_skipped_view = 0
    n_skipped_fen = 0
    for row in label_rows:
        image_name = row["image_name"]
        fen = row["fen"]
        view = row["view"]
        source = row.get("source_dataset", "")

        if view not in VALID_VIEWS:
            print(f"  skip {image_name}: unknown view {view!r}")
            n_skipped_view += 1
            continue
        try:
            grid = fen_to_label_grid(fen, view)
        except ValueError as e:
            print(f"  skip {image_name}: invalid FEN ({e})")
            n_skipped_fen += 1
            continue

        for r in range(8):
            for c in range(8):
                manifest_rows.append({
                    "source_image": image_name,
                    "view": view,
                    "board_row": r,
                    "board_col": c,
                    "label": int(grid[r, c]),
                    "fen": fen,
                    "source_dataset": source,
                })
        n_processed += 1

    print(f"\nProcessed: {n_processed}   "
          f"skipped (unknown view): {n_skipped_view}   "
          f"skipped (bad FEN): {n_skipped_fen}")

    expected_total = n_processed * 64
    assert len(manifest_rows) == expected_total, (
        f"row count mismatch: got {len(manifest_rows)}, expected {expected_total}"
    )

    # Exactly 64 rows per image
    per_image = Counter(r["source_image"] for r in manifest_rows)
    bad = [(k, v) for k, v in per_image.items() if v != 64]
    assert not bad, f"images with != 64 rows: {bad[:5]}"

    print(f"\nManifest:        {len(manifest_rows):>8d} rows")
    print(f"Expected:        {7665 * 64:>8d} rows  (7665 labelled × 64)")
    print(f"Matches:         {'YES' if len(manifest_rows) == 7665 * 64 else 'NO'}")

    # Write CSV
    print(f"\nWriting {OUT_CSV} ...")
    fieldnames = ["source_image", "view", "board_row", "board_col",
                  "label", "fen", "source_dataset"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"  wrote {len(manifest_rows)} rows  "
          f"({OUT_CSV.stat().st_size/1024/1024:.1f} MB)")

    # ---- Class distribution ----
    counts = Counter(r["label"] for r in manifest_rows)
    total = sum(counts.values())
    print(f"\nClass distribution (n={total}):")
    print(f"  {'cls':>3s}  {'name':<22s}  {'count':>9s}   {'pct':>7s}")
    for cls in range(13):
        n = counts.get(cls, 0)
        pct = 100 * n / total
        bar = "█" * int(pct / 2)
        print(f"  {cls:>3d}  {CLASS_NAMES[cls]:<22s}  {n:>9d}   {pct:>6.2f}%  {bar}")

    # ---- Sanity warnings ----
    warns = []
    empty_pct = 100 * counts.get(12, 0) / total
    if not 40 <= empty_pct <= 70:
        warns.append(f"empty cells {empty_pct:.2f}% outside expected [40%, 70%]")
    if counts.get(5, 0) == 0:
        warns.append("zero white kings (class 5)")
    if counts.get(11, 0) == 0:
        warns.append("zero black kings (class 11)")
    white_total = sum(counts.get(c, 0) for c in range(6))
    black_total = sum(counts.get(c, 0) for c in range(6, 12))
    if white_total > 0 and black_total > 0:
        ratio = max(white_total, black_total) / min(white_total, black_total)
        if ratio > 1.2:
            warns.append(f"white/black piece imbalance: white={white_total}, "
                         f"black={black_total} (ratio {ratio:.2f})")
    if warns:
        print(f"\n  WARNINGS:")
        for w in warns:
            print(f"    - {w}")
    else:
        print(f"\n  (no distribution warnings)")

    # ---- Per-view balance ----
    view_counts = Counter(r["view"] for r in manifest_rows)
    print(f"\nPer-view balance:")
    for view in VALID_VIEWS:
        n = view_counts[view]
        n_imgs = n // 64
        print(f"  {view:<10s}  {n:>9d} rows  ({n_imgs:>5d} images)")

    # ---- Per-source breakdown ----
    src_counts = Counter(r["source_dataset"] for r in manifest_rows)
    print(f"\nPer-source breakdown:")
    for src in VALID_SOURCES:
        n = src_counts[src]
        n_imgs = n // 64
        print(f"  {src:<10s}  {n:>9d} rows  ({n_imgs:>5d} images)")

    # ---- Per-(source × view) sanity ----
    sv_counts = Counter((r["source_dataset"], r["view"]) for r in manifest_rows)
    print(f"\nPer-(source × view) breakdown:")
    for src in VALID_SOURCES:
        for view in VALID_VIEWS:
            n_imgs = sv_counts.get((src, view), 0) // 64
            print(f"  {src:6s}  {view:8s}  {n_imgs:>5d} images")


if __name__ == "__main__":
    main()
