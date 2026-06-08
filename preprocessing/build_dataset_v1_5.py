"""
build_dataset_v1_5.py — create data/dataset_v1.5/ by merging:
    data/dataset_v1/   (6,135 images, 6,132 labelled — 3 fen_diag excluded)
    data/dataset/      (1,533 images, all labelled — the 1.5K legacy set)

Both sources use the same chess_position_api rectification family and the
same rot180 FEN→grid transform per VIEW_ORIENTATIONS (verified earlier).
No filename collisions between the two (v1 uses fen_XXXX_rY_Z_view.png,
legacy uses fen_XXXX_Z_view.png).

Strategy:
  - Symlink images into dataset_v1.5/images/  (no disk duplication; sources
    are stable synthetic renders so symlinks are safe).
  - Write a unified labels.csv with columns:
        image_name, fen, view, source_dataset, source_game, source_frame,
        fen_idx, run_idx
  - Validate end-to-end: every image has a label, every label has an
    existing image file, every FEN parses through fen_to_label_grid, every
    view is in {overhead, west, east}.
  - Write build_log.txt with counts + any warnings.
  - Write README.md documenting provenance.

Expected output: 7,665 images and 7,665 label rows.
"""

import csv
import sys
import time
from collections import Counter
from pathlib import Path

# fen_to_grid imports from `scripts.view_orientations`, so add the project root
# (parent of `scripts/`) to sys.path so the dotted import resolves.
sys.path.insert(0, "/home/eladbaum/chess_project")
sys.path.insert(0, "/home/eladbaum/chess_project/preprocessing")
from fen_to_grid import fen_to_label_grid


PROJECT_ROOT = Path("/home/eladbaum/chess_project")
SRC_V1 = PROJECT_ROOT / "data/dataset_v1"
SRC_LEGACY = PROJECT_ROOT / "data/dataset"
OUT = PROJECT_ROOT / "data/dataset_v1.5"

VALID_VIEWS = ("overhead", "west", "east")
UNIFIED_FIELDS = [
    "image_name", "fen", "view",
    "source_dataset", "source_game", "source_frame",
    "fen_idx", "run_idx",
]


def view_from_camera_field(camera):
    """'1_overhead' / '2_west' / '3_east'  →  'overhead' / 'west' / 'east'."""
    return camera.split("_", 1)[1] if "_" in camera else camera


def load_v1_labels():
    """Return list of unified dicts from dataset_v1/labels.csv (6,132 rows)."""
    out = []
    with (SRC_V1 / "labels.csv").open() as f:
        for r in csv.DictReader(f):
            image_name = Path(r["image_path"]).name
            out.append({
                "image_name": image_name,
                "fen": r["fen"],
                "view": view_from_camera_field(r["camera"]),
                "source_dataset": "v1",
                "source_game": r.get("source_game", ""),
                "source_frame": r.get("source_frame", ""),
                "fen_idx": r.get("fen_idx", ""),
                "run_idx": r.get("run_idx", ""),
            })
    return out


def load_legacy_labels():
    """Return list of unified dicts from dataset/labels.csv (1,533 rows).
    Legacy labels.csv has no run_idx — emit empty string for that column."""
    out = []
    with (SRC_LEGACY / "labels.csv").open() as f:
        for r in csv.DictReader(f):
            image_name = Path(r["image_path"]).name
            out.append({
                "image_name": image_name,
                "fen": r["fen"],
                "view": view_from_camera_field(r["camera"]),
                "source_dataset": "legacy",
                "source_game": r.get("source_game", ""),
                "source_frame": r.get("source_frame", ""),
                "fen_idx": r.get("fen_idx", ""),
                "run_idx": "",
            })
    return out


def make_symlinks(rows, log_lines):
    """For each row, create a symlink in OUT/images/ pointing at the source
    PNG. Raises if the source PNG is missing — that's a hard error."""
    images_dir = OUT / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    n_created = 0
    n_existed = 0
    missing = []
    for r in rows:
        name = r["image_name"]
        src_root = SRC_V1 if r["source_dataset"] == "v1" else SRC_LEGACY
        src = src_root / "images" / name
        if not src.exists():
            missing.append(str(src))
            continue
        dst = images_dir / name
        if dst.is_symlink() or dst.exists():
            # Verify it points at the right place; otherwise re-create
            if dst.is_symlink() and dst.resolve() == src.resolve():
                n_existed += 1
                continue
            dst.unlink()
        dst.symlink_to(src.resolve())
        n_created += 1
    log_lines.append(f"symlinks created: {n_created}")
    log_lines.append(f"symlinks already correct: {n_existed}")
    if missing:
        log_lines.append(f"MISSING SOURCE FILES ({len(missing)}):")
        for m in missing[:10]:
            log_lines.append(f"  {m}")
        raise RuntimeError(
            f"{len(missing)} source images missing; cannot build dataset_v1.5"
        )
    return n_created, n_existed


def write_labels_csv(rows):
    """Write the unified labels.csv. Returns the path."""
    path = OUT / "labels.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UNIFIED_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def validate(rows, log_lines):
    """End-to-end validation checks. Append findings to log_lines.
    Raises RuntimeError if any hard invariant fails."""
    errors = []
    warnings = []

    # 1. Every label has an existing image file in OUT/images/
    images_dir = OUT / "images"
    on_disk = {p.name for p in images_dir.iterdir() if p.is_symlink() or p.is_file()}
    label_names = [r["image_name"] for r in rows]
    label_name_set = set(label_names)

    missing_on_disk = [n for n in label_names if n not in on_disk]
    if missing_on_disk:
        errors.append(f"{len(missing_on_disk)} label rows point at missing files "
                      f"(e.g. {missing_on_disk[:3]})")

    extras_on_disk = on_disk - label_name_set
    if extras_on_disk:
        errors.append(f"{len(extras_on_disk)} image files on disk have no label row "
                      f"(e.g. {sorted(extras_on_disk)[:3]})")

    # 2. No duplicate image_name entries
    dup = [n for n, c in Counter(label_names).items() if c > 1]
    if dup:
        errors.append(f"{len(dup)} duplicate image_name entries (e.g. {dup[:3]})")

    # 3. Every view is in VALID_VIEWS
    bad_views = [r["image_name"] for r in rows if r["view"] not in VALID_VIEWS]
    if bad_views:
        errors.append(f"{len(bad_views)} rows have invalid view "
                      f"(e.g. {bad_views[:3]})")

    # 4. Every FEN parses cleanly through fen_to_label_grid
    bad_fens = []
    for r in rows:
        try:
            fen_to_label_grid(r["fen"], r["view"])
        except Exception as e:
            bad_fens.append((r["image_name"], r["fen"], str(e)))
            if len(bad_fens) >= 10:
                break
    if bad_fens:
        errors.append(f"{len(bad_fens)}+ rows with FEN that fails to parse "
                      f"(first: {bad_fens[0]})")

    # 5. Source-dataset balance + view balance (informational)
    src_counts = Counter(r["source_dataset"] for r in rows)
    view_counts = Counter(r["view"] for r in rows)
    log_lines.append(f"\nsource_dataset distribution:")
    for k, v in sorted(src_counts.items()):
        log_lines.append(f"  {k}: {v}")
    log_lines.append(f"\nview distribution:")
    for v in VALID_VIEWS:
        log_lines.append(f"  {v}: {view_counts.get(v, 0)}")
    # Each FEN typically has 3 views; check views are roughly balanced
    vmax, vmin = max(view_counts.values()), min(view_counts.values())
    if vmin > 0 and vmax / vmin > 1.05:
        warnings.append(f"per-view imbalance > 5% (max={vmax} min={vmin})")

    # 6. Per-source-per-view counts
    log_lines.append(f"\nsource_dataset × view counts:")
    src_view_counts = Counter((r["source_dataset"], r["view"]) for r in rows)
    for src in ("v1", "legacy"):
        for view in VALID_VIEWS:
            log_lines.append(f"  {src:6s} {view:8s}: {src_view_counts.get((src, view), 0)}")

    # Report
    if warnings:
        log_lines.append(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            log_lines.append(f"  {w}")
    if errors:
        log_lines.append(f"\nERRORS ({len(errors)}):")
        for e in errors:
            log_lines.append(f"  {e}")
        raise RuntimeError(f"validation failed with {len(errors)} errors")
    log_lines.append("\nVALIDATION OK")


def write_readme():
    readme = OUT / "README.md"
    readme.write_text(f"""# dataset_v1.5 — merged synthetic dataset

Combines two synthetic source sets into one training pool:
- **dataset_v1** (`data/dataset_v1/`): 6,135 PNG renders from
  `chess_position_api_v3.py` (May 2026). 6,132 are FEN-labelled; 3
  `fen_diag_*` diagnostic renders without FENs are EXCLUDED here.
- **dataset (legacy 1.5K)** (`data/dataset/`): 1,533 PNG renders from
  `chess_position_api_v2.py`. All FEN-labelled.

**Total: 7,665 labelled images.**

Both sources:
- Use the same 3 camera views: `overhead`, `west`, `east`.
- Use the same FEN→image-grid transform: **rot180** (locked in
  `scripts/view_orientations.py`; verified separately for both sources).
- Are tightly cropped to the playing surface (512×512, no table margin).

## Files

- `images/` — symlinks to the source PNGs. Read-only by convention; do not
  modify in place. To re-build, run `scripts/build_dataset_v1_5.py` again.
- `labels.csv` — one row per image.
  Columns: `image_name, fen, view, source_dataset, source_game,
  source_frame, fen_idx, run_idx`.
  `view ∈ {{overhead, west, east}}`. `source_dataset ∈ {{v1, legacy}}`.
  `run_idx` is empty for legacy rows (legacy renders don't have HDRI runs).
- `build_log.txt` — counts + validation report from the build run.

## Notes for downstream

- Filenames are unique across the two source datasets — no collision
  handling needed. `image_name` is the unique key.
- Both sources use rot180. Use `fen_to_label_grid(fen, view)` from
  `scripts/fen_to_grid.py` to get an 8×8 label grid in image coordinates.
- The corner cache `corners.json` covers dataset_v1 only. To use
  dataset_v1.5 for training, you need to extend the cache to legacy images
  — run `scripts/cache_all_corners.py` against the merged image dir, or
  build a separate cache for the legacy subset and merge JSONs.
""")
    return readme


def main():
    print(f"Loading source labels ...")
    v1_rows = load_v1_labels()
    legacy_rows = load_legacy_labels()
    print(f"  dataset_v1:      {len(v1_rows)} labelled rows")
    print(f"  dataset (legacy):{len(legacy_rows)} labelled rows")
    rows = v1_rows + legacy_rows
    print(f"  total:           {len(rows)} rows")

    log_lines = [
        f"# dataset_v1.5 build log",
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"source_v1:     {SRC_V1}  ({len(v1_rows)} labelled rows)",
        f"source_legacy: {SRC_LEGACY}  ({len(legacy_rows)} labelled rows)",
        f"output:        {OUT}",
        f"total rows:    {len(rows)}",
        "",
    ]

    print(f"\nCreating output directory {OUT} ...")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating symlinks ...")
    n_created, n_existed = make_symlinks(rows, log_lines)
    print(f"  created: {n_created}, existed: {n_existed}")

    print(f"\nWriting labels.csv ...")
    labels_path = write_labels_csv(rows)
    print(f"  wrote {labels_path}  ({labels_path.stat().st_size/1024:.1f} KB)")

    print(f"\nValidating ...")
    validate(rows, log_lines)

    log_path = OUT / "build_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"  wrote {log_path}")

    readme_path = write_readme()
    print(f"  wrote {readme_path}")

    print(f"\nDone. dataset_v1.5 ready at {OUT}")


if __name__ == "__main__":
    main()
