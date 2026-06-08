"""
extend_corners_for_v1_5.py — extend the corner cache to cover dataset_v1.5
(adds the 1,533 legacy images to the existing 6,135-entry v1 cache, then
recomputes the per-view empirical fallback over the full 7,665 detections).

Strategy:
  1. Load existing scripts/corners.json + corner_detection_log.csv (covers v1).
  2. Walk data/dataset_v1.5/images/. For any image NOT in the v1 cache, run
     find_corners + sanity check (same logic as cache_all_corners.py).
  3. Recompute the per-view fallback mean from ALL detected entries across
     v1 and legacy.
  4. Fill any failed-status entries with the freshly-computed fallback.
  5. Save:
       data/dataset_v1.5/corners.json
       data/dataset_v1.5/fallback_corners.json
       data/dataset_v1.5/corner_detection_log.csv

Expected runtime: ~10 min for 1,533 new images at ~310ms/img.
"""

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/eladbaum/chess_project")
from preprocessing.verify_woelflein_crops import find_corners, ChessboardNotLocatedException


# --------------------------------------------------------------------------
# Config — same thresholds as cache_all_corners.py so the cache stays homogeneous
# --------------------------------------------------------------------------
V1_5_DIR = Path("/home/eladbaum/chess_project/data/dataset_v1.5")
IMAGES_DIR = V1_5_DIR / "images"
LABELS_CSV = V1_5_DIR / "labels.csv"

EXISTING_CORNERS = Path("/home/eladbaum/chess_project/preprocessing/corners.json")
EXISTING_LOG = Path("/home/eladbaum/chess_project/preprocessing/corner_detection_log.csv")

OUT_JSON = V1_5_DIR / "corners.json"
OUT_FALLBACK = V1_5_DIR / "fallback_corners.json"
OUT_CSV = V1_5_DIR / "corner_detection_log.csv"

PROGRESS_EVERY = 200
SEED = 0
CORNER_OOB_TOL = 10
MIN_QUAD_AREA_FRACTION = 0.30
ASPECT_RATIO_RANGE = (0.6, 1.66)
VIEWS = ("overhead", "west", "east")


# --------------------------------------------------------------------------
def quad_sanity(corners, img_shape):
    H, W = img_shape[:2]
    if not np.all(
        (corners[:, 0] >= -CORNER_OOB_TOL) & (corners[:, 0] <= W + CORNER_OOB_TOL)
        & (corners[:, 1] >= -CORNER_OOB_TOL) & (corners[:, 1] <= H + CORNER_OOB_TOL)
    ):
        return False, "bad_geometry: corner OOB"
    area = cv2.contourArea(corners.astype(np.float32))
    if area < MIN_QUAD_AREA_FRACTION * H * W:
        return False, f"bad_geometry: area frac {area/(H*W):.2f}"
    xmin, xmax = corners[:, 0].min(), corners[:, 0].max()
    ymin, ymax = corners[:, 1].min(), corners[:, 1].max()
    if ymax - ymin <= 0:
        return False, "bad_geometry: zero height"
    ar = (xmax - xmin) / (ymax - ymin)
    if not ASPECT_RATIO_RANGE[0] <= ar <= ASPECT_RATIO_RANGE[1]:
        return False, f"bad_geometry: aspect {ar:.2f}"
    return True, ""


def categorize_exception(e):
    if isinstance(e, ChessboardNotLocatedException):
        return "ransac_timeout" if "RANSAC" in str(e) else "chessboard_not_located"
    return f"exception:{type(e).__name__}"


def view_of_v1_5(name, view_lookup):
    """v1.5 has two naming schemes — use the labels.csv mapping rather than
    parsing the filename (legacy names don't have _rY_, so trailing-_view
    parsing would happen to work, but using the explicit lookup is safer)."""
    return view_lookup[name]


# --------------------------------------------------------------------------
def load_existing_cache():
    """Returns (corners_by_name, log_rows_by_name) — both keyed by image_name.
    Each log row: dict with view/status/failure_reason/runtime_s."""
    print(f"Loading existing cache ...")
    if not EXISTING_CORNERS.exists():
        print(f"  no existing cache at {EXISTING_CORNERS}; will detect from scratch.")
        return {}, {}
    corners = json.loads(EXISTING_CORNERS.read_text())
    print(f"  loaded {len(corners)} corner entries from {EXISTING_CORNERS}")

    log_rows = {}
    if EXISTING_LOG.exists():
        with EXISTING_LOG.open() as f:
            for row in csv.DictReader(f):
                log_rows[row["image"]] = row
        print(f"  loaded {len(log_rows)} log rows from {EXISTING_LOG}")
    else:
        print(f"  no existing log at {EXISTING_LOG}; new log will be fresh.")
    return corners, log_rows


def load_view_lookup():
    """image_name → view, from labels.csv."""
    lookup = {}
    with LABELS_CSV.open() as f:
        for row in csv.DictReader(f):
            lookup[row["image_name"]] = row["view"]
    return lookup


# --------------------------------------------------------------------------
def detect_new_images(new_names, view_lookup):
    """Run find_corners on each new image. Returns dict
    image_name → {view, status, corners (or None), runtime_s, failure_reason}."""
    out = {}
    t0 = time.perf_counter()
    n_ok = 0
    for i, name in enumerate(new_names, 1):
        view = view_lookup[name]
        img_path = IMAGES_DIR / name
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            out[name] = dict(view=view, status="failed", corners=None,
                             runtime_s=0.0,
                             failure_reason="exception:imread_returned_None")
            continue
        np.random.seed(SEED)
        t_img = time.perf_counter()
        try:
            corners = find_corners(bgr)
            dt = time.perf_counter() - t_img
            ok, reason = quad_sanity(corners, bgr.shape)
            if ok:
                out[name] = dict(view=view, status="detected", corners=corners,
                                 runtime_s=dt, failure_reason="")
                n_ok += 1
            else:
                out[name] = dict(view=view, status="failed", corners=None,
                                 runtime_s=dt, failure_reason=reason)
        except Exception as e:
            dt = time.perf_counter() - t_img
            out[name] = dict(view=view, status="failed", corners=None,
                             runtime_s=dt,
                             failure_reason=categorize_exception(e))
        if i % PROGRESS_EVERY == 0 or i == len(new_names):
            elapsed = time.perf_counter() - t0
            rate = elapsed / i
            eta = rate * (len(new_names) - i)
            print(f"  [{i:5d}/{len(new_names)}]  elapsed {elapsed/60:5.1f}m  "
                  f"rate {rate*1000:4.0f}ms/img  ETA {eta/60:5.1f}m  "
                  f"detected={n_ok}  failed={i-n_ok}", flush=True)
    return out


def compute_fallback(merged_corners, view_lookup):
    """Per-view mean corners across DETECTED entries only."""
    by_view = {v: [] for v in VIEWS}
    for name, c in merged_corners.items():
        if name not in view_lookup:
            continue  # safety; shouldn't happen
        # 'detected' rows are those with the original detection-stable shape;
        # we just use ALL entries since fallback rows already match the per-view mean
        # — but to recompute, exclude rows that were filled with the OLD fallback.
        # Simpler: re-compute over rows that originally had status 'detected'.
        pass
    # We need the status info; this function gets called after we've merged
    # the full audit log too. See main() for the actual mean computation.
    raise NotImplementedError("not used; mean is computed inline in main()")


def main():
    print(f"Output dir: {V1_5_DIR}")
    t_overall = time.perf_counter()

    # 1. Load v1 cache + view lookup
    existing_corners, existing_log = load_existing_cache()
    view_lookup = load_view_lookup()
    print(f"\nlabels.csv lists {len(view_lookup)} images.")

    # 2. Identify which v1.5 images need detection
    all_v1_5 = sorted(view_lookup.keys())
    needs_detection = [n for n in all_v1_5 if n not in existing_corners]
    already_cached = [n for n in all_v1_5 if n in existing_corners]
    print(f"already cached:    {len(already_cached)}")
    print(f"needs detection:   {len(needs_detection)}")

    # 3. Run detection on new images (legacy + any v1 stragglers)
    print(f"\n=== Pass 1: detect on {len(needs_detection)} new images ===")
    new_detections = detect_new_images(needs_detection, view_lookup)

    # 4. Stitch together a unified "results" dict keyed by image_name.
    #    For existing images: pull status/runtime/failure_reason from existing
    #    log + corners. For new ones: from new_detections.
    results = {}

    # 4a. existing entries
    for name in already_cached:
        log_row = existing_log.get(name)
        view = view_lookup[name]
        corners = np.asarray(existing_corners[name], dtype=np.float64)
        # status: if old log says 'fallback' or 'failed', remember that — we'll
        # re-fill from the NEW fallback in step 6. Otherwise 'detected'.
        if log_row is None:
            status = "detected"
            failure_reason = ""
            runtime_s = 0.0
        else:
            status = log_row["source"]
            failure_reason = log_row["failure_reason"]
            try:
                runtime_s = float(log_row["runtime_s"])
            except (TypeError, ValueError):
                runtime_s = 0.0
        results[name] = dict(
            view=view, status=status, corners=corners,
            runtime_s=runtime_s, failure_reason=failure_reason,
        )

    # 4b. new entries
    for name, d in new_detections.items():
        results[name] = d

    # 5. Compute per-view fallback over ALL truly-detected entries
    print(f"\n=== Compute per-view empirical fallback (mean of detections) ===")
    fallback = {}
    for view in VIEWS:
        ok = [d["corners"] for d in results.values()
              if d["status"] == "detected" and d["view"] == view
              and d["corners"] is not None]
        if not ok:
            raise RuntimeError(f"no detections for view {view!r}")
        avg = np.mean(np.stack(ok), axis=0)
        std = np.std(np.stack(ok), axis=0)
        fallback[view] = avg
        print(f"  {view}: n={len(ok):5d}  "
              f"mean=[{', '.join(f'({x:.1f},{y:.1f})' for x, y in avg)}]")
        print(f"           σ=[{', '.join(f'({x:.1f},{y:.1f})' for x, y in std)}]")

    # 6. Fill failures with NEW fallback. Override any old 'fallback' corners
    #    too, because they were computed from v1-only means and we now have
    #    legacy data folded in.
    print(f"\n=== Pass 2: fill failures with empirical fallback ===")
    n_fallback = 0
    for name, d in results.items():
        if d["status"] != "detected":
            d["corners"] = fallback[d["view"]]
            d["status"] = "fallback"
            n_fallback += 1
    print(f"  filled {n_fallback} fallback entries")

    # 7. Save outputs
    print(f"\n=== Save outputs ===")
    corners_dict = {name: d["corners"].tolist() for name, d in results.items()}
    OUT_JSON.write_text(json.dumps(corners_dict))
    print(f"  wrote {OUT_JSON}  ({len(corners_dict)} entries, "
          f"{OUT_JSON.stat().st_size/1024/1024:.1f} MB)")

    OUT_FALLBACK.write_text(json.dumps(
        {v: fallback[v].tolist() for v in VIEWS}, indent=2))
    print(f"  wrote {OUT_FALLBACK}")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "view", "source", "failure_reason", "runtime_s",
                    "tl_x", "tl_y", "tr_x", "tr_y",
                    "br_x", "br_y", "bl_x", "bl_y"])
        for name in sorted(results.keys()):
            d = results[name]
            c = np.asarray(d["corners"])
            w.writerow([
                name, d["view"], d["status"], d["failure_reason"],
                f"{d['runtime_s']:.3f}",
                f"{c[0,0]:.2f}", f"{c[0,1]:.2f}",
                f"{c[1,0]:.2f}", f"{c[1,1]:.2f}",
                f"{c[2,0]:.2f}", f"{c[2,1]:.2f}",
                f"{c[3,0]:.2f}", f"{c[3,1]:.2f}",
            ])
    print(f"  wrote {OUT_CSV}")

    # 8. Summary
    by_view = {v: {"detected": 0, "fallback": 0, "reasons": {}} for v in VIEWS}
    for d in results.values():
        v = d["view"]
        if d["status"] == "detected":
            by_view[v]["detected"] += 1
        else:
            by_view[v]["fallback"] += 1
            by_view[v]["reasons"][d["failure_reason"]] = (
                by_view[v]["reasons"].get(d["failure_reason"], 0) + 1
            )

    n_total = len(results)
    total_runtime = time.perf_counter() - t_overall
    print(f"\n=== Summary ===")
    print(f"Total images in v1.5: {n_total}")
    print(f"Total runtime:        {total_runtime/60:.1f} min")
    print(f"Detected:             {n_total - n_fallback}/{n_total}  "
          f"({100*(1 - n_fallback/n_total):.2f}%)")
    print(f"Fallback:             {n_fallback}/{n_total}  "
          f"({100*n_fallback/n_total:.2f}%)")
    for v in VIEWS:
        b = by_view[v]
        n = b["detected"] + b["fallback"]
        print(f"  {v:8s}: {b['detected']}/{n} detected, {b['fallback']} fallback")
        for reason, count in sorted(b["reasons"].items(), key=lambda x: -x[1]):
            print(f"     {reason}: {count}")


if __name__ == "__main__":
    main()
