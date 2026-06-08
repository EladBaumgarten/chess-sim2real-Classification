"""
game7_orientation_diagnostics.py — Step 6a: lock the FEN→image-grid transform
for game7 real images. Same 4-panel diagnostic structure used in Step 4a
(synthetic).

For each of 3 chosen frames (opening / midgame / endgame):
  1. Load the JPG from data/game7_per_frame/images/.
  2. Look up the FEN from data/game7_per_frame/gt.csv.
  3. Run chesscog find_corners; if it fails, fall back to the image-corner
     quad (0,0), (W,0), (W,H), (0,H).
  4. Warp to chesscog's 500×500 (board at inner [50..450]).
  5. Build a 2×2 panel showing identity / fliplr / flipud / rot180 of the
     FEN-derived label grid overlaid on the warped board.

Outputs (one PNG per frame) → results/game7_orientation/:
  opening_<frame>.png
  midgame_<frame>.png
  endgame_<frame>.png

User picks which panel matches; that transform becomes GAME7_ORIENTATION.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/eladbaum/chess_project")
from preprocessing.verify_woelflein_crops import (
    SQUARE_SIZE, BOARD_SIZE, warp_chessboard_image,
    find_corners, ChessboardNotLocatedException,
)


GAME7_IMAGES = Path("/home/eladbaum/chess_project/data/game7_per_frame/images")
GAME7_GT = Path("/home/eladbaum/chess_project/data/game7_per_frame/gt.csv")
OUT_DIR = Path("/home/eladbaum/chess_project/results/game7_orientation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Class encoding (project spec)
PIECE_TO_CLASS = {
    "P": 0, "R": 1, "N": 2, "B": 3, "Q": 4, "K": 5,
    "p": 6, "r": 7, "n": 8, "b": 9, "q": 10, "k": 11,
}
CLASS_TO_PIECE = {v: k for k, v in PIECE_TO_CLASS.items()}
CLASS_TO_PIECE[12] = "."

# 3 frames chosen for FEN diversity (verified to exist in gt.csv).
SAMPLES = [
    ("opening", "frame_000696.jpg"),
    ("midgame", "frame_004516.jpg"),
    ("endgame", "frame_030820.jpg"),
]


def _font(size):
    for p in [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_fen_lookup():
    """image_name → fen."""
    lookup = {}
    with GAME7_GT.open() as f:
        for row in csv.DictReader(f):
            lookup[row["image_name"]] = row["fen"]
    return lookup


def raw_grid(fen):
    """FEN piece-placement → 8×8 FEN-native grid (row 0 = rank 8, col 0 = file a)."""
    board = np.full((8, 8), 12, dtype=np.int64)
    for r, rank in enumerate(fen.split()[0].split("/")):
        c = 0
        for ch in rank:
            if ch.isdigit():
                c += int(ch)
            else:
                board[r, c] = PIECE_TO_CLASS[ch]
                c += 1
    return board


def detect_or_fallback_corners(bgr, max_oob_px=8):
    """Run find_corners, but reject the result if any corner is more than
    `max_oob_px` outside the image — on game7 the board fills the frame
    and chesscog hallucinates extensions of 50+ px in that case. When
    rejected, fall back to the image-corner quad."""
    H, W = bgr.shape[:2]
    np.random.seed(0)
    try:
        corners = find_corners(bgr)
        lo, hi_x, hi_y = -max_oob_px, W + max_oob_px, H + max_oob_px
        in_bounds = bool(np.all(
            (corners[:, 0] >= lo) & (corners[:, 0] <= hi_x)
            & (corners[:, 1] >= lo) & (corners[:, 1] <= hi_y)
        ))
        if in_bounds:
            return corners, "find_corners"
        # detection succeeded but corners are wildly OOB — fall back
        fallback = np.array(
            [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
            dtype=np.float32,
        )
        return fallback, f"fallback (find_corners returned OOB)"
    except (ChessboardNotLocatedException, Exception) as e:
        fallback = np.array(
            [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
            dtype=np.float32,
        )
        return fallback, f"fallback (image corners): {type(e).__name__}"


def warp_inner(bgr, corners):
    """Warp to 500×500 then extract the inner 400×400 board (BGR→RGB)."""
    warped = warp_chessboard_image(bgr, corners.astype(np.float32))
    inner_bgr = warped[SQUARE_SIZE:SQUARE_SIZE + BOARD_SIZE,
                       SQUARE_SIZE:SQUARE_SIZE + BOARD_SIZE]
    return cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2RGB)


def overlay(inner_rgb, label_grid, title, scale=2):
    """Render the warped board at 2× with labels overlaid in
    yellow (white) / magenta (black) boxes."""
    h, w = inner_rgb.shape[:2]
    big = cv2.resize(inner_rgb, (w * scale, h * scale),
                     interpolation=cv2.INTER_LANCZOS4)
    im = Image.fromarray(big)
    d = ImageDraw.Draw(im)
    font = _font(28)
    sq = SQUARE_SIZE * scale
    for r in range(8):
        for c in range(8):
            cls = int(label_grid[r, c])
            ch = CLASS_TO_PIECE[cls]
            if ch == ".":
                continue
            cx = c * sq + sq // 2
            cy = r * sq + sq // 2
            bg = (255, 255, 0) if ch.isupper() else (200, 0, 200)
            d.rectangle([cx - 18, cy - 18, cx + 18, cy + 18], fill=bg)
            d.text((cx - 9, cy - 16), ch, fill=(0, 0, 0), font=font)
    out = Image.new("RGB", (im.width, im.height + 38), (240, 240, 240))
    ImageDraw.Draw(out).text((10, 8), title, fill=(0, 0, 0), font=_font(22))
    out.paste(im, (0, 38))
    return out


def diagnostic_for(label, frame_name, fen, fen_lookup):
    image_path = GAME7_IMAGES / frame_name
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"could not read {image_path}")
    H, W = bgr.shape[:2]
    corners, src_tag = detect_or_fallback_corners(bgr)

    inner = warp_inner(bgr, corners)
    raw = raw_grid(fen)
    candidates = {
        "identity (no transform)":   raw,
        "fliplr (cols mirrored)":    np.fliplr(raw),
        "flipud (rows mirrored)":    np.flipud(raw),
        "rot180 (rows+cols flipped)": np.rot90(raw, 2),
    }
    tiles = {name: overlay(inner, grid, name) for name, grid in candidates.items()}

    tw, th = tiles["identity (no transform)"].size
    pad = 12
    header_h = 100
    Wc = 2 * tw + 3 * pad
    Hc = header_h + 2 * th + 3 * pad

    canvas = Image.new("RGB", (Wc, Hc), (255, 255, 255))
    cd = ImageDraw.Draw(canvas)
    cd.text((pad, 8),
            f"{label.upper()}  —  {frame_name}  (real image, {H}×{W})",
            fill=(0, 0, 0), font=_font(22))
    cd.text((pad, 36), f"FEN: {fen}", fill=(40, 40, 40), font=_font(14))
    cd.text((pad, 56), f"corners: {src_tag}",
            fill=(40, 40, 40), font=_font(13))
    cd.text((pad, 74),
            "yellow = white piece, magenta = black piece.  "
            "Pick the panel where labels sit on the visible pieces.",
            fill=(40, 40, 40), font=_font(13))

    positions = [
        (pad,        header_h),
        (tw + 2*pad, header_h),
        (pad,        header_h + th + pad),
        (tw + 2*pad, header_h + th + pad),
    ]
    for (name, tile), pos in zip(tiles.items(), positions):
        canvas.paste(tile, pos)
    return canvas


def main():
    fen_lookup = load_fen_lookup()
    print(f"Loaded {len(fen_lookup)} game7 frame→FEN entries.")
    print(f"Generating 3 orientation diagnostics → {OUT_DIR}\n")

    for label, frame_name in SAMPLES:
        if frame_name not in fen_lookup:
            print(f"  WARNING: {frame_name} not in gt.csv, skipping")
            continue
        fen = fen_lookup[frame_name]
        n_pieces = sum(1 for c in fen.split()[0] if c.isalpha())
        print(f"--- {label}  ({n_pieces} pieces)  {frame_name}  FEN: {fen}")
        canvas = diagnostic_for(label, frame_name, fen, fen_lookup)
        out_path = OUT_DIR / f"{label}_{frame_name.replace('.jpg', '.png')}"
        canvas.save(out_path)
        print(f"  wrote {out_path.name}")

    print("\nDone. Inspect:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
