"""
view_orientations.py — FEN->image-grid transform per view.

dataset_v1 synthetic renders use rot180 (the renderer rectifies by world position,
so every camera view lands in one canonical orientation; mapping FEN-native
row0=rank8/col0=file a to image coords flips both axes). Real game frames are filmed
with the camera behind white (image (0,0) = a8 = FEN-native (0,0)), so no transform.
"""
import numpy as np

VIEW_ORIENTATIONS = {
    # dataset_v1 synthetic (verified 2026-05-20)
    "overhead": "rot180",
    "west":     "rot180",
    "east":     "rot180",
    # real games — camera behind white, no transform (verified per game)
    "game2":    "identity",
    "game4":    "identity",
    "game5":    "identity",
    "game6":    "identity",
    "game7":    "identity",
    "game8":    "identity",
    "game9":    "identity",
    "game10":   "identity",
    "game11":   "identity",
}

GAME7_ORIENTATION = "identity"


def apply_orientation(raw_board: np.ndarray, view: str) -> np.ndarray:
    """Map a FEN-native (8, 8) grid (row 0 = rank 8, col 0 = file a) into
    image-aligned coords (row 0 = top, col 0 = left) for `view`."""
    transform = VIEW_ORIENTATIONS[view]
    if transform == "identity":
        return raw_board.copy()
    if transform == "fliplr":
        return np.fliplr(raw_board).copy()
    if transform == "flipud":
        return np.flipud(raw_board).copy()
    if transform == "rot180":
        return np.ascontiguousarray(np.rot90(raw_board, 2))
    raise ValueError(f"unknown orientation transform: {transform!r}")
