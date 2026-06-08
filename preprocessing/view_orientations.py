"""Confirmed FEN→image-grid transforms per view.

dataset_v1's three camera views all share rot180: its rectifier orders the
source quad by world position, so every view lands in one canonical
orientation, and mapping FEN-native (row 0 = rank 8, col 0 = file a) to image
indexing flips both axes. The real-game views (camera behind white) need no
transform — image (0,0) = a8 = FEN-native (0,0).

dataset-specific: data/dataset_v2 uses a different rectification (fliplr per
labels.py) and is not verified for this pipeline.
"""
import numpy as np

# Synthetic dataset_v1 views use rot180; real-game views (camera behind white)
# use identity.
VIEW_ORIENTATIONS = {
    "overhead": "rot180",
    "west":     "rot180",
    "east":     "rot180",
    "game7":    "identity",
    "game2":    "identity",
    "game4":    "identity",
    "game5":    "identity",
    "game6":    "identity",
    "game8":    "identity",
    "game9":    "identity",
    "game10":   "identity",
    "game11":   "identity",
}

GAME7_ORIENTATION = "identity"


def apply_orientation(raw_board: np.ndarray, view: str) -> np.ndarray:
    """Transform a FEN-native (8, 8) grid into image-aligned coords for `view`."""
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
