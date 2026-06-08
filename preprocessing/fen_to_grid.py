"""Convert a FEN piece-placement string into an (8, 8) int64 label grid in
image coordinates, applying the per-view transform from view_orientations.py.

Class encoding (project spec, 13 classes): 0-5 white P/R/N/B/Q/K,
6-11 black p/r/n/b/q/k, 12 empty.
"""
import numpy as np

from preprocessing.view_orientations import apply_orientation


PIECE_TO_CLASS = {
    "P": 0, "R": 1, "N": 2, "B": 3, "Q": 4, "K": 5,
    "p": 6, "r": 7, "n": 8, "b": 9, "q": 10, "k": 11,
}
EMPTY_CLASS = 12


def _parse_fen_to_raw_grid(fen: str) -> np.ndarray:
    """Parse a FEN's piece-placement field into an 8×8 FEN-native grid
    (row 0 = rank 8, col 0 = file a; empty = EMPTY_CLASS). Un-oriented;
    callers apply the view transform to get image space.

    Raises ValueError on malformed FEN.
    """
    board = np.full((8, 8), EMPTY_CLASS, dtype=np.int64)
    placement = fen.split()[0]
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN must have 8 ranks, got {len(ranks)}: {fen!r}")
    for r, rank in enumerate(ranks):
        c = 0
        for ch in rank:
            if ch.isdigit():
                c += int(ch)
            else:
                if ch not in PIECE_TO_CLASS:
                    raise ValueError(f"Unknown piece {ch!r} in FEN {fen!r}")
                if c >= 8:
                    raise ValueError(
                        f"Rank {r} overflowed 8 columns at piece {ch!r}: {fen!r}")
                board[r, c] = PIECE_TO_CLASS[ch]
                c += 1
        if c != 8:
            raise ValueError(
                f"Rank {r} did not sum to 8 (got {c}): {fen!r}")
    return board


def fen_to_label_grid(fen: str, view: str) -> np.ndarray:
    """Convert a FEN to an (8, 8) int64 grid of labels (0-12) in image
    coordinates for `view`. grid[row, col] is the label at image (row, col).
    """
    raw = _parse_fen_to_raw_grid(fen)
    return apply_orientation(raw, view).astype(np.int64)
