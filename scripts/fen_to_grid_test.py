"""
fen_to_grid_test.py — hand-verified assertions for fen_to_label_grid().

Five distinctive ASYMMETRIC FENs are tested. Each is chosen so that any
orientation bug would visibly misplace a piece (e.g., a single piece in a
corner, or kings on diagonal corners). For each, specific image-space cells
are asserted against the expected piece class.

Plus an edge-case test for the empty board (8/8/8/8/8/8/8/8) — separates
parser correctness from orientation correctness.

Run:   /home/eladbaum/.conda/envs/chess/bin/python fen_to_grid_test.py
"""
import numpy as np

from scripts.fen_to_grid import EMPTY_CLASS, fen_to_label_grid


# Convenience name aliases (project class encoding 0-11; 12 = empty)
W_PAWN, W_ROOK, W_KNIGHT, W_BISHOP, W_QUEEN, W_KING = range(6)
B_PAWN, B_ROOK, B_KNIGHT, B_BISHOP, B_QUEEN, B_KING = range(6, 12)
E = EMPTY_CLASS  # 12

SYM = {
    W_PAWN: "P", W_ROOK: "R", W_KNIGHT: "N", W_BISHOP: "B",
    W_QUEEN: "Q", W_KING: "K",
    B_PAWN: "p", B_ROOK: "r", B_KNIGHT: "n", B_BISHOP: "b",
    B_QUEEN: "q", B_KING: "k",
    E: ".",
}


def grid_to_str(grid):
    return "\n".join(
        " ".join(SYM[int(grid[r, c])] for c in range(8)) for r in range(8)
    )


def _all_empty_except(grid, expected):
    """Assert grid has exactly the (r, c, class) entries in `expected` and
    everything else is EMPTY_CLASS."""
    occupied = set()
    for r, c, cls in expected:
        assert grid[r, c] == cls, (
            f"expected class {cls} ({SYM[cls]}) at ({r},{c}), "
            f"got {int(grid[r,c])} ({SYM[int(grid[r,c])]})\n{grid_to_str(grid)}"
        )
        occupied.add((r, c))
    for r in range(8):
        for c in range(8):
            if (r, c) in occupied:
                continue
            assert grid[r, c] == E, (
                f"expected empty at ({r},{c}), got {SYM[int(grid[r,c])]}\n"
                f"{grid_to_str(grid)}"
            )


# --------------------------------------------------------------------------
# Edge case: empty board
# --------------------------------------------------------------------------
def test_empty_board():
    """Isolates parser correctness: did digit-expansion produce all empties?"""
    fen = "8/8/8/8/8/8/8/8"
    for view in ("overhead", "west", "east"):
        grid = fen_to_label_grid(fen, view)
        assert grid.shape == (8, 8), f"shape {grid.shape}"
        assert grid.dtype == np.int64, f"dtype {grid.dtype}"
        assert (grid == E).all(), (
            f"empty board not all empty for view {view!r}:\n{grid_to_str(grid)}"
        )
    print("PASS  test_empty_board")


# --------------------------------------------------------------------------
# 5 hand-verified asymmetric FENs
# --------------------------------------------------------------------------
def test_1_white_king_on_e1():
    """8/8/8/8/8/8/8/4K3 — white K on e1.
    FEN-native: rank 1 = row 7, file e = col 4   →  board[7, 4] = K.
    After rot180:  (7, 4) → (0, 3).
    Distinctive: K in a unique non-symmetric cell (col 3 ≠ col 4)."""
    fen = "8/8/8/8/8/8/8/4K3"
    grid = fen_to_label_grid(fen, "overhead")
    _all_empty_except(grid, [(0, 3, W_KING)])
    print("PASS  test_1_white_king_on_e1")


def test_2_kings_on_diagonal_corners():
    """7K/8/8/8/8/8/8/k7 — white K on h8, black k on a1.
    FEN-native:  K at (0, 7),  k at (7, 0).
    After rot180: K → (7, 0),  k → (0, 7).
    Distinctive: both axes asymmetric AND opposite colors swap corners."""
    fen = "7K/8/8/8/8/8/8/k7"
    grid = fen_to_label_grid(fen, "overhead")
    _all_empty_except(grid, [(7, 0, W_KING), (0, 7, B_KING)])
    print("PASS  test_2_kings_on_diagonal_corners")


def test_3_back_ranks_only():
    """r3k2r/8/8/8/8/8/8/R3K2R — castling-ready back ranks, only those squares.
    FEN-native row 0 (rank 8): r at 0, k at 4, r at 7.
    FEN-native row 7 (rank 1): R at 0, K at 4, R at 7.
    After rot180:
      row 0 (was row 7): R at 7, K at 3, R at 0   (image top = white)
      row 7 (was row 0): r at 7, k at 3, r at 0   (image bottom = black)
    Note kings shift from col 4 to col 3 because cols flip (c → 7-c).
    Distinctive: catches row-mirror bugs (white must end up at top, not
    bottom) AND col-mirror bugs (K must be at col 3, not col 4)."""
    fen = "r3k2r/8/8/8/8/8/8/R3K2R"
    grid = fen_to_label_grid(fen, "overhead")
    _all_empty_except(grid, [
        (0, 0, W_ROOK), (0, 3, W_KING), (0, 7, W_ROOK),
        (7, 0, B_ROOK), (7, 3, B_KING), (7, 7, B_ROOK),
    ])
    print("PASS  test_3_back_ranks_only")


def test_4_diagonal_pawns_asymmetric():
    """8/p7/1p6/2p5/8/8/PPP5/8 — three black pawns climbing diagonally on
    a-b-c files in upper-left of FEN, plus a row of three white pawns at a2-c2.
    FEN-native:
      board[1,0]=p  board[2,1]=p  board[3,2]=p
      board[6,0]=P  board[6,1]=P  board[6,2]=P
    After rot180 (r,c → 7-r, 7-c):
      black pawns:  (6,7), (5,6), (4,5)   — diagonal in image lower-right
      white pawns:  (1,7), (1,6), (1,5)   — short row in image upper-right
    Distinctive: any row-only or col-only mirror produces a different shape."""
    fen = "8/p7/1p6/2p5/8/8/PPP5/8"
    grid = fen_to_label_grid(fen, "overhead")
    _all_empty_except(grid, [
        (6, 7, B_PAWN), (5, 6, B_PAWN), (4, 5, B_PAWN),
        (1, 7, W_PAWN), (1, 6, W_PAWN), (1, 5, W_PAWN),
    ])
    print("PASS  test_4_diagonal_pawns_asymmetric")


def test_5_white_queen_on_a1():
    """8/8/8/8/8/8/8/Q7 — single white Q on a1.
    FEN-native: rank 1 = row 7, file a = col 0   →  board[7, 0] = Q.
    After rot180:  (7, 0) → (0, 7).
    Distinctive: Q on a CORNER, and each of the 4 orientations lands it on
    a different corner — any wrong transform is immediately visible."""
    fen = "8/8/8/8/8/8/8/Q7"
    grid = fen_to_label_grid(fen, "overhead")
    _all_empty_except(grid, [(0, 7, W_QUEEN)])
    print("PASS  test_5_white_queen_on_a1")


# --------------------------------------------------------------------------
# Additional sanity checks
# --------------------------------------------------------------------------
def test_view_consistency():
    """rot180 is the same transform for all 3 dataset_v1 views — same FEN
    must produce identical grids across overhead / west / east."""
    fen = "r3k2r/8/8/8/8/8/8/R3K2R"
    g_oh = fen_to_label_grid(fen, "overhead")
    g_w = fen_to_label_grid(fen, "west")
    g_e = fen_to_label_grid(fen, "east")
    assert np.array_equal(g_oh, g_w), "overhead vs west differ"
    assert np.array_equal(g_oh, g_e), "overhead vs east differ"
    print("PASS  test_view_consistency")


def test_dtype_and_shape():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    grid = fen_to_label_grid(fen, "overhead")
    assert grid.shape == (8, 8), f"shape {grid.shape}"
    assert grid.dtype == np.int64, f"dtype {grid.dtype}"
    print("PASS  test_dtype_and_shape")


def test_malformed_fen_raises():
    """Parser must raise ValueError on malformed FEN."""
    for bad_fen in [
        "8/8/8/8/8/8/8",                                    # only 7 ranks
        "8/8/8/8/8/8/8/8/8",                                # 9 ranks
        "8/8/8/8/8/8/8/Q9",                                 # rank overflow
        "8/8/8/8/8/8/8/X7",                                 # unknown piece
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN",       # last rank only 7
    ]:
        try:
            fen_to_label_grid(bad_fen, "overhead")
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad_fen!r}")
    print("PASS  test_malformed_fen_raises")


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
HAND_VERIFIED_FENS = [
    ("1: single white king (e1)",          "8/8/8/8/8/8/8/4K3"),
    ("2: kings on diagonal corners",       "7K/8/8/8/8/8/8/k7"),
    ("3: back-rank-only castling-ready",   "r3k2r/8/8/8/8/8/8/R3K2R"),
    ("4: diagonal pawns (asymmetric)",     "8/p7/1p6/2p5/8/8/PPP5/8"),
    ("5: white queen on a1",               "8/8/8/8/8/8/8/Q7"),
]


def display_hand_verified():
    print("\n" + "=" * 70)
    print("Display of the 5 hand-verified FENs (image-space grid after rot180)")
    print("rows = image rows (0 = top of image), cols = image cols (0 = left)")
    print("=" * 70)
    for desc, fen in HAND_VERIFIED_FENS:
        print(f"\n{desc}")
        print(f"  FEN: {fen}")
        grid = fen_to_label_grid(fen, "overhead")
        for line in grid_to_str(grid).split("\n"):
            print(f"    {line}")


def main():
    tests = [
        test_empty_board,
        test_1_white_king_on_e1,
        test_2_kings_on_diagonal_corners,
        test_3_back_ranks_only,
        test_4_diagonal_pawns_asymmetric,
        test_5_white_queen_on_a1,
        test_view_consistency,
        test_dtype_and_shape,
        test_malformed_fen_raises,
    ]
    print(f"Running {len(tests)} tests...\n")
    for t in tests:
        t()
    print("\nAll tests passed.")
    display_hand_verified()


if __name__ == "__main__":
    main()
