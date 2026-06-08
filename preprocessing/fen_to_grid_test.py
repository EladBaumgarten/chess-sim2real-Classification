"""Hand-verified assertions for fen_to_label_grid().

Tests five asymmetric FENs (any orientation bug visibly misplaces a piece)
plus an empty-board case that isolates parser from orientation correctness.
"""
import numpy as np

from preprocessing.fen_to_grid import EMPTY_CLASS, fen_to_label_grid


W_PAWN, W_ROOK, W_KNIGHT, W_BISHOP, W_QUEEN, W_KING = range(6)
B_PAWN, B_ROOK, B_KNIGHT, B_BISHOP, B_QUEEN, B_KING = range(6, 12)
E = EMPTY_CLASS

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
    """Assert grid has exactly the `expected` (r, c, class) entries, rest empty."""
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


def test_1_white_king_on_e1():
    """8/8/8/8/8/8/8/4K3 — white K on e1 → image (0, 3) after rot180."""
    fen = "8/8/8/8/8/8/8/4K3"
    grid = fen_to_label_grid(fen, "overhead")
    _all_empty_except(grid, [(0, 3, W_KING)])
    print("PASS  test_1_white_king_on_e1")


def test_2_kings_on_diagonal_corners():
    """7K/8/8/8/8/8/8/k7 — white K h8, black k a1 → (7,0) and (0,7) after rot180."""
    fen = "7K/8/8/8/8/8/8/k7"
    grid = fen_to_label_grid(fen, "overhead")
    _all_empty_except(grid, [(7, 0, W_KING), (0, 7, B_KING)])
    print("PASS  test_2_kings_on_diagonal_corners")


def test_3_back_ranks_only():
    """r3k2r/8/8/8/8/8/8/R3K2R — castling-ready back ranks. After rot180 white
    lands on image-top row, black on image-bottom; kings at col 3 (cols flip).
    Catches both row-mirror and col-mirror bugs."""
    fen = "r3k2r/8/8/8/8/8/8/R3K2R"
    grid = fen_to_label_grid(fen, "overhead")
    _all_empty_except(grid, [
        (0, 0, W_ROOK), (0, 3, W_KING), (0, 7, W_ROOK),
        (7, 0, B_ROOK), (7, 3, B_KING), (7, 7, B_ROOK),
    ])
    print("PASS  test_3_back_ranks_only")


def test_4_diagonal_pawns_asymmetric():
    """8/p7/1p6/2p5/8/8/PPP5/8 — diagonal black pawns + a row of white pawns.
    After rot180 the diagonal/row shapes land such that any single-axis mirror
    produces a different layout."""
    fen = "8/p7/1p6/2p5/8/8/PPP5/8"
    grid = fen_to_label_grid(fen, "overhead")
    _all_empty_except(grid, [
        (6, 7, B_PAWN), (5, 6, B_PAWN), (4, 5, B_PAWN),
        (1, 7, W_PAWN), (1, 6, W_PAWN), (1, 5, W_PAWN),
    ])
    print("PASS  test_4_diagonal_pawns_asymmetric")


def test_5_white_queen_on_a1():
    """8/8/8/8/8/8/8/Q7 — white Q a1 → image corner (0, 7) after rot180.
    Each of the 4 orientations lands Q on a different corner."""
    fen = "8/8/8/8/8/8/8/Q7"
    grid = fen_to_label_grid(fen, "overhead")
    _all_empty_except(grid, [(0, 7, W_QUEEN)])
    print("PASS  test_5_white_queen_on_a1")


def test_view_consistency():
    """All 3 dataset_v1 views share rot180, so grids must be identical."""
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
