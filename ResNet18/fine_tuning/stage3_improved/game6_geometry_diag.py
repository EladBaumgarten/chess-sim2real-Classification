"""
game6 domain-wall diagnostic — NO inference, NO training. Uses the already-saved s00
held-out predictions (results/s00/predictions/game{2,6}_{preds,labels}.npy), mapped back
to board positions, to localize WHERE game6 fails.

Questions:
  - Is the error spatially structured by board ROW (far ranks worse => perspective/tall-piece
    clipping) or uniform (=> appearance) or a fixed offset on every board (=> orientation/corner)?
  - game2 (near-overhead, works) vs game6 (the wall) side by side.
Predictions are ordered exactly as RealGameDataset.manifest (sorted image_name, board_row,
board_col), so reshaping to (n_boards, 8, 8) gives image-space board grids.
"""
import sys; sys.path.insert(0, "/home/eladbaum/chess_project")
import csv
import numpy as np
import pandas as pd
from scripts.fen_to_grid import fen_to_label_grid

ROOT="/home/eladbaum/chess_project"; PRED=f"{ROOT}/fine_tuning/stage3_323/results/predictions"
CLASS=["wP","wR","wN","wB","wQ","wK","bP","bR","bN","bB","bQ","bK","empty"]


def manifest(game):
    rows=[]
    with open(f"{ROOT}/data/game{game}_per_frame/gt.csv") as f:
        for r in csv.DictReader(f):
            grid=fen_to_label_grid(r["fen"], f"game{game}")
            for br in range(8):
                for bc in range(8):
                    rows.append({"image":r["image_name"],"row":br,"col":bc,"label":int(grid[br,bc])})
    return pd.DataFrame(rows).sort_values(["image","row","col"]).reset_index(drop=True)


for g in [2,6]:
    m=manifest(g)
    p=np.load(f"{PRED}/game{g}_preds.npy"); y=np.load(f"{PRED}/game{g}_labels.npy")
    assert len(m)==len(p)==len(y), f"len mismatch game{g}: {len(m)},{len(p)},{len(y)}"
    m["pred"]=p; m["correct"]=(m["pred"]==m["label"])
    nb=m["image"].nunique()
    print(f"\n================ game{g}  ({nb} boards) overall per-sq={m['correct'].mean():.4f} ================")

    # 8x8 error-rate heatmap (image-space row,col)
    err=1.0-m.groupby(["row","col"])["correct"].mean().unstack()
    print("error-rate by board position (rows 0=image-top .. 7=bottom):")
    print((err.round(2)).to_string())

    # per-row error (collapse cols) — perspective signature if monotone in row
    perrow=(1.0-m.groupby("row")["correct"].mean())
    print("per-ROW error:   "+"  ".join(f"r{r}={perrow[r]:.2f}" for r in range(8)))
    percol=(1.0-m.groupby("col")["correct"].mean())
    print("per-COL error:   "+"  ".join(f"c{c}={percol[c]:.2f}" for c in range(8)))

    # error on PIECE squares only, by row (tall-piece/perspective signature)
    pm=m[m["label"]!=12]
    perrow_piece=(1.0-pm.groupby("row")["correct"].mean())
    print("per-ROW error (PIECES only): "+"  ".join(f"r{r}={perrow_piece.get(r,float('nan')):.2f}" for r in range(8)))

    # are the SAME squares wrong on most boards? (systematic offset signature)
    persq_err=(1.0-m.groupby(["row","col"])["correct"].mean())
    always_bad=(persq_err>0.8).sum(); always_ok=(persq_err<0.05).sum()
    print(f"squares wrong on >80% of boards: {always_bad}/64 | squares right on >95%: {always_ok}/64")
