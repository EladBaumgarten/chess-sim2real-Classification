"""Visual check of the clipping hypothesis: render per-square crops for a game6 frame
(steep angle, the wall) vs a game2 frame (near-overhead, works). Starting-position frames
chosen so the far rank (row 0) is full of TALL pieces (R N B Q K) — if the uniform 100px
crop clips them, we'll see it. No training; saves PNGs to plots/."""
import sys; sys.path.insert(0,"/home/eladbaum/chess_project")
import csv
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.fen_to_grid import fen_to_label_grid
from scripts.verify_woelflein_crops import (warp_chessboard_image, crop_square,
    find_corners, ChessboardNotLocatedException)
ROOT="/home/eladbaum/chess_project"; OUT=f"{ROOT}/fine_tuning/stage3_improved/plots"
CLASS=["wP","wR","wN","wB","wQ","wK","bP","bR","bN","bB","bQ","bK","·"]

def first_frame(game):
    with open(f"{ROOT}/data/game{game}_per_frame/gt.csv") as f:
        r=next(csv.DictReader(f)); return r["image_name"], r["fen"]

def corners(bgr):
    H,W=bgr.shape[:2]
    try:
        np.random.seed(42); c=find_corners(bgr)
        if not bool(np.all((c[:,0]>=-8)&(c[:,0]<=W+8)&(c[:,1]>=-8)&(c[:,1]<=H+8))):
            raise ChessboardNotLocatedException("oob")
        return c,"detected"
    except Exception:
        return np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype=np.float32),"FALLBACK"

for g in [2,6]:
    name,fen=first_frame(g)
    bgr=cv2.imread(f"{ROOT}/data/game{g}_per_frame/images/{name}")
    c,status=corners(bgr); warped=warp_chessboard_image(bgr,c)
    rgb=cv2.cvtColor(warped,cv2.COLOR_BGR2RGB); grid=fen_to_label_grid(fen,f"game{g}")
    # show rows 0 (far), 4 (middle), 7 (near) across all 8 cols
    rows=[0,4,7]
    fig,axes=plt.subplots(len(rows),8,figsize=(16,6))
    for ri,row in enumerate(rows):
        for col in range(8):
            crop=cv2.cvtColor(crop_square(warped,row,col),cv2.COLOR_BGR2RGB)
            ax=axes[ri,col]; ax.imshow(crop); ax.set_xticks([]); ax.set_yticks([])
            lab=int(grid[row,col]); ax.set_title(f"r{row}c{col}:{CLASS[lab]}",fontsize=7)
            if col==0: ax.set_ylabel(f"row {row}",fontsize=9)
    fig.suptitle(f"game{g} crops ({status}) — first frame {name} — rows: 0=far(top), 4=mid, 7=near(bottom)",fontsize=11)
    plt.tight_layout()
    p=f"{OUT}/cropdiag_game{g}.png"; plt.savefig(p,dpi=110); plt.close()
    print(f"wrote {p}  (warped {rgb.shape}, corners {status})")
    # also save the full warped board for context
    plt.figure(figsize=(5,5)); plt.imshow(rgb); plt.title(f"game{g} warped board ({status})"); plt.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/warped_game{g}.png",dpi=110); plt.close()
    print(f"wrote {OUT}/warped_game{g}.png")
