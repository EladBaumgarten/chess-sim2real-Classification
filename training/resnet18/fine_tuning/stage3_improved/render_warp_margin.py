"""Warp-margin gating check (no train): re-warp game6 (steep) + game2 (control) with a LARGE
extra top/far margin to see whether far-rank piece-tops are recoverable from the source photo
(vs already cropped out by the photographer). Draws the CURRENT board top edge (where the 50px
warp ring ends) so we can see what the current pipeline discards above the far rank."""
import sys; sys.path.insert(0,"/home/eladbaum/chess_project")
import csv
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import preprocessing.verify_woelflein_crops as V
from preprocessing.verify_woelflein_crops import sort_corner_points, find_corners, ChessboardNotLocatedException
ROOT="/home/eladbaum/chess_project"; OUT=f"{ROOT}/fine_tuning/stage3_improved/plots"
S=V.SQUARE_SIZE; B=V.BOARD_SIZE; I=V.IMG_SIZE   # 50, 400, 500
EXTRA_TOP=200   # extra px of far-side margin to reveal (4 squares)

def corners(bgr):
    H,W=bgr.shape[:2]
    try:
        np.random.seed(42); c=find_corners(bgr)
        if not bool(np.all((c[:,0]>=-8)&(c[:,0]<=W+8)&(c[:,1]>=-8)&(c[:,1]<=H+8))): raise ValueError
        return c,"detected"
    except Exception:
        return np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype=np.float32),"FALLBACK"

def first(game):
    with open(f"{ROOT}/data/game{game}_per_frame/gt.csv") as f:
        r=next(csv.DictReader(f)); return r["image_name"]

for g in [6,2]:
    name=first(g); bgr=cv2.imread(f"{ROOT}/data/game{g}_per_frame/images/{name}")
    c,status=corners(bgr); src=sort_corner_points(c.astype(np.float32))
    dst=np.array([[S,S],[B+S,S],[B+S,B+S],[S,B+S]],dtype=np.float32)
    M,_=cv2.findHomography(src,dst)
    T=np.array([[1,0,0],[0,1,EXTRA_TOP],[0,0,1]],dtype=np.float64)  # shift down -> reveal above
    wide=cv2.warpPerspective(bgr,(T@M).astype(np.float32),(I, I+EXTRA_TOP))
    rgb=cv2.cvtColor(wide,cv2.COLOR_BGR2RGB)
    fig,ax=plt.subplots(figsize=(6,7)); ax.imshow(rgb)
    # board now occupies y in [S+EXTRA_TOP, B+S+EXTRA_TOP]; far edge (rank-0 top) at y=S+EXTRA_TOP
    far_edge=S+EXTRA_TOP
    ax.axhline(far_edge,color="lime",lw=2)
    ax.axhline(far_edge - S,color="red",lw=1.5,ls="--")   # where the CURRENT 50px ring ends
    ax.text(5, far_edge-6, "board far edge (rank 0 top)", color="lime", fontsize=9)
    ax.text(5, far_edge-S-6, "current crop/warp ceiling (50px ring)", color="red", fontsize=9)
    ax.set_title(f"game{g} WIDE warp (+{EXTRA_TOP}px far margin, {status})\nIs there piece above the green line, and is it kept by the red ceiling?",fontsize=10)
    ax.axis("on")
    plt.tight_layout(); p=f"{OUT}/warpmargin_game{g}.png"; plt.savefig(p,dpi=120); plt.close()
    print(f"wrote {p}  ({status})")
