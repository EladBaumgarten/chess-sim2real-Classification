"""Visual-only test of the upward-extended crop (no training). Shows OLD (100px, square-
centered) vs NEW (extended ~1 square upward, resized to 100x100) crops for game6's failing
far rank (row0) and near rank (row7), plus game2 row0 as control. If NEW captures the full
leaning piece where OLD clips it, the retrain is justified."""
import sys; sys.path.insert(0,"/home/eladbaum/chess_project")
import csv
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocessing.fen_to_grid import fen_to_label_grid
import preprocessing.verify_woelflein_crops as V
from preprocessing.verify_woelflein_crops import (warp_chessboard_image, crop_square,
    find_corners, ChessboardNotLocatedException)
ROOT="/home/eladbaum/chess_project"; OUT=f"{ROOT}/fine_tuning/stage3_improved/plots"
CLASS=["wP","wR","wN","wB","wQ","wK","bP","bR","bN","bB","bQ","bK","·"]
S=V.SQUARE_SIZE  # 50
print(f"SQUARE_SIZE={S}")

def extended_crop(img, row, col, up=1.0):
    """Extend the crop UPWARD by `up` squares (square sits lower, captures leaning piece);
    keep width = 2 squares; pad if it runs off the warped image; resize to 100x100."""
    y0=int(S*(row+0.5-up)); y1=int(S*(row+2.5)); x0=int(S*(col+0.5)); x1=int(S*(col+2.5))
    H,W=img.shape[:2]
    pt=max(0,-y0); pl=max(0,-x0); pb=max(0,y1-H); pr=max(0,x1-W)
    ys,ye=max(0,y0),min(H,y1); xs,xe=max(0,x0),min(W,x1)
    patch=img[ys:ye, xs:xe]
    if pt or pb or pl or pr:
        patch=cv2.copyMakeBorder(patch,pt,pb,pl,pr,cv2.BORDER_CONSTANT,value=(0,0,0))
    return cv2.resize(patch,(100,100),interpolation=cv2.INTER_AREA)

def setup(game):
    with open(f"{ROOT}/data/game{game}_per_frame/gt.csv") as f:
        r=next(csv.DictReader(f)); name,fen=r["image_name"],r["fen"]
    bgr=cv2.imread(f"{ROOT}/data/game{game}_per_frame/images/{name}"); H,W=bgr.shape[:2]
    try:
        np.random.seed(42); c=find_corners(bgr)
        if not bool(np.all((c[:,0]>=-8)&(c[:,0]<=W+8)&(c[:,1]>=-8)&(c[:,1]<=H+8))): raise ValueError
    except Exception:
        c=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype=np.float32)
    return warp_chessboard_image(bgr,c), fen_to_label_grid(fen,f"game{game}")

def panel(game,row):
    warped,grid=setup(game)
    fig,axes=plt.subplots(2,8,figsize=(16,4.2))
    for col in range(8):
        old=cv2.cvtColor(crop_square(warped,row,col),cv2.COLOR_BGR2RGB)
        new=cv2.cvtColor(extended_crop(warped,row,col),cv2.COLOR_BGR2RGB)
        axes[0,col].imshow(old); axes[1,col].imshow(new)
        lab=int(grid[row,col])
        axes[0,col].set_title(f"c{col}:{CLASS[lab]}",fontsize=8)
        for rr in (0,1): axes[rr,col].set_xticks([]); axes[rr,col].set_yticks([])
    axes[0,0].set_ylabel("OLD 100px",fontsize=10); axes[1,0].set_ylabel("NEW +1sq up",fontsize=10)
    fig.suptitle(f"game{game} row {row} ({'far/top' if row<4 else 'near/bottom'}) — OLD vs NEW crop",fontsize=12)
    plt.tight_layout(); p=f"{OUT}/extcrop_game{game}_row{row}.png"; plt.savefig(p,dpi=110); plt.close()
    print(f"wrote {p}")

panel(6,0); panel(6,7); panel(2,0)
