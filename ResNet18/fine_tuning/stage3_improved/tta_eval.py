"""
Test-Time Augmentation (TTA) on the EXISTING s00 best_real checkpoint — NO retraining.
Averages softmax over a few perspective-consistent views per square (Wölflein never tried
TTA). Views stay inside the training-augmentation envelope and preserve which square is
being classified (NO flips, NO large rotation/translation).

Validation: the identity-only pass MUST reproduce s00's held-out 2/6 (per-sq 0.9085 /
piece-only 0.7556). If it doesn't, the pipeline diverged and the TTA comparison is invalid.

RealGameDataset is a verbatim copy of train.py Cell 5 (returns the uint8 HWC crop so we can
augment), so crops match the headline eval exactly. Eval = ImageNet-normalize then argmax,
identical to train.py's evaluate().
"""
import sys
sys.path.insert(0, "/home/eladbaum/chess_project")
import csv, json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import torchvision.transforms.functional as TF
from PIL import Image

from scripts.fen_to_grid import fen_to_label_grid
from scripts.verify_woelflein_crops import (
    warp_chessboard_image, crop_square, find_corners, ChessboardNotLocatedException)

PROJECT_ROOT = "/home/eladbaum/chess_project"
EXP = f"{PROJECT_ROOT}/fine_tuning/stage3_improved"
CKPT = f"{EXP}/checkpoints/s00/best_real.pt"
SEED = 42
NUM_CLASSES = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(DEVICE)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(DEVICE)
CLASS_SHORT=["wP","wR","wN","wB","wQ","wK","bP","bR","bN","bB","bQ","bK","empty"]


class RealGameDataset(Dataset):   # verbatim crop pipeline; returns uint8 HWC RGB + label
    CORNER_OOB_TOLERANCE = 8
    def __init__(self, gt_csv, images_dir, game_name):
        self.images_dir = Path(images_dir); self.game_name = game_name
        import pandas as pd
        rows=[]
        with open(gt_csv) as f:
            for r in csv.DictReader(f):
                grid=fen_to_label_grid(r["fen"], game_name)
                for br in range(8):
                    for bc in range(8):
                        rows.append({"image_name":r["image_name"],"board_row":br,
                                     "board_col":bc,"label":int(grid[br,bc])})
        self.manifest=pd.DataFrame(rows).sort_values(
            ["image_name","board_row","board_col"]).reset_index(drop=True)
        self._cc={}
    def __len__(self): return len(self.manifest)
    def _corners(self, name, bgr):
        if name in self._cc: return self._cc[name]
        H,W=bgr.shape[:2]
        try:
            np.random.seed(SEED); c=find_corners(bgr)
            lo,hx,hy=-self.CORNER_OOB_TOLERANCE,W+self.CORNER_OOB_TOLERANCE,H+self.CORNER_OOB_TOLERANCE
            if not bool(np.all((c[:,0]>=lo)&(c[:,0]<=hx)&(c[:,1]>=lo)&(c[:,1]<=hy))):
                raise ChessboardNotLocatedException("oob")
        except Exception:
            c=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype=np.float32)
        self._cc[name]=c; return c
    def __getitem__(self, i):
        r=self.manifest.iloc[i]; name=r["image_name"]
        bgr=cv2.imread(str(self.images_dir/name))
        warped=warp_chessboard_image(bgr,self._corners(name,bgr))
        crop=crop_square(warped,int(r["board_row"]),int(r["board_col"]))
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.ascontiguousarray(rgb)), int(r["label"])  # HWC uint8


# --- TTA views (PIL->PIL), within the train-aug envelope, orientation-preserving ---
VIEWS = {
    "identity":   lambda im: im,
    "bright+":    lambda im: TF.adjust_brightness(im, 1.15),
    "bright-":    lambda im: TF.adjust_brightness(im, 0.85),
    "contrast+":  lambda im: TF.adjust_contrast(im, 1.15),
    "contrast-":  lambda im: TF.adjust_contrast(im, 0.85),
    "shear+8":    lambda im: TF.affine(im, angle=0, translate=(0,0), scale=1.0, shear=[8.0,0.0]),
    "shear-8":    lambda im: TF.affine(im, angle=0, translate=(0,0), scale=1.0, shear=[-8.0,0.0]),
    "scale1.05":  lambda im: TF.affine(im, angle=0, translate=(0,0), scale=1.05, shear=[0.0,0.0]),
}


def to_norm_tensor(pil_img):
    arr = torch.from_numpy(np.ascontiguousarray(np.array(pil_img))).permute(2,0,1).float()/255.0
    return (arr.to(DEVICE) - MEAN) / STD


@torch.no_grad()
def run(model, loader, view_names):
    model.eval()
    all_p=[]; all_y=[]
    for crops, ys in loader:                  # crops: B,H,W,C uint8
        probs = torch.zeros(crops.size(0), NUM_CLASSES, device=DEVICE)
        for v in view_names:
            fn = VIEWS[v]
            batch = torch.stack([to_norm_tensor(fn(Image.fromarray(c.numpy()))) for c in crops])
            probs += F.softmax(model(batch), dim=1)
        all_p.append((probs/len(view_names)).argmax(1).cpu().numpy()); all_y.append(ys.numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


def metrics(p,y):
    persq=float((p==y).mean()); m=y!=12
    piece=float((p[m]==y[m]).mean()) if m.any() else float("nan")
    emp=y==12; empty=float((p[emp]==y[emp]).mean()) if emp.any() else float("nan")
    return persq,piece,empty


def main():
    model=resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT,map_location=DEVICE,weights_only=False)["model_state_dict"])
    model=model.to(DEVICE)
    loaders={}
    for g in [2,6]:
        ds=RealGameDataset(f"{PROJECT_ROOT}/data/game{g}_per_frame/gt.csv",
                           f"{PROJECT_ROOT}/data/game{g}_per_frame/images", f"game{g}")
        loaders[g]=DataLoader(ds,batch_size=64,shuffle=False,num_workers=4)
    full=list(VIEWS.keys())
    for tag,views in [("IDENTITY-only (must==s00 0.9085/0.7556)",["identity"]),
                      (f"TTA ({len(full)} views)", full)]:
        ps=[];ys=[];perg={}
        for g in [2,6]:
            p,y=run(model,loaders[g],views); perg[g]=metrics(p,y); ps.append(p);ys.append(y)
        p=np.concatenate(ps); y=np.concatenate(ys); persq,piece,empty=metrics(p,y)
        print(f"\n=== {tag} ===")
        print(f"  AGG  2/6: per-sq={persq:.4f}  piece-only={piece:.4f}  empty={empty:.4f}")
        print(f"  game2: per-sq={perg[2][0]:.4f} piece={perg[2][1]:.4f} | game6: per-sq={perg[6][0]:.4f} piece={perg[6][1]:.4f}")
        # per-class dark pieces
        pcs={c:(float((p[y==i]==i).mean()) if (y==i).any() else float('nan')) for i,c in enumerate(CLASS_SHORT)}
        print("  dark: "+" ".join(f"{c}={pcs[c]:.3f}" for c in ["bN","bB","bK","bQ","wN","wQ"]))
        if tag.startswith("TTA"):
            json.dump({"views":full,"agg":{"per_square":persq,"piece_only":piece},
                       "game2":perg[2],"game6":perg[6],"per_class":pcs},
                      open(f"{EXP}/results/tta_s00.json","w"),indent=2)
            print(f"  wrote {EXP}/results/tta_s00.json")


if __name__=="__main__":
    main()
