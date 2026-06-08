"""PyTorch Dataset for per-square classification. One sample = (source image,
board square) → (crop_tensor[3,100,100] float32 in [0,1], label).

Crops are computed on-the-fly: warp via cached corners, then crop_square.
Augmentation is passed in as `transform` (applied to HWC uint8 RGB) so it can
be ablated independently of the Dataset.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, "/home/eladbaum/chess_project")
from preprocessing.verify_woelflein_crops import warp_chessboard_image, crop_square


DEFAULT_DATASET_DIR = Path(
    "/home/eladbaum/chess_project/syn_data_generation/dataset_v1/images"
)


class ChessSquareDataset(Dataset):
    """One sample per (image × board square); 392,448 samples in dataset_v1."""

    def __init__(
        self,
        manifest,
        corners_json_path,
        dataset_dir=None,
        transform=None,
    ):
        """
        Args:
            manifest: path to manifest.csv or a preloaded/filtered DataFrame.
            corners_json_path: path to corners.json (image_name → 4 corners).
            dataset_dir: source PNG directory; defaults to dataset_v1/images/.
            transform: optional callable on the (100,100,3) uint8 RGB crop
                before tensorization.
        """
        if isinstance(manifest, pd.DataFrame):
            self.manifest = manifest.reset_index(drop=True)
        else:
            self.manifest = pd.read_csv(manifest)
        with open(corners_json_path) as f:
            self.corners = json.load(f)
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        image_name = row["source_image"]
        board_row = int(row["board_row"])
        board_col = int(row["board_col"])
        label = int(row["label"])

        bgr = cv2.imread(str(self.dataset_dir / image_name))
        corners = np.array(self.corners[image_name], dtype=np.float32)
        warped = warp_chessboard_image(bgr, corners)
        crop_bgr = crop_square(warped, board_row, board_col)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            crop_rgb = self.transform(crop_rgb)

        tensor = torch.from_numpy(np.ascontiguousarray(crop_rgb)) \
                      .permute(2, 0, 1).float() / 255.0

        return tensor, label
