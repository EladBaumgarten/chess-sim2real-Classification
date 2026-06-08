"""STEP-0 CONFIRMATION (no training, no writes outside the convnext/ guard).

Confirms, before we build the training runs, that:
  1. torchvision exposes convnext_tiny with ImageNet weights (IMAGENET1K_V1) and it loads.
  2. The head is swapped to 13 classes: ConvNeXt's classifier is
     Sequential(LayerNorm2d, Flatten, Linear) — we replace classifier[2].
  3. A forward pass on a 100x100 crop batch (our crop size, NOT 224) yields finite
     (B, 13) logits — ConvNeXt is fully convolutional + adaptive pool, so no resize.
  4. Reports ConvNeXt-Tiny param count vs our ResNet-18 (~11.7M) for the report.

Run:  python convnext/training_scripts/confirm_convnext.py
"""
import sys
sys.path.insert(0, "/home/eladbaum/chess_project")

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    resnet18, ResNet18_Weights,
)

NUM_CLASSES = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def build_convnext(num_classes=NUM_CLASSES, pretrained=True):
    """ConvNeXt-Tiny with ImageNet weights and the final Linear swapped to 13 classes.
    classifier = Sequential(LayerNorm2d(768), Flatten(1), Linear(768, 1000)); we replace
    classifier[2]. No BatchNorm anywhere — ConvNeXt uses LayerNorm (no running stats)."""
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    m = convnext_tiny(weights=weights)
    in_features = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_features, num_classes)
    return m, in_features


print("\n[1] Loading convnext_tiny(IMAGENET1K_V1) ...")
model, in_features = build_convnext(pretrained=True)
print(f"  loaded. classifier[2].in_features = {in_features}")
assert in_features == 768, f"expected 768 head in-features, got {in_features}"
assert isinstance(model.classifier[2], nn.Linear)
assert model.classifier[2].out_features == NUM_CLASSES, "head not swapped to 13 classes"
print(f"  head swapped: classifier[2] = Linear({in_features}, {model.classifier[2].out_features})")

# Confirm there is genuinely no BatchNorm (the BN-freeze lever is moot).
bn_count = sum(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
               for m in model.modules())
ln_count = sum(isinstance(m, nn.LayerNorm) for m in model.modules())
# torchvision ConvNeXt uses a custom LayerNorm2d (not nn.LayerNorm) for the channel norm.
print(f"  norm layers: BatchNorm={bn_count} (expect 0)  nn.LayerNorm={ln_count}")
assert bn_count == 0, "ConvNeXt unexpectedly has BatchNorm — re-check the architecture"

print("\n[2] Forward pass on a 100x100 crop batch (our crop size) ...")
model = model.to(DEVICE).eval()
x = torch.rand(8, 3, 100, 100, device=DEVICE)
imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
with torch.no_grad():
    logits = model((x - imagenet_mean) / imagenet_std)
print(f"  input {tuple(x.shape)} -> logits {tuple(logits.shape)}")
assert logits.shape == (8, NUM_CLASSES), f"bad logits shape {logits.shape}"
assert torch.isfinite(logits).all(), "non-finite logits on 100x100 input"
print("  finite (B, 13) logits on 100x100 input — no resize needed.")

print("\n[3] Param counts (for the report) ...")
def count(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

cvx_total, _ = count(model)
# Backbone (features) vs head split, for the freeze scheme.
feat_params = sum(p.numel() for p in model.features.parameters())
head_params = sum(p.numel() for p in model.classifier.parameters())

resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
res_total, _ = count(resnet)

print(f"  ConvNeXt-Tiny (13-class head): {cvx_total:,} params  "
      f"(features={feat_params:,}, classifier={head_params:,})")
print(f"  ResNet-18    (13-class head): {res_total:,} params")
print(f"  ratio ConvNeXt / ResNet     : {cvx_total / res_total:.2f}x")

# Stage structure that the freeze scheme will target.
print("\n[4] model.features stage structure (freeze target):")
for i, block in enumerate(model.features):
    np_ = sum(p.numel() for p in block.parameters())
    print(f"    features[{i}]: {type(block).__name__:<22} {np_:>10,} params")

print("\n\033[92m✓ ConvNeXt-Tiny confirmed: loads, head swapped to 13, runs on 100x100, "
      "no BatchNorm.\033[0m")
