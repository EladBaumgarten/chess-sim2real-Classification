"""Aggregate the three ConvNeXt runs into the architecture-comparison table (md + csv),
alongside the published ResNet-18 numbers. Reads each run's games_2_6_eval.json,
synth_monitor_results.json, and recipe.json.

Usage:  python build_report.py
"""
import sys, os, json
sys.path.insert(0, "/home/eladbaum/chess_project")
import csv as _csv

EXP_DIR = "/home/eladbaum/chess_project/training/convnext"
RESULTS = f"{EXP_DIR}/results"

# Published ResNet-18 numbers (games 2/6) — hard-coded references.
RESNET = {
    "zeroshot": {"per_sq": 0.5138, "piece": None},
    "stage3":   {"per_sq": 0.9085, "piece": 0.7556},
    "stage5":   {"per_sq": 0.9160, "piece": 0.7748},
}
ROWS = [("synth-only (zero-shot)", "convnext_zeroshot", "zeroshot"),
        ("real fine-tune (Stage 3)", "convnext_stage3", "stage3"),
        ("combined (Stage 5)", "convnext_stage5", "stage5")]
CLASS_SHORT = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK", "empty"]


def _load(run, name):
    p = f"{RESULTS}/{run}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


lines = ["# ConvNeXt-Tiny vs ResNet-18 — architecture comparison (games 2/6)\n"]
lines.append("| model (games 2/6) | ResNet-18 per-sq | ConvNeXt per-sq | "
             "ResNet piece-only | ConvNeXt piece-only | forgetting Δ (ConvNeXt) |")
lines.append("|---|---|---|---|---|---|")
table_csv = [["row", "resnet_per_sq", "convnext_per_sq", "resnet_piece", "convnext_piece",
              "convnext_forgetting_delta", "convnext_game7_per_sq", "selected_epoch"]]

for label, run, mode in ROWS:
    g26 = _load(run, "games_2_6_eval.json")
    forget = _load(run, "synth_monitor_results.json")
    recipe = _load(run, "recipe.json")
    cvx_persq = g26["per_square_acc"] if g26 else None
    cvx_piece = g26["piece_only_acc"] if g26 else None
    delta = forget["forgetting_delta"] if forget else None
    g7 = recipe["results"]["game7_per_square"] if recipe else None
    sel_ep = recipe.get("selected_epoch") if recipe else None
    r = RESNET[mode]
    lines.append(f"| {label} | {fmt(r['per_sq'])} | {fmt(cvx_persq)} | "
                 f"{fmt(r['piece'])} | {fmt(cvx_piece)} | {fmt(delta, 4) if delta is not None else '—'} |")
    table_csv.append([label, r["per_sq"], cvx_persq, r["piece"], cvx_piece, delta, g7, sel_ep])

# Per-class on held-out for stage3 and stage5.
lines.append("\n## Per-class held-out (games 2/6) accuracy — ConvNeXt\n")
lines.append("| class | " + " | ".join(c for c in CLASS_SHORT) + " |")
lines.append("|---|" + "---|" * len(CLASS_SHORT))
for label, run, mode in ROWS:
    if mode == "zeroshot":
        continue
    g26 = _load(run, "games_2_6_eval.json")
    if not g26:
        continue
    pc = g26["per_class_acc"]
    lines.append(f"| {label} | " + " | ".join(
        (f"{pc[c]:.3f}" if pc.get(c) is not None else "—") for c in CLASS_SHORT) + " |")

# Recipe note.
lines.append("\n## Recipe (each architecture done right)\n")
lines.append("ResNet-18 used SGD + two-phase freeze. ConvNeXt-Tiny (~27.8M params, 2.49× "
             "ResNet's 11.2M) used AdamW + cosine + weight-decay + a ConvNeXt-stage-structured "
             "two-phase freeze. ConvNeXt uses LayerNorm (no BatchNorm running stats), so the "
             "BN-freeze lever does not apply; forgetting Δ is still logged above.")
for label, run, mode in ROWS:
    recipe = _load(run, "recipe.json")
    if recipe:
        lines.append(f"\n- **{label}** (`{run}`): {recipe['optimizer']}, lr_head={recipe['lr_head']}, "
                     f"lr_backbone={recipe['lr_backbone']}, wd={recipe['weight_decay']}, "
                     f"epochs={recipe['epochs']} (phaseA={recipe['warmup_phaseA_epochs']}), "
                     f"select on {recipe['selection_metric']} @ epoch {recipe.get('selected_epoch')}, "
                     f"source={recipe['source_weights']}.")

md = "\n".join(lines) + "\n"
open(f"{RESULTS}/comparison_report.md", "w").write(md)
with open(f"{RESULTS}/comparison_report.csv", "w", newline="") as f:
    _csv.writer(f).writerows(table_csv)
print(md)
print(f"wrote {RESULTS}/comparison_report.md + .csv")
