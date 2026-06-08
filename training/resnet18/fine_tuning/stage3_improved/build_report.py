"""
Build the Stage-3-improved sweep report: balance_strength vs. accuracy metrics,
including rare-piece per-class accuracy and the over-correction guard on `empty`.

Scans fine_tuning/stage3_improved/results/<run_name>/ for game7_results.json,
held_out_aggregate.json, class_weights.json, over_correction_guard.json and emits
results/sweep_report.md + results/sweep_report.csv. Recommends the strength with
the best game7 real_val acc whose `empty` accuracy is not degraded vs. baseline.

Usage:
    python build_report.py
"""
import csv
import json
from pathlib import Path

EXP_DIR = Path("/home/eladbaum/chess_project/training/resnet18/fine_tuning/stage3_improved")
RESULTS_ROOT = EXP_DIR / "results"
BASELINE_DIR = Path("/home/eladbaum/chess_project/training/resnet18/fine_tuning/stage3_323/results")
RARE = ["wN", "wB", "wQ", "wK", "bN", "bB", "bQ", "bK"]
EMPTY_TOL = 0.01


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _collect():
    rows = []
    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        g7 = _load(run_dir / "game7_results.json")
        ho = _load(run_dir / "held_out_aggregate.json")
        cw = _load(run_dir / "class_weights.json")
        guard = _load(run_dir / "over_correction_guard.json")
        if not (g7 and ho and cw):
            continue
        row = {
            "run": run_dir.name,
            "strength": cw.get("balance_strength"),
            "formula": cw.get("balance_formula"),
            "game7_persq": g7["per_square_acc"],
            "ho_persq": ho["per_square_acc"],
            "ho_piece_only": ho["piece_only_acc"],
            "empty": ho["per_class_acc"].get("empty"),
            "over_correction": (guard or {}).get("over_correction_flag"),
        }
        for c in RARE:
            row[c] = ho["per_class_acc"].get(c)
        rows.append(row)
    rows.sort(key=lambda r: (r["strength"] if r["strength"] is not None else -1))
    return rows


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"


def main():
    rows = _collect()
    if not rows:
        print(f"No completed runs found under {RESULTS_ROOT}.")
        return 1

    base_ho = _load(BASELINE_DIR / "held_out_aggregate.json") or {}
    base_g7 = _load(BASELINE_DIR / "game7_results.json") or {}
    base_empty = base_ho.get("per_class_acc", {}).get("empty", float("nan"))

    cols = (["run", "strength", "game7_persq", "ho_persq", "ho_piece_only", "empty"]
            + RARE + ["over_correction"])

    # CSV
    csv_path = RESULTS_ROOT / "sweep_report.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    # Markdown
    head = ("| strength | run | game7 per-sq | 2/6 per-sq | piece-only | empty | "
            + " | ".join(RARE) + " | over-corr? |")
    sep = "|" + "---|" * (6 + len(RARE) + 1)
    lines = ["# Stage 3 improved — class-balanced fine-tuning sweep", "",
             f"Baseline (frozen stage3_323): game7 per-sq "
             f"{_fmt(base_g7.get('per_square_acc'))}, 2/6 per-sq "
             f"{_fmt(base_ho.get('per_square_acc'))}, empty {_fmt(base_empty)}.", "",
             head, sep]
    base_row = ("| — | baseline | "
                f"{_fmt(base_g7.get('per_square_acc'))} | "
                f"{_fmt(base_ho.get('per_square_acc'))} | "
                f"{_fmt(base_ho.get('piece_only_acc'))} | {_fmt(base_empty)} | "
                + " | ".join(_fmt(base_ho.get('per_class_acc', {}).get(c)) for c in RARE)
                + " | — |")
    lines.append(base_row)
    for r in rows:
        lines.append(
            f"| {r['strength']} | {r['run']} | {_fmt(r['game7_persq'])} | "
            f"{_fmt(r['ho_persq'])} | {_fmt(r['ho_piece_only'])} | {_fmt(r['empty'])} | "
            + " | ".join(_fmt(r.get(c)) for c in RARE)
            + f" | {'YES' if r['over_correction'] else 'no'} |")

    # Recommendation: best game7 per-sq among runs that did NOT over-correct empty.
    def empty_ok(r):
        e = r.get("empty")
        return (e is not None and not (isinstance(base_empty, float) and base_empty != base_empty)
                and e >= base_empty - EMPTY_TOL and not r.get("over_correction"))

    safe = [r for r in rows if empty_ok(r)]
    pool = safe if safe else rows
    best = max(pool, key=lambda r: r["game7_persq"])
    rec = (f"**Recommended:** `{best['run']}` (strength={best['strength']}) — "
           f"best game7 per-square {_fmt(best['game7_persq'])} among runs that did "
           f"NOT degrade `empty` (within {EMPTY_TOL} of baseline {_fmt(base_empty)}).")
    if not safe:
        rec += " ⚠️ NOTE: every run degraded `empty`; recommendation is the least-bad — reconsider strengths."
    lines += ["", rec, ""]

    md_path = RESULTS_ROOT / "sweep_report.md"
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {md_path}\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
