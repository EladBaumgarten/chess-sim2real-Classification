"""
Sanity check: confirm the strength=0.0 run (plain CE) reproduces the FROZEN
stage3_323 baseline within tolerance. Reproduction is tolerance-based, NOT
bit-exact — the original ran with num_workers>0 and no cuDNN-determinism flag,
so small run-to-run noise is expected. A per-square gap > TOL means the pipeline
diverged somewhere and the comparison would be invalid; stop and investigate.

Usage:
    python check_reproduction.py [--run_name s00] [--tol 0.005]
"""
import argparse
import json
from pathlib import Path

EXP_DIR = Path("/home/eladbaum/chess_project/training/resnet18/fine_tuning/stage3_improved")
BASELINE_DIR = Path("/home/eladbaum/chess_project/training/resnet18/fine_tuning/stage3_323/results")
CLASS_SHORT = ["wP", "wR", "wN", "wB", "wQ", "wK",
               "bP", "bR", "bN", "bB", "bQ", "bK", "empty"]


def _load(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text())


def _cmp_file(name, run_dir, tol):
    new = _load(run_dir / name)
    base = _load(BASELINE_DIR / name)
    a, b = new["per_square_acc"], base["per_square_acc"]
    gap = abs(a - b)
    ok = gap <= tol
    print(f"\n[{name}]  per-square: run={a:.4f}  baseline={b:.4f}  "
          f"|Δ|={gap:.4f}  {'PASS' if ok else 'FAIL'} (tol {tol})")
    # Per-class deltas (informational; not gating).
    np_, bp = new.get("per_class_acc", {}), base.get("per_class_acc", {})
    worst = []
    for c in CLASS_SHORT:
        if c in np_ and c in bp and np_[c] is not None and bp[c] is not None:
            worst.append((abs(np_[c] - bp[c]), c, np_[c], bp[c]))
    worst.sort(reverse=True)
    for d, c, x, y in worst[:4]:
        print(f"    {c:>6}: run={x:.4f}  baseline={y:.4f}  |Δ|={d:.4f}")
    return ok, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="s00")
    ap.add_argument("--tol", type=float, default=0.005)
    args = ap.parse_args()
    run_dir = EXP_DIR / "results" / args.run_name
    print(f"Comparing {run_dir}  vs  {BASELINE_DIR}  (tol={args.tol})")

    # Guard: confirm strength was actually 0.0 for this run.
    cw = _load(run_dir / "class_weights.json")
    s = cw.get("balance_strength")
    print(f"run balance_strength = {s} (expected 0.0 for a reproduction check)")
    if s not in (0.0, 0):
        print("  WARNING: this run was NOT strength 0.0 — reproduction not expected.")

    ok1, _ = _cmp_file("game7_results.json", run_dir, args.tol)
    ok2, _ = _cmp_file("held_out_aggregate.json", run_dir, args.tol)
    verdict = ok1 and ok2
    print("\n" + ("=" * 60))
    print(f"REPRODUCTION: {'PASS — pipeline matches baseline.' if verdict else 'FAIL — pipeline diverged; investigate before sweeping.'}")
    print("=" * 60)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
