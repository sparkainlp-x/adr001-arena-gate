"""Run the full ADR-001 arena-gate grid and write deterministic results.

Grid (the task's grid, with steps fixed at the arena benchmark's
50-step sweep length for tractability -- see README for the timing
reason; a 200-step exact-mode run takes ~60 s, so the 200-step grid
would take hours):
  seeds 0-19, dt in {0.01, 0.05}, operator_mode in {legacy, exact},
  residual in {stock, adr001}  =>  160 runs.

The two arms differ ONLY in the residual function. Outputs contain
no timestamps, so re-running the grid reproduces the same JSON.

Usage (from the package root):
  python3 run_experiment.py [--steps 50] [--out results]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adr001_arena_gate.runner import run_one  # noqa: E402
from adr001_arena_gate.stats import paired_summary, summarise_arm  # noqa: E402

SEEDS = list(range(20))
DTS = [0.01, 0.05]
MODES = ["legacy", "exact"]
RESIDUALS = ["stock", "adr001"]


def grid_configs(steps: int = 50) -> list[dict]:
    """Enumerate the grid without running anything."""
    configs = []
    for mode in MODES:
        for dt in DTS:
            for seed in SEEDS:
                for residual in RESIDUALS:
                    configs.append(
                        {
                            "seed": seed,
                            "dt": dt,
                            "operator_mode": mode,
                            "residual": residual,
                            "steps": steps,
                        }
                    )
    return configs


def aggregate(rows: list[dict]) -> dict:
    """Aggregate per (mode, dt) cell: survival + acceptance + paired delta."""
    cells = {}
    for mode in MODES:
        for dt in DTS:
            stock_rows = [r for r in rows if r["operator_mode"] == mode and r["dt"] == dt and r["residual"] == "stock"]
            adr_rows = [r for r in rows if r["operator_mode"] == mode and r["dt"] == dt and r["residual"] == "adr001"]
            stock_rows.sort(key=lambda r: r["seed"])
            adr_rows.sort(key=lambda r: r["seed"])
            stock_acc = [r["accepted_fraction"] for r in stock_rows]
            adr_acc = [r["accepted_fraction"] for r in adr_rows]
            acc_deltas = [a - s for s, a in zip(stock_acc, adr_acc)]
            cells[f"{mode}/dt={dt}"] = {
                "stock": summarise_arm([r["survived"] for r in stock_rows]),
                "adr001": summarise_arm([r["survived"] for r in adr_rows]),
                "paired_survival": paired_summary(
                    [r["survived"] for r in stock_rows], [r["survived"] for r in adr_rows]
                ),
                "stock_mean_accepted_fraction": sum(stock_acc) / len(stock_acc) if stock_acc else 0.0,
                "adr001_mean_accepted_fraction": sum(adr_acc) / len(adr_acc) if adr_acc else 0.0,
                "mean_paired_acceptance_delta": sum(acc_deltas) / len(acc_deltas) if acc_deltas else 0.0,
                "runs_with_nonzero_acceptance_delta": sum(1 for d in acc_deltas if d != 0.0),
                "acceptance_deltas": acc_deltas,
            }
    total_stock = [r["survived"] for r in rows if r["residual"] == "stock"]
    total_adr = [r["survived"] for r in rows if r["residual"] == "adr001"]
    return {
        "cells": cells,
        "overall_stock": summarise_arm(total_stock),
        "overall_adr001": summarise_arm(total_adr),
        "total_runs": len(rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = grid_configs(steps=args.steps)
    rows: list[dict] = []
    for i, cfg in enumerate(configs, 1):
        row = run_one(
            seed=cfg["seed"],
            dt=cfg["dt"],
            operator_mode=cfg["operator_mode"],
            residual=cfg["residual"],
            steps=cfg["steps"],
        )
        rows.append(row)
        print(
            f"[{i}/{len(configs)}] {cfg['operator_mode']} dt={cfg['dt']} "
            f"seed={cfg['seed']} {cfg['residual']}: {row['status']} "
            f"executed={row['steps_executed']} accepted={row['accepted_steps']}",
            flush=True,
        )
    payload = {
        "grid": {
            "seeds": SEEDS,
            "dts": DTS,
            "operator_modes": MODES,
            "residuals": RESIDUALS,
            "steps": args.steps,
            "only_difference_between_arms": "residual function (stock full-matrix "
            "max-abs vs ADR-001 diagonal max-abs, same tolerance 0.20, strict >)",
        },
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out_dir / 'results.json'} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
