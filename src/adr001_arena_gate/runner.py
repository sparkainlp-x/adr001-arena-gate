"""Run the arena with the stock residual or the ADR-001 residual.

This module is the only part of the package that needs the arena
tree and numpy: the arena itself is numpy-based, so a pure-stdlib
re-implementation of its 32-dimensional density-matrix evolution
would be a different engine, not the arena. The package's own code
(``adr001.py``, ``stats.py``) is standard-library only; this runner
is an integration layer, and it says so.

Swap mechanics: the arena observer resolves ``residual_infinity``
from the ``membrane_arena_3x3.shields`` module namespace, so the
ADR arm patches exactly that name to ``adr001_residual_matrix``
(diagonal max-abs) for the duration of the run and restores the
stock function afterwards. No arena file is modified, and the
stock function in ``core.py`` is never touched. All other settings
-- guards, failure_limit, adaptive retry, calibrator -- are identical
between arms; the residual function is the only difference.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from .arena_mapping import adr001_residual_matrix

DEFAULT_ARENA_TREE = Path.home() / "workspace" / "membrane-arena-3x3-upgrade"


def find_arena_tree() -> Path:
    """Locate the upgraded arena tree, or raise a clear error."""
    env = os.environ.get("ADR001_ARENA_TREE")
    candidates = [Path(env)] if env else []
    candidates.append(DEFAULT_ARENA_TREE)
    for candidate in candidates:
        if (candidate / "membrane_arena_3x3" / "engine.py").exists():
            return candidate
    raise RuntimeError(
        "membrane-arena-3x3-upgrade tree not found; set ADR001_ARENA_TREE "
        "to the directory containing membrane_arena_3x3/engine.py"
    )


def load_arena(arena_tree: Path):
    """Import the arena engine from the given tree. Returns modules."""
    tree = str(arena_tree)
    if tree not in sys.path:
        sys.path.insert(0, tree)
    from membrane_arena_3x3 import shields  # noqa: F401  (patched below)
    from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
    from membrane_arena_3x3.lattice import LatticeConfig
    from membrane_arena_3x3.shields import GuardConfig

    return EngineConfig, MembraneArena3x3, LatticeConfig, GuardConfig, shields


@contextmanager
def adr001_residual_active(shields_module):
    """Temporarily replace the observer's residual with ADR-001."""
    original = shields_module.residual_infinity
    shields_module.residual_infinity = adr001_residual_matrix
    try:
        yield
    finally:
        shields_module.residual_infinity = original


def run_one(
    seed: int,
    dt: float,
    operator_mode: str,
    residual: str,
    steps: int = 50,
    arena_tree: Path | None = None,
) -> dict:
    """Run one arena configuration. ``residual`` is 'stock' or 'adr001'."""
    if residual not in ("stock", "adr001"):
        raise ValueError("residual must be 'stock' or 'adr001'")
    if operator_mode not in ("legacy", "exact"):
        raise ValueError("operator_mode must be 'legacy' or 'exact'")
    tree = arena_tree or find_arena_tree()
    EngineConfig, MembraneArena3x3, LatticeConfig, GuardConfig, shields = load_arena(tree)
    config = EngineConfig(
        lattice=LatticeConfig(dt=dt, operator_mode=operator_mode),
        guards=GuardConfig(),
        steps=steps,
        seed=seed,
    )
    arena = MembraneArena3x3(config)
    if residual == "adr001":
        with adr001_residual_active(shields):
            summary = arena.run()
    else:
        summary = arena.run()
    final = summary.get("final") or {}
    reasons: dict[str, int] = {}
    for item in summary.get("rejections", []):
        reason = item.get("failed_guard") or item.get("reason") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    executed = summary["steps_executed"]
    accepted = summary["accepted_steps"]
    return {
        "seed": seed,
        "dt": dt,
        "operator_mode": operator_mode,
        "residual": residual,
        "steps_requested": steps,
        "steps_executed": executed,
        "accepted_steps": accepted,
        "rejected_steps": summary["rejected_steps"],
        "accepted_fraction": accepted / executed if executed else 0.0,
        "status": summary["status"],
        "survived": summary["status"] == "completed",
        "repair_accepts": summary["repair_accepts"],
        "repair_attempts_total": summary["repair_attempts_total"],
        "final_max_residual": final.get("max_residual"),
        "final_min_purity": final.get("min_purity"),
        "final_max_divergence": final.get("max_divergence"),
        "rejection_reasons": reasons,
    }
