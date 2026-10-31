"""Tests for the ADR-001 arena-gate package.

Unit tests (ADR rule, statistics, grid enumeration) are standard-
library only and always run. Integration tests need the upgraded
arena tree and numpy; they skip cleanly when the tree is absent, so
a clean extraction without the arena tree still reports them as
skipped rather than failed. Run from the package root:

    python3 tests/tests.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adr001_arena_gate.adr001 import (  # noqa: E402
    DIMENSION,
    calculate_residual,
    evaluate_residual,
)
from adr001_arena_gate.stats import (  # noqa: E402
    paired_summary,
    summarise_arm,
    wilson_interval,
)

try:
    from adr001_arena_gate.runner import find_arena_tree  # noqa: E402

    ARENA_TREE = find_arena_tree()
except Exception:
    ARENA_TREE = None

ARENA_AVAILABLE = ARENA_TREE is not None


def vec(base: float = 0.0, changes: dict | None = None) -> list[float]:
    """A 32-vector of `base` with the given index/value overrides."""
    out = [base] * DIMENSION
    for index, value in (changes or {}).items():
        out[index] = value
    return out


class TestAdr001(unittest.TestCase):
    def test_dimension_is_32(self):
        self.assertEqual(DIMENSION, 32)

    def test_calculate_known_values(self):
        components, aggregate = calculate_residual(vec(0.0), vec(0.0, {3: 0.25, 7: -0.10}))
        self.assertEqual(components[3], 0.25)
        self.assertEqual(components[7], 0.10)
        self.assertEqual(aggregate, 0.25)
        self.assertTrue(all(c == 0.0 for i, c in enumerate(components) if i not in (3, 7)))

    def test_strict_greater_failure_rule(self):
        # Exactly at tolerance: passes. Any epsilon above: fails.
        self.assertTrue(evaluate_residual(vec(0.0), vec(0.0, {0: 0.20}), 0.20).passed)
        self.assertFalse(evaluate_residual(vec(0.0), vec(0.0, {0: 0.20}), 0.20).failed)
        result = evaluate_residual(vec(0.0), vec(0.0, {0: 0.2000001}), 0.20)
        self.assertTrue(result.failed)
        self.assertFalse(result.passed)

    def test_zero_tolerance(self):
        self.assertTrue(evaluate_residual(vec(1.0), vec(1.0), 0.0).passed)
        self.assertTrue(evaluate_residual(vec(1.0), vec(1.0, {5: 1.0001}), 0.0).failed)

    def test_validation_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            calculate_residual([0.0] * 31, [0.0] * 31)
        with self.assertRaises(ValueError):
            calculate_residual([0.0] * 33, [0.0] * 33)
        with self.assertRaises(ValueError):
            calculate_residual(vec(0.0, {0: float("inf")}), vec(0.0))
        with self.assertRaises(ValueError):
            calculate_residual(vec(0.0, {0: True}), vec(0.0))
        with self.assertRaises(ValueError):
            evaluate_residual(vec(0.0), vec(0.0), -0.1)
        with self.assertRaises(ValueError):
            evaluate_residual(vec(0.0), vec(0.0), float("nan"))

    def test_determinism(self):
        a = evaluate_residual(vec(0.0), vec(0.0, {2: 0.4}), 0.20)
        b = evaluate_residual(vec(0.0), vec(0.0, {2: 0.4}), 0.20)
        self.assertEqual(a, b)


class TestStats(unittest.TestCase):
    def test_wilson_known_values(self):
        low, high = wilson_interval(10, 20)
        self.assertAlmostEqual(low, 0.299, places=2)
        self.assertAlmostEqual(high, 0.701, places=2)
        low0, high0 = wilson_interval(0, 20)
        self.assertEqual(low0, 0.0)
        self.assertGreater(high0, 0.0)
        low1, high1 = wilson_interval(20, 20)
        self.assertEqual(high1, 1.0)

    def test_wilson_rejects_bad_counts(self):
        with self.assertRaises(ValueError):
            wilson_interval(3, 2)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)

    def test_summarise_arm(self):
        summary = summarise_arm([True, False, True])
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["survived"], 2)
        self.assertAlmostEqual(summary["survival_fraction"], 2 / 3)

    def test_paired_summary_counts(self):
        paired = paired_summary([True, False, True, False], [True, True, False, False])
        self.assertEqual(paired["both_survive"], 1)
        self.assertEqual(paired["both_fail"], 1)
        self.assertEqual(paired["adr_only_survives"], 1)
        self.assertEqual(paired["stock_only_survives"], 1)
        self.assertEqual(paired["discordant"], 2)
        self.assertAlmostEqual(paired["survival_delta_adr_minus_stock"], 0.0)

    def test_paired_summary_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            paired_summary([True], [True, False])


class TestGrid(unittest.TestCase):
    def test_grid_enumeration(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from run_experiment import grid_configs

        configs = grid_configs(steps=50)
        self.assertEqual(len(configs), 160)
        self.assertEqual(len({(c["seed"], c["dt"], c["operator_mode"]) for c in configs}), 80)
        for cfg in configs:
            self.assertIn(cfg["residual"], ("stock", "adr001"))
            self.assertIn(cfg["operator_mode"], ("legacy", "exact"))
            self.assertIn(cfg["dt"], (0.01, 0.05))
            self.assertEqual(cfg["steps"], 50)


@unittest.skipUnless(ARENA_AVAILABLE, "arena tree not found")
class TestSwapMechanics(unittest.TestCase):
    def test_off_diagonal_counts_for_stock_not_adr(self):
        import numpy as np

        from adr001_arena_gate.runner import load_arena

        _, _, _, _, shields = load_arena(ARENA_TREE)
        from adr001_arena_gate.arena_mapping import adr001_residual_matrix
        from adr001_arena_gate.runner import adr001_residual_active

        ref = np.zeros((3, 3, 32, 32), dtype=complex)
        state = np.zeros((3, 3, 32, 32), dtype=complex)
        for i in range(3):
            for j in range(3):
                ref[i, j] = np.eye(32, dtype=complex) / 32.0
                state[i, j] = np.eye(32, dtype=complex) / 32.0
        state[0, 0, 0, 1] = 0.30
        state[0, 0, 1, 0] = 0.30
        observer = shields.YangObserverShield()
        stock_obs = observer.observe(state, ref, 0)
        self.assertAlmostEqual(stock_obs.max_residual, 0.30, places=9)
        with adr001_residual_active(shields):
            adr_obs = observer.observe(state, ref, 0)
        self.assertAlmostEqual(adr_obs.max_residual, 0.0, places=9)
        # Patch is restored afterwards.
        from membrane_arena_3x3.core import residual_infinity

        self.assertIs(shields.residual_infinity, residual_infinity)

    def test_diagonal_counts_for_both(self):
        import numpy as np

        from adr001_arena_gate.runner import load_arena

        _, _, _, _, shields = load_arena(ARENA_TREE)
        from adr001_arena_gate.runner import adr001_residual_active

        ref = np.zeros((3, 3, 32, 32), dtype=complex)
        state = np.zeros((3, 3, 32, 32), dtype=complex)
        for i in range(3):
            for j in range(3):
                ref[i, j] = np.eye(32, dtype=complex) / 32.0
                state[i, j] = np.eye(32, dtype=complex) / 32.0
        state[0, 1, 5, 5] = 1.0 / 32.0 + 0.10
        observer = shields.YangObserverShield()
        stock_obs = observer.observe(state, ref, 0)
        self.assertAlmostEqual(stock_obs.max_residual, 0.10, places=9)
        with adr001_residual_active(shields):
            adr_obs = observer.observe(state, ref, 0)
        self.assertAlmostEqual(adr_obs.max_residual, 0.10, places=9)

    def test_runner_rejects_bad_arguments(self):
        from adr001_arena_gate.runner import run_one

        with self.assertRaises(ValueError):
            run_one(0, 0.05, "legacy", "weighted")
        with self.assertRaises(ValueError):
            run_one(0, 0.05, "bogus", "stock")

    def test_runner_determinism_small(self):
        from adr001_arena_gate.runner import run_one

        first = run_one(0, 0.05, "legacy", "stock", steps=8, arena_tree=ARENA_TREE)
        second = run_one(0, 0.05, "legacy", "stock", steps=8, arena_tree=ARENA_TREE)
        self.assertEqual(first, second)
        adr = run_one(0, 0.05, "legacy", "adr001", steps=8, arena_tree=ARENA_TREE)
        self.assertEqual(adr["residual"], "adr001")


if __name__ == "__main__":
    unittest.main(verbosity=2)
