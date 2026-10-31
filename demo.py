"""Demo: the ADR-001 rule on example vectors, plus a Wilson interval.

No arena run: this demo is instant and standard-library only. For the
arena experiment, run ``python3 run_experiment.py`` (see README).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adr001_arena_gate.adr001 import evaluate_residual
from adr001_arena_gate.stats import wilson_interval


def main() -> int:
    reference = [0.0] * 32
    observed = [0.0] * 32
    observed[7] = 0.25
    result = evaluate_residual(reference, observed, 0.20)
    print(f"ADR-001 example: aggregate={result.aggregate_residual:.4f}, "
          f"tolerance={result.tolerance:.2f}, passed={result.passed}, "
          f"failed={result.failed}  (strict > rule: 0.25 > 0.20 fails)")
    edge = evaluate_residual(reference, [0.0] * 31 + [0.20], 0.20)
    print(f"Edge case (aggregate == tolerance): passed={edge.passed} "
          f"(equality passes under strict >)")
    low, high = wilson_interval(12, 20)
    print(f"Wilson 95% for 12/20: [{low:.3f}, {high:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
