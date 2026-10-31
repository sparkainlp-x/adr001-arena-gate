"""ADR-001 arena-gate experiment package."""

from .adr001 import ResidualResult, calculate_residual, evaluate_residual
from .stats import paired_summary, summarise_arm, wilson_interval

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ResidualResult",
    "calculate_residual",
    "evaluate_residual",
    "paired_summary",
    "summarise_arm",
    "wilson_interval",
]
