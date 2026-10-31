"""ADR-001 max-abs residual, implemented exactly as defined.

Provenance: sparkainlp-x/oes32-residual, file
``src/residual_reference.py`` (blob SHA
f187d8f198a6682927de8cc84cdb1905ee7be2ce, read 2026-10-30), itself
the implementation of the ADR-001 contract stated in that repo's
README and ``residual_definition.md`` / ``failure_criterion.md``:

- Inputs: two finite real-valued vectors of length 32 and a finite,
  non-negative tolerance.
- Component residual: ``abs(observed[i] - reference[i])``.
- Aggregate residual: the maximum of the 32 component residuals.
- Failure rule: failure occurs if and only if the aggregate residual
  is strictly greater than the tolerance (``>``, not ``>=``).
- Invalid input is rejected with ``ValueError``; no status is produced.

This module is standard-library only and is deliberately a faithful
re-implementation, not a copy of the repo's test suite. The arena
mapping (diagonal populations as the 32-vector) lives in
``arena_mapping.py`` and is a separate, documented decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fabs, isfinite
from numbers import Real
from typing import Sequence

DIMENSION = 32


@dataclass(frozen=True)
class ResidualResult:
    """The deterministic result of one valid OES-32 residual evaluation."""

    component_residuals: tuple[float, ...]
    aggregate_residual: float
    tolerance: float
    passed: bool
    failed: bool


def _validate_vector(values: Sequence[Real], name: str) -> tuple[float, ...]:
    """Return a finite 32-component real vector or raise ``ValueError``."""
    try:
        vector = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of {DIMENSION} real values") from error
    if len(vector) != DIMENSION:
        raise ValueError(f"{name} must contain exactly {DIMENSION} values")
    converted: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name}[{index}] must be a real value")
        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise ValueError(f"{name}[{index}] must be finite")
        converted.append(numeric_value)
    return tuple(converted)


def _validate_tolerance(tolerance: Real) -> float:
    """Return a finite non-negative tolerance or raise ``ValueError``."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise ValueError("tolerance must be a real value")
    numeric_tolerance = float(tolerance)
    if not isfinite(numeric_tolerance) or numeric_tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return numeric_tolerance


def calculate_residual(
    reference: Sequence[Real], observed: Sequence[Real]
) -> tuple[tuple[float, ...], float]:
    """Calculate component and aggregate residuals for two valid vectors."""
    reference_vector = _validate_vector(reference, "reference")
    observed_vector = _validate_vector(observed, "observed")
    component_residuals = tuple(
        fabs(observed_value - reference_value)
        for reference_value, observed_value in zip(reference_vector, observed_vector)
    )
    return component_residuals, max(component_residuals)


def evaluate_residual(
    reference: Sequence[Real], observed: Sequence[Real], tolerance: Real
) -> ResidualResult:
    """Evaluate against the ADR-001 failure criterion (strict ``>``)."""
    numeric_tolerance = _validate_tolerance(tolerance)
    component_residuals, aggregate_residual = calculate_residual(reference, observed)
    passed = aggregate_residual <= numeric_tolerance
    return ResidualResult(
        component_residuals=component_residuals,
        aggregate_residual=aggregate_residual,
        tolerance=numeric_tolerance,
        passed=passed,
        failed=not passed,
    )


__all__ = ["DIMENSION", "ResidualResult", "calculate_residual", "evaluate_residual"]
