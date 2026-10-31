"""Mapping ADR-001 onto the membrane arena, stated before any result.

The stock arena guard (``membrane_arena_3x3/shields.py``,
``YangObserverShield``) computes, per cell, ``residual_infinity`` from
``core.py``: ``max(abs(rho - reference))`` over the full 32x32 density
matrix, then takes the maximum over the nine cells as ``max_residual``
and rejects when ``max_residual > residual_target`` (default 0.20).

ADR-001, by contrast, is defined only for 32-vectors. An arena cell is
32-dimensional, so its natural 32-vector is its diagonal: the
populations ``rho[i, i]`` (real for a density matrix, up to float
noise). The ADR-001 arm of this experiment therefore replaces the
residual function with, per cell,

    max_i |diag(rho)[i] - diag(reference)[i]|

and takes the maximum over the nine cells, rejecting when that value
is strictly greater than the same tolerance (0.20). Everything else
-- divergence guards, purity guard, calibrator, adaptive retry,
failure_limit -- is identical between the two arms, so the measured
delta is exactly the contribution of off-diagonal drift to the stock
residual guard. Off-diagonal drift is NOT unpoliced in the ADR arm:
it still enters the divergence and purity guards.

Two consequences to keep in mind when reading the results:

- If the delta is zero, that does not prove the stock residual "is"
  ADR-001; it proves only that, on this grid, off-diagonal drift
  never decided a step that the diagonal residual would not also
  decide (or that other guards decided those steps first).
- The diagonal mapping is a modelling choice of this experiment, not
  a claim from ADR-001 itself. ADR-001 does not say anything about
  density matrices, and this package does not pretend otherwise.
"""

from __future__ import annotations


def adr001_residual_matrix(rho, reference) -> float:
    """ADR-001 aggregate residual of one cell, via its diagonal.

    ``rho`` and ``reference`` are 32x32 density matrices (numpy arrays
    in the arena). The diagonal is taken as real populations. The
    result is the ADR-001 aggregate for that cell's 32-vector.
    """
    import numpy as np

    diag = np.diag(rho).real
    ref_diag = np.diag(reference).real
    return float(np.max(np.abs(diag - ref_diag)))


def stock_residual_matrix(rho, reference) -> float:
    """The stock arena residual: max abs entry over the full matrix."""
    import numpy as np

    return float(np.max(np.abs(rho - reference)))
