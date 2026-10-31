# adr001-arena-gate

Swap the ADR-001 max-abs residual into the membrane arena's guard and
measure what changes. Classical synthetic toy only.

## Question

The 2026-10-29 code exploration found that "OES residual" means two
things in the sparkainlp-x repos: the normative ADR-001 definition in
`oes32-residual` (max-abs over two 32-vectors, failure iff strictly
greater than tolerance) and a weighted score in `oes-resilience`.
This package asks the narrower, falsifiable version of that question
inside one system: if the arena's guard used the ADR-001 residual
instead of its stock residual, would survival or acceptance change?

## The two residuals, exactly

**Stock arena residual** (`membrane-arena-3x3-upgrade`,
`membrane_arena_3x3/core.py:residual_infinity`, used by
`shields.py:YangObserverShield`): per cell,
`max(abs(rho - reference))` over the full 32x32 density matrix;
the guard uses the maximum over the nine cells and rejects when it
is `> residual_target` (default 0.20).

**ADR-001 residual** (`sparkainlp-x/oes32-residual`,
`src/residual_reference.py`, blob SHA
`f187d8f198a6682927de8cc84cdb1905ee7be2ce`, read 2026-10-30):
for two 32-vectors, component residual `abs(observed[i] -
reference[i])`, aggregate = max over i, failure iff aggregate `>`
tolerance. Implemented here in `src/adr001_arena_gate/adr001.py`,
standard-library only, with the strict-`>` rule pinned by tests
(equality with tolerance passes).

**Mapping (this experiment's decision, stated before the results).**
ADR-001 is defined only for 32-vectors, and an arena cell is
32-dimensional, so the ADR arm uses each cell's diagonal
(populations `rho[i, i]`) as its 32-vector: per cell
`max_i |diag(rho)[i] - diag(reference)[i]|`, max over cells, same
tolerance 0.20, same strict `>`. The swap is implemented at the
residual-function level (`runner.py` patches
`membrane_arena_3x3.shields.residual_infinity` for the ADR run and
restores it afterwards; no arena file is modified). Everything else
-- divergence guards, purity guard, calibrator, adaptive retry,
failure_limit 5 -- is identical between arms, so the measured delta
is exactly the contribution of off-diagonal drift to the stock
residual guard. Off-diagonal drift is not unpoliced in the ADR arm:
it still enters the divergence and purity guards.

## Bar

- The ADR implementation reproduces the ADR-001 contract (tests).
- The swap mechanics are pinned: a purely off-diagonal difference
  trips the stock residual and not the ADR residual; a diagonal
  difference trips both (tests).
- Runner determinism: the same configuration run twice gives the
  same result (tests, small step count).
- A null delta is a publishable result, not a failure. No threshold
  was tuned to manufacture a difference; the tolerance is the stock
  0.20 in both arms.

## Grid

Seeds 0-19, dt in {0.01, 0.05}, operator_mode in {legacy, exact},
residual in {stock, adr001}: 160 runs. Steps are 50, the arena
benchmark's sweep length: a 200-step exact-mode run takes ~60 s on
this machine, so the 200-step grid would take hours; 50 steps keeps
the full seed x dt x mode grid tractable while still measuring
survival (completed all 50 steps without the breaker latching) and
acceptance fraction. Survival is defined per run as status
`completed`; acceptance fraction is accepted / executed steps.

Statistics: Wilson 95% intervals for survival fractions; paired
comparison by (seed, dt, mode) -- discordant counts and mean paired
acceptance delta -- because the arms differ only in the residual
function. n = 20 per cell is small; the intervals show it.

## Results

Grid completed 2026-10-30: 160/160 runs, 50 steps each, full data in
`results/results.json`. **The delta is exactly zero, in every cell,
on every decision metric.** This is reported as the result, not as
a failure, and no threshold was touched to change it.

| Cell | Stock survival (n=20) | ADR-001 survival (n=20) | Survival delta | Mean accepted: stock / ADR |
|---|---|---|---|---|
| legacy, dt=0.01 | 13 (0.650, Wilson [0.433, 0.819]) | 13 (0.650, Wilson [0.433, 0.819]) | 0.000 (discordant 0) | 0.9575 / 0.9575 |
| legacy, dt=0.05 | 0 (0.000, Wilson [0.000, 0.161]) | 0 (0.000, Wilson [0.000, 0.161]) | 0.000 (discordant 0) | 0.5315 / 0.5315 |
| exact, dt=0.01 | 20 (1.000, Wilson [0.839, 1.000]) | 20 (1.000, Wilson [0.839, 1.000]) | 0.000 (discordant 0) | 1.0000 / 1.0000 |
| exact, dt=0.05 | 0 (0.000, Wilson [0.000, 0.161]) | 0 (0.000, Wilson [0.000, 0.161]) | 0.000 (discordant 0) | 0.3080 / 0.3080 |
| **Overall (n=80/arm)** | **33 (0.4125, Wilson [0.311, 0.522])** | **33 (0.4125, Wilson [0.311, 0.522])** | **0.000** | — |

Stronger than "same survival": **0 of 80 paired runs differ in
status, steps executed, accepted steps, repair accepts, or repair
attempts**, and rejection-reason counts are identical in both arms
(residual_above_target 110, purity_too_low 100,
divergence_above_maximum 28, per arm). The two arms took the same
trajectories on this grid.

**Why the null, and why it is not a broken patch.** Two checks
rule out a dead swap. First, the swap-mechanics test proves the
patch changes the observer's residual (a purely off-diagonal
difference scores 0.30 stock, 0.00 ADR). Second, the final-state
telemetry in the grid itself differs in 11 of 80 pairs -- always
ADR <= stock, e.g. exact dt=0.01 seed 0: 0.0401 stock vs 0.0357 ADR
-- so the ADR function really was active during the runs. The
decisions coincide because, on the states this arena actually
visits, the largest entrywise drift is attained on the diagonal
(population drift dominates off-diagonal drift, and the guard
threshold 0.20 is far above the observed off-diagonal excess).
A follow-up that could break the tie: a regime where off-diagonal
drift is driven large while populations stay put (stronger
dephasing/mirror settings, or a dedicated probe state) -- that
regime is not in this grid, and this package does not claim the
null generalises to it.

Context numbers (not deltas): survival is governed by dt and mode,
not by the residual -- exact dt=0.01 survives 20/20, dt=0.05
survives 0/20 in both modes, legacy dt=0.01 survives 13/20. The
residual swap moves none of it.

## How to run

```bash
python3 tests/tests.py          # unit tests always run; arena tests skip if the tree is absent
python3 demo.py                 # ADR-001 rule example, no arena run
python3 run_experiment.py --steps 50 --out results   # full grid (slow: tens of minutes)
```

The arena tree is located via `ADR001_ARENA_TREE` or defaults to
`~/workspace/membrane-arena-3x3-upgrade`. The package's own code
(`adr001.py`, `stats.py`) is standard-library only; `runner.py`
imports the numpy-based arena as an external integration layer and
says so -- a pure-stdlib re-implementation of 32-dimensional
density-matrix evolution would be a different engine, not the arena.

## Honest limits

- Classical synthetic toy only. No quantum-hardware, medical, brain,
  or consciousness claim. ADR-001 is a numerical contract, not a
  physical model, and this experiment does not validate any choice
  of vectors, tolerance, or system outside the arena.
- The diagonal mapping is this experiment's choice, not ADR-001's:
  ADR-001 says nothing about density matrices. A zero delta proves
  only that, on this grid, off-diagonal drift never decided a step
  that the diagonal residual or another guard would not also decide.
- 50 steps, 20 seeds per cell: the grid measures the guard swap in
  the arena's early regime, not long-run behaviour.
- The weighted `oes-resilience` score is NOT tested here; that is a
  separate comparison (weighted vs max-abs on the same vectors).
