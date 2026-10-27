# ADR-005: water_distance learns its reference houses in fit()

Status: Accepted (2026-10-27)

## Context
The notebook computes `water_distance` against all waterfront houses in the full dataset, before the train/test split. Waterfront locations are inputs, not the target, so this is not target leakage, but test rows do shape a training feature. For a reusable pipeline, new houses need a fixed reference set.

## Decision
`WaterDistance.fit(X)` stores the coordinates of rows with `waterfront == 1`; `transform` uses only those. Phase 3 parity fits on the full cleaned data (as the notebook does). Phase 4 runs both variants, fitted on all rows (parity with the notebook R^2) and fitted on training rows only (the correct procedure), and report the R^2 difference.

## Consequences
Test `test_water_distance_uses_reference_from_fit`. Waterfront houses get distance 0 to themselves when they are in the reference set, as in the notebook. The write-up names the full-data fit as a limitation of the original notebook.
