# ADR-003: Drop the implausible row by rule (bedrooms == 33), not by index 15856

Status: Accepted (2026-10-27)

## Context
The notebook runs `kc_data.drop(15856)`. A hard-coded index only works on this exact file in this exact order, and a reusable pipeline must work on new data. The brief says nothing about outliers.

## Decision
`cleaning.py` drops rows where `bedrooms == config.IMPLAUSIBLE_BEDROOMS` (33). On the real CSV this is exactly row 15856, so parity holds. The cleaning step returns or logs the number of dropped rows.

## Consequences
Unit test `test_drops_rows_with_implausible_bedrooms`; data gate `test_real_data_loses_exactly_one_row`. Row dropping happens before the sklearn feature pipeline, because sklearn transformers cannot remove rows (ADR-004). At inference (API, new houses) no row is dropped.
