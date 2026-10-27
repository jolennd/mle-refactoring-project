# ADR-001: Keep King-County.ipynb frozen as the reference; refactor in a copy

Status: Accepted (2026-10-27)

## Context
The brief says the modeling section "stays in the notebook", and the course PR template asks for "a copy of the notebook for refactoring it". To check parity (ADR-002) we need an unchanged reference.

## Decision
`King-County.ipynb` is never edited. Its cleaning and feature cells (26-70) are copied verbatim into `tests/fixtures/legacy_notebook.py` as the parity oracle. The working notebook is `notebooks/king_county_refactored.ipynb`: it calls `kc_housing.pipeline` and keeps the modeling cells.

## Consequences
Two notebooks live in the repo; the README says which is which. A test checks that the refactored notebook imports the pipeline and contains no cleaning code. The reference notebook keeps writing to `model/` (ignored) if someone runs it.
