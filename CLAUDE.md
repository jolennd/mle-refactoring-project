# King County housing: refactoring project

## Fixed scope (do not change without an ADR and human approval)
Goal: move the notebook's cleaning and feature engineering into a tested, reusable pipeline in
src/kc_housing/ whose output matches the notebook (King-County.ipynb is the frozen reference, ADR-001/002).
Label: price (USD).   Features: config.FEATURES (19 columns, notebook order).   Leakage: never use config.LEAKAGE_COLUMNS as inputs.
Metric (stretch modeling): test R^2, reproducing the notebook's own value (ADR-006).
Split: train_test_split(test_size=0.3, random_state=42); GridSearchCV 5 folds (config).

## Conventions
- Logic in src/kc_housing/; notebooks thin; paths and thresholds only from config.py.
- Parity first: reproduce the notebook exactly, then improve only through an ADR.
- Tests: `make test-fast` while working, `make test` before a gate. Never weaken a test.
- Every non-obvious decision becomes an ADR in docs/decisions/.
- The human approves every phase gate in PROJECT_WORKFLOW.md.

## Not in this repository
Agent, command and phase-spec definitions live in a private folder that is not committed.
A fresh clone runs and tests fully without them; they only guide the AI-assisted workflow.
