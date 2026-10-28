# ADR-006: Stretch modeling target = the notebook's own R^2, locked before refactoring

Status: Accepted (2026-10-27)

## Context
The brief sets no metric or threshold. The notebook reports test R^2 (about 0.83 in its conclusion), with `train_test_split(test_size=0.3, random_state=42)` and `GridSearchCV(cv=5, scoring="r2")` over ElasticNet. A number from a markdown cell is not a reliable target, and repeat sales of one `id` can fall into both train and test (mild optimism, inherited from the notebook).

## Decision
Before writing `models.py`, run the frozen notebook procedure once, record its CV R^2, best parameters and test R^2 in `reports/results/baseline_lock.json`, and commit that file alone. The refactored training must reproduce the locked test R^2 within `config.R2_PARITY_ATOL = 0.001` (the notebook prints 3 decimals). Split, folds, grid and seed come from `config.py`. The split stays random and per row, as in the notebook.

## Consequences
`test_lock_precedes_test_eval` checks the commit order; `test_test_r2_matches_locked_notebook_r2` is the data gate. A group split by `id` is listed as a possible improvement in the write-up, not done.

Amendment (2026-10-27, Phase 3 gate): the Phase 3 hand run of `notebooks/king_county_refactored.ipynb` (nbconvert, exit 0) printed Test R^2 0.844 before `baseline_lock.json` exists. The modeling cells were verbatim and nothing was tuned, so no decision used it. It is not the lock: the Phase 4 lock still comes from running the frozen `King-County.ipynb`, and the Phase 4 notebook re-run happens after the lock commit. 0.844 vs the conclusion's "roughly 83%" suggests that text is rounded or stale.

Amendment (2026-10-28, Phase 4): CV R^2 parity is also checked against the lock's `cv_r2`, reusing `config.R2_PARITY_ATOL` (no new constant). Same code, same folds and same data, so the mean of the 5 fold R^2 values should match to the same tolerance as test R^2. Human-approved; the data gate is `test_cv_r2_matches_locked_cv_r2`.

Amendment (2026-10-28, Phase 4): the test R^2 difference A - B gets a paired percentile bootstrap interval: the test rows are resampled with replacement and both variants are scored on the same draw, `config.BOOTSTRAP_N_RESAMPLES = 10000` draws (drawn from one generator in chunks of 1,000 to bound memory), `config.BOOTSTRAP_CONFIDENCE = 0.95` (the 2.5% and 97.5% quantiles), seed `config.RANDOM_STATE`. It covers test-set sampling only, not a different split or seed, nor CV or model refit variability. An interval that includes 0 is reported as no significant difference. Human-approved; reported as `r2_a_minus_b_ci` in `metrics.json`.

Amendment (2026-10-28, Phase 4 gate review): lock provenance. The cell appended to an unmodified copy of `King-County.ipynb` to write `baseline_lock.json` is committed verbatim as `reports/results/baseline_lock_cell.py` (sha256 `ee7612199397ac56f1bc9103ad3ac19c3eda7f22cb32a3f5f2ebaebc44e831ce`; excluded from ruff so it stays byte-identical). It appends the cell with nbformat (`python baseline_lock_cell.py <notebook copy> <notebook sha256>`); the copy then ran with `jupyter nbconvert --to notebook --execute` from a working directory holding the notebook's `data/` and `assets/`. The lock file itself is unchanged. Human-approved.
