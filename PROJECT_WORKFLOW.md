# King County Refactoring Project Workflow

Status: todo · doing · done. Each phase ends with a gate: make target runs + gate tests pass + reviewer PASS + human approval.
Phases 4 to 6 are stretch goals from the brief; Phase 7 is required. Phase plan: ADR-009.

| Phase | Work | Owner | Notebook / module | Make target | Gate tests | Human check | Status |
|---|---|---|---|---|---|---|---|
| 0 Framing | Scope, parity definition, success criteria, ADRs | human | docs/00 to 03, docs/decisions/ | (none) | traceability table complete | scope and ADRs approved | done 2026-10-27: ADR-001 to 009 accepted; tolerances 1e-9 (features), 0.001 (R^2) |
| 1 Load and clean | Load raw CSV; drop the implausible row; recompute sqft_basement; fill view/waterfront; build last_known_change | data-engineer | cleaning.py | make data | tests/unit/test_cleaning.py, tests/integration/test_contract_clean.py, tests/data_gates/test_clean_real.py | dropped rows = 1, reported | doing |
| 2 Features | center_distance, water_distance (vectorized, fitted reference), column selection to config.FEATURES | feature-engineer | features.py | make features | tests/unit/test_features.py, tests/integration/test_contract_features.py | formulas match the notebook cell by cell | todo |
| 3 Pipeline (parity) | One reusable sklearn-compatible pipeline raw -> features; refactored notebook copy uses it | feature-engineer | pipeline.py, notebooks/king_county_refactored.ipynb | make pipeline | tests/unit/test_pipeline.py, tests/data_gates/test_parity.py, tests/unit/test_notebook_thin.py | parity report: max abs diff per column | todo |
| 4 Modeling (stretch) | Split, ElasticNet grid search, lock CV R^2, test R^2, save model | modeler, evaluator | splits.py, models.py, evaluate.py | make train evaluate | tests/model/test_models.py, tests/integration/test_experiment_order.py, tests/data_gates/test_r2_parity.py | lock committed alone; R^2 matches notebook | todo |
| 5 FastAPI CRUD (stretch) | Create/read/update/delete houses (5 features) in a database; GET / health; `api` dependency group | api-engineer | api/ (main, db, models, schemas) | make api | tests/integration/test_api_crud.py | manual run through /docs | todo |
| 6 Docker (stretch) | Dockerfile (api group only), compose with Postgres and healthchecks, run in container | api-engineer | Dockerfile, docker-compose.yaml | make docker-up | tests/unit/test_docker_files.py | container serves /docs; CRUD smoke test | todo |
| 7 Write-up | README: steps, challenges, what I'd do differently; results; how to run | presentation-coach | README.md | (none) | tests/unit/test_readme.py | read-through; numbers traceable | todo |
