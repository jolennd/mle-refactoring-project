# ADR-008: CRUD API for 5 house fields; SQLite in tests, Postgres in Docker

Status: Accepted (2026-10-27)

## Context
The brief asks for a FastAPI CRUD app "for houses stored in a database, using 5 features", a Dockerfile and a container run. It does not name the 5 features, the database or whether the API predicts prices. `bonus_solution/` is a reference, "not a required structure".

## Decision
Use the same 5 fields as `bonus_solution` (`config.API_FEATURES`: bedrooms, bathrooms, sqft_living, grade, zipcode). The code lives in `src/kc_housing/api/` (main, db, models, schemas) with SQLAlchemy, so it is importable and tested. The database URL comes from the `DATABASE_URL` environment variable: SQLite in tests and `make api`, Postgres in docker-compose. Validation ranges are the same as `bonus_solution/app/schemas.py`: bedrooms 0-20, bathrooms 0-20, sqft_living >= 100, grade 1-13, zipcode a string of 5-10 characters. `GET /` returns `{"status": "ok"}` as the health endpoint for the compose healthcheck. CRUD only: no prediction endpoint, because the model needs 19 features and the API stores 5. A new `api-engineer` agent owns Phases 5 and 6.

The API packages go in a separate uv dependency group `api`, not in the main dependencies. The Docker image installs only that group (`uv sync --locked --only-group api`) and copies `src/kc_housing/` onto `PYTHONPATH`, so pandas, numpy, pyarrow and scikit-learn stay out of the image. For this to work, `kc_housing/__init__.py`, `config.py` and everything under `kc_housing/api/` must import nothing outside the standard library and the `api` group.

## Consequences
New dependencies at Phase 5: `uv add --group api fastapi uvicorn sqlalchemy "psycopg[binary]" pydantic`; `uv add --dev httpx` for tests. `make api` runs with `--group api`. A test checks the import rule, and another checks that the Dockerfile installs only the `api` group. `bonus_solution/` stays untouched as reference. A prediction endpoint is an out-of-scope idea for the write-up.
