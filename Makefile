.PHONY: data features pipeline train evaluate api docker-up docker-down test-fast test lint format

PKG = kc_housing

# Phase 1: raw CSV -> data/processed/houses_clean.parquet
data:
	uv run python -m $(PKG).cleaning

# Phase 2: clean table -> data/processed/features.parquet
features:
	uv run python -m $(PKG).features

# Phase 3: raw CSV -> features in one reusable pipeline; parity checked by the data gates
pipeline:
	uv run python -m $(PKG).pipeline
	uv run pytest -q -m data tests/data_gates

# Phase 4 (stretch): split, grid search, baseline lock, saved model
train:
	uv run python -m $(PKG).models

evaluate:
	uv run python -m $(PKG).evaluate

# Phase 5 (stretch): CRUD API on a local SQLite database
api:
	uv run --group api uvicorn $(PKG).api.main:app --reload

# Phase 6 (stretch): API + Postgres in containers
docker-up:
	docker compose up --build --wait

docker-down:
	docker compose down

test-fast:
	uv run pytest -q -m "not integration and not model and not data and not llm"

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .
