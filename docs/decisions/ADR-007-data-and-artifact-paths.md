# ADR-007: Keep the raw CSV committed; save models to models/

Status: Accepted (2026-10-27)

## Context
The playbook ignores everything in `data/`, but the course template already tracks `data/King_County_House_prices_dataset.csv` (2.4 MB). Without it a fresh clone cannot run the pipeline or the parity gate. The notebook saves its model to `model/model.bin` (ignored); the playbook uses `models/`.

## Decision
The raw CSV stays committed at its current path (whitelisted in `.gitignore`; config `RAW_DATA_PATH`). Processed files (`data/processed/*.parquet`) are ignored and rebuilt by `make data features`. Refactored models are saved to `models/elastic_net.skops` (ignored). `model/` stays ignored for the reference notebook.

## Consequences
CI could run the data gates, because the raw data is in the repo; for now CI excludes the `data` marker to stay fast, and the gates run locally before each gate. `pyarrow` was added as a dependency for Parquet.
