# Data sources

## King County house sales
URL / access: committed with the course template at `data/King_County_House_prices_dataset.csv` (ADR-007). Originally the public Kaggle "House Sales in King County, USA" dataset.
Retrieved: 2026-10-27 (template commit dcb4be8)        License: CC0 (verify: assumed from the Kaggle original; the template does not state one)
Grain and keys: one row per sale. `id` is a parcel id, not a row key: 21,597 rows, 21,420 unique ids (repeat sales).
Coverage: sales from May 2014 to May 2015, King County, WA. 21 columns.
Update frequency: static snapshot.
Known issues (checked 2026-10-27):
- Missing values: `waterfront` 2,376, `yr_renovated` 3,842, `view` 63.
- `sqft_basement` is text (contains "?"); the notebook replaces it with `sqft_living - sqft_above`.
- One implausible row: `bedrooms == 33` (index 15856); the notebook drops it.
- `yr_renovated == 0` means "never renovated".
- Only 146 waterfront houses: the reference set for `water_distance`.
To verify: none of the columns is recorded after the sale (no lookahead risk); repeat sales of the same `id` can land on both sides of the random split (ADR-006).
Owner module: cleaning.py      Contract test: tests/integration/test_contract_clean.py
