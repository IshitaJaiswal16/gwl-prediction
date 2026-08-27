# Groundwater Level Prediction — Comparative ML/DL Study

BTech final-year project: predicting groundwater depth from climatic,
hydrological, and (where available) anthropogenic features, comparing
Random Forest / XGBoost / LSTM baselines against a proposed LSTM→XGBoost
hybrid model.

**Current phase:** data pipeline (fetch → preprocess → feature engineering).
Baseline models (RF, XGBoost, LSTM) are the next phase, due before the
1st progress presentation.

## Project structure

```
gwl-prediction/
├── src/
│   ├── config.py              # all settings: well IDs, date range, paths
│   ├── data_fetch.py           # USGS (NGWMN) + NASA POWER API calls
│   ├── preprocessing.py        # resample to monthly, handle missing data
│   └── feature_engineering.py  # lag features, temporal features, scaling, split
├── tests/
│   └── test_pipeline_logic.py  # synthetic-data tests (no internet needed)
├── data/
│   ├── raw/                    # raw API pulls, saved as CSV (gitignored)
│   └── processed/              # final feature tables (gitignored)
├── run_pipeline.py             # entry point: runs the whole pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Step 1 — find a well.** Run this once to see candidate wells in your
chosen state/region, with their lat/lon:

```bash
python -m src.data_fetch
```

Pick a well ID and its lat/lon from the printed table, and paste it into
`WELL_IDS` in `src/config.py`:

```python
WELL_IDS = [("MWA-57655", 34.912, -117.171)]
```

**Step 2 — run the pipeline:**

```bash
python run_pipeline.py
```

This fetches raw GWL + climate data into `data/raw/`, then builds the
final feature table into `data/processed/`.

**Step 3 — validate the logic without hitting any API** (useful for
quickly checking changes to preprocessing/feature engineering):

```bash
python -m tests.test_pipeline_logic
```

## Data sources

- Groundwater levels: USGS National Ground Water Monitoring Network
  (NGWMN), via the `dataretrieval` Python package.
- Climate: NASA POWER daily point API (rainfall, temperature) — free,
  no API key required.
- Backup (if a chosen region has too little USGS history): Jasechko et
  al. 2024 global groundwater dataset, downloaded manually from
  HydroShare — see `data_fetch.load_backup_dataset()`.

## Known limitation

Anthropogenic features (groundwater extraction rate, land-use) are not
available from USGS/NASA POWER for an arbitrary well. This is documented
as a limitation consistent with Aderemi et al. (2023). A coarse
`cumulative_year_trend` feature stands in as a proxy for now.
