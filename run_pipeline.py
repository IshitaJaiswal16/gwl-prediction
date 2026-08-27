"""Entry point: fetch -> preprocess -> engineer features -> save.

Run `python -m src.data_fetch` first to pick a well, set WELL_IDS in
src/config.py, then run this.
"""

import sys
import pandas as pd

from src import config
from src import data_fetch
from src import preprocessing
from src import feature_engineering as fe


def fetch_raw(well_id: str, lat: float, lon: float):
    print(f"[1/4] Fetching GWL data for well {well_id} ...")
    gwl_df = data_fetch.get_gwl_daily_timeseries(well_id, config.START_DATE, config.END_DATE)
    gwl_df.to_csv(config.DATA_RAW_DIR / f"{well_id}_gwl_raw.csv", index=False)
    print(f"      {len(gwl_df)} readings saved.")

    print(f"[2/4] Fetching climate data for lat={lat}, lon={lon} ...")
    climate_df = data_fetch.get_climate_data(lat, lon, config.START_DATE, config.END_DATE)
    climate_df.to_csv(config.DATA_RAW_DIR / f"{well_id}_climate_raw.csv", index=False)
    print(f"      {len(climate_df)} daily records saved.")

    return gwl_df, climate_df


def build_dataset(well_id: str, gwl_df: pd.DataFrame, climate_df: pd.DataFrame) -> pd.DataFrame:
    print("[3/4] Resampling to monthly + handling missing values ...")
    monthly = preprocessing.resample_monthly(gwl_df, climate_df)
    monthly = preprocessing.handle_missing(monthly)

    print("[4/4] Building lag/temporal features ...")
    feature_table = fe.build_feature_table(monthly)

    out_path = config.DATA_PROCESSED_DIR / f"{well_id}_features.csv"
    feature_table.to_csv(out_path, index=False)
    print(f"      Final table: {feature_table.shape[0]} rows x "
          f"{feature_table.shape[1]} cols -> {out_path}")
    return feature_table


def main():
    if not config.WELL_IDS:
        print(
            "config.WELL_IDS is empty. Run `python -m src.data_fetch`, "
            "pick a well, and add it as (well_id, lat, lon) in src/config.py."
        )
        sys.exit(1)

    for well_id, lat, lon in config.WELL_IDS:
        gwl_df, climate_df = fetch_raw(well_id, lat, lon)
        feature_table = build_dataset(well_id, gwl_df, climate_df)
        train_df, test_df = fe.time_based_split(feature_table)
        print(f"      train: {len(train_df)} rows, test: {len(test_df)} rows "
              f"(most recent {config.TEST_FRACTION:.0%})\n")


if __name__ == "__main__":
    main()
