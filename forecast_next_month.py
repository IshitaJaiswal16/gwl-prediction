"""One-step-ahead operational forecast using the tuned XGBoost model.

Genuine forecast for the month after the latest available data - distinct
from the held-out test evaluation (that's what final_evaluation.py and
the dashboard's actual-vs-predicted chart show).

Two things make this an operational forecast rather than a re-evaluation:
1. Trained on the FULL historical feature table, not the 80% train split
   - appropriate once evaluation is already locked.
2. Rainfall and temperature for the forecasted month are NOT known in
   advance. This script substitutes the climatological mean for that
   calendar month (the average across all prior years at this well) as a
   documented estimate. Lag features, month encoding, and year trend are
   all genuine, computed from real historical data - only these two
   weather inputs are estimated.

Run after new monthly data is added / before viewing the dashboard.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src import config
from src import feature_engineering as fe

RESULTS_DIR = config.PROJECT_ROOT / "results"
WELL_ID = config.WELL_IDS[0][0]

TUNED_XGB_PARAMS = dict(subsample=1.0, n_estimators=200, min_child_weight=5,
                         max_depth=4, learning_rate=0.1, colsample_bytree=1.0)


def build_next_month_row(monthly_df: pd.DataFrame):
    monthly_df = monthly_df.sort_values("date").reset_index(drop=True)
    last_date = monthly_df["date"].max()
    next_date = last_date + pd.DateOffset(months=1)

    recent = monthly_df.tail(6)["date"].reset_index(drop=True)
    gaps_ok = all(
        (recent[i + 1].year - recent[i].year) * 12 + (recent[i + 1].month - recent[i].month) == 1
        for i in range(len(recent) - 1)
    )
    if not gaps_ok:
        print("WARNING: recent monthly data has a gap - lag features for the "
              "forecast may not reflect exactly 1/3/6 months prior.")

    lag_1 = monthly_df.iloc[-1]["gwl"]
    lag_3 = monthly_df.iloc[-3]["gwl"]
    lag_6 = monthly_df.iloc[-6]["gwl"]

    climatology = monthly_df.groupby(monthly_df["date"].dt.month)[["rainfall_mm", "temp_c"]].mean()
    rainfall_est = climatology.loc[next_date.month, "rainfall_mm"]
    temp_est = climatology.loc[next_date.month, "temp_c"]
    year_min = monthly_df["date"].dt.year.min()

    row = pd.DataFrame([{
        "gwl_lag_1": lag_1, "gwl_lag_3": lag_3, "gwl_lag_6": lag_6,
        "rainfall_mm": rainfall_est, "temp_c": temp_est,
        "month_sin": np.sin(2 * np.pi * next_date.month / 12),
        "month_cos": np.cos(2 * np.pi * next_date.month / 12),
        "cumulative_year_trend": next_date.year - year_min,
    }])
    return row, next_date, last_date, monthly_df.iloc[-1]["gwl"], rainfall_est, temp_est


def main():
    monthly = pd.read_csv(config.DATA_PROCESSED_DIR / f"{WELL_ID}_monthly.csv", parse_dates=["date"])
    features = pd.read_csv(config.DATA_PROCESSED_DIR / f"{WELL_ID}_features.csv", parse_dates=["date"])

    next_row, forecast_date, current_date, current_gwl, rainfall_est, temp_est = build_next_month_row(monthly)

    print(f"Training XGBoost (tuned) on full history ({len(features)} rows) for operational forecast ...")
    model = XGBRegressor(**TUNED_XGB_PARAMS, random_state=config.RANDOM_SEED, n_jobs=-1)
    model.fit(features[fe.FEATURE_COLUMNS], features[fe.TARGET_COLUMN])

    predicted_gwl = float(model.predict(next_row[fe.FEATURE_COLUMNS])[0])
    change_ft = predicted_gwl - current_gwl

    out = pd.DataFrame([{
        "forecast_date": forecast_date, "current_date": current_date,
        "current_gwl": current_gwl, "predicted_gwl": predicted_gwl,
        "change_ft": change_ft, "model_used": "XGBoost (tuned)",
        "rainfall_assumption_mm": rainfall_est, "temp_assumption_c": temp_est,
        "trained_on_n_rows": len(features),
    }])
    out_path = RESULTS_DIR / f"{WELL_ID}_next_month_forecast.csv"
    out.to_csv(out_path, index=False)

    print(f"\nForecast for {forecast_date.date()}: {predicted_gwl:.2f} ft "
          f"(change of {change_ft:+.2f} ft from {current_gwl:.2f} ft)")
    print(f"Rainfall/temp assumption (climatological avg for that month): "
          f"{rainfall_est:.1f} mm, {temp_est:.1f} °C")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()