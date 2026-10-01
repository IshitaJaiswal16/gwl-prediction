"""Operational one-step-ahead forecast using the tuned hybrid model,
trained on the FULL available history (not the 80% train split - that
split exists for evaluation, not for a deployed forecast).

Same honesty rule as before: lag features, month encoding, and year trend
for the forecasted month are genuine. Rainfall/temperature are not known
in advance and are substituted with the climatological average for that
calendar month - documented, not hidden.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from xgboost import XGBRegressor

from src import config
from src import feature_engineering as fe
from src import sequence_features as seqf
from src import baselines
from src import hybrid

RESULTS_DIR = config.PROJECT_ROOT / "results"
WELL_ID = config.WELL_IDS[0][0]
LOOKBACK = 6

TUNED_HYBRID_XGB_PARAMS = dict(subsample=0.8, n_estimators=500, min_child_weight=3,
                                max_depth=4, learning_rate=0.05, colsample_bytree=1.0)


def main():
    monthly = pd.read_csv(config.DATA_PROCESSED_DIR / f"{WELL_ID}_monthly.csv", parse_dates=["date"])
    features = pd.read_csv(config.DATA_PROCESSED_DIR / f"{WELL_ID}_features.csv", parse_dates=["date"])
    monthly = monthly.sort_values("date").reset_index(drop=True)

    last_date = monthly["date"].max()
    forecast_date = last_date + pd.DateOffset(months=1)
    current_gwl = monthly.iloc[-1]["gwl"]

    # --- LSTM encoder trained on ALL history ---
    hybrid.set_all_seeds()
    df = seqf.add_month_encoding(monthly)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = df.copy()
    scaled[seqf.SEQUENCE_FEATURE_COLUMNS] = scaler.fit_transform(df[seqf.SEQUENCE_FEATURE_COLUMNS])
    X_all, y_all = seqf.build_sequences(scaled, LOOKBACK)

    print(f"Training LSTM encoder on full history ({len(X_all)} sequences) ...")
    n_features = len(seqf.SEQUENCE_FEATURE_COLUMNS)
    lstm_model = baselines.build_lstm_model(LOOKBACK, n_features)
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    lstm_model.fit(X_all, y_all, epochs=100, batch_size=8, verbose=0, callbacks=[early_stop])
    encoder = hybrid.build_encoder_submodel(lstm_model, LOOKBACK, n_features)

    # --- Embeddings for every historical month, merged with tabular features ---
    emb_df = hybrid.extract_embeddings(encoder, monthly, scaler, LOOKBACK)
    merged, hybrid_feature_columns = hybrid.build_hybrid_dataset(features, emb_df)

    print(f"Training XGBoost (tuned hybrid stage) on full merged history ({len(merged)} rows) ...")
    xgb_model = XGBRegressor(**TUNED_HYBRID_XGB_PARAMS, random_state=config.RANDOM_SEED, n_jobs=-1)
    xgb_model.fit(merged[hybrid_feature_columns], merged[fe.TARGET_COLUMN])

    # --- Build the next-month embedding from the last LOOKBACK real months ---
    tail_scaled = scaled.tail(LOOKBACK)[seqf.SEQUENCE_FEATURE_COLUMNS].to_numpy()
    next_month_embedding = encoder.predict(tail_scaled[np.newaxis, :, :], verbose=0)[0]

    # --- Build the next-month tabular row (genuine lags/trend + estimated climate) ---
    lag_1, lag_3, lag_6 = monthly.iloc[-1]["gwl"], monthly.iloc[-3]["gwl"], monthly.iloc[-6]["gwl"]
    climatology = monthly.groupby(monthly["date"].dt.month)[["rainfall_mm", "temp_c"]].mean()
    rainfall_est = climatology.loc[forecast_date.month, "rainfall_mm"]
    temp_est = climatology.loc[forecast_date.month, "temp_c"]
    year_min = monthly["date"].dt.year.min()

    next_row = {
        "gwl_lag_1": lag_1, "gwl_lag_3": lag_3, "gwl_lag_6": lag_6,
        "rainfall_mm": rainfall_est, "temp_c": temp_est,
        "month_sin": np.sin(2 * np.pi * forecast_date.month / 12),
        "month_cos": np.cos(2 * np.pi * forecast_date.month / 12),
        "cumulative_year_trend": forecast_date.year - year_min,
    }
    for j, val in enumerate(next_month_embedding):
        next_row[f"lstm_emb_{j}"] = val
    next_row_df = pd.DataFrame([next_row])[hybrid_feature_columns]

    predicted_gwl = float(xgb_model.predict(next_row_df)[0])
    change_ft = predicted_gwl - current_gwl

    out = pd.DataFrame([{
        "forecast_date": forecast_date, "current_date": last_date,
        "current_gwl": current_gwl, "predicted_gwl": predicted_gwl,
        "change_ft": change_ft, "rainfall_assumption_mm": rainfall_est,
        "temp_assumption_c": temp_est,
    }])
    out.to_csv(RESULTS_DIR / f"{WELL_ID}_next_month_forecast.csv", index=False)
    print(f"\nForecast for {forecast_date.date()}: {predicted_gwl:.2f} ft "
          f"(change {change_ft:+.2f} ft from {current_gwl:.2f} ft)")
    print(f"Saved results/{WELL_ID}_next_month_forecast.csv")


if __name__ == "__main__":
    main()