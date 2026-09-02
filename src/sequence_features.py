"""Turns the monthly (gwl, rainfall_mm, temp_c, ...) table into 3D sequences
for the LSTM: (samples, lookback_months, n_features) -> next month's gwl.

This is deliberately separate from feature_engineering.py's flat lag
columns. RF/XGBoost work on tabular snapshots (gwl_lag_1, gwl_lag_3, ...),
which is the right shape for a tree model. An LSTM should instead see a
genuine ordered sequence and learn the temporal dependency itself - handing
it pre-lagged columns would just make it a worse tree model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SEQUENCE_FEATURE_COLUMNS = ["gwl", "rainfall_mm", "temp_c", "month_sin", "month_cos"]


def add_month_encoding(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = monthly_df.copy()
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def time_based_split_raw(df: pd.DataFrame, test_fraction: float):
    """Same chronological split logic as feature_engineering.time_based_split,
    but kept local so this module has no dependency on the tabular pipeline."""
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def build_sequences(df: pd.DataFrame, lookback: int):
    """Slide a `lookback`-month window across df, predicting the month right
    after each window. Returns X (n, lookback, n_features) and y (n,).
    """
    values = df[SEQUENCE_FEATURE_COLUMNS].to_numpy()
    target_idx = SEQUENCE_FEATURE_COLUMNS.index("gwl")

    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i, target_idx])
    return np.array(X), np.array(y)


def prepare_lstm_data(monthly_df: pd.DataFrame, lookback: int, test_fraction: float):
    """End-to-end: encode month, split chronologically, scale (fit on train
    only), then window into sequences. Splitting BEFORE windowing means no
    window straddles the train/test boundary with test-set information.
    """
    df = add_month_encoding(monthly_df)
    train_df, test_df = time_based_split_raw(df, test_fraction)

    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[SEQUENCE_FEATURE_COLUMNS] = scaler.fit_transform(train_df[SEQUENCE_FEATURE_COLUMNS])
    test_scaled[SEQUENCE_FEATURE_COLUMNS] = scaler.transform(test_df[SEQUENCE_FEATURE_COLUMNS])

    # Give the test window some lookback history from the tail of train,
    # otherwise the first `lookback` test months would have nothing to
    # predict from.
    combined = pd.concat([train_scaled.tail(lookback), test_scaled], ignore_index=True)

    X_train, y_train = build_sequences(train_scaled, lookback)
    X_test, y_test = build_sequences(combined, lookback)

    gwl_col_idx = SEQUENCE_FEATURE_COLUMNS.index("gwl")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "gwl_col_idx": gwl_col_idx,
        "test_dates": test_df["date"].reset_index(drop=True),
    }


def inverse_transform_gwl(scaler: MinMaxScaler, scaled_values: np.ndarray, gwl_col_idx: int) -> np.ndarray:
    """Undo scaling for just the gwl column, since the scaler was fit on all
    sequence features jointly."""
    dummy = np.zeros((len(scaled_values), len(SEQUENCE_FEATURE_COLUMNS)))
    dummy[:, gwl_col_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, gwl_col_idx]
