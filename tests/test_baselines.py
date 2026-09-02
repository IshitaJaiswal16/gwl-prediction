"""Validates src/baselines.py and src/sequence_features.py against synthetic
monthly data. Run with: python -m tests.test_baselines
"""

import numpy as np
import pandas as pd

from src import feature_engineering as fe
from src import baselines
from src import metrics


def make_synthetic_monthly(n_months=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-01", periods=n_months, freq="MS")
    trend = np.linspace(20, 28, n_months)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    noise = rng.normal(0, 0.3, n_months)
    gwl = trend + seasonal + noise

    rainfall = np.clip(rng.gamma(2, 20, n_months), 0, None)
    temp = 15 + 8 * np.sin(2 * np.pi * (np.arange(n_months) - 2) / 12) + rng.normal(0, 1, n_months)

    return pd.DataFrame({"date": dates, "gwl": gwl, "rainfall_mm": rainfall, "temp_c": temp})


def test_tree_baselines(features_df):
    train_df, test_df = fe.time_based_split(features_df)
    X_train, y_train = train_df[fe.FEATURE_COLUMNS], train_df[fe.TARGET_COLUMN]
    X_test, y_test = test_df[fe.FEATURE_COLUMNS], test_df[fe.TARGET_COLUMN].to_numpy()

    rf_model, rf_pred = baselines.train_random_forest(X_train, y_train, X_test)
    rf_scores = metrics.evaluate(y_test, rf_pred)
    assert rf_scores["R2"] > 0.5, f"RF R2 too low for synthetic trend data: {rf_scores}"
    print(f"[OK] Random Forest: {rf_scores}")

    xgb_model, xgb_pred = baselines.train_xgboost(X_train, y_train, X_test)
    xgb_scores = metrics.evaluate(y_test, xgb_pred)
    assert xgb_scores["R2"] > 0.3, f"XGBoost R2 too low: {xgb_scores}"
    print(f"[OK] XGBoost: {xgb_scores}")

    assert len(rf_model.feature_importances_) == len(fe.FEATURE_COLUMNS)
    print("[OK] feature_importances_ shape matches FEATURE_COLUMNS")


def test_lstm_baseline(monthly_df):
    model, y_pred, y_test, test_dates = baselines.train_lstm(monthly_df, lookback=6, epochs=15)
    assert len(y_pred) == len(y_test) == len(test_dates)
    scores = metrics.evaluate(y_test, y_pred)
    print(f"[OK] LSTM: {scores} (few epochs, so scores may look mediocre - that's expected)")


if __name__ == "__main__":
    monthly_df = make_synthetic_monthly()
    features_df = fe.build_feature_table(monthly_df)

    test_tree_baselines(features_df)
    test_lstm_baseline(monthly_df)
    print("\nAll baseline tests passed.")
