"""
test_pipeline_logic.py
------------------------
Validates preprocessing.py and feature_engineering.py against synthetic
data that mimics what real USGS/NASA POWER responses look like, WITHOUT
needing a live internet connection. Run with:

    python -m tests.test_pipeline_logic

This does NOT test data_fetch.py itself (that needs real network access -
run `python -m src.data_fetch` on your own machine for that).
"""

import numpy as np
import pandas as pd

from src import preprocessing
from src import feature_engineering as fe


def make_synthetic_gwl(n_days=3000, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="10D")
    # Slow declining trend + seasonal wobble + noise, roughly like a real well
    trend = np.linspace(20, 28, n_days)
    seasonal = 2 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = rng.normal(0, 0.5, n_days)
    gwl = trend + seasonal + noise
    df = pd.DataFrame({"date": dates, "gwl_ft_below_surface": gwl})
    # Punch a few gaps to test missing-value handling
    df.loc[50:52, "gwl_ft_below_surface"] = np.nan
    return df


def make_synthetic_climate(n_days=3650, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    rainfall = np.clip(rng.gamma(1.5, 2.0, n_days) * (dates.month.isin([6, 7, 8, 9])), 0, None)
    temp = 15 + 10 * np.sin(2 * np.pi * (dates.dayofyear - 60) / 365) + rng.normal(0, 1, n_days)
    return pd.DataFrame({"date": dates, "rainfall_mm": rainfall, "temp_c": temp})


def test_resample_and_missing():
    gwl_df = make_synthetic_gwl()
    climate_df = make_synthetic_climate()

    monthly = preprocessing.resample_monthly(gwl_df, climate_df)
    assert {"date", "gwl", "rainfall_mm", "temp_c"}.issubset(monthly.columns)
    assert len(monthly) > 100, "expected several years of monthly rows"

    clean = preprocessing.handle_missing(monthly)
    assert clean.isna().sum().sum() == 0, "no NaNs should remain after handle_missing"
    print(f"[OK] resample_monthly + handle_missing: {len(clean)} clean monthly rows")
    return clean


def test_feature_engineering(clean_monthly):
    features = fe.build_feature_table(clean_monthly)
    expected_cols = set(fe.FEATURE_COLUMNS + [fe.TARGET_COLUMN, "date"])
    assert expected_cols.issubset(features.columns), (
        f"missing columns: {expected_cols - set(features.columns)}"
    )
    assert features.isna().sum().sum() == 0, "no NaNs should remain after build_feature_table"
    print(f"[OK] build_feature_table: {features.shape[0]} rows x {features.shape[1]} cols")

    train_df, test_df = fe.time_based_split(features)
    assert train_df["date"].max() < test_df["date"].min(), "train must end before test begins"
    assert len(test_df) / len(features) == pytest_approx(0.20, tol=0.02)
    print(f"[OK] time_based_split: train={len(train_df)}, test={len(test_df)}, "
          f"train ends {train_df['date'].max().date()}, test starts {test_df['date'].min().date()}")

    train_scaled, test_scaled, scaler = fe.scale_features(train_df, test_df, fe.FEATURE_COLUMNS)
    assert train_scaled.min() >= -1e-9 and train_scaled.max() <= 1 + 1e-9, "train should be in [0,1]"
    print(f"[OK] scale_features: train_scaled shape {train_scaled.shape}, "
          f"test_scaled shape {test_scaled.shape}")


def pytest_approx(target, tol):
    class _Approx:
        def __eq__(self, other):
            return abs(other - target) <= tol
    return _Approx()


if __name__ == "__main__":
    clean_monthly = test_resample_and_missing()
    test_feature_engineering(clean_monthly)
    print("\nAll pipeline logic tests passed.")
