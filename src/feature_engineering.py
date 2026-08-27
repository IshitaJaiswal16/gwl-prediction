"""Lag/temporal features, scaling, and the chronological train/test split.

The split is time-based, not shuffled: shuffling would let a model train
on 2022 and get tested on 2020, effectively peeking at the future. Every
model in this project (baselines and the hybrid) uses the same split so
their scores are directly comparable.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from src import config


def add_lag_features(df: pd.DataFrame, lags=config.LAG_MONTHS) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"gwl_lag_{lag}"] = df["gwl"].shift(lag)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Month is encoded as sin/cos so December and January end up close
    together instead of 11 units apart."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_anthropogenic_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """No extraction-rate or land-use data is available from USGS/NASA
    POWER for an arbitrary well (a known limitation - see Aderemi et al.,
    2023). A cumulative year trend stands in as a coarse proxy for now."""
    df = df.copy()
    df["cumulative_year_trend"] = df["year"] - df["year"].min()
    return df


def build_feature_table(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(monthly_df)
    df = add_temporal_features(df)
    df = add_anthropogenic_placeholder(df)
    return df.dropna().reset_index(drop=True)


def time_based_split(df: pd.DataFrame, test_fraction: float = config.TEST_FRACTION):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)

from sklearn.model_selection import TimeSeriesSplit


def walk_forward_splits(df: pd.DataFrame, n_splits: int = 4):
    """Yield (train_df, test_df) pairs using an expanding-window, walk-forward
    split instead of one fixed train/test cut.

    Use this when a single held-out test set is too small to trust (e.g. a
    sparse well). Each fold trains on everything up to a point and tests on
    the chunk right after it; results get averaged across folds rather than
    resting on one split.
    """
    df = df.sort_values("date").reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(df):
        yield (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )

def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    """Fit on train only - fitting on the full dataset would leak test-set
    min/max values into training."""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled, scaler


FEATURE_COLUMNS = (
    [f"gwl_lag_{lag}" for lag in config.LAG_MONTHS]
    + ["rainfall_mm", "temp_c", "month_sin", "month_cos", "cumulative_year_trend"]
)
TARGET_COLUMN = "gwl"
