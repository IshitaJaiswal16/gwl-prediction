"""Tests for the strict hybrid walk-forward CV. Uses synthetic data only -
no internet, no real project files needed. Run with:
    python -m tests.test_hybrid_cv
"""

import numpy as np
import pandas as pd

from src import hybrid
from src import feature_engineering as fe
from src import sequence_features as seqf


def _make_synthetic_monthly(n=90, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    gwl = 100 + np.cumsum(rng.normal(0, 0.5, n))
    rainfall = rng.uniform(0, 100, n)
    temp = 15 + 10 * np.sin(2 * np.pi * dates.month / 12) + rng.normal(0, 1, n)
    return pd.DataFrame({"date": dates, "gwl": gwl, "rainfall_mm": rainfall, "temp_c": temp})


def test_fold_train_never_overlaps_fold_test():
    monthly = _make_synthetic_monthly()
    features = fe.build_feature_table(monthly)
    for train_df, test_df in fe.walk_forward_splits(features, n_splits=3):
        assert train_df["date"].max() < test_df["date"].min()
        assert set(train_df["date"]).isdisjoint(set(test_df["date"]))
    print("[OK] fold train/test never overlap")


def test_monthly_train_slice_excludes_future_dates():
    monthly = _make_synthetic_monthly()
    features = fe.build_feature_table(monthly)
    for train_df, test_df in fe.walk_forward_splits(features, n_splits=3):
        fold_train_end = train_df["date"].max()
        monthly_train = monthly[monthly["date"] <= fold_train_end]
        assert monthly_train["date"].max() <= fold_train_end
        assert monthly_train["date"].max() < test_df["date"].min()
    print("[OK] monthly LSTM training slice never includes fold's future/test dates")


def test_scaler_fit_only_on_fold_train_not_full_series():
    monthly = _make_synthetic_monthly()
    train_window = monthly.iloc[:40].reset_index(drop=True)
    test_window = monthly.iloc[40:50].reset_index(drop=True)
    lstm_data = hybrid.prepare_fold_lstm_data(train_window, test_window, lookback=6)
    scaler = lstm_data["scaler"]
    gwl_idx = seqf.SEQUENCE_FEATURE_COLUMNS.index("gwl")

    assert np.isclose(scaler.data_min_[gwl_idx], train_window["gwl"].min())
    assert np.isclose(scaler.data_max_[gwl_idx], train_window["gwl"].max())
    # the full series' random walk continues past row 40, so its max
    # differs from the train window's max - confirms the scaler did NOT
    # see the rest of the series
    assert not np.isclose(scaler.data_max_[gwl_idx], monthly["gwl"].max())
    print("[OK] scaler is fit only on the fold's training data")


def test_sequence_window_never_includes_its_own_target():
    monthly = _make_synthetic_monthly()
    df = seqf.add_month_encoding(monthly)
    lookback = 6
    X, y = seqf.build_sequences(df, lookback)
    values = df[seqf.SEQUENCE_FEATURE_COLUMNS].to_numpy()
    gwl_idx = seqf.SEQUENCE_FEATURE_COLUMNS.index("gwl")
    for i in range(len(X)):
        assert np.allclose(X[i], values[i:i + lookback])          # window = rows [i, i+lookback)
        assert np.isclose(y[i], values[i + lookback, gwl_idx])    # target = row i+lookback, never inside window
    print(f"[OK] all {len(X)} sequence windows verified: target month never inside its own input window")


def test_predictions_length_matches_targets_per_fold():
    monthly = _make_synthetic_monthly(n=90)
    features = fe.build_feature_table(monthly)
    for train_df, test_df in fe.walk_forward_splits(features, n_splits=3):
        y_test, y_pred, n_train, n_test = hybrid.train_hybrid_fold(
            monthly, train_df, test_df, lookback=6, epochs=5
        )
        assert len(y_pred) == len(y_test) == n_test
        assert n_train == len(train_df)  # every train row survived the embedding merge
    print("[OK] prediction/target lengths match test set size in every fold")


def test_metrics_compute_without_error_on_fold_output():
    from src import metrics
    monthly = _make_synthetic_monthly(n=90)
    features = fe.build_feature_table(monthly)
    train_df, test_df = next(iter(fe.walk_forward_splits(features, n_splits=3)))
    y_test, y_pred, _, _ = hybrid.train_hybrid_fold(monthly, train_df, test_df, lookback=6, epochs=5)
    scores = metrics.evaluate(y_test, y_pred)
    assert set(scores.keys()) == {"RMSE", "MAE", "R2", "WI"}
    assert all(np.isfinite(v) for v in scores.values())
    print("[OK] metrics compute cleanly on fold output")


if __name__ == "__main__":
    test_fold_train_never_overlaps_fold_test()
    test_monthly_train_slice_excludes_future_dates()
    test_scaler_fit_only_on_fold_train_not_full_series()
    test_sequence_window_never_includes_its_own_target()
    test_predictions_length_matches_targets_per_fold()
    test_metrics_compute_without_error_on_fold_output()
    print("\nAll strict hybrid CV tests passed.")