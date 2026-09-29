"""Hyperparameter search for RF, XGBoost, and the hybrid's XGBoost stage -
all evaluated with the same walk-forward CV protocol already validated.

LSTM architecture is NOT searched here (deliberate scoping decision):
retraining it per candidate x per fold is not tractable on CPU. The
hybrid's LSTM stays fixed at its current validated config; only the
XGBoost stage on top of it is tuned, reusing each fold's pre-computed
embeddings across every candidate.

NOTE ON None-VALUED PARAMS: candidate dicts (e.g. max_features=None) are
kept in their original Python form for determining the "best" config -
never read back out of a pandas DataFrame, since mixed None/numeric
columns get silently upcast to float64 and None becomes NaN on the way
into a DataFrame. The DataFrame is only used for the saved CSV / full
results table, with None stringified there so it displays correctly.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

from src import config
from src import feature_engineering as fe
from src import hybrid
from src import metrics

RESULTS_DIR = config.PROJECT_ROOT / "results"
N_CANDIDATES = 20  # randomized search size per model; raise if time allows


def cv_score_tabular(features_df, model_class, params, n_splits=4):
    """Mean walk-forward RMSE for a tabular model (RF or XGBoost) with a
    given hyperparameter set."""
    fold_rmses = []
    for train_df, test_df in fe.walk_forward_splits(features_df, n_splits=n_splits):
        X_train, y_train = train_df[fe.FEATURE_COLUMNS], train_df[fe.TARGET_COLUMN]
        X_test, y_test = test_df[fe.FEATURE_COLUMNS], test_df[fe.TARGET_COLUMN].to_numpy()
        model = model_class(**params, random_state=config.RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_rmses.append(metrics.evaluate(y_test, y_pred)["RMSE"])
    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))


def tune_random_forest(features_df):
    param_space = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 15],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": ["sqrt", "log2", None],
    }
    return _run_search("Random Forest", features_df, RandomForestRegressor, param_space)


def tune_xgboost(features_df):
    param_space = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    return _run_search("XGBoost", features_df, XGBRegressor, param_space)


def _run_search(name, features_df, model_class, param_space):
    print(f"\nTuning {name} ({N_CANDIDATES} candidates, walk-forward CV) ...")
    candidates = list(ParameterSampler(param_space, n_iter=N_CANDIDATES, random_state=config.RANDOM_SEED))
    results = []
    for i, params in enumerate(candidates, start=1):
        mean_rmse, std_rmse = cv_score_tabular(features_df, model_class, params)
        results.append({**params, "RMSE_mean": mean_rmse, "RMSE_std": std_rmse})
        print(f"  [{i}/{len(candidates)}] RMSE={mean_rmse:.3f} params={params}")

    best_idx = min(range(len(results)), key=lambda j: results[j]["RMSE_mean"])
    best_params = candidates[best_idx]  # untouched original dict - no DataFrame round-trip
    print(f"\n  Best {name} config (verified from source, not DataFrame): {best_params}")
    print(f"  Best {name} RMSE: {results[best_idx]['RMSE_mean']:.3f} +/- {results[best_idx]['RMSE_std']:.3f}")

    # display/CSV copy only - stringify None so pandas doesn't silently
    # upcast mixed None/numeric columns to float and turn None into NaN
    display_results = [
        {k: ("None" if v is None else v) for k, v in r.items()} for r in results
    ]
    results_df = pd.DataFrame(display_results).sort_values("RMSE_mean").reset_index(drop=True)
    return results_df, best_params


def tune_hybrid_xgb_stage(monthly_df, features_df, n_splits=4, lookback=6, epochs=100):
    print(f"\nComputing embeddings once per fold ({n_splits} folds) for hybrid tuning ...")
    fold_data_list = []
    for fold_i, (train_df, test_df) in enumerate(fe.walk_forward_splits(features_df, n_splits=n_splits), start=1):
        fold_data = hybrid.compute_fold_embeddings(monthly_df, train_df, test_df, lookback, epochs)
        fold_data_list.append(fold_data)
        print(f"  Fold {fold_i} embeddings ready.")

    param_space = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    candidates = list(ParameterSampler(param_space, n_iter=N_CANDIDATES, random_state=config.RANDOM_SEED))

    print(f"\nTuning hybrid XGBoost stage ({len(candidates)} candidates, reusing cached embeddings) ...")
    results = []
    for i, xgb_params in enumerate(candidates, start=1):
        fold_rmses = []
        for fold_data in fold_data_list:
            _, _, scores = hybrid.train_xgb_on_fold_embeddings(fold_data, xgb_params)
            fold_rmses.append(scores["RMSE"])
        mean_rmse, std_rmse = float(np.mean(fold_rmses)), float(np.std(fold_rmses))
        results.append({**xgb_params, "RMSE_mean": mean_rmse, "RMSE_std": std_rmse})
        print(f"  [{i}/{len(candidates)}] RMSE={mean_rmse:.3f} params={xgb_params}")

    best_idx = min(range(len(results)), key=lambda j: results[j]["RMSE_mean"])
    best_params = candidates[best_idx]  # untouched original dict - no DataFrame round-trip
    print(f"\n  Best Hybrid XGBoost stage config (verified from source, not DataFrame): {best_params}")
    print(f"  Best Hybrid RMSE: {results[best_idx]['RMSE_mean']:.3f} +/- {results[best_idx]['RMSE_std']:.3f}")

    display_results = [
        {k: ("None" if v is None else v) for k, v in r.items()} for r in results
    ]
    results_df = pd.DataFrame(display_results).sort_values("RMSE_mean").reset_index(drop=True)
    return results_df, best_params


def main():
    well_id = config.WELL_IDS[0][0]
    monthly = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_monthly.csv", parse_dates=["date"])
    features = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_features.csv", parse_dates=["date"])

    rf_results, rf_best = tune_random_forest(features)
    rf_results.to_csv(RESULTS_DIR / f"{well_id}_rf_tuning_results.csv", index=False)
    print(f"\nBest RF (verified): {rf_best}")

    xgb_results, xgb_best = tune_xgboost(features)
    xgb_results.to_csv(RESULTS_DIR / f"{well_id}_xgboost_tuning_results.csv", index=False)
    print(f"\nBest XGBoost (verified): {xgb_best}")

    hybrid_results, hybrid_best = tune_hybrid_xgb_stage(monthly, features)
    hybrid_results.to_csv(RESULTS_DIR / f"{well_id}_hybrid_xgb_tuning_results.csv", index=False)
    print(f"\nBest Hybrid XGBoost stage (verified): {hybrid_best}")

    print("\n=== Tuning Summary (best RMSE per model) ===")
    print(f"Random Forest:  {rf_results.iloc[0]['RMSE_mean']:.3f}  config: {rf_best}")
    print(f"XGBoost:        {xgb_results.iloc[0]['RMSE_mean']:.3f}  config: {xgb_best}")
    print(f"Hybrid:         {hybrid_results.iloc[0]['RMSE_mean']:.3f}  config: {hybrid_best}")


if __name__ == "__main__":
    main()