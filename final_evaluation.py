"""Final model evaluation: RF, XGBoost, and Hybrid using their LOCKED,
tuned configurations (from tune_hyperparameters.py), compared against
Persistence and the original untuned baselines, on both the single
chronological split and the strict walk-forward CV.

This is the definitive results table for the report - run only after
tuning is locked and confirmed reproducible.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src import config
from src import feature_engineering as fe
from src import hybrid
from src import metrics

RESULTS_DIR = config.PROJECT_ROOT / "results"

# Locked configs, confirmed reproducible across two identical tuning runs.
TUNED_RF_PARAMS = dict(n_estimators=500, min_samples_leaf=5, max_features=None, max_depth=10)
TUNED_XGB_PARAMS = dict(subsample=1.0, n_estimators=200, min_child_weight=5,
                         max_depth=4, learning_rate=0.1, colsample_bytree=1.0)
TUNED_HYBRID_XGB_PARAMS = dict(subsample=0.8, n_estimators=500, min_child_weight=3,
                                max_depth=4, learning_rate=0.05, colsample_bytree=1.0)


def load_data(well_id: str):
    monthly = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_monthly.csv", parse_dates=["date"])
    features = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_features.csv", parse_dates=["date"])
    return monthly, features


def eval_single_split_tabular(features_df, model_class, params, test_fraction=config.TEST_FRACTION):
    train_df, test_df = fe.time_based_split(features_df, test_fraction)
    X_train, y_train = train_df[fe.FEATURE_COLUMNS], train_df[fe.TARGET_COLUMN]
    X_test, y_test = test_df[fe.FEATURE_COLUMNS], test_df[fe.TARGET_COLUMN].to_numpy()
    model = model_class(**params, random_state=config.RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metrics.evaluate(y_test, y_pred)


def eval_walk_forward_tabular(features_df, model_class, params, n_splits=4):
    fold_scores = []
    for train_df, test_df in fe.walk_forward_splits(features_df, n_splits=n_splits):
        X_train, y_train = train_df[fe.FEATURE_COLUMNS], train_df[fe.TARGET_COLUMN]
        X_test, y_test = test_df[fe.FEATURE_COLUMNS], test_df[fe.TARGET_COLUMN].to_numpy()
        model = model_class(**params, random_state=config.RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_scores.append(metrics.evaluate(y_test, y_pred))
    return fold_scores


def eval_persistence_single_split(features_df, test_fraction=config.TEST_FRACTION):
    _, test_df = fe.time_based_split(features_df, test_fraction)
    y_test = test_df[fe.TARGET_COLUMN].to_numpy()
    y_pred = test_df["gwl_lag_1"].to_numpy()
    return metrics.evaluate(y_test, y_pred)


def eval_persistence_walk_forward(features_df, n_splits=4):
    fold_scores = []
    for _, test_df in fe.walk_forward_splits(features_df, n_splits=n_splits):
        y_test = test_df[fe.TARGET_COLUMN].to_numpy()
        y_pred = test_df["gwl_lag_1"].to_numpy()
        fold_scores.append(metrics.evaluate(y_test, y_pred))
    return fold_scores


def summarize_folds(fold_scores):
    summary = {}
    for m in fold_scores[0].keys():
        vals = [s[m] for s in fold_scores]
        summary[f"{m}_mean"] = sum(vals) / len(vals)
        summary[f"{m}_std"] = (sum((v - summary[f"{m}_mean"]) ** 2 for v in vals) / len(vals)) ** 0.5
    return summary


def main():
    well_id = config.WELL_IDS[0][0]
    monthly, features = load_data(well_id)

    print("=" * 60)
    print("FINAL EVALUATION - single chronological split")
    print("=" * 60)

    single_split_results = {}
    single_split_results["Persistence"] = eval_persistence_single_split(features)
    single_split_results["Random Forest (tuned)"] = eval_single_split_tabular(
        features, RandomForestRegressor, TUNED_RF_PARAMS)
    single_split_results["XGBoost (tuned)"] = eval_single_split_tabular(
        features, XGBRegressor, TUNED_XGB_PARAMS)

    hybrid_result = hybrid.train_hybrid(monthly, features, xgb_params=TUNED_HYBRID_XGB_PARAMS)
    single_split_results["Hybrid (tuned)"] = metrics.evaluate(hybrid_result["y_test"], hybrid_result["y_pred"])

    single_df = pd.DataFrame(single_split_results).T.sort_values("RMSE")
    print(single_df.round(3))
    single_df.to_csv(RESULTS_DIR / f"{well_id}_final_single_split.csv")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION - strict walk-forward CV")
    print("=" * 60)

    wf_summary = {}
    wf_summary["Persistence"] = summarize_folds(eval_persistence_walk_forward(features))
    wf_summary["Random Forest (tuned)"] = summarize_folds(
        eval_walk_forward_tabular(features, RandomForestRegressor, TUNED_RF_PARAMS))
    wf_summary["XGBoost (tuned)"] = summarize_folds(
        eval_walk_forward_tabular(features, XGBRegressor, TUNED_XGB_PARAMS))

    print("\nRunning tuned hybrid strict walk-forward CV (fresh LSTM per fold, tuned XGBoost stage) ...")
    hybrid_fold_scores = []
    for fold_i, (train_df, test_df) in enumerate(fe.walk_forward_splits(features, n_splits=4), start=1):
        fold_data = hybrid.compute_fold_embeddings(monthly, train_df, test_df)
        _, y_pred, scores = hybrid.train_xgb_on_fold_embeddings(fold_data, TUNED_HYBRID_XGB_PARAMS)
        hybrid_fold_scores.append(scores)
        print(f"  Fold {fold_i}: RMSE={scores['RMSE']:.3f} R2={scores['R2']:.3f}")
    wf_summary["Hybrid (tuned)"] = summarize_folds(hybrid_fold_scores)

    wf_df = pd.DataFrame(wf_summary).T.sort_values("RMSE_mean")
    print("\n" + str(wf_df.round(3)))
    wf_df.to_csv(RESULTS_DIR / f"{well_id}_final_walk_forward.csv")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print("\nSingle split RMSE:")
    print(single_df["RMSE"].round(3))
    print("\nWalk-forward mean RMSE:")
    print(wf_df["RMSE_mean"].round(3))


if __name__ == "__main__":
    main()