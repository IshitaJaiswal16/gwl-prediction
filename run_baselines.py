"""Entry point: train all three baselines on the processed feature table,
compare them, and save plots. Run after run_pipeline.py has produced
data/processed/{well_id}_monthly.csv and {well_id}_features.csv.
"""

import matplotlib
matplotlib.use("Agg")  # no display in this environment; just save to file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src import config
from src import feature_engineering as fe
from src import baselines
from src import metrics

RESULTS_DIR = config.PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data(well_id: str):
    monthly = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_monthly.csv", parse_dates=["date"])
    features = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_features.csv", parse_dates=["date"])
    return monthly, features


def run_tree_baseline(name, train_fn, train_df, test_df):
    X_train = train_df[fe.FEATURE_COLUMNS]
    y_train = train_df[fe.TARGET_COLUMN]
    X_test = test_df[fe.FEATURE_COLUMNS]
    y_test = test_df[fe.TARGET_COLUMN].to_numpy()

    model, y_pred = train_fn(X_train, y_train, X_test)
    scores = metrics.evaluate(y_test, y_pred)
    return model, y_pred, y_test, test_df["date"].reset_index(drop=True), scores


def plot_predicted_vs_actual(dates, y_test, results: dict, well_id: str):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_test, label="Actual", color="black", linewidth=2)
    for name, (y_pred, _) in results.items():
        plt.plot(dates, y_pred, label=name, linestyle="--")
    plt.gca().invert_yaxis()
    plt.ylabel("Depth to water (ft)")
    plt.title(f"Predicted vs. actual GWL - {well_id}")
    plt.legend()
    plt.tight_layout()
    out_path = RESULTS_DIR / f"{well_id}_predicted_vs_actual.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_feature_importance(rf_model, xgb_model, well_id: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].barh(fe.FEATURE_COLUMNS, rf_model.feature_importances_)
    axes[0].set_title("Random Forest")
    axes[1].barh(fe.FEATURE_COLUMNS, xgb_model.feature_importances_)
    axes[1].set_title("XGBoost")
    plt.tight_layout()
    out_path = RESULTS_DIR / f"{well_id}_feature_importance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def run_walk_forward_cv(features_df, well_id: str, n_splits: int = 4):
    print(f"\nRunning walk-forward CV ({n_splits} folds) ...")
    fold_scores = {"Random Forest": [], "XGBoost": [], "Persistence": []}

    for fold_i, (train_df, test_df) in enumerate(fe.walk_forward_splits(features_df, n_splits=n_splits), start=1):
        y_test = test_df[fe.TARGET_COLUMN].to_numpy()

        _, rf_pred = baselines.train_random_forest(
            train_df[fe.FEATURE_COLUMNS], train_df[fe.TARGET_COLUMN], test_df[fe.FEATURE_COLUMNS]
        )
        _, xgb_pred = baselines.train_xgboost(
            train_df[fe.FEATURE_COLUMNS], train_df[fe.TARGET_COLUMN], test_df[fe.FEATURE_COLUMNS]
        )
        persist_pred = test_df["gwl_lag_1"].to_numpy()

        fold_scores["Random Forest"].append(metrics.evaluate(y_test, rf_pred))
        fold_scores["XGBoost"].append(metrics.evaluate(y_test, xgb_pred))
        fold_scores["Persistence"].append(metrics.evaluate(y_test, persist_pred))

        print(f"  Fold {fold_i}: train={len(train_df)}, test={len(test_df)}")

    summary = {}
    for model_name, scores_list in fold_scores.items():
        metric_names = scores_list[0].keys()
        summary[model_name] = {}
        for m in metric_names:
            vals = [s[m] for s in scores_list]
            summary[model_name][f"{m}_mean"] = np.mean(vals)
            summary[model_name][f"{m}_std"] = np.std(vals)

    summary_df = pd.DataFrame(summary).T
    print("\n=== Walk-Forward CV Results (mean ± std across folds) ===")
    print(summary_df.round(3))
    summary_df.to_csv(RESULTS_DIR / f"{well_id}_walk_forward_results.csv")
    return summary_df

def main():
    well_id = config.WELL_IDS[0][0]
    monthly, features = load_data(well_id)
    train_df, test_df = fe.time_based_split(features)

    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows\n")

    results = {}
    all_scores = {}

    print("Training Random Forest ...")
    rf_model, rf_pred, y_test, dates, rf_scores = run_tree_baseline(
        "Random Forest", baselines.train_random_forest, train_df, test_df
    )
    results["Random Forest"] = (rf_pred, rf_scores)
    all_scores["Random Forest"] = rf_scores

    print("Training XGBoost ...")
    xgb_model, xgb_pred, _, _, xgb_scores = run_tree_baseline(
        "XGBoost", baselines.train_xgboost, train_df, test_df
    )
    results["XGBoost"] = (xgb_pred, xgb_scores)
    all_scores["XGBoost"] = xgb_scores

    print("Computing Persistence baseline ...")
    persist_pred = test_df["gwl_lag_1"].to_numpy()
    persist_scores = metrics.evaluate(y_test, persist_pred)
    results["Persistence"] = (persist_pred, persist_scores)
    all_scores["Persistence"] = persist_scores

    print("Training LSTM (this takes a bit longer) ...")
    lstm_model, lstm_pred, lstm_y_test, lstm_dates = baselines.train_lstm(monthly)
    lstm_scores = metrics.evaluate(lstm_y_test, lstm_pred)
    results["LSTM"] = (lstm_pred, lstm_scores)
    all_scores["LSTM"] = lstm_scores

    print("\n=== Results ===")
    results_df = pd.DataFrame(all_scores).T
    print(results_df.round(3))
    results_df.to_csv(RESULTS_DIR / f"{well_id}_baseline_results.csv")

    plot_predicted_vs_actual(dates, y_test, {"Random Forest": results["Random Forest"],
                                             "XGBoost": results["XGBoost"],
                                             "Persistence": results["Persistence"]}, well_id)
    plot_feature_importance(rf_model, xgb_model, well_id)
    run_walk_forward_cv(features, well_id)


if __name__ == "__main__":
    main()
