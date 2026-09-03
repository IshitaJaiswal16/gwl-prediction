"""Entry point: train the LSTM->XGBoost hybrid, evaluate it against every
baseline (including Persistence) on the same single split AND walk-forward
CV, save comparison plots + a SHAP summary.

Run this AFTER run_pipeline.py and run_baselines.py - it reads their saved
results CSVs to build the combined comparison table.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor

from src import config
from src import feature_engineering as fe
from src import hybrid
from src import metrics

RESULTS_DIR = config.PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data(well_id: str):
    monthly = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_monthly.csv", parse_dates=["date"])
    features = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_features.csv", parse_dates=["date"])
    return monthly, features


def plot_hybrid_vs_actual(dates, y_test, hybrid_pred, well_id):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_test, label="Actual", color="black", linewidth=2)
    plt.plot(dates, hybrid_pred, label="Hybrid (LSTM->XGBoost)", linestyle="--", color="red")
    plt.gca().invert_yaxis()
    plt.ylabel("Depth to water (ft)")
    plt.title(f"Hybrid: Predicted vs. actual GWL - {well_id}")
    plt.legend()
    plt.tight_layout()
    out_path = RESULTS_DIR / f"{well_id}_hybrid_predicted_vs_actual.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_shap_summary(xgb_model, X_test, well_id):
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    out_path = RESULTS_DIR / f"{well_id}_hybrid_shap_summary.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def run_hybrid_walk_forward_cv(merged_df, feature_columns, n_splits=4):
    """Retrains only the XGBoost stage per fold; the LSTM encoder is
    trained once (on the original train split) and reused across folds.
    Documented simplification: for early folds this means the embedding
    was produced by an encoder that saw slightly more of the series than
    a strict walk-forward setup would allow. Flagged as a limitation."""
    print(f"\nRunning hybrid walk-forward CV ({n_splits} folds) ...")
    fold_scores = []
    for fold_i, (train_df, test_df) in enumerate(fe.walk_forward_splits(merged_df, n_splits=n_splits), start=1):
        X_train = train_df[feature_columns]
        y_train = train_df[fe.TARGET_COLUMN]
        X_test = test_df[feature_columns]
        y_test = test_df[fe.TARGET_COLUMN].to_numpy()

        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                              random_state=config.RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores = metrics.evaluate(y_test, y_pred)
        fold_scores.append(scores)
        print(f"  Fold {fold_i}: train={len(train_df)}, test={len(test_df)} -> RMSE={scores['RMSE']:.3f}")

    summary = {}
    for m in fold_scores[0].keys():
        vals = [s[m] for s in fold_scores]
        summary[f"{m}_mean"] = np.mean(vals)
        summary[f"{m}_std"] = np.std(vals)
    return pd.DataFrame({"Hybrid": summary}).T


def main():
    well_id = config.WELL_IDS[0][0]
    monthly, features = load_data(well_id)

    print("Training hybrid (LSTM encoder -> XGBoost) ...")
    result = hybrid.train_hybrid(monthly, features)

    scores = metrics.evaluate(result["y_test"], result["y_pred"])
    print("\n=== Hybrid Single-Split Result ===")
    print(pd.Series(scores).round(3))

    baseline_results = pd.read_csv(RESULTS_DIR / f"{well_id}_baseline_results.csv", index_col=0)
    combined = pd.concat([baseline_results, pd.DataFrame({"Hybrid": scores}).T]).sort_values("RMSE")
    print("\n=== Full Comparison (single split) ===")
    print(combined.round(3))
    combined.to_csv(RESULTS_DIR / f"{well_id}_full_comparison.csv")

    plot_hybrid_vs_actual(result["test_dates"], result["y_test"], result["y_pred"], well_id)
    plot_shap_summary(result["model"], result["test_df"][result["feature_columns"]], well_id)

    per_fold_df, cv_summary, fold_predictions = hybrid.run_strict_hybrid_walk_forward_cv(monthly, features)

    fold3 = fold_predictions[3]
    plt.figure(figsize=(10, 5))
    plt.plot(fold3["dates"], fold3["y_test"], label="Actual", color="black", linewidth=2)
    plt.plot(fold3["dates"], fold3["y_pred"], label="Hybrid (fold 3)", linestyle="--", color="red")
    plt.gca().invert_yaxis()
    plt.ylabel("Depth to water (ft)")
    plt.title(f"Fold 3 diagnostic: Predicted vs. actual GWL - {well_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{well_id}_hybrid_fold3_diagnostic.png")
    plt.close()
    print(f"Saved {RESULTS_DIR / f'{well_id}_hybrid_fold3_diagnostic.png'}")

    print("\n=== Strict Hybrid Walk-Forward: Per-Fold Detail ===")
    print(per_fold_df.round(3))
    per_fold_df.to_csv(RESULTS_DIR / f"{well_id}_hybrid_walk_forward_strict_per_fold.csv")

    baseline_cv = pd.read_csv(RESULTS_DIR / f"{well_id}_walk_forward_results.csv", index_col=0)
    combined_cv = pd.concat([baseline_cv, cv_summary])
    print("\n=== Full Walk-Forward Comparison (strict) ===")
    print(combined_cv.round(3))
    combined_cv.to_csv(RESULTS_DIR / f"{well_id}_hybrid_walk_forward_strict.csv")

    print("\nPREVIOUS PRELIMINARY RESULT (DISCARDED - had fold-wise encoder leakage):")
    print("  Hybrid RMSE = 0.861 +/- 0.528  [INVALID - do not cite]")


if __name__ == "__main__":
    main()