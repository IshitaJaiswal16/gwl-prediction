"""Final SHAP interpretability analysis on the tuned hybrid model
(single chronological split). Supersedes the earlier preliminary SHAP
run, which used the pre-tuning hybrid config - this uses the locked,
tuned XGBoost stage confirmed in final_evaluation.py.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src import config
from src import hybrid

RESULTS_DIR = config.PROJECT_ROOT / "results"

TUNED_HYBRID_XGB_PARAMS = dict(subsample=0.8, n_estimators=500, min_child_weight=3,
                                max_depth=4, learning_rate=0.05, colsample_bytree=1.0)


def load_data(well_id: str):
    monthly = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_monthly.csv", parse_dates=["date"])
    features = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_features.csv", parse_dates=["date"])
    return monthly, features


def main():
    well_id = config.WELL_IDS[0][0]
    monthly, features = load_data(well_id)

    print("Training final tuned hybrid (single split) for SHAP analysis ...")
    result = hybrid.train_hybrid(monthly, features, xgb_params=TUNED_HYBRID_XGB_PARAMS)

    xgb_model = result["model"]
    X_test = result["test_df"][result["feature_columns"]]

    print("Computing SHAP values ...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    out_path = RESULTS_DIR / f"{well_id}_final_hybrid_shap_summary.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # Mean |SHAP value| per feature - a compact ranked table for the report,
    # alongside the summary plot.
    mean_abs_shap = pd.DataFrame({
        "feature": result["feature_columns"],
        "mean_abs_shap": abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\n=== Top 10 features by mean |SHAP value| ===")
    print(mean_abs_shap.head(10).round(4))
    mean_abs_shap.to_csv(RESULTS_DIR / f"{well_id}_final_hybrid_shap_ranking.csv", index=False)

    total_shap = mean_abs_shap["mean_abs_shap"].sum()
    lag1_share = mean_abs_shap.loc[mean_abs_shap["feature"] == "gwl_lag_1", "mean_abs_shap"].values[0] / total_shap
    emb_cols = [c for c in mean_abs_shap["feature"] if c.startswith("lstm_emb_")]
    emb_share = mean_abs_shap[mean_abs_shap["feature"].isin(emb_cols)]["mean_abs_shap"].sum() / total_shap

    print(f"\ngwl_lag_1 share of total |SHAP|: {lag1_share:.1%}")
    print(f"Combined LSTM embedding share of total |SHAP|: {emb_share:.1%}")


if __name__ == "__main__":
    main()