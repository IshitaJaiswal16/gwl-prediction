"""Run once (outside Streamlit) to export the final tuned hybrid's
test-set predictions to CSV, so the dashboard can display them without
needing to import TensorFlow itself - avoids a known Windows issue where
TF's native DLL loader can fail inside Streamlit's worker thread.
"""

import pandas as pd

from src import config
from src import hybrid

RESULTS_DIR = config.PROJECT_ROOT / "results"
WELL_ID = config.WELL_IDS[0][0]

TUNED_HYBRID_XGB_PARAMS = dict(subsample=0.8, n_estimators=500, min_child_weight=3,
                                max_depth=4, learning_rate=0.05, colsample_bytree=1.0)


def main():
    monthly = pd.read_csv(config.DATA_PROCESSED_DIR / f"{WELL_ID}_monthly.csv", parse_dates=["date"])
    features = pd.read_csv(config.DATA_PROCESSED_DIR / f"{WELL_ID}_features.csv", parse_dates=["date"])

    print("Training final tuned hybrid to export predictions for dashboard ...")
    result = hybrid.train_hybrid(monthly, features, xgb_params=TUNED_HYBRID_XGB_PARAMS)

    out_df = pd.DataFrame({
        "date": result["test_dates"],
        "actual": result["y_test"],
        "predicted": result["y_pred"],
    })
    out_path = RESULTS_DIR / f"{WELL_ID}_dashboard_hybrid_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()