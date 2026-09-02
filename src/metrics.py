"""RMSE, MAE, R2, and Willmott's Index - the four metrics every model gets
scored on. Willmott's Index isn't in sklearn, hence a hand-rolled version.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def willmott_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Willmott's Index of Agreement (1981), bounded 0-1, 1 = perfect fit.

    Unlike R2, WI is bounded and penalizes both over- and under-prediction
    symmetrically relative to the mean, which is why it shows up so often
    in hydrology forecasting papers alongside RMSE/MAE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_obs = y_true.mean()

    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs)) ** 2)
    if denominator == 0:
        return 1.0
    return 1 - numerator / denominator


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return all four metrics as a plain dict, keyed by name."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "WI": float(willmott_index(y_true, y_pred)),
    }
