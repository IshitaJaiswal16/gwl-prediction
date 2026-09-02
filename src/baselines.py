"""Train and predict with the three standalone baselines: Random Forest,
XGBoost, and LSTM. Each function takes prepared data in and returns test-set
predictions out, so run_baselines.py can score and plot them identically.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras import layers

from src import config
from src import sequence_features as seqf


def train_random_forest(X_train, y_train, X_test):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, model.predict(X_test)


def train_xgboost(X_train, y_train, X_test):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, model.predict(X_test)


def build_lstm_model(lookback: int, n_features: int) -> keras.Model:
    """Small LSTM: one recurrent layer feeding a dense output. Kept simple
    deliberately - this is the standalone baseline, not the hybrid's
    feature-extraction stage."""
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(32, name="lstm_encoder"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(monthly_df, lookback: int = 6, test_fraction: float = config.TEST_FRACTION,
                epochs: int = 100):
    data = seqf.prepare_lstm_data(monthly_df, lookback, test_fraction)

    model = build_lstm_model(lookback, n_features=data["X_train"].shape[2])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True
    )
    model.fit(
        data["X_train"], data["y_train"],
        epochs=epochs, batch_size=8, verbose=0, callbacks=[early_stop],
    )

    y_pred_scaled = model.predict(data["X_test"], verbose=0).flatten()
    y_pred = seqf.inverse_transform_gwl(data["scaler"], y_pred_scaled, data["gwl_col_idx"])
    y_test = seqf.inverse_transform_gwl(data["scaler"], data["y_test"], data["gwl_col_idx"])

    return model, y_pred, y_test, data["test_dates"]
