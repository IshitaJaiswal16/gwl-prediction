"""LSTM -> XGBoost hybrid: the LSTM's recurrent-layer output (a learned
temporal embedding) is extracted for every month and combined with the
existing static/tabular features; XGBoost trains on the combined table.

Unlike the standalone LSTM baseline (which reads the final dense-layer
prediction), here the LSTM is cut open right after its recurrent layer -
that 32-dim vector is treated as engineered features, the same role
gwl_lag_1/3/6 play, handed to XGBoost alongside them. This tests whether
XGBoost can use temporal information when it's shaped as features it can
split on, given the baselines showed it otherwise ignores rainfall/temp/
seasonality entirely in favor of gwl_lag_1.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from xgboost import XGBRegressor
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src import metrics

from src import config
from src import sequence_features as seqf
from src import feature_engineering as fe
from src import baselines

EMBEDDING_DIM = 32


def train_lstm_encoder(monthly_df: pd.DataFrame, lookback: int, test_fraction: float, epochs: int = 100):
    """Same architecture, same train-only fit as the standalone LSTM
    baseline - so the embedding is learned under identical conditions."""
    data = seqf.prepare_lstm_data(monthly_df, lookback, test_fraction)
    model = baselines.build_lstm_model(lookback, n_features=data["X_train"].shape[2])
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=8, verbose=0, callbacks=[early_stop])
    return model, data["scaler"]


def build_encoder_submodel(trained_lstm: keras.Model, lookback: int, n_features: int) -> keras.Model:
    """Rebuilds the encoder as an explicit functional model, reusing the
    trained LSTM layer's weights directly (layers are callable, so this
    reuses weights without retraining - it's just a graph-construction
    workaround for a Keras 3 issue where Sequential.input doesn't resolve
    reliably even after training)."""
    encoder_input = keras.Input(shape=(lookback, n_features))
    lstm_layer = trained_lstm.get_layer("lstm_encoder")
    encoder_output = lstm_layer(encoder_input)
    return keras.Model(inputs=encoder_input, outputs=encoder_output)


def extract_embeddings(encoder: keras.Model, monthly_df: pd.DataFrame, scaler, lookback: int) -> pd.DataFrame:
    """Slide the lookback window across the WHOLE monthly series (train and
    test both - this is inference only, no weights change here) so every
    date gets an embedding that can be merged with the tabular table by
    date. Scaling uses the scaler fit on the train split only."""
    df = seqf.add_month_encoding(monthly_df).sort_values("date").reset_index(drop=True)
    scaled = df.copy()
    scaled[seqf.SEQUENCE_FEATURE_COLUMNS] = scaler.transform(df[seqf.SEQUENCE_FEATURE_COLUMNS])

    values = scaled[seqf.SEQUENCE_FEATURE_COLUMNS].to_numpy()
    X, dates = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        dates.append(df["date"].iloc[i])
    X = np.array(X)

    embeddings = encoder.predict(X, verbose=0)
    emb_cols = [f"lstm_emb_{j}" for j in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    emb_df["date"] = dates
    return emb_df


def build_hybrid_dataset(features_df: pd.DataFrame, emb_df: pd.DataFrame):
    """Inner-join tabular features with LSTM embeddings on date, so every
    row has both the flat lag/climate features AND the learned temporal
    embedding."""
    merged = features_df.merge(emb_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    emb_cols = [c for c in emb_df.columns if c.startswith("lstm_emb_")]
    hybrid_feature_columns = fe.FEATURE_COLUMNS + emb_cols
    return merged, hybrid_feature_columns


def train_hybrid(monthly_df: pd.DataFrame, features_df: pd.DataFrame,
                  lookback: int = 6, test_fraction: float = config.TEST_FRACTION,
                  epochs: int = 100):
    """End-to-end: train LSTM encoder on the train split -> extract
    embeddings for every date -> merge with tabular features -> re-split
    chronologically -> train XGBoost on train, predict test."""
    set_all_seeds()
    lstm_model, scaler = train_lstm_encoder(monthly_df, lookback, test_fraction, epochs)
    n_features = len(seqf.SEQUENCE_FEATURE_COLUMNS)
    encoder = build_encoder_submodel(lstm_model, lookback, n_features)
    emb_df = extract_embeddings(encoder, monthly_df, scaler, lookback)

    merged, hybrid_feature_columns = build_hybrid_dataset(features_df, emb_df)
    train_df, test_df = fe.time_based_split(merged, test_fraction)

    X_train = train_df[hybrid_feature_columns]
    y_train = train_df[fe.TARGET_COLUMN]
    X_test = test_df[hybrid_feature_columns]
    y_test = test_df[fe.TARGET_COLUMN].to_numpy()

    xgb_model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    return {
        "model": xgb_model,
        "encoder": encoder,
        "y_pred": y_pred,
        "y_test": y_test,
        "test_dates": test_df["date"].reset_index(drop=True),
        "feature_columns": hybrid_feature_columns,
        "train_df": train_df,
        "test_df": test_df,
    }
def set_all_seeds(seed: int = config.RANDOM_SEED):
    """Seeds Python's random, NumPy, and TensorFlow. Note: TensorFlow on
    CPU can still show minor nondeterminism from multi-threaded op
    scheduling even with all seeds set - this is a known TF limitation,
    not something these seeds fail to address. Documented rather than
    silently assumed away."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_fold_lstm_data(monthly_train: pd.DataFrame, monthly_test_window: pd.DataFrame, lookback: int):
    """Fold-specific replacement for seqf.prepare_lstm_data. Unlike that
    function, there is no further internal split here - monthly_train is
    used in full for scaler fitting and LSTM training, because the fold's
    external test set (monthly_test_window) is the only held-out data for
    this fold. Test embeddings are built the same way the single-split
    code does: the last `lookback` months of train are prepended so the
    first test month has history to look back on, without any test month
    ever appearing inside another test month's own input window."""
    train_df = seqf.add_month_encoding(monthly_train).sort_values("date").reset_index(drop=True)
    test_df = seqf.add_month_encoding(monthly_test_window).sort_values("date").reset_index(drop=True)

    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    train_scaled[seqf.SEQUENCE_FEATURE_COLUMNS] = scaler.fit_transform(train_df[seqf.SEQUENCE_FEATURE_COLUMNS])

    test_scaled = test_df.copy()
    test_scaled[seqf.SEQUENCE_FEATURE_COLUMNS] = scaler.transform(test_df[seqf.SEQUENCE_FEATURE_COLUMNS])

    X_train, y_train = seqf.build_sequences(train_scaled, lookback)

    combined = pd.concat([train_scaled.tail(lookback), test_scaled], ignore_index=True)
    X_test, _ = seqf.build_sequences(combined, lookback)

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "scaler": scaler}


def train_hybrid_fold(monthly_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       lookback: int = 6, epochs: int = 100):
    """One fold of strict hybrid walk-forward CV. A brand-new LSTM encoder
    and a brand-new XGBoost model are trained from scratch here, using
    ONLY this fold's training period. monthly_train never includes any
    date at or after the fold's test start; the fold's test rows only
    ever see frozen inference (embedding extraction, XGBoost.predict),
    never training."""
    fold_train_end = train_df["date"].max()
    fold_test_start = test_df["date"].min()
    fold_test_end = test_df["date"].max()

    monthly_train = monthly_df[monthly_df["date"] <= fold_train_end].reset_index(drop=True)
    monthly_test_window = monthly_df[
        (monthly_df["date"] >= fold_test_start) & (monthly_df["date"] <= fold_test_end)
    ].reset_index(drop=True)
    assert monthly_train["date"].max() < fold_test_start, "monthly_train leaked into the test period"

    lstm_data = prepare_fold_lstm_data(monthly_train, monthly_test_window, lookback)

    n_features = len(seqf.SEQUENCE_FEATURE_COLUMNS)
    lstm_model = baselines.build_lstm_model(lookback, n_features)
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    # NOTE: monitors training loss only - no validation split is used, so
    # no future/test data influences when training stops. Matches the
    # existing baseline LSTM's behaviour.
    lstm_model.fit(lstm_data["X_train"], lstm_data["y_train"], epochs=epochs,
                    batch_size=8, verbose=0, callbacks=[early_stop])

    encoder = build_encoder_submodel(lstm_model, lookback, n_features)
    train_embeddings = encoder.predict(lstm_data["X_train"], verbose=0)
    test_embeddings = encoder.predict(lstm_data["X_test"], verbose=0)

    emb_cols = [f"lstm_emb_{j}" for j in range(train_embeddings.shape[1])]
    # Sequence row i uses raw rows [i, i+lookback) as input and predicts
    # raw row (i+lookback) - so embedding row i belongs to date
    # monthly_train.date[i + lookback].
    train_emb_dates = monthly_train["date"].iloc[lookback:].reset_index(drop=True)
    test_emb_dates = test_df["date"].reset_index(drop=True)

    train_emb_df = pd.DataFrame(train_embeddings, columns=emb_cols)
    train_emb_df["date"] = train_emb_dates
    test_emb_df = pd.DataFrame(test_embeddings, columns=emb_cols)
    test_emb_df["date"] = test_emb_dates

    fold_train_merged = train_df.merge(train_emb_df, on="date", how="inner")
    fold_test_merged = test_df.merge(test_emb_df, on="date", how="inner")
    assert len(fold_test_merged) == len(test_df), "test embedding alignment dropped rows unexpectedly"

    hybrid_feature_columns = fe.FEATURE_COLUMNS + emb_cols
    X_train = fold_train_merged[hybrid_feature_columns]
    y_train = fold_train_merged[fe.TARGET_COLUMN]
    X_test = fold_test_merged[hybrid_feature_columns]
    y_test = fold_test_merged[fe.TARGET_COLUMN].to_numpy()

    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                              random_state=config.RANDOM_SEED, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    return y_test, y_pred, len(fold_train_merged), len(fold_test_merged)


def run_strict_hybrid_walk_forward_cv(monthly_df: pd.DataFrame, features_df: pd.DataFrame,
                                       n_splits: int = 4, lookback: int = 6, epochs: int = 100):
    """Leakage-free hybrid walk-forward CV. Replaces the old
    run_hybrid_walk_forward_cv, which reused one pre-computed embedding
    table (from a single LSTM trained once on the original 80% split)
    across all folds - meaning folds 1-3's 'test' data was literally part
    of that encoder's training data. Here, every fold gets its own LSTM,
    its own scaler, and its own XGBoost, trained only on that fold's
    training period."""
    set_all_seeds()
    print(f"\nRunning STRICT hybrid walk-forward CV ({n_splits} folds) ...")
    fold_scores, fold_sizes = [], []
    fold_predictions = {}
    for fold_i, (train_df, test_df) in enumerate(fe.walk_forward_splits(features_df, n_splits=n_splits), start=1):
        y_test, y_pred, n_train, n_test = train_hybrid_fold(monthly_df, train_df, test_df, lookback, epochs)
        scores = metrics.evaluate(y_test, y_pred)
        fold_scores.append(scores)
        fold_sizes.append((n_train, n_test))
        fold_predictions[fold_i] = {"dates": test_df["date"].reset_index(drop=True), "y_test": y_test, "y_pred": y_pred}
        print(f"  Fold {fold_i}: train={n_train}, test={n_test} -> "
              f"RMSE={scores['RMSE']:.3f} MAE={scores['MAE']:.3f} R2={scores['R2']:.3f} WI={scores['WI']:.3f}")
    per_fold_df = pd.DataFrame(fold_scores, index=[f"fold_{i+1}" for i in range(len(fold_scores))])
    per_fold_df["n_train"] = [s[0] for s in fold_sizes]
    per_fold_df["n_test"] = [s[1] for s in fold_sizes]

    summary = {}
    for m in fold_scores[0].keys():
        vals = [s[m] for s in fold_scores]
        summary[f"{m}_mean"] = np.mean(vals)
        summary[f"{m}_std"] = np.std(vals)
    summary_df = pd.DataFrame({"Hybrid_strict": summary}).T
    return per_fold_df, summary_df, fold_predictions