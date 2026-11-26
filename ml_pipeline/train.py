"""
ml_pipeline/train.py
-------------------
Full end-to-end training pipeline for the NASA C-MAPSS FD001 dataset.
Run with:
    python -m ml_pipeline.train
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data_loader import load_data
from .features import (
    drop_constant_sensors,
    scale_operating_settings,
    scale_sensors,
    piecewise_rul,
    compute_rolling_features,
    add_pca_health_index,
    build_sliding_windows,
)
from .metrics import rmse, nasa_asymmetric_score


def prepare_training_dataframe(df: pd.DataFrame):
    df = drop_constant_sensors(df)
    df, op_scaler = scale_operating_settings(df)
    df, sensor_scaler = scale_sensors(df)

    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = piecewise_rul(max_cycle - df["cycle"], max_rul=125)

    df = compute_rolling_features(df)
    df, pca = add_pca_health_index(df)
    return df, {"op_scaler": op_scaler, "sensor_scaler": sensor_scaler, "pca": pca}


def prepare_test_dataframe(df: pd.DataFrame, transformers: dict):
    df = drop_constant_sensors(df)
    df, _ = scale_operating_settings(df, transformers["op_scaler"])
    df, _ = scale_sensors(df, transformers["sensor_scaler"])
    df = compute_rolling_features(df)
    df, _ = add_pca_health_index(df, pca=transformers["pca"])
    return df


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


def asymmetric_xgb_objective(preds: np.ndarray, dtrain: xgb.DMatrix):
    """Smooth asymmetric squared-error objective penalizing late predictions (preds > labels)."""
    labels = dtrain.get_label()
    diff = preds - labels
    weights = np.where(diff >= 0.0, 2.0, 1.0)
    grad = weights * diff
    hess = weights
    return grad, hess


def train_xgboost(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "eval_metric": "rmse",
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "tree_method": "hist",
    }
    evals = [(dtrain, "train"), (dval, "eval")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        obj=asymmetric_xgb_objective,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return model


def train_lstm(X_train, y_train, X_val, y_val, device: str = "cpu"):
    torch.manual_seed(42)
    input_dim = X_train.shape[2]
    model = LSTMRegressor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 25
    batch_size = 64

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_rmse = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        vals, trgs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(device))
                vals.append(preds.cpu().numpy())
                trgs.append(yb.numpy())
        val_pred = np.concatenate(vals)
        val_true = np.concatenate(trgs)
        val_rmse = rmse(val_true, val_pred)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} - Val RMSE: {val_rmse:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    print("Loading NASA C-MAPSS FD001 dataset...")
    train_df, test_df, rul_series = load_data("FD001")
    print(f"Loaded train rows={len(train_df)}, test rows={len(test_df)}, test engines={len(rul_series)}")

    print("Preprocessing and engineering rolling + PCA Health Index features...")
    train_processed, transformers = prepare_training_dataframe(train_df)

    # Split by engine unit ID to avoid temporal data leakage
    unit_ids = train_processed["unit"].unique().copy()
    rng = np.random.default_rng(42)
    rng.shuffle(unit_ids)
    split = int(0.8 * len(unit_ids))
    train_units, val_units = unit_ids[:split], unit_ids[split:]

    train_mask = train_processed["unit"].isin(train_units)
    val_mask = train_processed["unit"].isin(val_units)

    X_seq_train, y_seq_train, feature_names, _, _ = build_sliding_windows(train_processed[train_mask], window_size=30)
    X_seq_val, y_seq_val, _, _, _ = build_sliding_windows(train_processed[val_mask], window_size=30, feature_cols=feature_names)

    # For tabular XGBoost: use the last cycle's engineered features (which already encode rolling 5/10/20 stats)
    X_flat_train = X_seq_train[:, -1, :]
    X_flat_val = X_seq_val[:, -1, :]

    print(f"Training XGBoost baseline (features={len(feature_names)}, train_samples={len(X_flat_train)})...")
    xgb_model = train_xgboost(X_flat_train, y_seq_train, X_flat_val, y_seq_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training PyTorch LSTM on {device} (seq_shape={X_seq_train.shape})...")
    lstm_model = train_lstm(X_seq_train, y_seq_train, X_seq_val, y_seq_val, device=device)

    # Prepare test set with true RUL at every cycle:
    # RUL_FD001 gives remaining cycles AFTER each test unit's last recorded cycle.
    test_units = sorted(test_df["unit"].unique())
    rul_map = dict(zip(test_units, rul_series.to_numpy()))
    test_max_cycle = test_df.groupby("unit")["cycle"].transform("max")
    test_end_rul = test_df["unit"].map(rul_map)
    test_df["rul"] = piecewise_rul(test_end_rul + (test_max_cycle - test_df["cycle"]), max_rul=125)

    test_processed = prepare_test_dataframe(test_df, transformers)
    X_test_seq, y_test_seq, _, test_win_units, test_win_cycles = build_sliding_windows(
        test_processed, window_size=30, feature_cols=feature_names
    )
    X_test_flat = X_test_seq[:, -1, :]

    # Evaluate across all test windows AND on the final cycle per engine (standard C-MAPSS benchmark)
    xgb_test_pred = np.clip(xgb_model.predict(xgb.DMatrix(X_test_flat)), 0.0, 125.0)
    lstm_model.eval()
    with torch.no_grad():
        lstm_test_pred = np.clip(
            lstm_model(torch.from_numpy(X_test_seq).float().to(device)).cpu().numpy(),
            0.0,
            125.0,
        )

    # Extract last window index per test engine
    last_indices = []
    for unit in test_units:
        idxs = np.where(test_win_units == unit)[0]
        if len(idxs) > 0:
            last_indices.append(idxs[-1])
    last_indices = np.asarray(last_indices, dtype=np.int32)

    y_test_last = y_test_seq[last_indices]
    xgb_last_pred = xgb_test_pred[last_indices]
    lstm_last_pred = lstm_test_pred[last_indices]

    xgb_last_rmse = rmse(y_test_last, xgb_last_pred)
    xgb_last_score = nasa_asymmetric_score(y_test_last, xgb_last_pred)
    lstm_last_rmse = rmse(y_test_last, lstm_last_pred)
    lstm_last_score = nasa_asymmetric_score(y_test_last, lstm_last_pred)

    xgb_all_rmse = rmse(y_test_seq, xgb_test_pred)
    xgb_all_score = nasa_asymmetric_score(y_test_seq, xgb_test_pred)
    lstm_all_rmse = rmse(y_test_seq, lstm_test_pred)
    lstm_all_score = nasa_asymmetric_score(y_test_seq, lstm_test_pred)

    best_type = "xgboost" if xgb_last_score <= lstm_last_score else "lstm"
    best_rmse = xgb_last_rmse if best_type == "xgboost" else lstm_last_rmse
    best_score = xgb_last_score if best_type == "xgboost" else lstm_last_score

    # Persist artifacts
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(transformers, artifacts_dir / "preprocess.pkl")
    xgb_model.save_model(str(artifacts_dir / "xgb_best.json"))
    torch.save({"state_dict": lstm_model.state_dict(), "input_dim": X_test_seq.shape[2]}, artifacts_dir / "lstm_best.pt")

    np.save(artifacts_dir / "X_test_seq.npy", X_test_seq)
    np.save(artifacts_dir / "y_test_seq.npy", y_test_seq)
    np.save(artifacts_dir / "test_win_units.npy", test_win_units)
    np.save(artifacts_dir / "test_win_cycles.npy", test_win_cycles)

    # Save sample test engine JSON payloads for quick API/demo consumption
    sample_engines = {}
    for unit in test_units[:10]:
        unit_mask = test_win_units == unit
        sample_engines[int(unit)] = {
            "engine_id": int(unit),
            "cycles": test_win_cycles[unit_mask].tolist(),
            "true_rul": y_test_seq[unit_mask].tolist(),
            "xgb_pred_rul": xgb_test_pred[unit_mask].tolist(),
            "lstm_pred_rul": lstm_test_pred[unit_mask].tolist(),
        }
    with open(artifacts_dir / "sample_test_engines.json", "w", encoding="utf-8") as f:
        json.dump(sample_engines, f, indent=2)

    summary = {
        "dataset": "FD001",
        "window_size": 30,
        "max_rul": 125,
        "features": feature_names,
        "best_model": best_type,
        "test_last_cycle_metrics": {
            "xgboost": {"rmse": float(xgb_last_rmse), "nasa_asymmetric_score": float(xgb_last_score)},
            "lstm": {"rmse": float(lstm_last_rmse), "nasa_asymmetric_score": float(lstm_last_score)},
        },
        "test_all_windows_metrics": {
            "xgboost": {"rmse": float(xgb_all_rmse), "nasa_asymmetric_score": float(xgb_all_score)},
            "lstm": {"rmse": float(lstm_all_rmse), "nasa_asymmetric_score": float(lstm_all_score)},
        },
        "test_rmse": float(best_rmse),
        "test_asymmetric_score": float(best_score),
    }
    with open(artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Model Comparison on Test Set (Final Cycle per Engine, N={}) ===".format(len(last_indices)))
    print(f"{'Model':<12} {'RMSE':>10} {'NASA Asym. Score':>20}")
    print(f"{'XGBoost':<12} {xgb_last_rmse:10.3f} {xgb_last_score:20.3f}")
    print(f"{'LSTM':<12} {lstm_last_rmse:10.3f} {lstm_last_score:20.3f}")
    print(f"\nBest model: {best_type.upper()} | Artifacts saved to {artifacts_dir}")


if __name__ == "__main__":
    main()
