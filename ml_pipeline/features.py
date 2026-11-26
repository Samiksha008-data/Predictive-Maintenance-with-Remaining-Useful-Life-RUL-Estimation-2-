import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Sensors to drop (zero-variance in FD001)
DROP_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}


def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c in DROP_SENSORS]).copy()


def scale_operating_settings(df: pd.DataFrame, scaler: StandardScaler = None):
    op_cols = [c for c in df.columns if c.startswith("op")]
    if scaler is None:
        scaler = StandardScaler()
        df[op_cols] = scaler.fit_transform(df[op_cols])
    else:
        df[op_cols] = scaler.transform(df[op_cols])
    return df, scaler


def scale_sensors(df: pd.DataFrame, scaler: MinMaxScaler = None):
    sensor_cols = [c for c in df.columns if c.startswith("s") and "_" not in c]
    if scaler is None:
        scaler = MinMaxScaler()
        df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    else:
        df[sensor_cols] = scaler.transform(df[sensor_cols])
    return df, scaler


def piecewise_rul(series: pd.Series, max_rul: int = 125) -> pd.Series:
    """Clip RUL to max_rul so steady-state healthy cycles remain flat."""
    return series.clip(upper=max_rul)


def _fast_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64) - (n - 1) * 0.5
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def compute_rolling_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    if windows is None:
        windows = [5, 10, 20]
    sensor_cols = [c for c in df.columns if c.startswith("s") and "_" not in c]
    new_cols = {}
    grouped = df.groupby("unit", sort=False)

    for w in windows:
        roll = grouped[sensor_cols].rolling(window=w, min_periods=1)
        mean_df = roll.mean().reset_index(level=0, drop=True)
        std_df = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        ema_df = grouped[sensor_cols].transform(lambda s: s.ewm(span=w, adjust=False).mean())
        slope_df = roll.apply(_fast_slope, raw=True).reset_index(level=0, drop=True)

        for col in sensor_cols:
            new_cols[f"{col}_mean_{w}"] = mean_df[col].to_numpy()
            new_cols[f"{col}_std_{w}"] = std_df[col].to_numpy()
            new_cols[f"{col}_ema_{w}"] = ema_df[col].to_numpy()
            new_cols[f"{col}_slope_{w}"] = slope_df[col].to_numpy()

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_pca_health_index(df: pd.DataFrame, n_components: int = 2, pca: PCA = None):
    sensor_cols = [c for c in df.columns if c.startswith("s") and "_" not in c]
    if pca is None:
        pca = PCA(n_components=n_components)
        pca_fit = pca.fit_transform(df[sensor_cols])
    else:
        pca_fit = pca.transform(df[sensor_cols])
    recon = pca.inverse_transform(pca_fit)
    error = np.mean((df[sensor_cols].to_numpy() - recon) ** 2, axis=1)
    df = df.copy()
    df["health_index"] = error
    return df, pca


def build_sliding_windows(df: pd.DataFrame, window_size: int = 30, feature_cols: list = None):
    """Build sliding windows (num_samples, window_size, num_features) and targets."""
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c.startswith("s") or c.startswith("op") or c == "health_index"
        ]
    X, y, units, cycles = [], [], [], []
    for unit, group in df.groupby("unit"):
        group = group.sort_values("cycle")
        values = group[feature_cols].to_numpy(dtype=np.float32)
        rul = group["rul"].to_numpy(dtype=np.float32)
        cyc = group["cycle"].to_numpy(dtype=np.int32)
        for i in range(len(group) - window_size + 1):
            X.append(values[i : i + window_size])
            y.append(rul[i + window_size - 1])
            units.append(int(unit))
            cycles.append(int(cyc[i + window_size - 1]))
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        feature_cols,
        np.asarray(units, dtype=np.int32),
        np.asarray(cycles, dtype=np.int32),
    )
