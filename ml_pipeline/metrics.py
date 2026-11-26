import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def nasa_asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA asymmetric scoring function.
    S = sum(exp(-d/13) - 1) for d < 0 (early prediction)
        + sum(exp(d/10) - 1)   for d >= 0 (late prediction)
    where d = y_pred - y_true.
    """
    d = y_pred - y_true
    early = d < 0
    late = ~early
    score = np.sum(np.exp(-d[early] / 13) - 1) + np.sum(np.exp(d[late] / 10) - 1)
    return float(score)
