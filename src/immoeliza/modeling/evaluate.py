# src/immoeliza/modeling/evaluate.py
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

def kfold_regression_metrics(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    k: int = 5,
    log_target: bool = True,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    KFold CV with optional log1p transform for y during fit; RMSE/MAE are on original scale.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    rmses, maes, mapes, r2s = [], [], [], []
    n_total = 0

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        est = clone(estimator)

        if log_target:
            # wrap with TransformedTargetRegressor on the fly
            from sklearn.compose import TransformedTargetRegressor
            est = TransformedTargetRegressor(
                regressor=est,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )

        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_te)

        # all metrics are calculated on the original scale
        rmse = mean_squared_error(y_te, y_pred, squared=False)
        mae  = mean_absolute_error(y_te, y_pred)
        mape = float(np.mean(np.abs((y_te - y_pred) / np.clip(np.abs(y_te), 1e-9, None))))
        r2   = r2_score(y_te, y_pred)

        rmses.append(rmse); maes.append(mae); mapes.append(mape); r2s.append(r2)
        n_total += len(X_te)

    return {
        "rmse": float(np.mean(rmses)),
        "mae":  float(np.mean(maes)),
        "mape": float(np.mean(mapes)),
        "r2":   float(np.mean(r2s)),
        "n":    int(n_total),
    }
