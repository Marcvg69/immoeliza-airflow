# src/immoeliza/modeling/train.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from .features import select_feature_columns, make_preprocessor
from .evaluate import kfold_regression_metrics
from .registry import save_model, save_metadata

logger = logging.getLogger("immoeliza.scraping")

def _build_pipelines(numeric, categorical) -> Dict[str, Pipeline]:
    pre = make_preprocessor(numeric, categorical)
    linear = Pipeline([("prep", pre), ("reg", LinearRegression())])
    rf = Pipeline([("prep", pre), ("reg", RandomForestRegressor(
        n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
    ))])
    return {"linear": linear, "rf": rf}

def train(training_parquet: str, *, target: str = "price") -> Dict:
    """
    Clean entrypoint (keeps your existing train_regression.py untouched).
    - log-RMSE CV (5-fold)
    - chooses best by RMSE
    - saves model + metadata
    """
    df = pd.read_parquet(training_parquet)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in training file: {training_parquet}")

    # drop rows without target
    df = df.dropna(subset=[target]).copy()
    y = df[target].astype(float)

    num, cat = select_feature_columns(df, target)
    X = df[num + cat]

    # KFold + log target (price scales well in log)
    # For price_per_m2 we also keep log=True; disable if you prefer linear scale.
    pipelines = _build_pipelines(num, cat)

    metrics_all = {}
    for name, pipe in pipelines.items():
        m = kfold_regression_metrics(pipe, X, y, k=5, log_target=True)
        metrics_all[name] = m

    # pick the best by RMSE
    best_name = min(metrics_all, key=lambda k: metrics_all[k]["rmse"])
    best = pipelines[best_name].fit(X, y)  # final fit on full data (log handled inside evaluate CV)
    model_path = save_model(best, best_name, target)
    meta_path = save_metadata(model_path, metrics_all, {"chosen": best_name})

    logger.info("Model saved -> %s", model_path)
    logger.info("Metrics: %s", metrics_all)

    return {
        "model_path": str(model_path),
        "metadata_path": str(meta_path),
        "metrics": metrics_all | {"chosen": best_name},
    }
