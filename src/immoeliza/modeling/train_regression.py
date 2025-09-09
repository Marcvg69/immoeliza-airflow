"""
Model training (dual-target) with KFold CV and log-RMSE evaluation.

- Targets: "price" (default) or "price_per_m2"
- KFold (k=5 by default; small-sample fallback to holdout)
- Log-target via TransformedTargetRegressor (log1p / expm1); RMSE on original scale
- Preprocessing:
    * Numeric: SimpleImputer(median) + StandardScaler
    * Categorical: OneHotEncoder(handle_unknown="ignore")
- Leakage guards:
    * If target == "price"       -> DROP 'price_per_m2' as feature
    * If target == "price_per_m2'-> DROP 'price'        as feature
- Drops features that are entirely missing
- Saves:
    * models/<chosen>_<target>_<timestamp>.joblib
    * models/metadata_<timestamp>.json   (cv metrics, params, feature lists, data hash, etc.)

Environment knobs (optional):
- IMMO_TRAIN_TARGET: "price" | "price_per_m2" (default "price" unless provided as function arg)
- IMMO_KFOLD: integer (default 5)
- IMMO_TRAIN_LOG_TARGET: "true" | "false" | "auto"  (default "auto" -> logs both targets)
"""

from __future__ import annotations

import os
import io
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import logging

logger = logging.getLogger("immoeliza.scraping")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# ---------------------------
# Utilities
# ---------------------------

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _data_sha256(parquet_path: str) -> str:
    with open(parquet_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-9
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))

def _ohe_compat_kwargs() -> Dict:
    """
    scikit-learn >=1.2: OneHotEncoder(sparse_output=...)
    older versions:     OneHotEncoder(sparse=...)
    We prefer dense output here.
    """
    try:
        # Try new API
        _ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return {"sparse_output": False, "handle_unknown": "ignore"}
    except TypeError:
        return {"sparse": False, "handle_unknown": "ignore"}


def _build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = OneHotEncoder(**_ohe_compat_kwargs())
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # ensure dense
    )


def _build_estimator(model_name: str,
                     num_cols: List[str],
                     cat_cols: List[str],
                     log_target: bool) -> Pipeline | TransformedTargetRegressor:
    pre = _build_preprocessor(num_cols, cat_cols)

    if model_name == "linear":
        base = Pipeline([("pre", pre), ("model", LinearRegression())])
    elif model_name == "rf":
        base = Pipeline([("pre", pre),
                         ("model", RandomForestRegressor(
                             n_estimators=400,
                             random_state=42,
                             n_jobs=-1
                         ))])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if log_target:
        # Wrap so .predict returns values back on the original scale (expm1)
        return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)
    return base


def _cv_scores(X: pd.DataFrame,
               y: pd.Series,
               num_cols: List[str],
               cat_cols: List[str],
               log_target: bool,
               model_name: str,
               kfold: int) -> Dict[str, float]:
    """
    Manual CV so we can handle log-target and compute metrics on original scale.
    """
    n = len(X)
    metrics = {"rmse": [], "mae": [], "mape": [], "r2": []}

    if n < 6 or kfold < 2:
        # Small-sample fallback: holdout 80/20
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        est = _build_estimator(model_name, num_cols, cat_cols, log_target)
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_te)
        metrics["rmse"].append(np.sqrt(mean_squared_error(y_te, y_pred)))
        metrics["mae"].append(mean_absolute_error(y_te, y_pred))
        metrics["mape"].append(_mape(y_te, y_pred))
        metrics["r2"].append(r2_score(y_te, y_pred))
    else:
        k = min(kfold, n)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            est = _build_estimator(model_name, num_cols, cat_cols, log_target)
            est.fit(X_tr, y_tr)
            y_pred = est.predict(X_te)
            metrics["rmse"].append(np.sqrt(mean_squared_error(y_te, y_pred)))
            metrics["mae"].append(mean_absolute_error(y_te, y_pred))
            metrics["mape"].append(_mape(y_te, y_pred))
            metrics["r2"].append(r2_score(y_te, y_pred))

    # aggregate
    out = {m: float(np.mean(v)) for m, v in metrics.items()}
    out.update({f"{m}_std": float(np.std(v)) for m, v in metrics.items()})
    out["n"] = int(n)
    out["folds"] = int(len(metrics["rmse"]))
    return out


def _choose_features(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """
    Choose existing features and drop fully-missing ones.
    Leakage guards applied depending on target.
    """
    # Candidate sets
    numeric_cand = ["surface_m2", "bedrooms", "bathrooms", "year_built", "price_per_m2", "price"]
    cat_cand = ["postal_code", "city", "region", "property_type", "energy_label"]

    # Leakage: drop counterpart target
    if target == "price":
        if "price_per_m2" in numeric_cand:
            numeric_cand.remove("price_per_m2")
    elif target == "price_per_m2":
        if "price" in numeric_cand:
            numeric_cand.remove("price")

    # Keep only columns that exist
    numeric = [c for c in numeric_cand if c in df.columns]
    categor = [c for c in cat_cand if c in df.columns]

    # Drop fully-missing numeric
    all_missing = [c for c in numeric if df[c].isna().all()]
    if all_missing:
        logger.warning("Dropping numeric features with all-missing values: %s", all_missing)
        numeric = [c for c in numeric if c not in all_missing]

    # Drop fully-missing categoricals
    all_miss_cat = [c for c in categor if df[c].isna().all()]
    if all_miss_cat:
        logger.warning("Dropping categorical features with all-missing values: %s", all_miss_cat)
        categor = [c for c in categor if c not in all_miss_cat]

    return numeric, categor


def _log_choice(target: str) -> bool:
    """Decide whether to log-transform the target."""
    env = os.getenv("IMMO_TRAIN_LOG_TARGET", "auto").strip().lower()
    if env in {"true", "1", "yes"}:
        return True
    if env in {"false", "0", "no"}:
        return False
    # auto: log both price and price_per_m2 (reasonable magnitudes)
    return True


# ---------------------------
# Public API
# ---------------------------

def train(training_parquet: str, target: Optional[str] = None) -> Dict:
    """
    Train and select a model for the given target using KFold CV.
    Returns dict with paths and metrics.
    """
    if not target:
        target = os.getenv("IMMO_TRAIN_TARGET", "price").strip().lower()
    if target not in {"price", "price_per_m2"}:
        raise ValueError("target must be 'price' or 'price_per_m2'")

    df = pd.read_parquet(training_parquet)
    if target not in df.columns:
        raise ValueError(f"Training file does not contain target '{target}'")

    # Basic filtering: need at least 2 non-null targets
    df = df.copy()
    df = df[df[target].notna()]
    if len(df) < 2:
        raise ValueError(f"Not enough rows with target '{target}' to train (have {len(df)}).")

    # Prevent obvious leakage (log hints)
    if target == "price":
        if "price_per_m2" in df.columns and df["price_per_m2"].notna().any():
            logger.warning("Leakage hint: price_per_m2 present alongside price (potential leakage).")
    else:
        if "price" in df.columns and df["price"].notna().any():
            logger.warning("Leakage hint: price present while predicting price_per_m2 (potential leakage).")

    # Feature selection
    num_cols, cat_cols = _choose_features(df, target)
    feature_cols = num_cols + cat_cols
    if not feature_cols:
        raise ValueError("No usable features available after filtering.")

    X = df[feature_cols]
    y = df[target].astype(float)

    # Make sure categorical columns are string-typed to avoid NA ambiguity
    for c in cat_cols:
        X[c] = X[c].astype("string")

    log_target = _log_choice(target)
    kfold = int(os.getenv("IMMO_KFOLD", "5"))

    # Evaluate models by CV
    cv_lin = _cv_scores(X, y, num_cols, cat_cols, log_target, model_name="linear", kfold=kfold)
    cv_rf  = _cv_scores(X, y, num_cols, cat_cols, log_target, model_name="rf",     kfold=kfold)

    # Pick by RMSE (lower is better)
    chosen = "linear" if cv_lin["rmse"] <= cv_rf["rmse"] else "rf"
    cv_all = {"linear": cv_lin, "rf": cv_rf, "chosen": chosen}

    # Fit chosen on full data
    final_est = _build_estimator(chosen, num_cols, cat_cols, log_target)
    final_est.fit(X, y)

    # Save artifacts
    Path_models = "models"
    os.makedirs(Path_models, exist_ok=True)
    tag = _now_tag()
    model_fname = f"{chosen}_{target}_{tag}.joblib"
    model_path = os.path.join(Path_models, model_fname)
    dump(final_est, model_path)

    # Build metadata
    meta = {
        "timestamp": tag,
        "training_file": training_parquet,
        "training_file_sha256": _data_sha256(training_parquet),
        "n_rows": int(len(df)),
        "target": target,
        "log_target": bool(log_target),
        "features": {
            "numeric": num_cols,
            "categorical": cat_cols,
        },
        "cv": cv_all,
        "models": {
            "linear": {"class": "LinearRegression"},
            "rf": {"class": "RandomForestRegressor", "params": {"n_estimators": 400, "random_state": 42, "n_jobs": -1}},
        },
        "chosen": chosen,
        "model_path": model_path,
    }

    meta_path = os.path.join(Path_models, f"metadata_{tag}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Log concise summary
    logger.info("🤖 Model saved -> %s", model_path)
    logger.info("📊 CV Metrics (mean over %s fold%s): %s",
                cv_all[chosen]["folds"],
                "" if cv_all[chosen]["folds"] == 1 else "s",
                {k: v for k, v in cv_all[chosen].items() if k in ("rmse", "mae", "mape", "r2")})
    logger.info("🧾 Metadata -> %s", meta_path)

    return {"model_path": model_path, "metadata_path": meta_path, "metrics": cv_all}
