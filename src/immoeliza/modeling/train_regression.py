"""
Train baseline models on the processed training parquet.

- Reads a single training parquet path (argument: training_parquet)
- Builds a preprocessing pipeline (num median-impute + scale, cat one-hot)
- Auto-drops numeric features that are entirely NaN to avoid imputer warnings
- Trains LinearRegression (log-price) and RandomForestRegressor; saves best model
- Returns paths + metrics dict

Public API:
    train(training_parquet: str) -> dict
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ---------- logger ---------- #
logger = logging.getLogger("immoeliza.scraping")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ---------- helpers ---------- #

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")

def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")

def _mape(y_true, y_pred) -> float:
    # only compute where y_true > 0 to avoid div-by-zero
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- main ---------- #

def train(training_parquet: str) -> Dict:
    p = Path(training_parquet)
    if not p.exists():
        raise FileNotFoundError(f"Training parquet not found: {p}")

    df = pd.read_parquet(p)
    if df.empty:
        raise ValueError(f"Training file has 0 rows: {training_parquet}")

    # Expected columns
    target_col = "price"

    base_num_cols = ["surface_m2", "price_per_m2", "bedrooms", "bathrooms", "year_built"]
    base_cat_cols = ["postal_code", "city", "region", "property_type", "energy_label"]

    # Keep only columns that exist
    present_num_cols = [c for c in base_num_cols if c in df.columns]
    present_cat_cols = [c for c in base_cat_cols if c in df.columns]

    # Drop numeric columns that are entirely NaN (avoid imputer warnings & silent drops)
    nonempty_num_cols = [c for c in present_num_cols if pd.to_numeric(df[c], errors="coerce").notna().any()]
    dropped_num_cols = sorted(list(set(present_num_cols) - set(nonempty_num_cols)))

    if dropped_num_cols:
        logger.warning("Dropping numeric features with all-missing values: %s", dropped_num_cols)

    # Final feature lists
    num_cols: List[str] = nonempty_num_cols
    cat_cols: List[str] = present_cat_cols

    # Minimal safety: if *all* numeric ended up dropped, keep none; model can still run on cats.
    if len(num_cols) == 0 and len(cat_cols) == 0:
        raise ValueError("No usable features available after filtering.")

    # Split
    X = df[num_cols + cat_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    keep = y.notna()
    X, y = X.loc[keep], y.loc[keep]

    if len(X) < 6:
        logger.warning("Very small dataset (%d rows) — metrics will be unstable.", len(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=max(1, int(len(X) * 0.2)), random_state=42)

    # Pipelines
    num_pipe = Pipeline([
        ("cast", SimpleImputer(strategy="median")),   # impute
        ("scale", StandardScaler(with_mean=False)),   # sparse-friendly
    ])

    # Use new OneHotEncoder kw (sparse_output) for recent sklearn; falls back if older
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", ohe),
    ])

    transformers = []
    if len(num_cols):
        transformers.append(("num", num_pipe, num_cols))
    if len(cat_cols):
        transformers.append(("cat", cat_pipe, cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    # Models
    lin = Pipeline([("pre", pre), ("lin", LinearRegression())])
    rf = Pipeline([("pre", pre),
                   ("rf", RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42,
                   ))])

    # Fit on log-price; evaluate in original scale
    lin.fit(X_train, np.log1p(y_train))
    rf.fit(X_train, np.log1p(y_train))

    def predict_to_euros(model, X):
        return np.expm1(model.predict(X))

    preds_lin = predict_to_euros(lin, X_test)
    preds_rf  = predict_to_euros(rf,  X_test)

    m_lin = {
        "rmse": _rmse(y_test.values, preds_lin),
        "mae":  _mae(y_test.values, preds_lin),
        "mape": _mape(y_test.values, preds_lin),
        "r2":   float(np.corrcoef(y_test.values, preds_lin)[0,1]**2) if len(y_test) >= 2 else float("nan"),
    }
    m_rf = {
        "rmse": _rmse(y_test.values, preds_rf),
        "mae":  _mae(y_test.values, preds_rf),
        "mape": _mape(y_test.values, preds_rf),
        "r2":   float(np.corrcoef(y_test.values, preds_rf)[0,1]**2) if len(y_test) >= 2 else float("nan"),
    }

    # Pick best by RMSE (lower is better)
    best_name, best_model, best_metrics = ("linear", lin, m_lin) if m_lin["rmse"] <= m_rf["rmse"] else ("rf", rf, m_rf)

    tag = _now_tag()
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{best_name}_price_{tag}.joblib"
    dump(best_model, model_path)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "training_parquet": str(p),
        "n_rows_total": int(len(df)),
        "n_rows_used": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": {
            "numeric_used": num_cols,
            "categorical_used": cat_cols,
            "numeric_dropped_all_missing": dropped_num_cols,
        },
        "metrics": {"linear": m_lin, "rf": m_rf, "chosen": best_name},
    }
    meta_path = models_dir / f"metadata_{tag}.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    logger.info("🤖 Model saved -> %s", model_path)
    logger.info("📊 Metrics: %s", metadata["metrics"])
    logger.info("🧾 Metadata -> %s", meta_path)

    return {
        "model_path": str(model_path),
        "metadata_path": str(meta_path),
        "metrics": metadata["metrics"],
    }


__all__ = ["train"]
