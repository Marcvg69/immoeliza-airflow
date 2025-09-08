# src/immoeliza/modeling/train_regression.py
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import joblib

from immoeliza.scraping.common import logger

NUM = ["surface_m2","bedrooms","bathrooms","year_built","price_per_m2"]
CAT = ["postal_code","city","property_type","energy_label","region"]

def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _preprocess():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])

def _rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train(training_parquet: str) -> dict:
    df = pd.read_parquet(training_parquet)
    if len(df) == 0:
        raise ValueError(f"Training file has 0 rows: {training_parquet}")

    y = df["price"].values.astype(float)
    X = df.drop(columns=["price"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    prep = _preprocess()

    # Baseline: LinearRegression on log(price), evaluate on original scale
    lin = Pipeline([("prep", prep), ("reg", LinearRegression())])
    lin.fit(X_train, np.log1p(y_train))
    y_pred_lin = np.expm1(lin.predict(X_test))

    # RandomForest
    rf = Pipeline([("prep", prep), ("reg", RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2
    ))])
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    metrics = {
        "linear": {
            "rmse": _rmse(y_test, y_pred_lin),
            "mae":  float(mean_absolute_error(y_test, y_pred_lin)),
            "mape": float(mean_absolute_percentage_error(y_test, y_pred_lin)),
            "r2":   float(r2_score(y_test, y_pred_lin)),
        },
        "rf": {
            "rmse": _rmse(y_test, y_pred_rf),
            "mae":  float(mean_absolute_error(y_test, y_pred_rf)),
            "mape": float(mean_absolute_percentage_error(y_test, y_pred_rf)),
            "r2":   float(r2_score(y_test, y_pred_rf)),
        },
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
    }

    # pick best by RMSE
    best_name = "linear" if metrics["linear"]["rmse"] <= metrics["rf"]["rmse"] else "rf"
    best = lin if best_name == "linear" else rf

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True)
    model_path = models_dir / (f"{'lin' if best_name=='linear' else 'rf'}_price_{ts}.joblib")
    joblib.dump(best, model_path)

    meta = {
        "timestamp": ts,
        "training_file": training_parquet,
        "training_md5": _md5(training_parquet),
        "features_num": NUM,
        "features_cat": CAT,
        "metrics": metrics,
        "best_model": best_name,
        "model_path": str(model_path),
    }
    meta_path = models_dir / f"metadata_{ts}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("🤖 Model saved -> %s", model_path)
    logger.info("📊 Metrics: %s", metrics)
    logger.info("🧾 Metadata -> %s", meta_path)
    return {"model_path": str(model_path), "metadata_path": str(meta_path), "metrics": metrics}
