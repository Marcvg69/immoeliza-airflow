from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import settings
from ..scraping.common import logger

@dataclass
class TrainIO:
    models_dir: Path = Path(settings.models_dir)
    def model_path(self, stamp: str) -> Path:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self.models_dir / f"rf_price_{stamp}.joblib"
    def metrics_path(self, stamp: str) -> Path:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self.models_dir / f"rf_price_{stamp}_metrics.json"

def train(training_parquet: str | Path) -> dict:
    df = pd.read_parquet(training_parquet)
    y = df["price"].values
    X = df.drop(columns=["price"])

    num_cols = ["surface_m2","bedrooms","bathrooms"]
    cat_cols = ["postal_code","city","property_type","year_built","energy_label"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=True), cat_cols),
        ]
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "mape": float(mean_absolute_percentage_error(y_test, preds)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    io = TrainIO()
    mpath = io.model_path(stamp)
    joblib.dump(pipe, mpath)
    with open(io.metrics_path(stamp), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("🤖 Model saved -> %s", mpath)
    logger.info("📈 Metrics: %s", metrics)
    return {"model_path": str(mpath), "metrics": metrics}
