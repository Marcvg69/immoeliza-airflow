# src/immoeliza/modeling/train_regression.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _con(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))

@dataclass
class TrainReport:
    rows: int
    features: list
    r2: float
    mae: float
    model_path: str
    trained_at: str

def _load_latest(db_path: Path) -> pd.DataFrame:
    with _con(db_path) as con:
        return con.execute("SELECT * FROM listings_latest;").df()

def _prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Keep essentials
    keep_cols = [
        "city","postal_code","property_type","is_rent",
        "bedrooms","bathrooms","surface_m2","price"
    ]
    df = df[[c for c in keep_cols if c in df]].copy()

    # Sanity filtering (mirror from cleaner)
    df = df[
        df["price"].notna() &
        df["surface_m2"].notna() &
        (df["surface_m2"] >= 10) & (df["surface_m2"] <= 1000)
    ].copy()

    # Types: make categoricals strings; numerics float
    cat_cols = [c for c in ["city","postal_code","property_type","is_rent"] if c in df]
    num_cols = [c for c in ["bedrooms","bathrooms","surface_m2"] if c in df]

    for c in cat_cols:
        if c == "is_rent":
            # Map to 'Yes'/'No' strings to keep uniform for encoder
            df[c] = df[c].map(lambda v: "Yes" if bool(v) else "No")
        else:
            df[c] = df[c].astype("string")

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    y = df["price"].astype("float64")
    X = df.drop(columns=["price"]).copy()

    # Build the sklearn pipeline
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return X, y, pipe

def rebuild_training_and_retrain(models_dir: Path | str, db_path: Path | str) -> dict:
    models_dir = Path(models_dir)
    db_path = Path(db_path)

    df = _load_latest(db_path)
    if df.empty:
        raise RuntimeError("listings_latest is empty. Rebuild analytics first.")

    X, y, pipe = _prepare(df)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)

    preds = pipe.predict(Xte)
    r2 = float(r2_score(yte, preds))
    mae = float(mean_absolute_error(yte, preds))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_path = models_dir / f"rf_immoeliza_{ts}.joblib"
    joblib.dump(pipe, model_path)

    meta = TrainReport(
        rows=len(df),
        features=list(X.columns),
        r2=r2,
        mae=mae,
        model_path=str(model_path),
        trained_at=ts,
    )
    (models_dir / "last_train.json").write_text(json.dumps(asdict(meta), indent=2))

    return asdict(meta)

if __name__ == "__main__":
    # default locations
    print(rebuild_training_and_retrain(MODELS_DIR, REPO_ROOT / "analytics" / "immoeliza.duckdb"))
