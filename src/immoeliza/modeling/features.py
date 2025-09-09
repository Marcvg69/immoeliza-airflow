# src/immoeliza/modeling/features.py
from __future__ import annotations

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

DEFAULT_NUMERIC = ["surface_m2", "bedrooms", "bathrooms", "year_built"]
DEFAULT_CATEG  = ["postal_code", "city", "property_type", "energy_label", "region"]

def select_feature_columns(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """
    Choose numeric+categorical features, avoid leakage depending on target.
    """
    num = [c for c in DEFAULT_NUMERIC if c in df.columns]
    cat = [c for c in DEFAULT_CATEG if c in df.columns]

    if target == "price":
        # price_per_m2 could leak target via surface
        if "price_per_m2" in num: num.remove("price_per_m2")
        if "price_per_m2" in cat: cat.remove("price_per_m2")
    elif target == "price_per_m2":
        # raw price could leak target
        if "price" in num: num.remove("price")
        if "price" in cat: cat.remove("price")

    return num, cat

def make_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    num_pipe = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler(with_mean=True)),
    ]
    cat_pipe = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[
            ("num", PipelineCompat(num_pipe), numeric),
            ("cat", PipelineCompat(cat_pipe), categorical),
        ],
        remainder="drop",
        n_jobs=None,
    )

# small compatibility shim so we don’t import sklearn.pipeline.Pipeline here
class PipelineCompat:
    def __init__(self, steps):
        self.steps = steps
        # lazily build real sklearn Pipeline on first fit
        self._pl = None

    def _ensure(self):
        if self._pl is None:
            from sklearn.pipeline import Pipeline
            self._pl = Pipeline(self.steps)

    def fit(self, X, y=None):
        self._ensure()
        return self._pl.fit(X, y)

    def transform(self, X):
        self._ensure()
        return self._pl.transform(X)

    def fit_transform(self, X, y=None):
        self._ensure()
        return self._pl.fit_transform(X, y)
