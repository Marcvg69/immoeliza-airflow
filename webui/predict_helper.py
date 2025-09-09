# webui/predict_helper.py
from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


MODELS_DIR = Path("models")


@dataclass
class PickedModel:
    path: Path
    target: str         # "price" or "price_per_m2"
    meta_path: Optional[Path] = None
    meta: Optional[Dict[str, Any]] = None


def _latest(path_pattern: str) -> Optional[Path]:
    files = sorted(glob.glob(path_pattern))
    return Path(files[-1]) if files else None


def pick_model_path(has_surface_m2: bool) -> PickedModel:
    """
    If surface_m2 is provided, prefer an absolute-price model.
    If not, prefer a price_per_m2 model.
    Fall back to whatever exists most recently.
    """
    price_model = _latest(str(MODELS_DIR / "*price_*.joblib"))
    ppm2_model = _latest(str(MODELS_DIR / "*price_per_m2_*.joblib"))

    prefer = "price" if has_surface_m2 else "price_per_m2"

    chosen_path: Optional[Path] = None
    chosen_target = prefer

    if prefer == "price":
        chosen_path = price_model or ppm2_model
        if chosen_path and "price_per_m2" in chosen_path.name:
            chosen_target = "price_per_m2"
    else:
        chosen_path = ppm2_model or price_model
        if chosen_path and "price_" in chosen_path.name and "price_per_m2" not in chosen_path.name:
            chosen_target = "price"

    if not chosen_path:
        raise FileNotFoundError(
            "No model files found in ./models. Train a model first."
        )

    # try to find a sibling metadata json (timestamp in filename is shared)
    stamp = re.findall(r"(\d{8}_\d{6})", chosen_path.name)
    meta_path = None
    meta = None
    if stamp:
        maybe = MODELS_DIR / f"metadata_{stamp[0]}.json"
        if maybe.exists():
            meta_path = maybe
            try:
                meta = json.loads(maybe.read_text())
            except Exception:
                meta = None

    return PickedModel(path=chosen_path, target=chosen_target, meta_path=meta_path, meta=meta)


_EXPECTED_FEATURES = [
    # numeric the pipeline knows how to impute/scale
    "surface_m2", "year_built", "bedrooms", "bathrooms",
    # categoricals the encoder handles
    "postal_code", "city", "property_type", "energy_label", "region",
]

def _to_row_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Coerce an arbitrary user dict into the columns the training pipeline expects.
    Extra keys are ignored; missing keys are filled with NA.
    """
    row = {k: payload.get(k, pd.NA) for k in _EXPECTED_FEATURES}
    # light coercions
    for k in ("surface_m2", "year_built", "bedrooms", "bathrooms"):
        v = row.get(k, pd.NA)
        if v is not pd.NA and v is not None:
            try:
                row[k] = float(v) if k == "surface_m2" else int(float(v))
            except Exception:
                row[k] = pd.NA
    # strings
    for k in ("postal_code", "city", "property_type", "energy_label", "region"):
        v = row.get(k, pd.NA)
        if v is not pd.NA and v is not None:
            row[k] = str(v)
    return pd.DataFrame([row])


def auto_predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Choose a model based on presence of surface_m2 and return a prediction dict.
    If the selected model predicts price_per_m2 and surface is present, we also
    return a derived 'price' = price_per_m2 * surface_m2 for convenience.
    """
    has_surface = bool(features.get("surface_m2"))
    picked = pick_model_path(has_surface_m2=has_surface)

    model = joblib.load(picked.path)

    X = _to_row_df(features)
    y_hat = model.predict(X)
    y_hat = float(np.expm1(y_hat)) if getattr(model, "target_is_log", False) else float(y_hat)

    out: Dict[str, Any] = {
        "used_model": picked.path.name,
        "target": picked.target,
        "prediction": y_hat,
        "meta": picked.meta or {},
    }

    # derive absolute price if ppm2 and we got a surface
    if picked.target == "price_per_m2" and has_surface:
        try:
            price = y_hat * float(features["surface_m2"])
            out["derived_price"] = float(price)
        except Exception:
            pass

    return out
