# src/immoeliza/modeling/registry.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import joblib

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def model_filename(kind: str, target: str) -> str:
    # kind: "linear" | "rf" | etc.
    return f"{kind}_{target}_{_ts()}.joblib"

def save_model(model, kind: str, target: str) -> Path:
    path = MODELS_DIR / model_filename(kind, target)
    joblib.dump(model, path)
    return path

def save_metadata(path: Path, metrics: Dict, extra: Optional[Dict] = None) -> Path:
    meta = {"model_path": str(path), "metrics": metrics}
    if extra:
        meta.update(extra)
    meta_path = MODELS_DIR / f"metadata_{_ts()}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path

def latest_model_path(target: str) -> Optional[Path]:
    pats = sorted(MODELS_DIR.glob(f"*_{target}_*.joblib"), key=lambda p: p.stat().st_mtime)
    return pats[-1] if pats else None
