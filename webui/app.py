# webui/app.py
from __future__ import annotations

import os
import sys
import glob
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Optional: DuckDB for market views (safe if not present)
try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None  # UI will degrade gracefully

# --- Project paths (env overrides allowed) ---
REPO_ROOT = Path(os.environ.get("IMMO_REPO_ROOT", Path.cwd()))
ANALYTICS_DUCKDB = Path(os.environ.get("IMMO_ANALYTICS_DB", REPO_ROOT / "analytics" / "immoeliza.duckdb"))
ANALYTICS_GLOB = os.environ.get("IMMO_ANALYTICS_GLOB", str(REPO_ROOT / "data" / "processed" / "analysis" / "*" / "listings.parquet"))
TRAINING_GLOB = os.environ.get("IMMO_TRAINING_GLOB", str(REPO_ROOT / "data" / "processed" / "training" / "*" / "training.parquet"))
MODELS_DIR = Path(os.environ.get("IMMO_MODELS_DIR", REPO_ROOT / "models"))

APP_TITLE = "ImmoEliza — Market & Predict"

# Try to import the helper the way you structured it (same folder as app.py)
# i.e. webui/predict_helper.py -> "from predict_helper import auto_predict"
def _import_auto_predict():
    try:
        from predict_helper import auto_predict  # local helper preferred
        return auto_predict
    except Exception:
        # Fallback: allow running app without helper (degraded)
        def _fallback_auto_predict(features: dict) -> dict:
            # crude pick: use €/m² model if surface is present & > 0
            has_m2 = float(features.get("surface_m2") or 0) > 0
            target = "price_per_m2" if has_m2 else "price"
            # choose latest model/metadata on disk
            meta = sorted(MODELS_DIR.glob(f"metadata_*_{'price_per_m2' if target=='price_per_m2' else 'price'}*.json")) \
                or sorted(MODELS_DIR.glob("metadata_*.json"))
            model = sorted(MODELS_DIR.glob(f"*{target}*.joblib")) or sorted(MODELS_DIR.glob("*.joblib"))
            return {
                "target": target,
                "pred": {"price": None, "price_per_m2": None},  # nothing computed in fallback
                "model_path": str(model[-1]) if model else None,
                "metadata_path": str(meta[-1]) if meta else None,
                "features_used": [],
            }
        return _fallback_auto_predict

auto_predict = _import_auto_predict()

# ---------- Tiny utils ----------
def _mtime(path: Path | None) -> str:
    if not path or not Path(path).exists():
        return "—"
    ts = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M:%S")

def _latest(path_glob: str) -> Path | None:
    files = sorted(glob.glob(path_glob))
    return Path(files[-1]) if files else None

@st.cache_data(show_spinner=False)
def _read_parquet(p: Path | None) -> pd.DataFrame:
    if not p or not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)

def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return "—"

def _fmt_eur(x) -> str:
    try:
        return f"{float(x):,.0f} €".replace(",", " ")
    except Exception:
        return "—"

# ---------- Rebuild actions (call your package code) ----------
def _rebuild_analytics():
    with st.spinner("Rebuilding analytics (clean → upsert)…"):
        try:
            # Your module paths (as you have them)
            from immoeliza.cleaning.analysis_clean import run as run_analysis
            res = run_analysis()
            st.success(f"Analytics rebuilt: {res}")
        except Exception as e:
            st.error(f"Analytics rebuild failed: {e}")

def _rebuild_training_and_train():
    with st.spinner("Rebuilding training set and retraining…"):
        try:
            from immoeliza.cleaning.training_clean import run as run_training
            from immoeliza.modeling.train_regression import train as train_model

            training_path = run_training()
            if isinstance(training_path, (list, tuple)):
                training_path = training_path[-1]
            elif isinstance(training_path, dict):
                training_path = training_path.get("training_path", training_path.get("path", ""))

            if not training_path or not Path(training_path).exists():
                raise RuntimeError(f"Training parquet not found: {training_path}")

            result = train_model(str(training_path))
            st.success(f"Training OK. Chosen: {result.get('metrics', {}).get('chosen')}")
            with st.expander("Training details"):
                st.json(result)
        except Exception as e:
            st.error(f"Training failed: {e}")

# ---------- Sanity summary (non-destructive) ----------
@st.cache_data(show_spinner=False)
def sanity_summary(analytics_parquet: Path | None) -> dict:
    """
    Read latest analytics parquet and compute a small summary of what would be dropped
    by obvious rules (not writing anything; just reporting).
    """
    df = _read_parquet(analytics_parquet).copy()
    if df.empty:
        return {"total": 0, "drop_missing_price": 0, "drop_missing_m2": 0,
                "drop_bad_m2": 0, "drop_bad_ppm2": 0, "kept": 0}

    # normalize columns we care about
    for col in ["price", "surface_m2", "price_per_m2"]:
        if col not in df.columns:
            df[col] = np.nan

    total = len(df)
    m_missing_price = df["price"].isna()
    m_missing_m2 = df["surface_m2"].isna()
    # basic plausibility
    m_bad_m2 = (~m_missing_m2) & ((df["surface_m2"] < 10) | (df["surface_m2"] > 1000))
    ppm2 = np.where((~df["price"].isna()) & (~df["surface_m2"].isna()) & (df["surface_m2"] > 0),
                    df["price"] / df["surface_m2"], np.nan)
    m_bad_ppm2 = (~np.isnan(ppm2)) & ((ppm2 < 500) | (ppm2 > 50000))

    drop = m_missing_price | m_missing_m2 | m_bad_m2 | m_bad_ppm2
    kept = total - int(drop.sum())

    return {
        "total": int(total),
        "drop_missing_price": int(m_missing_price.sum()),
        "drop_missing_m2": int(m_missing_m2.sum()),
        "drop_bad_m2": int(m_bad_m2.sum()),
        "drop_bad_ppm2": int(m_bad_ppm2.sum()),
        "kept": int(kept),
    }

# ---------- DuckDB helpers ----------
def _have_duckdb() -> bool:
    return duckdb is not None and ANALYTICS_DUCKDB.exists()

def _q(sql: str, params: tuple = ()):
    if not _have_duckdb():
        return pd.DataFrame()
    con = duckdb.connect(str(ANALYTICS_DUCKDB))
    try:
        return con.execute(sql, params).df()
    finally:
        con.close()

# ===================== UI =====================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Sidebar actions
with st.sidebar:
    st.subheader("Actions")
    if st.button("Rebuild analytics (clean → upsert)", use_container_width=True):
        _rebuild_analytics()

    if st.button("Rebuild training + retrain", use_container_width=True):
        _rebuild_training_and_train()

    st.divider()
    st.caption(f"Analytics DB: `{ANALYTICS_DUCKDB}`")
    st.caption(f"Models dir: `{MODELS_DIR}`")

tabs = st.tabs(["Market", "Predict"])

# -------- Market tab --------
with tabs[0]:
    c1, c2 = st.columns([2, 3])

    with c1:
        st.subheader("Latest listings (sample)")
        if _have_duckdb():
            try:
                df_latest = _q("SELECT * FROM listings_latest ORDER BY snapshot_ts DESC LIMIT 200;")
                st.dataframe(df_latest, use_container_width=True, hide_index=True)
            except Exception as e:
                st.info(f"Could not read `listings_latest`: {e}")
        else:
            st.info("No DuckDB found yet. Rebuild analytics or run the DAG.")

        st.subheader("Daily market summary")
        if _have_duckdb():
            try:
                df_sum = _q("SELECT * FROM market_daily_summary ORDER BY snapshot_date DESC LIMIT 60;")
                st.dataframe(df_sum, use_container_width=True, hide_index=True)
            except Exception as e:
                st.info(f"Could not read `market_daily_summary`: {e}")

    with c2:
        st.subheader("Sanity summary")
        latest_analytics = _latest(ANALYTICS_GLOB)
        s = sanity_summary(latest_analytics)
        st.markdown(
            f"""
**Total rows:** {_fmt_int(s['total'])}  
- Missing price: {_fmt_int(s['drop_missing_price'])}  
- Missing m²: {_fmt_int(s['drop_missing_m2'])}  
- Implausible m² (<10 or >1000): {_fmt_int(s['drop_bad_m2'])}  
- Implausible €/m² (<500 or >50,000): {_fmt_int(s['drop_bad_ppm2'])}  

**Kept after sanity:** {_fmt_int(s['kept'])}
"""
        )

# -------- Predict tab --------
with tabs[1]:
    st.subheader("Quick predict")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            price = st.number_input("Advertised price (€)", min_value=0, step=1000, value=0)
            surface_m2 = st.number_input("Living area (m²)", min_value=0, step=1, value=0)
            bedrooms = st.number_input("Bedrooms", min_value=0, step=1, value=2)

        with c2:
            bathrooms = st.number_input("Bathrooms", min_value=0, step=1, value=1)
            year_built = st.number_input("Year built", min_value=1800, max_value=2100, step=1, value=2000)
            energy_label = st.selectbox("Energy label", ["", "A", "B", "C", "D", "E", "F", "G"])

        with c3:
            postal_code = st.text_input("Postal code", value="")
            city = st.text_input("City", value="")
            property_type = st.selectbox("Property type", ["", "apartment", "house", "duplex", "studio", "villa"])

        submitted = st.form_submit_button("Predict")
        if submitted:
            try:
                features = {
                    "price": float(price) if price else None,
                    "surface_m2": float(surface_m2) if surface_m2 else None,
                    "bedrooms": int(bedrooms),
                    "bathrooms": int(bathrooms),
                    "year_built": int(year_built) if year_built else None,
                    "postal_code": postal_code.strip() or None,
                    "city": city.strip() or None,
                    "property_type": property_type or None,
                    "energy_label": energy_label or None,
                }
                res = auto_predict(features)  # helper decides the right model

                pred = res.get("pred", {})
                cc1, cc2 = st.columns(2)
                with cc1:
                    p = pred.get("price")
                    st.metric("Predicted price", _fmt_eur(p) if p else "—")
                with cc2:
                    p2 = pred.get("price_per_m2")
                    st.metric("Predicted €/m²", f"{p2:,.0f} €/m²".replace(",", " ") if p2 else "—")

                with st.expander("Model details"):
                    st.write("**Model file**:", Path(res["model_path"]).name if res.get("model_path") else "—")
                    st.write("**Metadata**:", Path(res["metadata_path"]).name if res.get("metadata_path") else "—")
                    st.json({"features_used": res.get("features_used", []), "target": res.get("target")})
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -------- Footer --------
st.divider()
latest_analytics = _latest(ANALYTICS_GLOB)
latest_training = _latest(TRAINING_GLOB)
latest_meta = max(MODELS_DIR.glob("metadata_*.json"), default=None, key=lambda p: p.stat().st_mtime) if MODELS_DIR.exists() else None

st.caption(
    "📦 **Paths** — "
    f"Analytics parquet: `{latest_analytics or '—'}` | "
    f"Training parquet: `{latest_training or '—'}` | "
    f"Models dir: `{MODELS_DIR}` | "
    f"DuckDB: `{ANALYTICS_DUCKDB}`"
)
st.caption(
    "⏱ **Last rebuild** — "
    f"analytics: {_mtime(latest_analytics)} | training: {_mtime(latest_training)} | model: {_mtime(latest_meta)}"
)
