# webui/app.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import streamlit as st

# local modules
# If your package name differs, adjust these imports
try:
    from immoeliza.cleaning.analysis_clean import rebuild_analytics, ANALYTICS_DB_PATH, SANITY_JSON_PATH, LAST_REBUILD_PATH
except Exception:
    # Fallback if package layout differs; keep app usable
    ANALYTICS_DB_PATH = Path("analytics/immoeliza.duckdb")
    SANITY_JSON_PATH = Path("analytics/_sanity_summary.json")
    LAST_REBUILD_PATH = Path("analytics/_last_rebuild.txt")

    def rebuild_analytics(clean: bool = True) -> dict:
        raise RuntimeError("analysis_clean.rebuild_analytics not importable.")

try:
    from immoeliza.modeling.train_regression import rebuild_training_and_retrain, MODELS_DIR
except Exception:
    MODELS_DIR = Path("models")

    def rebuild_training_and_retrain(models_dir: Path | str = MODELS_DIR,
                                     db_path: Path | str = ANALYTICS_DB_PATH) -> dict:
        raise RuntimeError("train_regression.rebuild_training_and_retrain not importable.")


st.set_page_config(page_title="ImmoEliza — Market & Predict", layout="wide")
st.title("ImmoEliza — Market & Predict")

# Sidebar actions
with st.sidebar:
    st.header("Actions")
    if st.button("Rebuild analytics (clean → upsert)"):
        try:
            info = rebuild_analytics(clean=True)
            st.success(f"Analytics rebuilt. Kept rows: {info.get('kept_after_sanity', 'n/a')}")
        except Exception as e:
            st.error(f"Analytics rebuild failed: {e}")

    if st.button("Rebuild training + retrain"):
        try:
            info = rebuild_training_and_retrain(MODELS_DIR, ANALYTICS_DB_PATH)
            st.success(f"Training ok. R²: {info.get('r2','n/a'):.3f}, MAE: {info.get('mae','n/a'):.1f}")
        except Exception as e:
            st.error(f"Training failed: {e}")

    st.write("### Analytics DB:")
    st.code(str(Path(ANALYTICS_DB_PATH).resolve()))

    st.write("### Models dir:")
    st.code(str(Path(MODELS_DIR).resolve()))

tab_market, tab_predict = st.tabs(["Market", "Predict"])

def _conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(ANALYTICS_DB_PATH))

def _load_df(sql: str) -> pd.DataFrame:
    with _conn() as con:
        return con.execute(sql).df()

with tab_market:
    st.subheader("Latest listings (sample)")
    sample_err: Optional[str] = None
    df_latest = pd.DataFrame()
    try:
        # NOTE: order by snapshot_date (not snapshot_ts)
        df_latest = _load_df("SELECT * FROM listings_latest ORDER BY snapshot_date DESC LIMIT 200;")
        st.dataframe(df_latest, use_container_width=True, height=400)
    except Exception as e:
        sample_err = f"Could not read listings_latest: {e}"
        st.code(sample_err)

    st.subheader("Sanity summary")
    kept_after = 0
    try:
        if SANITY_JSON_PATH.exists():
            info = json.loads(SANITY_JSON_PATH.read_text())
            kept_after = int(info.get("kept_after_sanity", 0))
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Total rows:** {info.get('total_rows',0)}  
                • **Missing price:** {info.get('missing_price',0)}  
                • **Missing m²:** {info.get('missing_m2',0)}  
                """)
            with col2:
                st.markdown(f"""
                • **Implausible m² (<10 or >1000):** {info.get('bad_m2',0)}  
                • **Implausible €/m² (<500 or >50,000):** {info.get('bad_eur_m2',0)}  
                **Kept after sanity:** {kept_after}
                """)
        else:
            st.info("No sanity file yet. Rebuild analytics first.")
    except Exception as e:
        st.error(f"Could not load sanity summary: {e}")

    st.subheader("Daily market summary")
    try:
        df_summary = _load_df("""
            SELECT snapshot_date, city, property_type,
                   COUNT(*) AS n,
                   MEDIAN(price) AS median_price,
                   MEDIAN(price_per_m2) AS median_eur_m2
            FROM listings_history
            GROUP BY ALL
            ORDER BY snapshot_date DESC, n DESC
            LIMIT 500;
        """)
        st.dataframe(df_summary, use_container_width=True, height=400)
    except Exception as e:
        st.code(f"Could not read listings_history: {e}")

with tab_predict:
    st.subheader("Quick prediction")
    with st.form("predict_form"):
        city = st.text_input("City")
        postal_code = st.text_input("Postal code")
        property_type = st.selectbox("Property type", ["Apartment", "House", "Residence", "Villa", "Other"])
        is_rent = st.selectbox("Is rent?", ["No", "Yes"]) == "Yes"
        bedrooms = st.number_input("Bedrooms", 0, 20, 2)
        bathrooms = st.number_input("Bathrooms", 0, 10, 1)
        surface_m2 = st.number_input("Surface (m²)", 0, 2000, 80)
        submitted = st.form_submit_button("Predict price")

    if submitted:
        model_path = next(Path(MODELS_DIR).glob("*.joblib"), None)
        if not model_path:
            st.error("No model found. Click ‘Rebuild training + retrain’ first.")
        else:
            import joblib
            pipe = joblib.load(model_path)
            X = pd.DataFrame([{
                "city": city or None,
                "postal_code": str(postal_code or ""),
                "property_type": property_type or None,
                "is_rent": bool(is_rent),
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "surface_m2": surface_m2,
            }])
            try:
                y = pipe.predict(X)[0]
                st.success(f"Predicted price: €{y:,.0f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Footer: last rebuild time
st.markdown("---")
try:
    if LAST_REBUILD_PATH.exists():
        st.caption(f"Last rebuild: {LAST_REBUILD_PATH.read_text().strip()}")
except Exception:
    pass
