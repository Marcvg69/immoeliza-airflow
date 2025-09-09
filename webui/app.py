# webui/app.py
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import List

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

# local helper (same folder)
from predict_helper import auto_predict

# ---------- Config ----------
DB = os.getenv("IMMO_ANALYTICS_DB", "analytics/immoeliza.duckdb")
st.set_page_config(page_title="ImmoEliza Market", layout="wide")
st.title("🏢 ImmoEliza Market Dashboard")


# ---------- DuckDB helpers (short-lived connections) ----------
def list_tables(db_path: str) -> List[str]:
    try:
        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
        """).fetchall()
        con.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def load_latest(db_path: str) -> pd.DataFrame:
    try:
        con = duckdb.connect(db_path, read_only=True)
        df = con.execute(
            "SELECT * FROM listings_latest ORDER BY snapshot_date DESC"
        ).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("**DB:**")
    st.code(DB)
    tabs = list_tables(DB)
    if tabs:
        st.caption("Tables: " + ", ".join(tabs))

    st.divider()

    # Rebuild analytics (clean -> upsert) via subprocess to avoid DB locks
    if st.button("🔁 Rebuild analytics (clean → upsert)", use_container_width=True):
        with st.spinner("Rebuilding analytics…"):
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from immoeliza.cleaning.analysis_clean import run as r; print(r())",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                st.success("Analytics rebuilt.")
                with st.expander("Build logs", expanded=False):
                    st.code(proc.stdout or "(no stdout)")
                    if proc.stderr:
                        st.code(proc.stderr)
                # Force a fresh page load so we read the new DuckDB contents
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error("Rebuild failed.")
                with st.expander("Error logs", expanded=True):
                    st.code(e.stdout or "(no stdout)")
                    st.code(e.stderr or "(no stderr)")

# ---------- Data + Filters ----------
df_latest = load_latest(DB)

flt1, flt2, flt3, flt4 = st.columns([1, 1, 1, 1])

cities = ["(all)"] + sorted(
    c for c in df_latest.get("city", pd.Series([], dtype=str)).dropna().astype(str).unique() if c and c != "None"
)
ptypes = ["(all)"] + sorted(
    c for c in df_latest.get("property_type", pd.Series([], dtype=str)).dropna().astype(str).unique() if c and c != "None"
)

with flt1:
    city = st.selectbox("City", cities, index=0)
with flt2:
    ptype = st.selectbox("Property type", ptypes, index=0)
with flt3:
    postal_filter = st.text_input("Postal code contains", "")
with flt4:
    only_complete = st.checkbox("Hide rows with missing\ncity/postal/price/surface", value=False)

q = df_latest.copy()
if not q.empty:
    if city != "(all)":
        q = q[q["city"].astype(str) == city]
    if ptype != "(all)":
        q = q[q["property_type"].astype(str) == ptype]
    if postal_filter:
        q = q[q["postal_code"].astype(str).str.contains(postal_filter, na=False)]
    if only_complete:
        q = q.dropna(subset=["city", "postal_code", "price", "surface_m2"])

st.caption(f"Listings ({len(q):,})")
st.dataframe(
    q[
        [
            "url",
            "price",
            "bedrooms",
            "bathrooms",
            "surface_m2",
            "year_built",
            "title",
            "city",
            "postal_code",
            "property_type",
            "price_per_m2",
            "snapshot_date",
        ]
    ].reset_index(drop=True),
    use_container_width=True,
    height=320,
)

# Optional: download filtered CSV
csv = q.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", csv, file_name="listings_filtered.csv", mime="text/csv")

st.divider()

# ---------- Quick chart + group stats ----------
col_l, col_r = st.columns([1, 1])

with col_r:
    if "price_per_m2" in q.columns and q["price_per_m2"].notna().any():
        fig = px.histogram(q[q["price_per_m2"].notna()], x="price_per_m2", nbins=20, title="Price per m²")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price_per_m2 values to plot.")

with col_l:
    if {"city", "property_type", "price_per_m2"}.issubset(q.columns) and q["price_per_m2"].notna().any():
        g = (
            q.dropna(subset=["city", "property_type", "price_per_m2"])
            .groupby(["city", "property_type"], as_index=False)["price_per_m2"]
            .median()
            .sort_values("price_per_m2", ascending=False)
        )
        st.subheader("Median €/m² by City & Property Type")
        st.dataframe(g, use_container_width=True, height=360)
    else:
        st.info("No €/m² data yet for group stats.")

st.divider()

# ---------- Predict (auto-pick model) ----------
st.subheader("🔮 Quick predict")

with st.form("predict_form", clear_on_submit=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        surface_m2 = st.number_input("Surface (m²)", min_value=0.0, step=1.0, value=0.0)
        bedrooms = st.number_input("Bedrooms", min_value=0, step=1, value=0)
        bathrooms = st.number_input("Bathrooms", min_value=0, step=1, value=0)
    with c2:
        city_in = st.text_input("City", value="")
        postal_in = st.text_input("Postal code", value="")
        year_built = st.number_input("Year built", min_value=0, step=1, value=0)
    with c3:
        ptype_in = st.text_input("Property type", value="Apartment")
        energy = st.text_input("Energy label", value="")
        region = st.text_input("Region", value="")

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "surface_m2": surface_m2 or None,
        "bedrooms": bedrooms or None,
        "bathrooms": bathrooms or None,
        "city": city_in or None,
        "postal_code": postal_in or None,
        "year_built": year_built or None,
        "property_type": ptype_in or None,
        "energy_label": energy or None,
        "region": region or None,
    }
    try:
        result = auto_predict(payload)
        msg = f"Used **{result['used_model']}** (target: `{result['target']}`). Prediction: **{result['prediction']:.2f}**"
        if "derived_price" in result:
            msg += f"  •  Derived price: **{result['derived_price']:.0f}**"
        st.success(msg)
        if result.get("meta"):
            st.caption(f"Metrics: {result['meta'].get('metrics')}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Tip: set IMMO_ANALYTICS_DB env var to point at a different DuckDB file.")
