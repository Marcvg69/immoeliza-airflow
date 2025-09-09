# webui/app.py
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

# (optional) if you want to call auto-predict elsewhere
# from predict_helper import auto_predict

DB = os.getenv("IMMO_ANALYTICS_DB", "analytics/immoeliza.duckdb")
PROCESSED_ANALYSIS_DIR = Path("data/processed/analysis")  # where the cleaner writes daily outputs

st.set_page_config(page_title="ImmoEliza Market", layout="wide")
st.title("🏢 ImmoEliza Market Dashboard")


# ---------- DuckDB helpers ----------
def list_tables(db_path: str) -> List[str]:
    try:
        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
        ).fetchall()
        con.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def load_latest(db_path: str) -> pd.DataFrame:
    try:
        con = duckdb.connect(db_path, read_only=True)
        df = con.execute("SELECT * FROM listings_latest ORDER BY snapshot_date DESC").df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def _find_latest_sanity_json() -> Optional[Path]:
    if not PROCESSED_ANALYSIS_DIR.exists():
        return None
    dated = sorted([p for p in PROCESSED_ANALYSIS_DIR.iterdir() if p.is_dir()])
    if not dated:
        return None
    cand = dated[-1] / "sanity_summary.json"
    return cand if cand.exists() else None


def _load_sanity_summary() -> Optional[dict]:
    p = _find_latest_sanity_json()
    if not p:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("**DB:**")
    st.code(DB)
    tabs = list_tables(DB)
    st.caption("Tables: " + (", ".join(tabs) if tabs else "—"))

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
                # Try to extract path lines (optional)
                if proc.stdout:
                    st.expander("Build logs", expanded=False).code(proc.stdout)
                if proc.stderr:
                    st.expander("Warnings/Errors", expanded=False).code(proc.stderr)
                # refresh session-level cache for footer
                st.session_state["last_rebuild_at"] = datetime.now().isoformat(timespec="seconds")
                st.session_state["sanity_summary"] = _load_sanity_summary()
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error("Rebuild failed.")
                with st.expander("Error logs", expanded=True):
                    st.code(e.stdout or "(no stdout)")
                    st.code(e.stderr or "(no stderr)")

# ---------- Data + Filters ----------
df_latest = load_latest(DB)
if df_latest.empty:
    st.info("No data in `listings_latest`. Run the pipeline or use the rebuild button in the sidebar.")
    # Footer (still show last summary if present)
    ss = st.session_state.get("sanity_summary") or _load_sanity_summary()
    if ss:
        st.caption(
            f"Last rebuild: {ss.get('when','?')} • dropped: "
            f"duplicates={ss.get('dropped_duplicates',0)}, "
            f"bad_price_for_sale={ss.get('dropped_bad_price_for_sale',0)}, "
            f"surface_outlier={ss.get('dropped_surface_outlier',0)}, "
            f"ppm2_outlier={ss.get('dropped_ppm2_outlier',0)}, "
            f"null_keys={ss.get('dropped_null_key',0)}"
        )
    st.stop()

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

# ---------- Quick chart + group stats ----------
col_l, col_r = st.columns([1, 1])
with col_r:
    if "price_per_m2" in q.columns and q["price_per_m2"].notna().any():
        fig = px.histogram(q[q["price_per_m2"].notna()], x="price_per_m2", nbins=20, title="Price per m²")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price_per_m2 values to plot.")

st.subheader("Median €/m² by City & Property Type")
roll = (
    q.dropna(subset=["price_per_m2"])
    .groupby(["city", "property_type"], dropna=False)["price_per_m2"]
    .median()
    .reset_index()
    .sort_values("price_per_m2", ascending=False)
)
st.dataframe(roll, use_container_width=True)

# ---------- Download filtered ----------
csv = q.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", csv, file_name="listings_filtered.csv", mime="text/csv")

# ---------- Footer: last rebuild + sanity summary ----------
ss = st.session_state.get("sanity_summary") or _load_sanity_summary()
when = st.session_state.get("last_rebuild_at")
if ss:
    when = when or ss.get("when")
    st.caption(
        f"Last rebuild: {when} • dropped: "
        f"duplicates={ss.get('dropped_duplicates',0)}, "
        f"bad_price_for_sale={ss.get('dropped_bad_price_for_sale',0)}, "
        f"surface_outlier={ss.get('dropped_surface_outlier',0)}, "
        f"ppm2_outlier={ss.get('dropped_ppm2_outlier',0)}, "
        f"null_keys={ss.get('dropped_null_key',0)}"
    )
else:
    st.caption("Last rebuild: –  •  (no sanity summary yet)")
