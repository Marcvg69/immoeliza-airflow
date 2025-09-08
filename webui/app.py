# webui/app.py
import os
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Config ---
DB = os.getenv("IMMO_ANALYTICS_DB", "analytics/immoeliza.duckdb")

st.set_page_config(page_title="ImmoEliza Market", layout="wide")
st.title("🏢 ImmoEliza Market Dashboard")

# ---------- DuckDB helpers (short-lived connections) ----------
def list_tables(db_path: str) -> list[str]:
    try:
        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute("""
            select table_name
            from information_schema.tables
            where table_schema = 'main'
            order by table_name
        """).fetchall()
        con.close()
        return [r[0] for r in rows]
    except Exception:
        return []

def load_latest(db_path: str) -> pd.DataFrame:
    try:
        con = duckdb.connect(db_path, read_only=True)
        df = con.execute("SELECT * FROM listings_latest").df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("**DB:**")
    st.code(DB, language="bash")

    tables = list_tables(DB)
    if tables:
        st.caption("Tables: " + ", ".join(tables))
    else:
        st.caption("No tables found.")

    st.divider()

    # Rebuild analytics on demand
    if st.button("🔄 Rebuild analytics (clean → upsert)"):
        from immoeliza.cleaning.analysis_clean import run as rebuild
        path = rebuild()
        st.success(f"Rebuilt analytics → {path}")
        st.rerun()

    st.divider()
    st.caption("Filters")

# ---------- Load data ----------
df = load_latest(DB)

if df.empty:
    st.info("No data in `listings_latest`. Run the pipeline or use the rebuild button in the sidebar.")
    st.stop()

# Ensure expected columns exist (fill missing with NA for display)
expected_cols = [
    "snapshot_date", "city", "postal_code", "property_type",
    "price", "price_per_m2", "surface_m2",
    "bedrooms", "bathrooms", "energy_label",
    "is_sale", "url"
]
for c in expected_cols:
    if c not in df.columns:
        df[c] = pd.Series([None] * len(df))

# ---------- Filters (safe lists) ----------
cities = sorted([c for c in df["city"].dropna().astype(str).unique().tolist() if c and c.lower() != "none"])
ptypes = sorted([p for p in df["property_type"].dropna().astype(str).unique().tolist() if p and p.lower() != "none"])

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    city = st.selectbox("City", options=["(all)"] + cities, index=0)
with col2:
    ptype = st.selectbox("Property type", options=["(all)"] + ptypes, index=0)
with col3:
    pc = st.text_input("Postal code contains", "")
with col4:
    non_null = st.checkbox("Hide rows with missing city/postal/price/surface", value=False)

# ---------- Apply filters ----------
q = df.copy()

if city != "(all)":
    q = q[q["city"].astype(str) == city]

if ptype != "(all)":
    q = q[q["property_type"].astype(str) == ptype]

if pc:
    q = q[q["postal_code"].astype(str).str.contains(pc, case=False, na=False)]

if non_null:
    q = q.dropna(subset=["city", "postal_code", "price", "surface_m2"])

# ---------- Summary & Table ----------
st.metric("Listings", len(q))
st.dataframe(q.fillna(""), use_container_width=True, height=420)

# ---------- Chart (only rows with price_per_m2) ----------
q_chart = q.dropna(subset=["price_per_m2"])
with st.container():
    st.subheader("Price per m²")
    if len(q_chart):
        st.plotly_chart(
            px.histogram(q_chart, x="price_per_m2", nbins=25),
            use_container_width=True,
        )
    else:
        st.info("No `price_per_m2` values to plot.")

# ---------- Rollups ----------
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
st.download_button(
    "⬇️ Download filtered CSV",
    q.to_csv(index=False).encode(),
    "immoeliza_filtered.csv",
    "text/csv",
)

st.caption("Tip: set IMMO_ANALYTICS_DB env var to point at a different DuckDB file.")
