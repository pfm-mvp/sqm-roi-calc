# pages/streamlit_sales_per_sqm_potential.py
import os, sys
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# === Imports / mapping
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from shop_mapping import SHOP_NAME_MAP

st.set_page_config(page_title="Sales-per-sqm Potentieel (CSm²I)", page_icon="📊", layout="wide")

# === Styling
PFM_RED = "#F04438"
st.markdown(f"""
<style>
.stButton>button {{
  background-color: {PFM_RED};
  color: white;
  border-radius: 12px;
  padding: .6rem 1rem;
  font-weight: 600; border: none;
}}
.block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)

st.title("📊 Sales-per-sqm Potentieel (CSm²I)")
st.caption("Selecteer winkels en periode. NVO (sq_meter) via API. Berekening op dag-niveau (STEP='day').")

# === Secrets
API_URL = st.secrets["API_URL"]

# === UI inputs (alleen periode + stores)
colA, colB = st.columns([1,1])
with colA:
    PRESETS = ["yesterday","today","this_week","last_week","this_month","last_month",
               "this_quarter","last_quarter","this_year","last_year"]
    period = st.selectbox("Periode", options=PRESETS, index=5)
with colB:
    pass  # plek gehouden voor evt. extra filters later

# Hardcode step (geen input)
STEP = "day"

# === Naam ↔ ID mapping
NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}
default_names = [ID_TO_NAME[i] for i in SHOP_NAME_MAP.keys()]
selected_names = st.multiselect("Select stores", list(NAME_TO_ID.keys()), default=default_names)
shop_ids = [NAME_TO_ID[n] for n in selected_names]
if not shop_ids:
    st.warning("Selecteer minimaal één store.")
    st.stop()

# === API call — identiek patroon als andere calcs (POST params, ZONDER [] in keys)
def fetch_report(api_url: str, shop_ids: list[int], period: str, step: str, metrics: list[str]):
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [
        ("source", "shops"),
        ("period", period),
        ("step", step),
        ("weather", "0"),
    ]

    r = requests.post(api_url, params=params, timeout=40)
    status = r.status_code
    text_preview = r.text[:2000] if r.text else ""
    try:
        js = r.json()
    except Exception:
        js = {}

    req_info = {"url": api_url, "params_list": params}
    return js, req_info, status, text_preview

# === Helpers
def _to_float(x):
    if x is None or x == "":
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

# === Parser (genest + plat) met aggregatie naar 1 regel per winkel
def parse_vemcount(payload: dict, shop_ids: list[int], fields: list[str], period_key: str) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()

    # Geneste structuur
    if "data" in payload and isinstance(payload["data"], dict) and period_key in payload["data"]:
        per = payload["data"].get(period_key, {})
        rows = []
        for sid in shop_ids:
            shop_block = per.get(str(sid)) or per.get(int(sid))
            if not isinstance(shop_block, dict):
                continue
            dates = shop_block.get("dates", {}) or {}
            for _, day_info in dates.items():
                day_data = (day_info or {}).get("data", {}) or {}
                row = {"shop_id": sid}
                for f in fields:
                    row[f] = _to_float(day_data.get(f))
                if row.get("sq_meter") is None or np.isnan(row.get("sq_meter")):
                    sm = ((shop_block.get("data") or {}).get("sq_meter"))
                    row["sq_meter"] = _to_float(sm)
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        agg = {}
        for f in fields:
            if f in ["count_in", "turnover"]:
                agg[f] = "sum"
            elif f in ["sq_meter"]:
                agg[f] = "max"
            else:
                agg[f] = "mean"
        return df.groupby("shop_id", as_index=False).agg(agg)

    # Platte structuur
    if "data" in payload and isinstance(payload["data"], dict):
        flat = []
        for sid in shop_ids:
            rec = payload["data"].get(str(sid), {}) or {}
            row = {"shop_id": sid}
            for f in fields:
                row[f] = _to_float(rec.get(f))
            flat.append(row)
        return pd.DataFrame(flat)

    return pd.DataFrame()

# === Analyse-run (altijd live data)
if st.button("Analyseer", type="primary"):
    try:
        metrics = [
            "count_in",
            "sales_per_visitor",
            "sales_per_sqm",
            "conversion_rate",
            "sales_per_transaction",
            "turnover",
            "sq_meter"
        ]

        payload, req_info, status, text_preview = fetch_report(API_URL, shop_ids, period, STEP, metrics)

        # Debug
        with st.expander("🔧 Request/Response Debug"):
            st.write("➡️  POST naar:", req_info["url"])
            st.write("➡️  Params list:", req_info["params_list"])
            st.write("⬅️  HTTP status:", status)
            st.write("⬅️  Response preview:"); st.code(text_preview or "<empty>")

        if status != 200:
            st.error(f"API gaf status {status}. Zie debug hierboven.")
            st.stop()

        df = parse_vemcount(payload, shop_ids, fields=metrics, period_key=period)

        if df.empty:
            st.error("Geen data (na parsen/aggregatie). Check periode en debug.")
            st.stop()

        # === Berekeningen (CSm²I & uplift)
        df["store_name"] = df["shop_id"].map(ID_TO_NAME)
        sq = df["sq_meter"].astype(float).replace(0, np.nan)

        df["visitors_per_sqm"] = df["count_in"].astype(float) / sq
        df["expected_spsqm"]   = df["sales_per_visitor"].astype(float) * df["visitors_per_sqm"]

        actual_chk = df["turnover"].astype(float) / sq
        sales_sqm = df.get("sales_per_sqm", pd.Series(np.nan, index=df.index)).astype(float)
        df["actual_spsqm"] = np.where(sales_sqm.notna(), sales_sqm, actual_chk)

        eps = 1e-9
        df["CSm2I"] = df["actual_spsqm"] / (df["expected_spsqm"] + eps)
        df["uplift_eur"] = np.maximum(0.0, df["expected_spsqm"] - df["actual_spsqm"]) * sq

        # === Output
        st.subheader("🏆 Top underperformers (laagste CSm²I)")
        topN = df.sort_values("CSm2I").head(10).copy()
        topN["uplift_eur_fmt"] = topN["uplift_eur"].map(lambda x: f"€{x:,.0f}".replace(",", "."))
        st.dataframe(topN[["store_name","shop_id","CSm2I","uplift_eur_fmt"]], use_container_width=True)

        st.subheader("💰 Uplift € per winkel")
        chart_df = df.sort_values("uplift_eur", ascending=False).head(20)
        fig = px.bar(chart_df, x="store_name", y="uplift_eur")
        fig.update_layout(yaxis_tickprefix="€", margin=dict(l=10,r=10,t=30,b=60))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧭 SPV vs Visitors/m²")
        df["uplift_size"] = df["uplift_eur"].fillna(0).clip(lower=0)
        fig2 = px.scatter(df, x="sales_per_visitor", y="visitors_per_sqm",
                          size="uplift_size", color="CSm2I",
                          hover_data=["store_name","shop_id","CSm2I","uplift_eur"])
        fig2.update_layout(margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📥 Export")
        export_cols = ["store_name","shop_id","sq_meter","count_in","sales_per_visitor",
                       "sales_per_sqm","turnover","visitors_per_sqm","expected_spsqm",
                       "actual_spsqm","CSm2I","uplift_eur"]
        csv = df[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="sales_per_sqm_potential.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Onverwachte fout: {e}")
