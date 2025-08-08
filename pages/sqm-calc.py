# pages/streamlit_sales_per_sqm_potential.py
import os, sys
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# Zorg dat shop_mapping vanuit root gevonden wordt (script staat in /pages)
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from shop_mapping import SHOP_NAME_MAP

# === PAGE CONFIG & HEADER ===
st.set_page_config(page_title="Sales-per-sqm Potentieel (CSm²I)", page_icon="📊", layout="wide")

# Global styling (Instrument Sans, paarse pills, PFM-rode knop)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
/* Paarse multiselect pills */
[data-baseweb="tag"] { background-color: #9E77ED !important; color: white !important; }
/* Rode knop (Analyseer / Run simulation) */
button[kind="secondary"], button[kind="primary"], button[data-testid="stBaseButton-secondary"], .stButton>button {
  background-color: #F04438 !important; color: white !important;
  border-radius: 16px !important; font-weight: 600 !important; padding: .6rem 1.4rem !important; border: none !important;
}
button[kind="secondary"]:hover, button[kind="primary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover, .stButton>button:hover { background-color: #d13c30 !important; cursor: pointer; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Sales-per-sqm Potentieel (CSm²I)")
st.caption("Selecteer winkels en periode. NVO (sq_meter) via API. Berekening op dag‑niveau (STEP='day').")

# === SECRETS ===
API_URL = st.secrets["API_URL"]

# === INPUTS ===
PRESETS = ["yesterday","today","this_week","last_week",
           "this_month","last_month","this_quarter","last_quarter",
           "this_year","last_year"]
period = st.selectbox("Select period", options=PRESETS, index=5)

NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}
default_names = [ID_TO_NAME[i] for i in SHOP_NAME_MAP.keys()]
selected_names = st.multiselect("Select stores", options=list(NAME_TO_ID.keys()), default=default_names)
shop_ids = [NAME_TO_ID[n] for n in selected_names]

if not shop_ids:
    st.warning("Please select at least one store.")
    st.stop()

STEP = "day"  # hardcoded period_step

# === HELPERS ===
def _to_float(x):
    if x is None or x == "":
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def _eur(num, decimals=2):
    if pd.isna(num):
        return ""
    s = f"{float(num):,.{decimals}f}"
    return "€" + s.replace(",", "X").replace(".", ",").replace("X", ".")

# === API CALL (identiek aan je andere calcs: POST met params, ZONDER [] in keys) ===
def fetch_report(api_url: str, shop_ids: list[int], period: str, step: str, metrics: list[str]):
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step", step), ("weather","0")]

    r = requests.post(api_url, params=params, timeout=40)
    status = r.status_code
    text_preview = r.text[:2000] if r.text else ""
    try:
        js = r.json()
    except Exception:
        js = {}

    req_info = {"url": api_url, "params_list": params}
    return js, req_info, status, text_preview

# === PARSER: ondersteunt geneste dates-structuur + aggregatie naar 1 regel per winkel ===
def parse_vemcount(payload: dict, shop_ids: list[int], fields: list[str], period_key: str) -> pd.DataFrame:
    if not isinstance(payload, dict): return pd.DataFrame()
    if "data" in payload and isinstance(payload["data"], dict) and period_key in payload["data"]:
        per = payload["data"].get(period_key, {})
        rows = []
        for sid in shop_ids:
            shop_block = per.get(str(sid)) or per.get(int(sid))
            if not isinstance(shop_block, dict): continue
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
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        agg = {}
        for f in fields:
            if f in ["count_in", "turnover"]: agg[f] = "sum"
            elif f in ["sq_meter"]: agg[f] = "max"
            else: agg[f] = "mean"
        return df.groupby("shop_id", as_index=False).agg(agg)

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

# === RUN SIMULATION (zelfde UX‑flow als je andere calcs) ===
if st.button("Analyseer", type="secondary"):
    with st.spinner("Calculating hidden location potential..."):
        metrics = ["count_in","sales_per_visitor","sales_per_sqm",
                   "conversion_rate","sales_per_transaction","turnover","sq_meter"]

        payload, req_info, status, text_preview = fetch_report(API_URL, shop_ids, period, STEP, metrics)

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

        # === BEREKENINGEN ===
        df["Store"] = df["shop_id"].map(ID_TO_NAME)
        sq = df["sq_meter"].astype(float).replace(0, np.nan)

        df["visitors_per_sqm"] = df["count_in"].astype(float) / sq
        df["expected_spsqm"]   = df["sales_per_visitor"].astype(float) * df["visitors_per_sqm"]

        actual_chk = df["turnover"].astype(float) / sq
        sales_sqm = df.get("sales_per_sqm", pd.Series(np.nan, index=df.index)).astype(float)
        df["actual_spsqm"] = np.where(sales_sqm.notna(), sales_sqm, actual_chk)

        eps = 1e-9
        df["CSm2I"] = df["actual_spsqm"] / (df["expected_spsqm"] + eps)   # index; 1.00 = op niveau
        df["uplift_eur"] = np.maximum(0.0, df["expected_spsqm"] - df["actual_spsqm"]) * sq

        # === KPI‑BANNER ===
        total_extra = float(df["uplift_eur"].sum())
        st.markdown(f"""
            <div style='background-color: #FEAC76; color: #000000;
                        padding: 1.5rem; border-radius: 0.75rem;
                        font-size: 1.25rem; font-weight: 600;
                        text-align: center; margin-bottom: 1.5rem;'>
                🚀 The potential revenue growth is <span style='font-size:1.5rem;'>{_eur(total_extra, 0)}</span>
            </div>
        """, unsafe_allow_html=True)

        # === TABEL (opgeschoond + EU formats) ===
        table = pd.DataFrame({
            "Store": df["Store"],
            "Square meters": sq.round(0).astype("Int64"),
            "Current Avg Sales per sqm": df["actual_spsqm"].round(2).map(lambda v: _eur(v, 2)),
            "CSm²I (index)": df["CSm2I"].round(2),
            "Potential revenue uplift (€)": df["uplift_eur"].round(0).map(lambda v: _eur(v, 0)),
        }).sort_values("Potential revenue uplift (€)", ascending=False)
        st.subheader("🏆 Stores with most potential (sorted by uplift)")
        st.dataframe(table, use_container_width=True)

        # === BAR CHART (kleur #762181) ===
        chart_df = df[["Store","uplift_eur","CSm2I"]].copy()
        chart_df["uplift_fmt"] = chart_df["uplift_eur"].map(lambda v: _eur(v, 0))
        fig = px.bar(
            chart_df.sort_values("uplift_eur", ascending=False).head(20),
            x="Store", y="uplift_eur",
            custom_data=["uplift_fmt","CSm2I"],
            color_discrete_sequence=["#762181"]
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Uplift: %{customdata[0]}<br>CSm²I: %{customdata[1]:.2f}<extra></extra>")
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=60),
                          yaxis_title="Potential revenue uplift (€)")
        st.plotly_chart(fig, use_container_width=True)

        # === SCATTER (Viridis + duidelijke legenda + slimme x-as) ===
        sc = df[["Store","sales_per_visitor","visitors_per_sqm","CSm2I","uplift_eur"]].copy()
        sc["uplift_fmt"] = sc["uplift_eur"].map(lambda v: _eur(v, 0))
        x_min, x_max = float(np.nanmin(sc["sales_per_visitor"])), float(np.nanmax(sc["sales_per_visitor"]))
        span = max(0.01, x_max - x_min); pad = span * 0.05
        x_range = [x_min - pad, x_max + pad]

        fig2 = px.scatter(
            sc, x="sales_per_visitor", y="visitors_per_sqm",
            size="uplift_eur", color="CSm2I", color_continuous_scale="Viridis",
            hover_data={"Store": True, "sales_per_visitor": ":.2f", "visitors_per_sqm": ":.2f"},
            custom_data=["Store","uplift_fmt","CSm2I"]
        )
        fig2.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Uplift: %{customdata[1]}<br>"
                          "CSm²I: %{customdata[2]:.2f}<br>"
                          "SPV: %{x:.2f}<br>"
                          "Visitors/m²: %{y:.2f}<extra></extra>"
        )
        fig2.update_layout(
            margin=dict(l=10,r=10,t=30,b=10),
            xaxis_title="Sales per Visitor", yaxis_title="Visitors per m²",
            xaxis=dict(range=x_range),
            coloraxis_colorbar=dict(title="CSm²I (index)")
        )
        st.plotly_chart(fig2, use_container_width=True)

        # === KORTE TOELICHTING ===
        st.markdown(
            """
            **Toelichting**  
            - **CSm²I (index)**: gerealiseerd t.o.v. verwacht omzet per m². *1,00* = op verwachting; *<1,00* = onderprestatie.  
            - **Potential revenue uplift (€)**: indicatie van **extra omzet op jaarbasis** als de winkel naar verwacht niveau groeit
              (geëxtrapoleerd uit de gekozen periode, berekend als *(expected − actual) × m²*).  
            """
        )
