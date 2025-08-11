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
/* Rode knop (Analyseer) */
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
colA, colB = st.columns([1, 1])
with colA:
    PRESETS = ["yesterday","today","this_week","last_week",
               "this_month","last_month","this_quarter","last_quarter",
               "this_year","last_year"]
    period = st.selectbox("Select period", options=PRESETS, index=5)

NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}
default_names = [ID_TO_NAME[i] for i in SHOP_NAME_MAP.keys()]
with colB:
    selected_names = st.multiselect("Select stores", options=list(NAME_TO_ID.keys()), default=default_names)

shop_ids = [NAME_TO_ID[n] for n in selected_names]
if not shop_ids:
    st.warning("Please select at least one store.")
    st.stop()

# Benchmark-modus (voor doel-CSm²I, niet voor conversie)
use_benchmark = st.toggle("📌 Gebruik benchmark-winkel i.p.v. vaste CSm²I-target", value=False)
benchmark_store_id, benchmark_store_name = None, None
if use_benchmark:
    benchmark_store_name = st.selectbox("Kies benchmark-winkel", options=selected_names)
    benchmark_store_id = NAME_TO_ID[benchmark_store_name]
else:
    # Slider voor doel-CSm²I als benchmark UIT staat
    target_csm2i = st.slider("Target CSm²I", min_value=0.10, max_value=1.00, value=0.85, step=0.05)

# 🎯 Nieuw: slider voor gewenste conversie (stuurt conversie-uplift)
target_conv_pct = st.slider("Target conversion rate (%)", min_value=10, max_value=50, value=35, step=1)

STEP = "day"  # hardcoded period_step

# === HELPERS ===
def _to_float(x):
    if x is None or x == "": return np.nan
    try: return float(x)
    except Exception: return np.nan

def _eur(num, decimals=2):
    if pd.isna(num): return ""
    s = f"{float(num):,.{decimals}f}"
    return "€" + s.replace(",", "X").replace(".", ",").replace("X", ".")

def smart_range(series):
    s = series.astype(float).replace([np.inf,-np.inf], np.nan).dropna()
    if s.empty: return None
    p2, p98 = np.percentile(s, [2, 98])
    span = max(0.001, p98 - p2)
    pad = span * 0.08
    return [p2 - pad, p98 + pad]

# === API CALL (POST met params, ZONDER [] – identiek aan je andere werkende calcs) ===
def fetch_report(api_url: str, shop_ids: list[int], period: str, step: str, metrics: list[str]):
    params = [("data", sid) for sid in shop_ids]
    params += [("data_output", m) for m in metrics]
    params += [("source","shops"), ("period", period), ("step", step), ("weather","0")]
    r = requests.post(api_url, params=params, timeout=40)
    status = r.status_code
    text_preview = r.text[:2000] if r.text else ""
    try: js = r.json()
    except Exception: js = {}
    req_info = {"url": api_url, "params_list": params}
    return js, req_info, status, text_preview

# === PARSER: geneste dates-structuur + aggregatie naar 1 regel per winkel ===
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
        agg = {f: ("sum" if f in ["count_in","turnover"] else "max" if f=="sq_meter" else "mean") for f in fields}
        return df.groupby("shop_id", as_index=False).agg(agg)

    if "data" in payload and isinstance(payload["data"], dict):
        flat = []
        for sid in shop_ids:
            rec = payload["data"].get(str(sid), {}) or {}
            row = {"shop_id": sid}
            for f in fields: row[f] = _to_float(rec.get(f))
            flat.append(row)
        return pd.DataFrame(flat)

    return pd.DataFrame()

# === RUN ===
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
        df["CSm2I"] = df["actual_spsqm"] / (df["expected_spsqm"] + eps)  # index; 1.00 = op niveau

        # Target CSm²I (slider of benchmark)
        if use_benchmark and benchmark_store_id in df["shop_id"].values:
            benchmark_row = df.loc[df["shop_id"] == benchmark_store_id]
            target_csm2i_eff = float((benchmark_row["actual_spsqm"] / (benchmark_row["expected_spsqm"] + eps)).iloc[0])
            target_label = f"benchmark: {benchmark_store_name} (CSm²I {target_csm2i_eff:.2f})"
        else:
            target_csm2i_eff = float(target_csm2i)  # van slider
            target_label = f"CSm²I target {target_csm2i_eff:.2f}"

        # Uplift t.o.v. CSm²I-target
        df["uplift_eur_csm"] = np.maximum(0.0, (target_csm2i_eff * df["expected_spsqm"] - df["actual_spsqm"])) * sq
        total_csm_uplift = float(df["uplift_eur_csm"].sum())

        # 🎯 Conversie‑uplift met slider (ongeacht benchmark)
        conv_target_frac = float(target_conv_pct) / 100.0
        spv_current = df["sales_per_visitor"].astype(float)
        spt = df["sales_per_transaction"].astype(float)
        spv_target = conv_target_frac * spt
        delta_spv = np.maximum(0.0, spv_target - spv_current)
        df["uplift_eur_conv"] = delta_spv * df["count_in"].astype(float)
        total_conv_uplift = float(df["uplift_eur_conv"].sum())

        # Totaal
        df["uplift_total"] = df["uplift_eur_csm"] + df["uplift_eur_conv"]
        total_all = float(df["uplift_total"].sum())

        # === KPI‑BANNERS ===
        st.markdown(f"""
            <div style='display:flex; gap:12px; flex-wrap:wrap;'>
              <div style='flex:1; min-width:260px; background:#FEAC76; color:#000; padding:1rem; border-radius:12px; text-align:center; font-weight:600;'>
                🚀 CSm²I potential <div style='opacity:.75'>({period}, target {target_label})</div>
                <div style='font-size:1.3rem; margin-top:.25rem;'>{_eur(total_csm_uplift, 0)}</div>
              </div>
              <div style='flex:1; min-width:260px; background:#FEAC76; color:#000; padding:1rem; border-radius:12px; text-align:center; font-weight:600;'>
                🎯 Conversion potential <div style='opacity:.75'>({period}, target conv = {target_conv_pct}%)</div>
                <div style='font-size:1.3rem; margin-top:.25rem;'>{_eur(total_conv_uplift, 0)}</div>
              </div>
              <div style='flex:1; min-width:260px; background:#FEAC76; color:#000; padding:1rem; border-radius:12px; text-align:center; font-weight:700;'>
                Σ Total potential <div style='opacity:.75'>({period})</div>
                <div style='font-size:1.35rem; margin-top:.25rem;'>{_eur(total_all, 0)}</div>
              </div>
            </div>
        """, unsafe_allow_html=True)

        # === TABEL (opgeschoond + EU formats) ===
        table = pd.DataFrame({
            "Store": df["Store"],
            "Square meters": sq.round(0).astype("Int64"),
            "Current Avg Sales per sqm": df["actual_spsqm"].round(2).map(lambda v: _eur(v, 2)),
            "CSm²I (index)": df["CSm2I"].round(2),
            "Uplift from CSm²I (€)": df["uplift_eur_csm"].round(0).map(lambda v: _eur(v, 0)),
            "Uplift from Conversion (€)": df["uplift_eur_conv"].round(0).map(lambda v: _eur(v, 0)),
            "Total Potential Uplift (€)": df["uplift_total"].round(0).map(lambda v: _eur(v, 0)),
        }).sort_values("Total Potential Uplift (€)", ascending=False)
        st.subheader("🏆 Stores with most potential")
        st.dataframe(table, use_container_width=True)

        # === BAR CHART (CSm²I uplift, kleur #762181) ===
        chart_df = df[["Store","uplift_eur_csm","CSm2I"]].copy()
        chart_df["uplift_fmt"] = chart_df["uplift_eur_csm"].map(lambda v: _eur(v, 0))
        fig = px.bar(
            chart_df.sort_values("uplift_eur_csm", ascending=False).head(20),
            x="Store", y="uplift_eur_csm",
            custom_data=["uplift_fmt","CSm2I"],
            color_discrete_sequence=["#762181"]
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>CSm²I uplift: %{customdata[0]}<br>CSm²I: %{customdata[1]:.2f}<extra></extra>")
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=60),
                          yaxis_title="CSm²I potential (€)")
        st.plotly_chart(fig, use_container_width=True)

        # === Extra bar chart (Conversion potential) ===
        conv_chart = df[["Store","uplift_eur_conv"]].copy()
        conv_chart["uplift_fmt"] = conv_chart["uplift_eur_conv"].map(lambda v: _eur(v, 0))
        figc = px.bar(
            conv_chart.sort_values("uplift_eur_conv", ascending=False).head(20),
            x="Store", y="uplift_eur_conv",
            custom_data=["uplift_fmt"],
            color_discrete_sequence=["#9E77ED"]
        )
        figc.update_traces(hovertemplate="<b>%{x}</b><br>Conversion uplift: %{customdata[0]}<extra></extra>")
        figc.update_layout(margin=dict(l=10,r=10,t=30,b=60),
                           yaxis_title="Conversion potential (€)")
        st.plotly_chart(figc, use_container_width=True)

        # === SCATTER (Viridis op CSm²I + symbolen voor driver) ===
        sc = df[["Store","sales_per_visitor","visitors_per_sqm","CSm2I","uplift_eur_csm","uplift_eur_conv","uplift_total"]].copy()
        # Bepaal dominante driver
        def driver(row):
            if row["uplift_eur_csm"] > row["uplift_eur_conv"] * 1.1:  # 10% marge
                return "CSm²I-led"
            elif row["uplift_eur_conv"] > row["uplift_eur_csm"] * 1.1:
                return "Conversion-led"
            else:
                return "Mixed"
        sc["Driver"] = sc.apply(driver, axis=1)
        x_rng = smart_range(sc["sales_per_visitor"])
        y_rng = smart_range(sc["visitors_per_sqm"])

        fig2 = px.scatter(
            sc, x="sales_per_visitor", y="visitors_per_sqm",
            size="uplift_total", color="CSm2I", color_continuous_scale="Viridis",
            symbol="Driver", symbol_sequence=["circle","diamond","triangle-up"],
            hover_data={"Store": True, "sales_per_visitor": ":.2f", "visitors_per_sqm": ":.2f", "Driver": True},
            custom_data=["Store","uplift_total","CSm2I","Driver"]
        )
        fig2.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Total uplift: %{customdata[1]:,.0f}<br>"
                          "CSm²I: %{customdata[2]:.2f}<br>"
                          "Driver: %{customdata[3]}<br>"
                          "SPV: %{x:.2f}<br>"
                          "Visitors/m²: %{y:.2f}<extra></extra>"
        )
        fig2.update_layout(
            margin=dict(l=10,r=10,t=30,b=10),
            xaxis_title="Sales per Visitor", yaxis_title="Visitors per m²",
            xaxis=dict(range=x_rng) if x_rng else None,
            yaxis=dict(range=y_rng) if y_rng else None,
            coloraxis_colorbar=dict(title="CSm²I (index)")
        )
        st.plotly_chart(fig2, use_container_width=True)

        # === KORTE TOELICHTING ===
        st.markdown(
            f"""
            **Toelichting**  
            - **CSm²I (index)** = *actual sales per m²* ÷ *expected sales per m²*. Verwacht = *(sales per visitor) × (visitors per m²)*.  
            - **CSm²I potential (€)** = *(target_CSm²I × expected − actual) × m²*: extra omzet in **{period}** als de winkel naar het target groeit.  
            - **Conversion potential (€)** = *(target_conv% × SPT − SPV) × bezoekers*: extra omzet in **{period}** bij verhogen van conversie naar {target_conv_pct}%.  
            - **Total** = som van beide componenten; geen dubbelcounting (CSm²I‑ en conversie‑uplift worden apart berekend).  
            """
        )
