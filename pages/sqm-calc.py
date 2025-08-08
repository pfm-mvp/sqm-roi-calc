# pages/streamlit_sales_per_sqm_potential.py
import os, sys
import numpy as np
import pandas as pd
# streamlit_sales_per_sqm_potential.py
import os
import sys
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# === Imports / mapping
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from fpdf import FPDF
from shop_mapping import SHOP_NAME_MAP

# === PAGE CONFIG ===
st.set_page_config(page_title="Sales-per-sqm Potentieel (CSm²I)", page_icon="📊", layout="wide")

# === Styling (paarse pills + PFM-rode knop; uniform met andere apps)
# === GLOBAL STYLING ===
st.markdown(
"""
   <style>
    /* Font import */
   @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');

    /* Forceer Instrument Sans overal */
   html, body, [class*="css"] {
       font-family: 'Instrument Sans', sans-serif !important;
   }

    /* 🎨 Multiselect pills in paars */
   [data-baseweb="tag"] {
       background-color: #9E77ED !important;
       color: white !important;
   }

    /* 🔴 Rode knop (cover zowel primary als secondary) */
    button[kind="secondary"], button[kind="primary"], button[data-testid="stBaseButton-secondary"], .stButton>button {
    button[data-testid="stBaseButton-secondary"] {
       background-color: #F04438 !important;
       color: white !important;
       border-radius: 16px !important;
       font-weight: 600 !important;
        font-family: "Instrument Sans", sans-serif !important;
       padding: 0.6rem 1.4rem !important;
       border: none !important;
        box-shadow: none !important;
        transition: background-color 0.2s ease-in-out;
   }
    button[kind="secondary"]:hover, button[kind="primary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover, .stButton>button:hover {
    button[data-testid="stBaseButton-secondary"]:hover {
       background-color: #d13c30 !important;
       cursor: pointer;
   }

    /* Compacte page paddings */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
   </style>
   """,
unsafe_allow_html=True
)

st.title("📊 Sales-per-sqm Potentieel (CSm²I)")
st.caption("Selecteer winkels en periode. NVO (sq_meter) via API. Berekening op dag-niveau (STEP='day').")
# === CONSTANTS ===
NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}
DEFAULT_SHOP_IDS = list(SHOP_NAME_MAP.keys())

# === Secrets
API_URL = st.secrets["API_URL"]

# === UI inputs (alleen periode + stores)
colA, colB = st.columns([1,1])
with colA:
    PRESETS = ["yesterday","today","this_week","last_week","this_month","last_month",
               "this_quarter","last_quarter","this_year","last_year"]
# === UI: Select period & stores ===
col1, col2 = st.columns([1, 2])
with col1:
    PRESETS = [
        "yesterday","today","this_week","last_week",
        "this_month","last_month","this_quarter","last_quarter",
        "this_year","last_year"
    ]
period = st.selectbox("Periode", options=PRESETS, index=5)
with colB:
    pass  # ruimte voor evt. filter later

# Hardcode step (geen input)
STEP = "day"
with col2:
    default_names = [ID_TO_NAME.get(sid, str(sid)) for sid in DEFAULT_SHOP_IDS]
    selected_names = st.multiselect("Select stores", options=list(NAME_TO_ID.keys()), default=default_names)
    shop_ids = [NAME_TO_ID[name] for name in selected_names]

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
# === FUNCTIONS ===
def get_kpi_data_for_stores(shop_ids, period="last_year", step="day"):
    params = [("data[]", shop_id) for shop_id in shop_ids]
params += [
        ("data_output[]", "count_in"),
        ("data_output[]", "sales_per_visitor"),
        ("data_output[]", "sales_per_sqm"),
        ("data_output[]", "conversion_rate"),
        ("data_output[]", "sales_per_transaction"),
        ("data_output[]", "turnover"),
        ("data_output[]", "sq_meter"),
("source", "shops"),
("period", period),
        ("step", step),
        ("period_step", step),
("weather", "0"),
        ("step", step)
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

def _eur(num, decimals=2):
    if pd.isna(num):
        return ""
    s = f"{float(num):,.{decimals}f}"
    return "€" + s.replace(",", "X").replace(".", ",").replace("X", ".")

# === Parser (genest + plat) met aggregatie naar 1 regel per winkel
def parse_vemcount(payload: dict, shop_ids: list[int], fields: list[str], period_key: str) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()

    # Geneste structuur: payload["data"][period_key][shop_id]["dates"][...]["data"]
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
                # sq_meter fallback uit shop meta indien dag-level ontbreekt
                if row.get("sq_meter") is None or np.isnan(row.get("sq_meter")):
                    sm = ((shop_block.get("data") or {}).get("sq_meter"))
                    row["sq_meter"] = _to_float(sm)
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Aggregatie: som voor telwaarden, gemiddelde voor ratio's, max voor sq_meter
        agg = {}
        for f in fields:
            if f in ["count_in", "turnover"]:
                agg[f] = "sum"
            elif f in ["sq_meter"]:
                agg[f] = "max"
            else:
                agg[f] = "mean"
        return df.groupby("shop_id", as_index=False).agg(agg)

    # Platte structuur {"data": {"<shop_id>": {metric: value}}}
    if "data" in payload and isinstance(payload["data"], dict):
        flat = []
        for sid in shop_ids:
            rec = payload["data"].get(str(sid), {}) or {}
            row = {"shop_id": sid}
            for f in fields:
                row[f] = _to_float(rec.get(f))
            flat.append(row)
        return pd.DataFrame(flat)

        response = requests.post(API_URL, params=params)
        if response.status_code == 200:
            full_response = response.json()
            if "data" in full_response and period in full_response["data"]:
                raw_data = full_response["data"][period]
                return normalize_sales_per_sqm(raw_data)
        else:
            st.error(f"❌ Error fetching data: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"🚨 API call exception: {e}")
return pd.DataFrame()

# === Analyse-run (uniforme workflow: Run simulation)
if st.button("Run simulation"):
    with st.spinner("Calculating hidden location potential..."):
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

            # Debug (consistent met je andere apps)
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
            df["Store"] = df["shop_id"].map(ID_TO_NAME)
            sq = df["sq_meter"].astype(float).replace(0, np.nan)

            df["visitors_per_sqm"] = df["count_in"].astype(float) / sq
            df["expected_spsqm"]   = df["sales_per_visitor"].astype(float) * df["visitors_per_sqm"]

            actual_chk = df["turnover"].astype(float) / sq
            sales_sqm = df.get("sales_per_sqm", pd.Series(np.nan, index=df.index)).astype(float)
            df["actual_spsqm"] = np.where(sales_sqm.notna(), sales_sqm, actual_chk)

            eps = 1e-9
            df["CSm2I"] = df["actual_spsqm"] / (df["expected_spsqm"] + eps)  # index: 1.00 = op lijn
            df["uplift_eur"] = np.maximum(0.0, df["expected_spsqm"] - df["actual_spsqm"]) * sq

            # === KPI-banner bovenaan
            total_extra_turnover = float(df["uplift_eur"].sum())
            st.markdown(f"""
                <div style='background-color: #FEAC76;
                            color: #000000;
                            padding: 1.5rem;
                            border-radius: 0.75rem;
                            font-size: 1.25rem;
                            font-weight: 600;
                            text-align: center;
                            margin-bottom: 1.5rem;'>
                    🚀 The potential revenue growth is
                    <span style='font-size:1.5rem;'>{_eur(total_extra_turnover, 0)}</span>
                </div>
            """, unsafe_allow_html=True)

            # === Tabel (opgeschoond + EU formats)
            df_view = pd.DataFrame({
                "Store": df["Store"],
                "Square meters": (sq).round(0).astype("Int64"),
                "Current Avg Sales per sqm": df["actual_spsqm"].round(2).map(lambda v: _eur(v, 2)),
                "CSm²I (index)": df["CSm2I"].round(2),
                "Potential revenue uplift (€)": df["uplift_eur"].round(0).map(lambda v: _eur(v, 0)),
def normalize_sales_per_sqm(response_json: dict) -> pd.DataFrame:
    rows = []
    for shop_id, shop_content in response_json.items():
        sq_meter = float(shop_content.get("data", {}).get("sq_meter", np.nan))
        dates = shop_content.get("dates", {})
        for date_label, day_info in dates.items():
            data = day_info.get("data", {})
            rows.append({
                "shop_id": int(shop_id),
                "date": pd.to_datetime(data.get("dt")),
                "sq_meter": sq_meter,
                "count_in": float(data.get("count_in", 0)),
                "sales_per_visitor": float(data.get("sales_per_visitor", 0)),
                "sales_per_sqm": float(data.get("sales_per_sqm", 0)),
                "conversion_rate": float(data.get("conversion_rate", 0)),
                "sales_per_transaction": float(data.get("sales_per_transaction", 0)),
                "turnover": float(data.get("turnover", 0)),
})
            st.subheader("🏆 Stores with most potential (sorted by uplift)")
            st.dataframe(
                df_view.sort_values("Potential revenue uplift (€)", ascending=False),
                use_container_width=True
            )

            # === Bar chart (EU hover, betere schaal)
            chart_df = df[["Store","uplift_eur","CSm2I"]].copy()
            chart_df["uplift_eur_fmt"] = chart_df["uplift_eur"].map(lambda v: _eur(v, 0))
            fig = px.bar(
                chart_df.sort_values("uplift_eur", ascending=False).head(20),
                x="Store", y="uplift_eur",
                custom_data=["uplift_eur_fmt","CSm2I"]
            )
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>Uplift: %{customdata[0]}<br>CSm²I: %{customdata[1]:.2f}<extra></extra>"
            )
            fig.update_layout(
                margin=dict(l=10,r=10,t=30,b=60),
                yaxis_title="Potential revenue uplift (€)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # === Scatter (EU hover + slimme x-as range)
            sc = df[["Store","sales_per_visitor","visitors_per_sqm","CSm2I","uplift_eur"]].copy()
            sc["uplift_eur_fmt"] = sc["uplift_eur"].map(lambda v: _eur(v, 0))

            # x-as range smart: 5% marge rond min/max; geen onnodige whitespace
            x_min = float(np.nanmin(sc["sales_per_visitor"]))
            x_max = float(np.nanmax(sc["sales_per_visitor"]))
            span = max(0.01, x_max - x_min)
            pad = span * 0.05
            x_range = [x_min - pad, x_max + pad]

            fig2 = px.scatter(
                sc, x="sales_per_visitor", y="visitors_per_sqm",
                size="uplift_eur", color="CSm2I",
                hover_data={"Store": True, "sales_per_visitor": ":.2f", "visitors_per_sqm": ":.2f"},
                custom_data=["Store","uplift_eur_fmt","CSm2I"]
            )
            fig2.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "Uplift: %{customdata[1]}<br>"
                              "CSm²I: %{customdata[2]:.2f}<br>"
                              "SPV: %{x:.2f}<br>"
                              "Visitors/m²: %{y:.2f}"
                              "<extra></extra>"
            )
            fig2.update_layout(
                margin=dict(l=10,r=10,t=30,b=10),
                xaxis_title="Sales per Visitor",
                yaxis_title="Visitors per m²",
                xaxis=dict(range=x_range)
            )
            st.plotly_chart(fig2, use_container_width=True)

            # === Korte toelichting
            st.markdown(
                """
                **Toelichting**  
                - **CSm²I (index)**: verhouding tussen gerealiseerde en verwachte omzet per m².  
                  *1,00* = conform verwachting; *<1,00* betekent onderprestatie.  
                - **Potential revenue uplift (€)**: indicatie van extra omzet wanneer winkel naar **verwacht niveau** groeit.
                - Verwacht niveau = *(sales per visitor) × (visitors per m²)*.
                """
            )

        except Exception as e:
            st.error(f"Onverwachte fout: {e}")
    return pd.DataFrame(rows)

def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Sales per sqm Potential Report", ln=True)
    pdf.ln(5)
    for i, row in df.iterrows():
        pdf.cell(0, 8, f"{row['Store']}: €{row['Potential Extra Revenue (€)']:,.2f}", ln=True)
    return pdf.output(dest="S").encode("latin1")

# === RUN SIMULATION ===
if st.button("Analyseer", type="secondary"):
    with st.spinner("Calculating hidden location potential..."):
        df = get_kpi_data_for_stores(shop_ids, period=period, step="day")

    if not df.empty:
        df["Store"] = df["shop_id"].map(ID_TO_NAME)
        df["visitors_per_sqm"] = df["count_in"] / df["sq_meter"].replace(0, np.nan)
        df["expected_spsqm"] = df["sales_per_visitor"] * df["visitors_per_sqm"]
        df["CSm2I"] = (df["sales_per_sqm"] / df["expected_spsqm"]).replace([np.inf, -np.inf], np.nan)
        df["Potential Extra Revenue (€)"] = np.maximum(0, df["expected_spsqm"] - df["sales_per_sqm"]) * df["sq_meter"]
        total_extra = df["Potential Extra Revenue (€)"].sum()

        st.markdown(f"""
        <div style='background-color: #FEAC76; color: #000000;
                    padding: 1.5rem; border-radius: 0.75rem;
                    font-size: 1.25rem; font-weight: 600;
                    text-align: center; margin-bottom: 1.5rem;'>
            🚀 The potential revenue growth is 
            <span style='font-size:1.5rem;'>€{str(f"{total_extra:,.0f}").replace(",", ".")}</span>
        </div>
        """, unsafe_allow_html=True)

        # === Bar Chart (Uplift) ===
        chart_df = df.groupby("Store", as_index=False)["Potential Extra Revenue (€)"].sum()
        fig_bar = px.bar(chart_df.sort_values("Potential Extra Revenue (€)", ascending=False),
                         x="Store", y="Potential Extra Revenue (€)",
                         color_discrete_sequence=["#762181"])
        fig_bar.update_yaxes(tickprefix="€", separatethousands=True)
        st.plotly_chart(fig_bar, use_container_width=True)

        # === Scatter Chart ===
        fig_scatter = px.scatter(df, x="sales_per_visitor", y="visitors_per_sqm",
                                 size="Potential Extra Revenue (€)",
                                 color="Store", color_continuous_scale="Viridis",
                                 hover_data=["Store", "CSm2I"])
        fig_scatter.update_xaxes(range=[df["sales_per_visitor"].min()*0.95,
                                        df["sales_per_visitor"].max()*1.05])
        st.plotly_chart(fig_scatter, use_container_width=True)

        # === Table ===
        table_cols = ["Store", "sq_meter", "sales_per_sqm", "CSm2I", "Potential Extra Revenue (€)"]
        df_table = df.groupby("Store", as_index=False).agg({
            "sq_meter": "mean",
            "sales_per_sqm": "mean",
            "CSm2I": "mean",
            "Potential Extra Revenue (€)": "sum"
        })
        df_table.rename(columns={
            "sq_meter": "Square meters",
            "sales_per_sqm": "Current Avg Sales per sqm"
        }, inplace=True)
        st.dataframe(df_table.style.format({
            "Square meters": "{:,.0f}",
            "Current Avg Sales per sqm": "€{:,.2f}",
            "CSm2I": "{:.2f}",
            "Potential Extra Revenue (€)": "€{:,.2f}"
        }), use_container_width=True)

        st.caption("💡 'Potential Extra Revenue (€)' represents the annualized uplift potential "
                   "based on the selected period's performance vs expected benchmark.")

        # === PDF Export ===
        pdf_bytes = generate_pdf(df_table)
        st.download_button("Download PDF", data=pdf_bytes, file_name="sales_per_sqm_potential.pdf", mime="application/pdf")
    else:
        st.warning("No data returned for the selected period.")
