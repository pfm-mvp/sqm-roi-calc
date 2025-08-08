# streamlit_sales_per_sqm_potential.py
import os
import sys
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from fpdf import FPDF

# 👇 Zet dit vóór de import!
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

# ✅ Nu pas importeren
from shop_mapping import SHOP_NAME_MAP

# === PAGE CONFIG ===
st.set_page_config(page_title="Sales-per-sqm Potentieel (CSm²I)", page_icon="📊", layout="wide")

# === GLOBAL STYLING ===
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Instrument Sans', sans-serif !important;
    }
    [data-baseweb="tag"] {
        background-color: #9E77ED !important;
        color: white !important;
    }
    button[data-testid="stBaseButton-secondary"] {
        background-color: #F04438 !important;
        color: white !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.4rem !important;
        border: none !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #d13c30 !important;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === CONSTANTS ===
NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}
DEFAULT_SHOP_IDS = list(SHOP_NAME_MAP.keys())

API_URL = st.secrets["API_URL"]

# === UI: Select period & stores ===
col1, col2 = st.columns([1, 2])
with col1:
    PRESETS = [
        "yesterday","today","this_week","last_week",
        "this_month","last_month","this_quarter","last_quarter",
        "this_year","last_year"
    ]
    period = st.selectbox("Periode", options=PRESETS, index=5)

with col2:
    default_names = [ID_TO_NAME.get(sid, str(sid)) for sid in DEFAULT_SHOP_IDS]
    selected_names = st.multiselect("Select stores", options=list(NAME_TO_ID.keys()), default=default_names)
    shop_ids = [NAME_TO_ID[name] for name in selected_names]

if not shop_ids:
    st.warning("Selecteer minimaal één store.")
    st.stop()

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
        ("period_step", step),
        ("weather", "0"),
        ("step", step)
    ]
    try:
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
