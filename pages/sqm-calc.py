# streamlit_sales_per_sqm_potential.py
import os
import sys
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# Voeg parent directory toe aan sys.path voor imports
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

from shop_mapping import SHOP_NAME_MAP

# === PAGE CONFIG ===
st.set_page_config(page_title="Sales-per-sqm Potentieel", page_icon="📊", layout="wide")

# === Styling (uniform) ===
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
        font-family: "Instrument Sans", sans-serif !important;
        padding: 0.6rem 1.4rem !important;
        border: none !important;
        box-shadow: none !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #d13c30 !important;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Sales-per-sqm Potentieel (CSm²I)")
st.caption("Analyseer locaties op basis van sales per m² en identificeer potentieel bij CSm²I = 1,00.")

# === API CONFIG ===
try:
    API_URL = st.secrets["API_URL"]
except KeyError:
    st.error("❌ API_URL ontbreekt in Streamlit secrets.")
    st.stop()

# === INPUTS ===
period = st.selectbox("Periode", [
    "yesterday", "today", "this_week", "last_week",
    "this_month", "last_month", "this_quarter", "last_quarter",
    "this_year", "last_year"
], index=5)

NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}
default_names = list(NAME_TO_ID.keys())

selected_names = st.multiselect("Select stores", options=list(NAME_TO_ID.keys()), default=default_names)
shop_ids = [NAME_TO_ID[n] for n in selected_names]

if not shop_ids:
    st.warning("Selecteer minimaal één store.")
    st.stop()

# === API CALL FUNCTION ===
def get_kpi_data_for_stores(shop_ids, period="last_month"):
    params = []
    for sid in shop_ids:
        params.append(("data[]", sid))
    metrics = [
        "count_in", "sales_per_visitor", "sales_per_sqm",
        "conversion_rate", "sales_per_transaction",
        "turnover", "sq_meter"
    ]
    for m in metrics:
        params.append(("data_output[]", m))

    params += [
        ("source", "shops"),
        ("period", period),
        ("period_step", "day"),  # Hardcoded
        ("weather", "0"),
        ("step", "day")
    ]

    try:
        response = requests.post(API_URL, params=params)
        if response.status_code == 200:
            full_response = response.json()
            if "data" in full_response and period in full_response["data"]:
                return normalize_vemcount_response(full_response["data"][period])
            else:
                st.error("⚠️ Geen 'data' gevonden in API-response.")
        else:
            st.error(f"❌ Error fetching data: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"🚨 API call exception: {e}")
    return pd.DataFrame()

# === NORMALIZE RESPONSE ===
def normalize_vemcount_response(response_json: dict) -> pd.DataFrame:
    rows = []
    for shop_id, shop_content in response_json.items():
        sq_meter = None
        if "data" in shop_content and "sq_meter" in shop_content["data"]:
            sq_meter = float(shop_content["data"]["sq_meter"] or 0)

        dates = shop_content.get("dates", {})
        for date_label, day_info in dates.items():
            data = day_info.get("data", {})
            row = {
                "shop_id": int(shop_id),
                "date": data.get("dt"),
                "sq_meter": sq_meter,
                "count_in": float(data.get("count_in") or 0),
                "sales_per_visitor": float(data.get("sales_per_visitor") or 0),
                "sales_per_sqm": float(data.get("sales_per_sqm") or 0),
                "conversion_rate": float(data.get("conversion_rate") or 0),
                "sales_per_transaction": float(data.get("sales_per_transaction") or 0),
                "turnover": float(data.get("turnover") or 0)
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

# === RUN ANALYSIS ===
if st.button("Analyseer", type="secondary"):
    df = get_kpi_data_for_stores(shop_ids, period=period)

    if not df.empty:
        # Aggregate per store
        df_group = df.groupby("shop_id").agg({
            "sq_meter": "first",
            "count_in": "sum",
            "sales_per_visitor": "mean",
            "sales_per_sqm": "mean",
            "conversion_rate": "mean",
            "sales_per_transaction": "mean",
            "turnover": "sum"
        }).reset_index()

        df_group["store_name"] = df_group["shop_id"].map(ID_TO_NAME)
        df_group["visitors_per_sqm"] = df_group["count_in"] / df_group["sq_meter"].replace(0, np.nan)
        df_group["expected_spsqm"] = df_group["sales_per_visitor"] * df_group["visitors_per_sqm"]
        df_group["CSm2I"] = df_group["sales_per_sqm"] / df_group["expected_spsqm"].replace(0, np.nan)
        df_group["uplift_eur"] = np.maximum(0.0, df_group["expected_spsqm"] - df_group["sales_per_sqm"]) * df_group["sq_meter"]

        total_extra_turnover = df_group["uplift_eur"].sum()

        # === TOP PANEL ===
        st.markdown(f"""
            <div style='background-color: #FEAC76;
                        padding: 1.5rem;
                        border-radius: 0.75rem;
                        font-size: 1.25rem;
                        font-weight: 600;
                        text-align: center;
                        margin-bottom: 1.5rem;'>
                🚀 Potentieel extra omzet ({period} bij CSm²I = 1,00): 
                <span style='font-size:1.5rem;'>€{str(f"{total_extra_turnover:,.0f}").replace(",", ".")}</span>
            </div>
        """, unsafe_allow_html=True)

        # === BAR CHART ===
        chart_df = df_group.sort_values("uplift_eur", ascending=False)
        fig_bar = px.bar(chart_df, x="store_name", y="uplift_eur", text="uplift_eur",
                         color_discrete_sequence=["#762181"])
        fig_bar.update_traces(texttemplate="€%{y:,.0f}", textposition="outside")
        fig_bar.update_yaxes(title="Potential Revenue Uplift (€)", tickprefix="€")
        st.plotly_chart(fig_bar, use_container_width=True)

        # === SCATTER CHART ===
        fig_scatter = px.scatter(
            df_group, x="sales_per_visitor", y="visitors_per_sqm",
            size="uplift_eur", color="CSm2I",
            color_continuous_scale="Viridis",
            hover_name="store_name"
        )
        fig_scatter.update_layout(
            xaxis=dict(range=[df_group["sales_per_visitor"].min() * 0.95,
                              df_group["sales_per_visitor"].max() * 1.05]),
            yaxis_title="Visitors per m²",
            xaxis_title="Sales per Visitor"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # === TABLE ===
        table_df = df_group[[
            "store_name", "sq_meter", "sales_per_sqm", "CSm2I", "uplift_eur"
        ]].copy()
        table_df.rename(columns={
            "store_name": "Store",
            "sq_meter": "Square meters",
            "sales_per_sqm": "Current Avg Sales per sqm",
            "uplift_eur": "Potential Revenue Uplift (€)"
        }, inplace=True)
        table_df["Potential Revenue Uplift (€)"] = table_df["Potential Revenue Uplift (€)"].map(lambda x: f"€{x:,.0f}".replace(",", "."))
        st.dataframe(table_df, use_container_width=True)

        st.caption("📌 CSm²I = Current Sales per sqm Index — verhouding actuele / verwachte omzet/m². Potentieel berekend als extra omzet indien CSm²I minimaal 1,00 wordt.")
    else:
        st.warning("Geen data opgehaald.")
