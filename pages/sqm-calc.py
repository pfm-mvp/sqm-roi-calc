# streamlit_sales_per_sqm_potential.py
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
from shop_mapping import SHOP_NAME_MAP

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="Sales-per-sqm Potentieel", page_icon="📊", layout="wide")

# ✅ Styling
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
        transition: background-color 0.2s ease-in-out;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #d13c30 !important;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Config & Mapping
# =========================
API_URL = st.secrets["API_URL"]
NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}
DEFAULT_SHOP_IDS = list(SHOP_NAME_MAP.keys())
default_names = [ID_TO_NAME.get(shop_id, str(shop_id)) for shop_id in DEFAULT_SHOP_IDS]

# =========================
# Inputs
# =========================
period_options = [
    "yesterday", "today", "this_week", "last_week",
    "this_month", "last_month", "this_quarter", "last_quarter",
    "this_year", "last_year"
]
period = st.selectbox("Select period", options=period_options, index=5)

selected_names = st.multiselect("Select stores", options=list(NAME_TO_ID.keys()), default=default_names)
shop_ids = [NAME_TO_ID[name] for name in selected_names]

if not shop_ids:
    st.warning("Please select at least one store.")
    st.stop()

# =========================
# API Call
# =========================
def get_sales_per_sqm_data(shop_ids, period):
    params = [("data[]", sid) for sid in shop_ids]
    metrics = [
        "count_in", "sales_per_visitor", "sales_per_sqm",
        "conversion_rate", "sales_per_transaction", "turnover", "sq_meter"
    ]
    params += [("data_output[]", m) for m in metrics]
    params += [
        ("source", "shops"),
        ("period", period),
        ("period_step", "day"),
        ("weather", "0"),
        ("step", "day")
    ]

    try:
        response = requests.post(API_URL, params=params)
        if response.status_code == 200:
            full_response = response.json()
            if "data" in full_response and period in full_response["data"]:
                raw_data = full_response["data"][period]
                return parse_nested_response(raw_data)
        else:
            st.error(f"❌ API error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"🚨 API call failed: {e}")
    return pd.DataFrame()

# =========================
# Parse Nested JSON
# =========================
def parse_nested_response(response_json: dict) -> pd.DataFrame:
    rows = []
    for shop_id, shop_content in response_json.items():
        sq_meter = None
        if isinstance(shop_content.get("data"), dict):
            sq_meter = float(shop_content["data"].get("sq_meter", np.nan))
        dates = shop_content.get("dates", {})
        for _, day_info in dates.items():
            data = day_info.get("data", {})
            row = {
                "shop_id": int(shop_id),
                "date": data.get("dt"),
                "count_in": float(data.get("count_in", 0)),
                "sales_per_visitor": float(data.get("sales_per_visitor", 0)),
                "sales_per_sqm": float(data.get("sales_per_sqm", 0)),
                "conversion_rate": float(data.get("conversion_rate", 0)),
                "sales_per_transaction": float(data.get("sales_per_transaction", 0)),
                "turnover": float(data.get("turnover", 0)),
                "sq_meter": sq_meter
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

# =========================
# Analyse
# =========================
if st.button("Analyseer", type="secondary"):
    df = get_sales_per_sqm_data(shop_ids, period)
    if not df.empty:
        df_grouped = df.groupby("shop_id").agg({
            "count_in": "sum",
            "sales_per_visitor": "mean",
            "sales_per_sqm": "mean",
            "conversion_rate": "mean",
            "sales_per_transaction": "mean",
            "turnover": "sum",
            "sq_meter": "first"
        }).reset_index()
        df_grouped["store"] = df_grouped["shop_id"].map(ID_TO_NAME)
        df_grouped["visitors_per_sqm"] = df_grouped["count_in"] / df_grouped["sq_meter"]
        df_grouped["expected_spsqm"] = df_grouped["sales_per_visitor"] * df_grouped["visitors_per_sqm"]
        df_grouped["CSm2I"] = (df_grouped["sales_per_sqm"] / df_grouped["expected_spsqm"]) * 100
        df_grouped["uplift_eur"] = np.maximum(0.0, df_grouped["expected_spsqm"] - df_grouped["sales_per_sqm"]) * df_grouped["sq_meter"]

        total_uplift = df_grouped["uplift_eur"].sum()

        # 🚀 Banner bovenaan
        st.markdown(f"""
            <div style='background-color: #FEAC76;
                        color: #000000;
                        padding: 1.5rem;
                        border-radius: 0.75rem;
                        font-size: 1.25rem;
                        font-weight: 600;
                        text-align: center;
                        margin-bottom: 1.5rem;'>
                🚀 Potential annual revenue growth: <span style='font-size:1.5rem;'>€{str(f"{total_uplift:,.0f}").replace(",", ".")}</span>
            </div>
         """, unsafe_allow_html=True)

        # 📊 Bar chart
        fig_bar = px.bar(
            df_grouped.sort_values("uplift_eur", ascending=False),
            x="store", y="uplift_eur",
            hover_data=["CSm2I"],
            color_discrete_sequence=["#762181"]
        )
        fig_bar.update_layout(
            yaxis_tickprefix="€",
            yaxis_tickformat=",",
            xaxis_title=None
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # 🧭 Scatter chart
        fig_scatter = px.scatter(
            df_grouped,
            x="sales_per_visitor", y="visitors_per_sqm",
            size="uplift_eur", color="CSm2I",
            color_continuous_scale="Viridis",
            hover_data=["store"]
        )
        fig_scatter.update_layout(
            coloraxis_colorbar=dict(title="CSm²I (%)"),
            xaxis=dict(range=[df_grouped["sales_per_visitor"].min() * 0.95, df_grouped["sales_per_visitor"].max() * 1.05])
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 📋 Tabel
        df_grouped["Potential Extra Revenue (€)"] = df_grouped["uplift_eur"].map(lambda x: f"€{x:,.0f}".replace(",", "."))
        df_grouped["Current Avg Sales per sqm"] = df_grouped["sales_per_sqm"].map(lambda x: f"€{x:,.2f}".replace(",", "."))
        df_grouped["CSm2I (%)"] = df_grouped["CSm2I"].map(lambda x: f"{x:.1f}%")

        table_cols = ["store", "sq_meter", "Current Avg Sales per sqm", "CSm2I (%)", "Potential Extra Revenue (€)"]
        st.dataframe(df_grouped[table_cols].rename(columns={"store": "Store", "sq_meter": "Square meters"}), use_container_width=True)

        st.caption("💡 *Potential Extra Revenue* is calculated as the annualised additional revenue if each store reached its expected sales per sqm.")
    else:
        st.warning("No data returned for the selected period/stores.")
