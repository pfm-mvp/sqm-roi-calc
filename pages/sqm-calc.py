# 📊 Sales-per-sqm Potentieel (CSm²I) met benchmark-optie

import streamlit as st
import pandas as pd
import numpy as np
import requests
from shop_mapping import SHOP_NAME_MAP

# ✅ Pagina-instellingen
st.set_page_config(page_title="Sales-per-sqm Potentieel", page_icon="📊", layout="wide")

# ✅ Styling: Instrument Sans, paarse pills & rode knop
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
    [data-baseweb="tag"] { background-color: #9E77ED !important; color: white !important; }
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
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #d13c30 !important;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# ===== API Config =====
try:
    API_URL = st.secrets["API_URL"]
except KeyError:
    st.error("API_URL ontbreekt in Streamlit secrets.")
    st.stop()

# ===== Naam <-> ID mapping =====
NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}
DEFAULT_SHOP_IDS = list(SHOP_NAME_MAP.keys())

# ===== Inputs =====
colA, colB = st.columns([1, 1])
with colA:
    period = st.selectbox("Periode", [
        "yesterday", "today", "this_week", "last_week",
        "this_month", "last_month", "this_quarter",
        "last_quarter", "this_year", "last_year"
    ], index=5)

with colB:
    selected_names = st.multiselect(
        "Select stores",
        options=list(NAME_TO_ID.keys()),
        default=[ID_TO_NAME.get(sid) for sid in DEFAULT_SHOP_IDS]
    )
    shop_ids = [NAME_TO_ID[name] for name in selected_names]

if not shop_ids:
    st.warning("Selecteer minimaal één store.")
    st.stop()

# ===== Benchmark Toggle =====
use_benchmark = st.toggle("Gebruik benchmark-winkel i.p.v. vaste CSm²I = 1.0", value=False)
benchmark_store_id = None
if use_benchmark:
    benchmark_store_name = st.selectbox(
        "Kies benchmark-winkel",
        options=selected_names
    )
    benchmark_store_id = NAME_TO_ID[benchmark_store_name]

# ===== Data ophalen =====
def fetch_report(shop_ids, period):
    params = [("data[]", sid) for sid in shop_ids]
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
        ("period_step", "day"),
        ("weather", "0"),
        ("step", "day")
    ]
    response = requests.post(API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API error {response.status_code}: {response.text}")
        return None

# ===== Normalizer =====
def normalize_response(response_json):
    rows = []
    if "data" in response_json and period in response_json["data"]:
        for sid, shop_content in response_json["data"][period].items():
            sq_meter = shop_content["data"].get("sq_meter", np.nan)
            dates = shop_content.get("dates", {})
            for date_label, day_info in dates.items():
                data = day_info.get("data", {})
                rows.append({
                    "shop_id": int(sid),
                    "date": pd.to_datetime(data.get("dt")),
                    "sq_meter": float(sq_meter),
                    "count_in": float(data.get("count_in", 0)),
                    "sales_per_visitor": float(data.get("sales_per_visitor", 0)),
                    "sales_per_sqm": float(data.get("sales_per_sqm", 0)),
                    "conversion_rate": float(data.get("conversion_rate", 0)),
                    "sales_per_transaction": float(data.get("sales_per_transaction", 0)),
                    "turnover": float(data.get("turnover", 0))
                })
    return pd.DataFrame(rows)

# ===== Analyse knop =====
if st.button("Analyseer potentieel", type="secondary"):
    raw_json = fetch_report(shop_ids, period)
    if raw_json:
        df = normalize_response(raw_json)
        if df.empty:
            st.warning("Geen data ontvangen.")
            st.stop()

        # Gemiddeld per winkel over gekozen periode
        df_grouped = df.groupby("shop_id").agg({
            "sq_meter": "first",
            "count_in": "sum",
            "sales_per_visitor": "mean",
            "sales_per_sqm": "mean",
            "conversion_rate": "mean",
            "sales_per_transaction": "mean",
            "turnover": "sum"
        }).reset_index()
        df_grouped["store_name"] = df_grouped["shop_id"].map(ID_TO_NAME)

        # Visitors per m²
        df_grouped["visitors_per_sqm"] = df_grouped["count_in"] / df_grouped["sq_meter"]

        # Verwachte sales_per_sqm
        df_grouped["expected_spsqm"] = df_grouped["sales_per_visitor"] * df_grouped["visitors_per_sqm"]

        # Bepaal target_CSm2I
        if use_benchmark and benchmark_store_id in df_grouped["shop_id"].values:
            target_csm2i = df_grouped.loc[df_grouped["shop_id"] == benchmark_store_id, "sales_per_sqm"].values[0] / \
                           df_grouped.loc[df_grouped["shop_id"] == benchmark_store_id, "expected_spsqm"].values[0]
        else:
            target_csm2i = 1.0

        # Werkelijke CSm²I
        df_grouped["CSm2I"] = df_grouped["sales_per_sqm"] / df_grouped["expected_spsqm"]

        # Potentieel berekenen
        df_grouped["uplift_eur"] = np.maximum(0, (target_csm2i * df_grouped["expected_spsqm"] - df_grouped["sales_per_sqm"])) * df_grouped["sq_meter"]

        # 🚀 Totale potentieel tonen
        total_extra_turnover = df_grouped["uplift_eur"].sum()
        st.markdown(f"""
            <div style='background-color: #FEAC76; color: #000000; padding: 1.5rem; border-radius: 0.75rem;
                        font-size: 1.25rem; font-weight: 600; text-align: center; margin-bottom: 1.5rem;'>
                🚀 Potentieel omzetgroei over geselecteerde periode: 
                <span style='font-size:1.5rem;'>€{str(f"{total_extra_turnover:,.0f}").replace(",", ".")}</span>
            </div>
        """, unsafe_allow_html=True)

        # ===== Tabel tonen =====
        st.subheader("📋 Potentieel per winkel")
        table_df = df_grouped[["store_name", "sq_meter", "sales_per_sqm", "CSm2I", "uplift_eur"]].copy()
        table_df.rename(columns={
            "store_name": "Store",
            "sq_meter": "Square meters",
            "sales_per_sqm": "Current Avg Sales per sqm",
            "uplift_eur": "Potential extra revenue (€)"
        }, inplace=True)
        table_df["Potential extra revenue (€)"] = table_df["Potential extra revenue (€)"].apply(lambda x: f"€{x:,.0f}".replace(",", "."))
        st.dataframe(table_df, use_container_width=True)

        # Toelichting
        if use_benchmark:
            st.caption(f"Berekening gebaseerd op benchmark: {benchmark_store_name} met CSm²I {target_csm2i:.2f}")
        else:
            st.caption("Berekening gebaseerd op doel CSm²I = 1.0 (theoretisch maximum)")
