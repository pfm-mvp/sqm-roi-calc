
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Sales-per-sqm Potentieel", page_icon="📊", layout="wide")

PFM_RED = "#F04438"
st.markdown(
    f"""
    <style>
    .stButton>button {{
        background-color: {PFM_RED};
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        border: none;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Sales-per-sqm Potentieel (CSm²I)")

with st.sidebar:
    st.header("Instellingen")
    fastapi_base = st.text_input("FastAPI base URL", value="https://vemcount-agent.onrender.com")
    endpoint = "/get-report"
    presets = [
        "yesterday","today",
        "this_week","last_week",
        "this_month","last_month",
        "this_quarter","last_quarter",
        "this_year","last_year"
    ]
    period = st.radio("Periode preset", options=presets, index=4)

    default_ids = [32224,31977,31831,32319,30058,32320,32204]
    shop_ids = st.multiselect("Select stores", options=default_ids, default=default_ids)

    storemap_file = st.file_uploader("Upload CSV met kolommen: shop_id,sq_meter,store_name", type=["csv"])

    mock_mode = st.toggle("Gebruik mock data", value=True)
    winsorize = st.checkbox("Winsorize 1e/99e percentiel", value=True)

def build_get_report_url(base, period, shop_ids, metrics):
    params = [("source","shops"),("period",period)]
    for sid in shop_ids:
        params.append(("data", str(sid)))
    for m in metrics:
        params.append(("data_output", m))
    q = "&".join([f"{k}={v}" for k,v in params])
    return f"{base}{endpoint}?{q}"

def fetch_report(base, period, shop_ids):
    metrics = [
        "count_in",
        "sales_per_visitor",
        "sales_per_sqm",
        "conversion_rate",
        "sales_per_transaction",
        "turnover"
    ]
    url = build_get_report_url(base, period, shop_ids, metrics)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json(), url, None
    except Exception as e:
        return None, url, e

def make_mock_dataframe(shop_ids):
    rng = np.random.default_rng(42)
    n = len(shop_ids)
    count_in = rng.integers(5000, 25000, size=n)
    spv = rng.uniform(10, 50, size=n)
    sq = rng.uniform(80, 250, size=n)
    visitors_per_sqm = count_in / sq
    expected_spsqm = spv * visitors_per_sqm
    actual_spsqm = expected_spsqm * rng.uniform(0.7, 1.3, size=n)
    turnover = actual_spsqm * sq
    return pd.DataFrame({
        "shop_id": shop_ids,
        "count_in": count_in,
        "sales_per_visitor": spv,
        "sales_per_sqm": actual_spsqm,
        "conversion_rate": rng.uniform(0.1,0.5,size=n),
        "sales_per_transaction": rng.uniform(20,100,size=n),
        "turnover": turnover,
        "sq_meter": sq
    })

if st.button("Analyseer"):
    if storemap_file is None:
        st.error("Upload eerst een storemap CSV.")
        st.stop()
    storemap = pd.read_csv(storemap_file)
    storemap["shop_id"] = storemap["shop_id"].astype(int)

    payload, url, err = fetch_report(fastapi_base, period, shop_ids)
    st.caption(f"API-call: {url}")
    if err is not None and not mock_mode:
        st.error(f"API-fout: {err}")
        st.stop()

    if mock_mode or payload is None:
        df = make_mock_dataframe(shop_ids)
    else:
        # Flatten logic should be adapted to real API response
        df = make_mock_dataframe(shop_ids) # placeholder for real parse

    df = df.merge(storemap, on="shop_id", suffixes=("","_map"))

    df["visitors_per_sqm"] = df["count_in"] / df["sq_meter"]
    df["expected_spsqm"] = df["sales_per_visitor"] * df["visitors_per_sqm"]
    df["actual_spsqm_chk"] = df["turnover"] / df["sq_meter"]
    cond = abs(df["actual_spsqm_chk"] - df["sales_per_sqm"]) / df["sales_per_sqm"] > 0.1
    df["actual_spsqm"] = df["sales_per_sqm"].where(~cond, df["actual_spsqm_chk"])
    df["CSm2I"] = df["actual_spsqm"] / df["expected_spsqm"]
    df["uplift_eur"] = np.maximum(0, df["expected_spsqm"] - df["actual_spsqm"]) * df["sq_meter"]

    st.subheader("Top underperformers")
    st.dataframe(df.sort_values("CSm2I").head(10))

    fig = px.bar(df.sort_values("uplift_eur", ascending=False), x="store_name", y="uplift_eur")
    st.plotly_chart(fig, use_container_width=True)
