import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import requests
import plotly.express as px
from datetime import date

# 👇 Zet dit vóór de import!
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

# ✅ Nu pas importeren
from data_transformer import normalize_vemcount_response
from shop_mapping import SHOP_NAME_MAP

st.set_page_config(page_title="Sales-per-sqm Potentieel", page_icon="📊", layout="wide")

# === Styling (compact, jouw primary-knop) ===
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
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Sales-per-sqm Potentieel (CSm²I)")
st.caption("Naamselectie via SHOP_NAME_MAP, NVO (sq_meter) via API, presets als in Storescan. URL uit Streamlit secrets.")

# =========================
# Inputs (in main)
# =========================
colA, colB, colC = st.columns([1,1,1])

with colA:
    st.markdown("**Testmodus**")
    mock_mode = st.toggle("Gebruik mock data", value=False)

with colB:
    st.markdown("**Periode**")
    PRESETS = [
        "yesterday","today",
        "this_week","last_week",
        "this_month","last_month",
        "this_quarter","last_quarter",
        "this_year","last_year"
    ]
    period = st.selectbox("", options=PRESETS, index=4, label_visibility="collapsed")

with colC:
    # Laat API_URL niet zien; we pakken 'm uit secrets
    try:
        API_URL = st.secrets["API_URL"]  # bv. "https://vemcount-agent.onrender.com/get-report"
        st.markdown("**API-config**")
        st.caption("API‑URL geladen uit secrets ✅")
    except KeyError:
        st.error("API_URL ontbreekt in Streamlit secrets. Voeg toe aan `.streamlit/secrets.toml`.")
        st.stop()

# =========================
# Naam <-> ID mapping
# =========================
NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}
DEFAULT_SHOP_IDS = list(SHOP_NAME_MAP.keys())
default_names = [ID_TO_NAME.get(shop_id, str(shop_id)) for shop_id in DEFAULT_SHOP_IDS]

# Selectie op NAAM (zoals je andere tools)
selected_names = st.multiselect("Select stores", options=list(NAME_TO_ID.keys()), default=default_names)
shop_ids = [NAME_TO_ID[name] for name in selected_names]

if not shop_ids:
    st.warning("Selecteer minimaal één store.")
    st.stop()

# =========================
# Helpers
# =========================
def build_query_url(api_full_url: str, period: str, shop_ids: list[int], metrics: list[str]) -> str:
    """
    Maakt een querystring met herhaalde data= en data_output= (zonder []), aansluitend achter API_URL.
    Voorbeeld: <API_URL>?source=shops&period=this_month&data=32224&data_output=count_in&...
    """
    params = [("source", "shops"), ("period", period)]
    for sid in shop_ids:
        params.append(("data", str(sid)))
    for m in metrics:
        params.append(("data_output", m))
    q = "&".join([f"{k}={v}" for k, v in params])
    sep = "&" if "?" in api_full_url else "?"
    return f"{api_full_url}{sep}{q}"

def fetch_report(api_full_url: str, period: str, shop_ids: list[int], metrics: list[str]):
    url = build_query_url(api_full_url, period, shop_ids, metrics)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json(), url

def parse_flat(payload: dict, shop_ids: list[int], fields: list[str]) -> pd.DataFrame:
    """
    Verwacht minimaal: {"data": {"<shop_id>": { metric: value, ...}}}
    """
    rows = []
    if payload and isinstance(payload, dict) and "data" in payload:
        data = payload["data"]
        for sid in shop_ids:
            rec = data.get(str(sid), {}) or {}
            row = {"shop_id": sid}
            for f in fields:
                row[f] = rec.get(f, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

def make_mock_dataframe(shop_ids: list[int], rng_seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    n = len(shop_ids)
    sq = rng.uniform(90, 250, size=n)
    count_in = rng.integers(6000, 24000, size=n)
    spv = rng.uniform(12, 48, size=n)
    vps = count_in / sq
    expected_spsqm = spv * vps
    actual_spsqm = expected_spsqm * rng.uniform(0.72, 1.28, size=n)
    turnover = actual_spsqm * sq
    return pd.DataFrame({
        "shop_id": shop_ids,
        "sq_meter": sq,
        "count_in": count_in,
        "sales_per_visitor": spv,
        "sales_per_sqm": actual_spsqm,
        "conversion_rate": rng.uniform(0.1,0.5,size=n),
        "sales_per_transaction": rng.uniform(20,140,size=n),
        "turnover": turnover
    })

# =========================
# Analyse-run
# =========================
if st.button("Analyseer", type="primary"):
    try:
        metrics = [
            "count_in",
            "sales_per_visitor",
            "sales_per_sqm",
            "conversion_rate",
            "sales_per_transaction",
            "turnover",
            "sq_meter"  # NVO via API
        ]

        with st.spinner("Data ophalen en berekenen…"):
            # API-call
            if mock_mode:
                payload, url = None, build_query_url(API_URL, period, shop_ids, metrics)
            else:
                payload, url = fetch_report(API_URL, period, shop_ids, metrics)

            st.caption(f"API‑call opgebouwd ✅")

            # Dataframe opbouwen
            if mock_mode or payload is None:
                df = make_mock_dataframe(shop_ids)
            else:
                df = parse_flat(payload, shop_ids, fields=metrics)

            # Store namen erbij
            df["store_name"] = df["shop_id"].map(ID_TO_NAME)

            # Kernberekeningen
            df["visitors_per_sqm"] = df["count_in"] / df["sq_meter"].replace(0, np.nan)
            df["expected_spsqm"]   = df["sales_per_visitor"] * df["visitors_per_sqm"]
            df["actual_spsqm_chk"] = df["turnover"] / df["sq_meter"].replace(0, np.nan)

            # Sanity check: als sales_per_sqm >10% afwijkt van turnover/sq_meter, gebruik turnover-based
            cond = (
                (df["sales_per_sqm"].astype(float) > 0) &
                (np.abs(df["actual_spsqm_chk"] - df["sales_per_sqm"]) / df["sales_per_sqm"] > 0.10)
            )
            df["actual_spsqm"] = df["sales_per_sqm"].where(~cond, df["actual_spsqm_chk"])

            # Index & uplift
            df["CSm2I"] = df["actual_spsqm"] / df["expected_spsqm"].replace(0, np.nan)
            df["uplift_eur"] = np.maximum(0.0, df["expected_spsqm"] - df["actual_spsqm"]) * df["sq_meter"]

            # Drivers (vs medians)
            med_spv = df["sales_per_visitor"].median(skipna=True)
            med_vps = df["visitors_per_sqm"].median(skipna=True)
            df["driver_flag"] = np.select(
                [
                    (df["sales_per_visitor"] < med_spv) & (df["visitors_per_sqm"] >= med_vps),
                    (df["sales_per_visitor"] >= med_spv) & (df["visitors_per_sqm"] < med_vps),
                    (df["sales_per_visitor"] < med_spv) & (df["visitors_per_sqm"] < med_vps),
                ],
                ["Low SPV", "Low density", "Low SPV + density"],
                default="OK"
            )

        # =========================
        # Output
        # =========================
        st.markdown("---")
        c1, c2 = st.columns([1,1])

        with c1:
            st.subheader("🏆 Top underperformers (laagste CSm²I)")
            topN = df.sort_values("CSm2I").head(10).copy()
            topN["uplift_eur_fmt"] = topN["uplift_eur"].map(lambda x: f"€{x:,.0f}".replace(",", "."))
            st.dataframe(
                topN[["store_name","shop_id","CSm2I","uplift_eur_fmt","driver_flag"]],
                use_container_width=True
            )

        with c2:
            st.subheader("💰 Uplift € per winkel")
            chart_df = df.sort_values("uplift_eur", ascending=False).head(20)
            fig = px.bar(chart_df, x="store_name", y="uplift_eur", hover_data=["CSm2I","driver_flag"])
            fig.update_layout(yaxis_tickprefix="€", margin=dict(l=10,r=10,t=30,b=60), xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📍 Waarom per winkel? (drivers)")
        drivers = df[["store_name","shop_id","sales_per_visitor","visitors_per_sqm","CSm2I","uplift_eur","driver_flag"]].copy()
        drivers["uplift_eur"] = drivers["uplift_eur"].round(0)
        st.dataframe(drivers.sort_values("uplift_eur", ascending=False), use_container_width=True)

        st.markdown("---")
        st.subheader("🧭 SPV vs Visitors/m²")
        fig2 = px.scatter(
            df, x="sales_per_visitor", y="visitors_per_sqm",
            size="uplift_eur", color="CSm2I",
            hover_data=["store_name","shop_id","CSm2I","uplift_eur"]
        )
        fig2.update_layout(margin=dict(l=10,r=10,t=30,b=10), xaxis_title="Sales per Visitor", yaxis_title="Visitors per m²")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("📥 Export")
        export_cols = ["store_name","shop_id","sq_meter","count_in","sales_per_visitor","sales_per_sqm","turnover",
                       "visitors_per_sqm","expected_spsqm","actual_spsqm","CSm2I","uplift_eur","driver_flag"]
        csv = df[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="sales_per_sqm_potential.csv", mime="text/csv")

    except requests.HTTPError as http_err:
        st.error(f"API HTTP-error: {http_err}")
    except Exception as e:
        st.error(f"Onverwachte fout: {e}")
