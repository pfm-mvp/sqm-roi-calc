# streamlit_sales_per_sqm_potential.py
import os
import sys
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# Pad één niveau omhoog zodat we shop_mapping kunnen importeren
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from shop_mapping import SHOP_NAME_MAP  # geen data_transformer import

st.set_page_config(page_title="Sales-per-sqm Potentieel", page_icon="📊", layout="wide")

# --- Styling ---
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
st.caption("Naamselectie via SHOP_NAME_MAP, NVO (sq_meter) via API. Periode + period_step instelbaar. API‑URL uit Streamlit secrets.")

# =========================
# Inputs
# =========================
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    mock_mode = st.toggle("Gebruik mock data", value=False)

with colB:
    PRESETS = [
        "yesterday","today",
        "this_week","last_week",
        "this_month","last_month",
        "this_quarter","last_quarter",
        "this_year","last_year"
    ]
    period = st.selectbox("Periode", options=PRESETS, index=4)

with colC:
    PERIOD_STEPS = ["hour", "day", "week", "month", "quarter", "year", "total"]
    period_step = st.selectbox("Period Step", options=PERIOD_STEPS, index=1)

with colD:
    # Fallback: als secrets ontbreken, tijdelijk via text_input laten invullen
    API_URL = st.secrets.get("API_URL", "").strip()
    if not API_URL:
        st.warning("API_URL ontbreekt in secrets. Vul tijdelijk hieronder in (zet 'm later in `.streamlit/secrets.toml`).")
        API_URL = st.text_input("API URL", value="", placeholder="https://vemcount-agent.onrender.com/get-report")
    if not API_URL:
        st.stop()

# =========================
# Mapping
# =========================
NAME_TO_ID = {v: k for k, v in SHOP_NAME_MAP.items()}
ID_TO_NAME = {k: v for k, v in SHOP_NAME_MAP.items()}
DEFAULT_SHOP_IDS = list(SHOP_NAME_MAP.keys())
default_names = [ID_TO_NAME.get(shop_id, str(shop_id)) for shop_id in DEFAULT_SHOP_IDS]

selected_names = st.multiselect("Select stores", options=list(NAME_TO_ID.keys()), default=default_names)
shop_ids = [NAME_TO_ID[name] for name in selected_names]

if not shop_ids:
    st.warning("Selecteer minimaal één store.")
    st.stop()

# =========================
# API call (POST met [] keys) + headers + DEBUG
# =========================
def fetch_report(api_full_url: str, period: str, shop_ids: list[int], metrics: list[str], step: str):
    # optionele auth header
    api_token = st.secrets.get("API_TOKEN", "").strip()
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    # x-www-form-urlencoded payload (meerdere waarden via []-sleutels)
    payload = [
        ("source", "shops"),
        ("period", period),
        ("period_step", step),   # veel backends lezen deze
        ("step", step),          # sommige lezen juist deze
        ("weather", "0"),        # zoals je logs lieten zien
    ]
    for sid in shop_ids:
        payload.append(("data[]", str(sid)))
    for m in metrics:
        payload.append(("data_output[]", m))

    # Debug info voor in de UI
    req_info = {"url": api_full_url, "headers": headers, "payload": payload}

    try:
        r = requests.post(api_full_url, data=payload, headers=headers, timeout=40)
        status = r.status_code
        text_preview = r.text[:2000] if r.text else ""
        try:
            js = r.json()
        except Exception:
            js = {}
        return js, req_info, status, text_preview
    except Exception as e:
        return None, req_info, None, f"Request error: {e}"

# =========================
# Slimme parser (platte 'data' of geneste 'dates')
# =========================
def parse_vemcount(payload: dict, shop_ids: list[int], fields: list[str]) -> pd.DataFrame:
    rows = []

    # Case 1: platte structuur {"data": {"<shop_id>": {metric: value, ...}}}
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], dict):
        for sid in shop_ids:
            rec = payload["data"].get(str(sid), {}) or {}
            row = {"shop_id": sid}
            for f in fields:
                row[f] = rec.get(f, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    # Case 2: geneste structuur { "<shop_id>": { "dates": { "<date>": { "data": {...} } } } }
    if isinstance(payload, dict):
        for sid, content in payload.items():
            if not isinstance(content, dict):
                continue
            dates = content.get("dates", {})
            for _, day_info in dates.items():
                data = day_info.get("data", {})
                row = {"shop_id": int(sid)}
                for f in fields:
                    row[f] = data.get(f, np.nan)
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Aggregatie naar 1 regel per winkel
    agg = {}
    for f in fields:
        if f in ["count_in", "turnover"]:
            agg[f] = "sum"     # telwaarden cumuleren
        else:
            agg[f] = "mean"    # ratio's gemiddeld over de gekozen periode
    return df.groupby("shop_id", as_index=False).agg(agg)

# =========================
# Mock data
# =========================
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
            "sq_meter"
        ]

        if mock_mode:
            df = make_mock_dataframe(shop_ids)
            st.info("Mock data gebruikt")
        else:
            payload, req_info, status, text_preview = fetch_report(API_URL, period, shop_ids, metrics, step=period_step)

            # Debug‑expander: request + response preview
            with st.expander("🔧 Request/Response Debug"):
                st.write("➡️  POST naar:", req_info.get("url"))
                st.write("➡️  Headers:", req_info.get("headers"))
                st.write("➡️  Payload:", req_info.get("payload"))
                st.write("⬅️  HTTP status:", status)
                st.write("⬅️  Response preview (first 2000 chars):")
                st.code(text_preview if text_preview else "<empty body>")

            if status != 200:
                st.error(f"API gaf status {status}. Check payload/headers hierboven.")
                st.stop()

            if payload is None:
                st.error("Geen payload ontvangen van API.")
                st.stop()

            df = parse_vemcount(payload, shop_ids, fields=metrics)

        if df.empty:
            st.error("Geen data (alle waarden leeg). Controleer debug‑sectie, periode en period_step.")
            st.stop()

        # Voeg namen toe
        df["store_name"] = df["shop_id"].map(ID_TO_NAME)

        # Berekeningen
        eps = 1e-9
        sq = df["sq_meter"].replace(0, np.nan)
        df["visitors_per_sqm"] = df["count_in"] / sq
        df["expected_spsqm"]   = df["sales_per_visitor"] * df["visitors_per_sqm"]

        actual_chk = df["turnover"] / sq
        sales_sqm = df.get("sales_per_sqm", pd.Series(np.nan, index=df.index))
        needs_chk = sales_sqm.isna() | (
            (sales_sqm > 0) & ((actual_chk - sales_sqm).abs() / (sales_sqm + eps) > 0.10)
        )
        df["actual_spsqm"] = sales_sqm
        df.loc[needs_chk, "actual_spsqm"] = actual_chk

        df["CSm2I"] = (df["actual_spsqm"] / (df["expected_spsqm"] + eps)).replace([np.inf, -np.inf], np.nan)
        df["uplift_eur"] = np.maximum(0.0, (df["expected_spsqm"] - df["actual_spsqm"]).fillna(0)) * sq

        # Output
        st.subheader("🏆 Top underperformers (laagste CSm²I)")
        topN = df.sort_values("CSm2I").head(10).copy()
        topN["uplift_eur_fmt"] = topN["uplift_eur"].map(lambda x: f"€{x:,.0f}".replace(",", "."))
        st.dataframe(topN[["store_name","shop_id","CSm2I","uplift_eur_fmt"]], use_container_width=True)

        st.subheader("💰 Uplift € per winkel")
        chart_df = df.sort_values("uplift_eur", ascending=False).head(20)
        fig = px.bar(chart_df, x="store_name", y="uplift_eur")
        fig.update_layout(yaxis_tickprefix="€", margin=dict(l=10,r=10,t=30,b=60), xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧭 SPV vs Visitors/m²")
        df["uplift_size"] = df["uplift_eur"].fillna(0).clip(lower=0)
        fig2 = px.scatter(
            df, x="sales_per_visitor", y="visitors_per_sqm",
            size="uplift_size", color="CSm2I",
            hover_data=["store_name","shop_id","CSm2I","uplift_eur"]
        )
        fig2.update_layout(margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Onverwachte fout: {e}")
