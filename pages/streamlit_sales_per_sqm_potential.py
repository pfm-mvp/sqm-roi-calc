# streamlit_sales_per_sqm_potential.py
import os
import sys
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# Pad één niveau omhoog zodat we gedeelde modules kunnen importeren
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

# Jouw util & mapping
from data_transformer import normalize_vemcount_response
from shop_mapping import SHOP_NAME_MAP

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
st.caption("Naamselectie via SHOP_NAME_MAP, NVO (sq_meter) via API, presets als in Storescan. URL uit Streamlit secrets.")

# =========================
# Inputs in main
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
    try:
        API_URL = st.secrets["API_URL"]  # bv. "https://vemcount-agent.onrender.com/get-report"
        st.markdown("**API-config**")
        st.caption("API‑URL geladen uit secrets ✅")
    except KeyError:
        st.error("API_URL ontbreekt in Streamlit secrets. Voeg toe aan `.streamlit/secrets.toml`.")
        st.stop()

# =========================
# Naam <-> ID mapping en selectie op NAAM
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
# Helpers
# =========================
def build_query_url(api_full_url: str, period: str, shop_ids: list[int], metrics: list[str]) -> str:
    """
    Bouwt querystring met herhaalde data= en data_output= (zonder []) achter de volledige API_URL.
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
    Eenvoudige parser voor payloads van vorm: {"data": {"<shop_id>": { metric: value, ...}}}
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
            # 1) Haal payload op (of mock)
            if mock_mode:
                payload, url = None, build_query_url(API_URL, period, shop_ids, metrics)
            else:
                payload, url = fetch_report(API_URL, period, shop_ids, metrics)

            # 2) Bouw dataframe met fallback op jouw normalizer
            if mock_mode or payload is None:
                df = make_mock_dataframe(shop_ids)
            else:
                try:
                    df_norm = normalize_vemcount_response(payload)
                    # Aggregeer naar 1 regel per shop_id
                    agg = {
                        "count_in": "sum",
                        "turnover": "sum",
                        "sq_meter": "mean",
                        "sales_per_sqm": "mean",
                        "sales_per_visitor": "mean",
                        "conversion_rate": "mean",
                        "sales_per_transaction": "mean",
                    }
                    use_cols = [c for c in agg.keys() if c in df_norm.columns]
                    df = (df_norm[df_norm["shop_id"].isin(shop_ids)]
                          .groupby("shop_id", as_index=False)[use_cols].agg(agg))
                except Exception:
                    # Val terug op platte parser
                    df = parse_flat(payload, shop_ids, fields=metrics)

            # 3) Guardrails: verplichte velden + non-empty
            required = ["shop_id","count_in","turnover","sq_meter"]
            missing = [c for c in required if c not in df.columns]
            if missing or df.empty:
                st.error(f"Geen bruikbare data uit API. Ontbrekende velden: {missing}. "
                         "Zet desnoods tijdelijk ‘Gebruik mock data’ aan om de UI te testen.")
                st.stop()

            # 4) Voeg store namen toe
            df["store_name"] = df["shop_id"].map(ID_TO_NAME)

            # 5) Veilige kernberekeningen (geen NaN/Inf)
            eps = 1e-9
            sq = df["sq_meter"].replace(0, np.nan)

            df["visitors_per_sqm"] = df["count_in"] / sq
            df["expected_spsqm"]   = df.get("sales_per_visitor", 0) * df["visitors_per_sqm"]

            actual_chk = df["turnover"] / sq
            sales_sqm = df.get("sales_per_sqm", pd.Series(np.nan, index=df.index))
            needs_chk = sales_sqm.isna() | (
                (sales_sqm > 0) & ((actual_chk - sales_sqm).abs() / (sales_sqm + eps) > 0.10)
            )
            df["actual_spsqm"] = sales_sqm
            df.loc[needs_chk, "actual_spsqm"] = actual_chk

            df["CSm2I"] = (df["actual_spsqm"] / (df["expected_spsqm"] + eps)).replace([np.inf, -np.inf], np.nan)
            df["uplift_eur"] = np.maximum(0.0, (df["expected_spsqm"] - df["actual_spsqm"]).fillna(0)) * sq

            # Drivers op basis van medianen
            med_spv = df.get("sales_per_visitor", pd.Series([np.nan])).median(skipna=True)
            med_vps = df["visitors_per_sqm"].median(skipna=True)
            df["driver_flag"] = np.select(
                [
                    (df.get("sales_per_visitor", 0) < med_spv) & (df["visitors_per_sqm"] >= med_vps),
                    (df.get("sales_per_visitor", 0) >= med_spv) & (df["visitors_per_sqm"] < med_vps),
                    (df.get("sales_per_visitor", 0) < med_spv) & (df["visitors_per_sqm"] < med_vps),
                ],
                ["Low SPV", "Low density", "Low SPV + density"],
                default="OK"
            )

            # Schoon voor visuals
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df["uplift_size"] = df["uplift_eur"].fillna(0).clip(lower=0)

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
            size="uplift_size",  # veilige size (geen NaN)
            color="CSm2I",
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
