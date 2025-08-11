import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

# ============ Page & Styling ============
st.set_page_config(page_title="SQM Calculator", page_icon="🟪", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
[data-baseweb="tag"] { background-color: #9E77ED !important; color: white !important; }
button[data-testid="stBaseButton-secondary"] {
  background-color: #F04438 !important; color: white !important; border-radius: 16px !important;
  font-weight: 600 !important; padding: 0.6rem 1.4rem !important; border: none !important;
}
button[data-testid="stBaseButton-secondary"]:hover { background-color: #d13c30 !important; cursor: pointer; }
.card { border: 1px solid #eee; border-radius: 12px; padding: 14px 16px; background:#fff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.kpi { font-size: 1.2rem; font-weight:700; }
.eur { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)

# ============ Helpers (gedeelde kern) ============
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce_numeric(df, ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"])
    if "conversion_rate" in out.columns and not out["conversion_rate"].empty:
        if out["conversion_rate"].max() > 1.5:  # % → fractie
            out["conversion_rate"] = out["conversion_rate"] / 100.0
    else:
        out["conversion_rate"] = out.get("transactions", 0.0) / (out.get("count_in", 0.0) + EPS)
    if ("sales_per_visitor" not in out.columns) or out["sales_per_visitor"].isna().all():
        out["sales_per_visitor"] = out.get("turnover", 0.0) / (out.get("count_in", 0.0) + EPS)
    if "sq_meter" in out.columns:
        sqm = pd.to_numeric(out["sq_meter"], errors="coerce")
        med = sqm.replace(0, np.nan).median()
        fallback = med if pd.notnull(med) and med > 0 else DEFAULT_SQ_METER
        out["sq_meter"] = sqm.replace(0, np.nan).fillna(fallback)
    else:
        out["sq_meter"] = DEFAULT_SQ_METER
    out["atv"] = out.get("turnover", 0.0) / (out.get("transactions", 0.0) + EPS)
    return out

def choose_ref_spv(df: pd.DataFrame, mode="portfolio", benchmark_shop_id=None, manual_spv=None, uplift_pct=0.0):
    if mode == "benchmark" and benchmark_shop_id is not None and benchmark_shop_id in df["shop_id"].values:
        sub = df[df["shop_id"] == int(benchmark_shop_id)]
        base = sub["turnover"].sum() / (sub["count_in"].sum() + EPS)
    elif mode == "manual" and manual_spv is not None:
        base = float(manual_spv)
    else:  # portfolio
        base = df["turnover"].sum() / (df["count_in"].sum() + EPS)
    return max(0.0, base) * (1.0 + float(uplift_pct))

def compute_csm2i_and_uplift(df: pd.DataFrame, ref_spv: float, csm2i_target: float):
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"] / (out["visitors"] + EPS)
    # CSm²I = actual_spv / ref_spv
    out["csm2i"] = out["actual_spv"] / (float(ref_spv) + EPS)
    # Optioneel spsqm (debug)
    out["visitors_per_sqm"] = out["count_in"] / (out["sq_meter"] + EPS)
    out["actual_spsqm"]     = out["turnover"]  / (out["sq_meter"] + EPS)
    out["expected_spsqm"]   = float(ref_spv)   * out["visitors_per_sqm"]
    # Uplift (€)
    out["uplift_eur_csm"] = np.maximum(0.0, out["visitors"] * (float(csm2i_target) * float(ref_spv) - out["actual_spv"]))
    return out

def fmt_eur(x: float) -> str:
    return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X", ".")

# ============ Inputs ============
st.title("🟪 SQM Calculator")
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP  # {id:int: "Naam"}
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}
names = sorted(NAME_TO_ID.keys(), key=str.lower)

today = date.today()
c1, c2 = st.columns(2)
with c1: date_from = st.date_input("Van", today - timedelta(days=7))
with c2: date_to   = st.date_input("Tot", today - timedelta(days=1))

sel_names = st.multiselect("Selecteer winkels", names, default=names[:1])
shop_ids = [NAME_TO_ID[n] for n in sel_names]

gran = st.selectbox("Granulariteit", ["Dag","Uur"], index=0)
period_step = "day" if gran == "Dag" else "hour"

c3, c4 = st.columns(2)
with c3:
    # Referentie: jouw keuze 1B → benchmark als gekozen, anders portfolio
    use_benchmark = st.checkbox("Benchmark‑winkel gebruiken?", value=False,
                                help="Indien aan: ref_SPV = SPV van gekozen winkel; anders portfolio‑SPV.")
    bm_name = st.selectbox("Benchmark‑winkel", names, index=0, disabled=not use_benchmark) if names else None
with c4:
    spv_uplift_pct = st.slider("SPV‑target uplift (%)", 0, 100, 10, 5,
                               help="Verhoog referentie‑SPV met dit % om een target‑scenario te testen.")

c5, c6 = st.columns(2)
with c5:
    csm2i_target = st.slider("CSm²I‑target (index)", 0.10, 2.00, 1.00, 0.05,
                             help="1.00 = op target; <1 is onder; >1 is boven target.")
with c6:
    st.info("**SPV = omzet per bezoeker**. We vergelijken jouw actuele SPV met een referentie‑SPV.\n"
            "• Portfolio‑SPV: gewogen (totaalomzet / totaalbezoekers)\n"
            "• Benchmark‑SPV: SPV van geselecteerde winkel\n"
            "• SPV‑target uplift: verhoogt de gekozen referentie met x%.")

run = st.button("🔍 Analyseer", type="secondary")

# ============ API fetch ============
def fetch_report(api_url, shop_ids, dfrom, dto, step, outputs, timeout=60):
    params = [("source","shops"), ("period","date"),
              ("form_date_from",str(dfrom)), ("form_date_to",str(dto)),
              ("period_step",step)]
    for sid in shop_ids:     params.append(("data", int(sid)))
    for out in outputs:      params.append(("data_output", out))
    r = requests.post(api_url, params=params, timeout=timeout); r.raise_for_status()
    return r.json()

def normalize(resp):
    rows = []
    data = resp.get("data", {})
    for _, shops in data.items():
        for sid, payload in shops.items():
            dates = (payload or {}).get("dates", {})
            for ts, obj in dates.items():
                rec = {"timestamp": ts, "shop_id": int(sid), "shop_name": SHOP_ID_TO_NAME.get(int(sid), str(sid))}
                rec.update((obj or {}).get("data", {}))
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = ts.dt.date; df["hour"] = ts.dt.hour
    return df

if run:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()
    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    outs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]
    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, period_step, outs)
        df = normalize(resp)
        if df.empty:
            st.info("Geen data."); st.stop()

    # ==== CSm²I uniform ====
    mode = "benchmark" if use_benchmark else "portfolio"  # keuze 1B
    bm_id = NAME_TO_ID.get(bm_name) if (use_benchmark and bm_name) else None
    ref_spv = choose_ref_spv(df, mode=mode, benchmark_shop_id=bm_id, uplift_pct=spv_uplift_pct/100.0)
    df = compute_csm2i_and_uplift(df, ref_spv=ref_spv, csm2i_target=csm2i_target)

    # ==== Output ====
    st.markdown("### Resultaten per winkel (geaggregeerd)")
    agg = df.groupby(["shop_id","shop_name"]).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
        csm2i=("csm2i","mean"),
        uplift_eur=("uplift_eur_csm","sum")
    ).reset_index()
    agg["uplift_eur"] = agg["uplift_eur"].round(0)
    st.dataframe(agg, use_container_width=True)
    st.markdown(f"**Referentie‑SPV (na uplift):** €{ref_spv:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
