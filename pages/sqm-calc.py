# pages/sqm-calc.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

# -----------------------------
# Page & styling
# -----------------------------
st.set_page_config(page_title="Sales‑per‑sqm Potentieel (CSm²I)", page_icon="📈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
[data-baseweb="tag"] { background-color: #9E77ED !important; color: white !important; }
button[data-testid="stBaseButton-secondary"] {
  background-color: #F04438 !important; color: white !important; border-radius: 16px !important;
  font-weight: 600 !important; padding: .6rem 1.4rem !important; border: none !important;
}
button[data-testid="stBaseButton-secondary"]:hover { background-color: #d13c30 !important; cursor: pointer; }
.card { border: 1px solid #eee; border-radius: 12px; padding: 14px 16px; background:#fff; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.kpi  { font-size: 1.2rem; font-weight: 800; }
.eur  { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
EPS = 1e-9

def fmt_eur(v: float) -> str:
    return ("€{:,.0f}".format(float(v))).replace(",", "X").replace(".", ",").replace("X",".")

def resolve_period(preset: str) -> tuple[date, date]:
    today = date.today()
    first_of_month = today.replace(day=1)
    last_month_last_day = first_of_month - timedelta(days=1)
    last_month_first_day = last_month_last_day.replace(day=1)
    p = (preset or "").lower()
    if p in ("7 dagen", "last_7_days"):   return today - timedelta(days=7),  today - timedelta(days=1)
    if p in ("30 dagen","last_30_days"):  return today - timedelta(days=30), today - timedelta(days=1)
    if p in ("last_month","vorige maand"):return last_month_first_day, last_month_last_day
    if p in ("month_to_date","mtd","deze maand"): return first_of_month, today
    if p in ("year_to_date","ytd","dit jaar"):    return today.replace(month=1, day=1), today
    if p in ("gisteren","yesterday"):      y=today-timedelta(days=1); return y,y
    return today - timedelta(days=7), today - timedelta(days=1)

def period_length_days(d_from: date, d_to: date) -> int:
    return max(1, (d_to - d_from).days + 1)

def remaining_year_days(today: date | None = None) -> int:
    today = today or date.today()
    end = date(today.year, 12, 31)
    return max(0, (end - today).days)

def coerce(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    return out

# -----------------------------
# Shop mapping
# -----------------------------
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP   # {id:int: "Naam"}
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}
names = sorted(NAME_TO_ID.keys(), key=str.lower)

# -----------------------------
# Header UI
# -----------------------------
st.title("Sales‑per‑sqm Potentieel (CSm²I)")

period_options = ["7 dagen", "30 dagen", "last_month", "month_to_date", "year_to_date"]
c1, c2 = st.columns([1,3])
with c1:
    period_label = st.selectbox("Select period", period_options, index=1)
with c2:
    selected_names = st.multiselect("Select stores", names, default=names[:6])
shop_ids = [NAME_TO_ID[n] for n in selected_names]

use_benchmark = st.toggle("📌 Gebruik benchmark‑winkel i.p.v. vaste CSm²I‑target", value=False)
csm2i_target = st.slider("Target CSm²I", .10, 2.00, .85, .05)
conv_target_pct = st.slider("Target conversion rate (%)", 1, 80, 35, 1)

# Projectie‑toggle
proj_rest_year = st.toggle("💡 Projecteer ‘Totaal potentieel’ naar resterend jaar", value=True)

# Run
run = st.button("Analyseer", type="secondary")

# -----------------------------
# Data fetch helpers
# -----------------------------
def fetch_report(api_url, shop_ids, dfrom, dto, step, outputs, timeout=60):
    params = [("source","shops"), ("period","date"),
              ("form_date_from",str(dfrom)), ("form_date_to",str(dto)), ("period_step",step)]
    for sid in shop_ids:  params.append(("data", int(sid)))
    for outp in outputs:  params.append(("data_output", outp))
    r = requests.post(api_url, params=params, timeout=timeout); r.raise_for_status()
    return r.json()

def normalize_resp(resp):
    rows = []
    data = resp.get("data", {})
    for _, shops in data.items():
        for sid, payload in shops.items():
            for ts, obj in (payload or {}).get("dates", {}).items():
                rec = {"timestamp": ts, "shop_id": int(sid), "shop_name": SHOP_ID_TO_NAME.get(int(sid), str(sid))}
                rec.update((obj or {}).get("data", {}))
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = ts.dt.date
    return df

# -----------------------------
# Compute
# -----------------------------
if run:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()

    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.error("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    date_from, date_to = resolve_period(period_label)
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]
    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, step="day", outputs=outputs)
        df = normalize_resp(resp)
        if df.empty:
            st.info("Geen data beschikbaar."); st.stop()

    # numeriek & basis‑kpi’s
    df = coerce(df, ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"])
    if df["conversion_rate"].max() > 1.5:
        df["conversion_rate"] = df["conversion_rate"] / 100.0
    df["spv"]    = np.where(df["count_in"]>0, df["turnover"]/ (df["count_in"]+EPS), 0.0)
    df["atv"]    = np.where(df["transactions"]>0, df["turnover"]/ (df["transactions"]+EPS), 0.0)
    df["spsqm"]  = np.where(df["sq_meter"]>0, df["turnover"]/ (df["sq_meter"]+EPS), 0.0)

    # referentie‑SPV (portfolio)
    ref_spv = (df["turnover"].sum() / (df["count_in"].sum()+EPS))
    # CSm²I en uplift
    df["csm2i"] = np.where(ref_spv>0, df["spv"]/ref_spv, 0.0)
    df["uplift_csm"]  = np.maximum(0.0, (csm2i_target*ref_spv - df["spv"])) * df["count_in"]
    conv_target = conv_target_pct/100.0
    df["uplift_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"])) * df["count_in"] * df["atv"]

    agg = df.groupby(["shop_id","shop_name"]).agg(
        sqm=("sq_meter","mean"),
        spsqm=("spsqm","mean"),
        csm2i=("csm2i","mean"),
        uplift_csm=("uplift_csm","sum"),
        uplift_conv=("uplift_conv","sum"),
    ).reset_index()
    agg["uplift_total"] = agg["uplift_csm"] + agg["uplift_conv"]

    # ---------------- KPI‑tegels ----------------
    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="card"><div>🚀 <b>CSm²I potential</b><br><small>({period_label}, target CSm²I {csm2i_target:.2f})</small></div><div class="kpi eur">{fmt_eur(agg["uplift_csm"].sum())}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><div>🎯 <b>Conversion potential</b><br><small>({period_label}, doel = {conv_target_pct}%)</small></div><div class="kpi eur">{fmt_eur(agg["uplift_conv"].sum())}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><div>∑ <b>Total potential</b><br><small>({period_label})</small></div><div class="kpi eur">{fmt_eur(agg["uplift_total"].sum())}</div></div>', unsafe_allow_html=True)

    # ---------- Projectie resterend jaar (toggle) ----------
    if proj_rest_year:
        days_left = remaining_year_days()
        days_in_period = period_length_days(date_from, date_to)
        projection = float(agg["uplift_total"].sum()) * (days_left / days_in_period)
        st.markdown(
            f"""
            <div style="margin:10px 0 0 0;padding:16px;border-radius:12px;background:#FEAC7622;border:1px solid #FEAC76">
              <div style="font-weight:700;">💰 Total extra potential in revenue (resterend jaar)</div>
              <div style="font-size:1.2rem;font-weight:800;">{fmt_eur(projection)}</div>
              <div style="opacity:.8;">Projectie: huidig {period_label.lower()}‑potentieel × (resterende dagen dit jaar / dagen in analyseperiode).</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------- Tabel + charts (ongewijzigd qua stijl) ---------
    st.subheader("🏆 Stores with most potential")
    view = agg[["shop_name","sqm","spsqm","csm2i","uplift_csm","uplift_conv","uplift_total"]].rename(columns={
        "shop_name":"Store","sqm":"Square meters","spsqm":"Current Avg Sales per sqm","csm2i":"CSm²I (index)",
        "uplift_csm":"Uplift from CSm²I (€)","uplift_conv":"Uplift from Conversion (€)","uplift_total":"Total Potential Uplift (€)"
    }).sort_values("uplift_total", ascending=False)
    st.dataframe(view, use_container_width=True, hide_index=True)
