# pages/sqm-calc.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.express as px
import sys, pathlib

# ─────────────────────────  Page & styling  ─────────────────────────
st.set_page_config(page_title="Sales-per-sqm Potentieel (CSm²I)", page_icon="📈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
[data-baseweb="tag"] { background:#9E77ED !important; color:#fff !important; }
button[data-testid="stBaseButton-secondary"]{
  background:#F04438!important;color:#fff!important;border-radius:16px!important;
  font-weight:600!important;border:none!important;padding:0.6rem 1.4rem!important;
}
button[data-testid="stBaseButton-secondary"]:hover{background:#d13c30!important;cursor:pointer;}
.card{border:1px solid #eee;border-radius:12px;padding:14px 16px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.kpi{font-size:1.2rem;font-weight:700;font-variant-numeric:tabular-nums}
.note{color:#667085}
.block-orange{background:#FFF7ED;border:1px solid #FEAC76;border-radius:12px;padding:18px}
.h-gap{height:14px}
</style>
""", unsafe_allow_html=True)

st.title("Sales-per-sqm Potentieel (CSm²I)")

# ─────────────────────────  Helpers  ─────────────────────────
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(x):
    try:
        return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "€0"

def fmt_eur2(x):
    try:
        return ("€{:,.2f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "€0,00"

def coerce(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out.get(c, 0.0), errors="coerce").fillna(0.0)
    return out

def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce(df, ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"])
    if out["conversion_rate"].max() > 1.5:
        out["conversion_rate"] = out["conversion_rate"]/100.0
    else:
        out["conversion_rate"] = out["conversion_rate"].fillna(out["transactions"]/(out["count_in"]+EPS))
    if out["sales_per_visitor"].isna().all() or out["sales_per_visitor"].eq(0).all():
        out["sales_per_visitor"] = out["turnover"]/(out["count_in"]+EPS)
    out["atv"] = out["turnover"]/(out["transactions"]+EPS)
    sqm = pd.to_numeric(out["sq_meter"], errors="coerce")
    med = sqm.replace(0, np.nan).median()
    out["sq_meter"] = sqm.replace(0, np.nan).fillna(med if pd.notnull(med) and med>0 else DEFAULT_SQ_METER)
    return out

def choose_ref_spv(df: pd.DataFrame, uplift_pct: float = 0.0) -> float:
    safe = coerce(df, ["turnover","count_in"])
    visitors = float(safe["count_in"].sum())
    base = 0.0 if visitors <= 0 else float(safe["turnover"].sum())/(visitors+EPS)
    return base * (1.0 + float(uplift_pct))

def compute_csm2i(df: pd.DataFrame, ref_spv: float, csm2i_target: float):
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"]/(out["visitors"]+EPS)
    out["csm2i"]      = out["actual_spv"]/(ref_spv+EPS)
    out["visitors_per_sqm"] = out["count_in"]/(out["sq_meter"]+EPS)
    out["actual_spsqm"]     = out["turnover"]/(out["sq_meter"]+EPS)
    out["expected_spsqm"]   = ref_spv * out["visitors_per_sqm"]
    out["uplift_eur_csm"] = np.maximum(0.0, out["visitors"]*(csm2i_target*ref_spv - out["actual_spv"]))
    return out

# ─────────────────────────  Shop mapping (ROBUUST)  ─────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MAP = {}
_import_err = None
try:
    import shop_mapping as sm
    for cand in ("SHOP_NAME_MAP", "SHOP_ID_TO_NAME", "SHOP_MAP", "MAP"):
        if hasattr(sm, cand):
            _MAP = getattr(sm, cand) or {}
            break
    if not _MAP:
        _import_err = "Geen geldige mapping-variabele in shop_mapping.py (verwacht bv. SHOP_NAME_MAP)."
except Exception as e:
    _import_err = f"Kon shop_mapping niet importeren: {e}"

try:
    SHOP_ID_TO_NAME = {int(k): str(v) for k, v in dict(_MAP).items() if str(v).strip()}
except Exception:
    SHOP_ID_TO_NAME = {}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}

if not NAME_TO_ID:
    msg = "Geen filialen geladen. "
    if _import_err:
        msg += f"Details: {_import_err}"
    msg += "\nControleer of `shop_mapping.py` in de project root staat en bv. bevat:\n\n" \
           "SHOP_NAME_MAP = { 31831: 'Den Bosch', 32224: 'Amersfoort', ... }"
    st.error(msg)
    st.stop()
else:
    st.caption(f"🔗 Shop mapping geladen: {len(NAME_TO_ID)} filialen.")

# ─────────────────────────  Inputs  ─────────────────────────
period_options = {
    "last_week": "Last week",
    "this_month": "This month",
    "last_month": "Last month",
    "this_quarter": "This quarter",
    "last_quarter": "Last quarter",
    "this_year": "This year",
    "last_year": "Last year"
}
period_label = st.selectbox("Select period", list(period_options.keys()),
                             index=1,
                             format_func=lambda s: period_options[s])

all_store_names = sorted(NAME_TO_ID.keys(), key=str.lower)
selected = st.multiselect("Select stores", all_store_names, default=all_store_names)
shop_ids = [NAME_TO_ID[n] for n in selected]

left, mid, right = st.columns([1,1,1])
with left:
    csm2i_target = st.slider("Target CSm²I", 0.10, 2.00, 0.85, 0.05)
with mid:
    conv_goal_pct = st.slider("Target conversion rate (%)", 1, 80, 35, 1)
with right:
    proj_toggle = st.toggle("Toon projectie voor resterend jaar", value=True)

run = st.button("🔍 Analyseer", type="secondary")

# ─────────────────────────  API helpers  ─────────────────────────
def fetch_report(api_url, shop_ids, dfrom, dto, step, outputs, timeout=60):
    params = [("source","shops"),("period",period_label)]
    for sid in shop_ids: params.append(("data", int(sid)))
    for outp in outputs: params.append(("data_output", outp))
    r = requests.post(api_url, params=params, timeout=timeout); r.raise_for_status()
    return r.json()

def normalize_resp(resp):
    rows = []
    for _, shops in (resp or {}).get("data", {}).items():
        for sid, payload in (shops or {}).items():
            for ts, obj in (payload or {}).get("dates", {}).items():
                rec = {"timestamp": ts, "shop_id": int(sid)}
                rec.update(((obj or {}).get("data", {})))
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = ts.dt.date; df["hour"] = ts.dt.hour
    df["shop_name"] = df["shop_id"].map(SHOP_ID_TO_NAME).fillna(df["shop_id"].astype(str))
    return df

# ─────────────────────────  Run  ─────────────────────────
if run:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()

    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, None, None, "day", outputs)
        df = normalize_resp(resp)
        if df.empty:
            st.info("Geen data voor de gekozen periode."); st.stop()

    ref_spv = choose_ref_spv(df, uplift_pct=0.0)
    df = compute_csm2i(df, ref_spv=ref_spv, csm2i_target=csm2i_target)
    conv_target = float(conv_goal_pct)/100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    agg = df.groupby(["shop_id","shop_name"]).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
        sqm=("sq_meter","mean"),
        spsqm=("actual_spsqm","mean"),
        csm2i=("csm2i","mean"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
    ).reset_index()
    agg["uplift_total"] = agg["uplift_csm"] + agg["uplift_conv"]

    # KPI's
    k1,k2,k3 = st.columns(3)
    k1.markdown(f'<div class="card"><div>🚀 <b>CSm²I potential</b></div><div class="kpi">{fmt_eur(agg["uplift_csm"].sum())}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><div>🎯 <b>Conversion potential</b></div><div class="kpi">{fmt_eur(agg["uplift_conv"].sum())}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><div>∑ <b>Total potential</b></div><div class="kpi">{fmt_eur(agg["uplift_total"].sum())}</div></div>', unsafe_allow_html=True)

    # Oranje blokken
    c1, c2 = st.columns([1.25,1])
    with c1:
        st.markdown(f"""
        <div class="block-orange">
          <div style="font-weight:700;font-size:1.05rem">💰 Total extra potential in revenue</div>
          <div class="kpi" style="margin-top:4px">{fmt_eur(agg["uplift_total"].sum())}</div>
          <div class="note">Som van CSm²I- en conversie-potentieel voor de geselecteerde periode.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        if proj_toggle:
            days_in_period = max(df["date"].nunique(), 1)
            daily_uplift = agg["uplift_total"].sum() / days_in_period
            remain_days = (date(today.year,12,31) - date.today()).days
            yearly_proj = daily_uplift * max(remain_days, 0)
            st.markdown(f"""
            <div class="block-orange">
              <div style="font-weight:700;font-size:1.05rem">📈 Projectie resterend jaar</div>
              <div class="kpi" style="margin-top:4px">{fmt_eur(yearly_proj)}</div>
              <div class="note">Daggemiddelde uit gekozen periode × resterende dagen dit jaar.</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="h-gap"></div>', unsafe_allow_html=True)

    # Tabel
    tbl_disp = pd.DataFrame({
        "Store": agg["shop_name"],
        "Square meters": agg["sqm"].round(0).astype(int),
        "Current Avg Sales per sqm": agg["spsqm"].map(fmt_eur2),
        "CSm²I (index)": agg["csm2i"].round(3),
        "Uplift from CSm²I (€)": agg["uplift_csm"].map(fmt_eur),
        "Uplift from Conversion (€)": agg["uplift_conv"].map(fmt_eur),
        "Total Potential Uplift (€)": agg["uplift_total"].map(fmt_eur),
    })
    st.dataframe(tbl_disp, use_container_width=True)

    # Bars met PFM kleur + EU hover
    agg_sorted_csm = agg.sort_values("uplift_csm", ascending=False).copy()
    agg_sorted_csm["hover_val"] = agg_sorted_csm["uplift_csm"].map(fmt_eur)
    fig1 = px.bar(agg_sorted_csm, x="shop_name", y="uplift_csm",
                  labels={"shop_name":"Store","uplift_csm":"CSm²I potential (€)"},
                  color_discrete_sequence=["#762181"],
                  custom_data=["hover_val"])
    fig1.update_traces(text=agg_sorted_csm["uplift_csm"].map(lambda v: ("{:,.0f}".format(v)).replace(",", ".")),
                       textposition="outside",
                       hovertemplate="%{x}<br>%{customdata[0]}")
    st.plotly_chart(fig1, use_container_width=True)

    agg_sorted_conv = agg.sort_values("uplift_conv", ascending=False).copy()
    agg_sorted_conv["hover_val"] = agg_sorted_conv["uplift_conv"].map(fmt_eur)
    fig2 = px.bar(agg_sorted_conv, x="shop_name", y="uplift_conv",
                  labels={"shop_name":"Store","uplift_conv":"Conversion potential (€)"},
                  color_discrete_sequence=["#762181"],
                  custom_data=["hover_val"])
    fig2.update_traces(text=agg_sorted_conv["uplift_conv"].map(lambda v: ("{:,.0f}".format(v)).replace(",", ".")),
                       textposition="outside",
                       hovertemplate="%{x}<br>%{customdata[0]}")
    st.plotly_chart(fig2, use_container_width=True)
