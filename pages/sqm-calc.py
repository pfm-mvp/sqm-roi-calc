l# pages/sqm-calc.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.express as px

# ─────────────────────────  Page & styling  ─────────────────────────
st.set_page_config(page_title="Sales‑per‑sqm Potentieel (CSm²I)", page_icon="📈", layout="wide")
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

st.title("Sales‑per‑sqm Potentieel (CSm²I)")

# ─────────────────────────  Helpers  ─────────────────────────
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(x):   # € 12.345
    try:
        return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "€0"

def fmt_eur2(x):  # € 12.345,67
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
    # conversie naar fractie
    if out["conversion_rate"].max() > 1.5:
        out["conversion_rate"] = out["conversion_rate"]/100.0
    else:
        out["conversion_rate"] = out["conversion_rate"].fillna(
            out["transactions"]/(out["count_in"]+EPS)
        )
    # SPV
    if out["sales_per_visitor"].isna().all() or out["sales_per_visitor"].eq(0).all():
        out["sales_per_visitor"] = out["turnover"]/(out["count_in"]+EPS)
    # ATV
    out["atv"] = out["turnover"]/(out["transactions"]+EPS)
    # m² fallback
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

# ─────────────────────────  Shop mapping  ─────────────────────────
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP  # {id:int: "Naam"}
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}

# ─────────────────────────  Inputs  ─────────────────────────
period_label = st.selectbox("Select period", ["7_dagen","30_dagen","last_month","this_month"], index=1,
                            format_func=lambda s: {"7_dagen":"7 dagen","30_dagen":"30 dagen",
                                                   "last_month":"last_month","this_month":"this_month"}[s])
selected = st.multiselect("Select stores", sorted(NAME_TO_ID.keys(), key=str.lower),
                          default=sorted(NAME_TO_ID.keys(), key=str.lower)[:5])
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
    params = [("source","shops"),("period","date"),
              ("form_date_from",str(dfrom)),("form_date_to",str(dto)),("period_step",step)]
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

    # periode
    today = date.today()
    if period_label == "last_month":
        first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last  = today.replace(day=1) - timedelta(days=1)
        date_from, date_to = first, last
    elif period_label == "this_month":
        date_from, date_to = today.replace(day=1), today
    elif period_label == "7_dagen":
        date_from, date_to = today - timedelta(days=7), today - timedelta(days=1)
    else:  # 30_dagen
        date_from, date_to = today - timedelta(days=30), today - timedelta(days=1)

    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, "day", outputs)
        df = normalize_resp(resp)
        if df.empty:
            st.info("Geen data voor de gekozen periode."); st.stop()

    # berekeningen
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

    # KPI‑tegels (boven)
    k1,k2,k3 = st.columns(3)
    k1.markdown(f'<div class="card"><div>🚀 <b>CSm²I potential</b><br/><small>({period_label.replace("_"," ")}, target {csm2i_target:.2f})</small></div><div class="kpi">{fmt_eur(agg["uplift_csm"].sum())}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><div>🎯 <b>Conversion potential</b><br/><small>({period_label.replace("_"," ")}, doel = {conv_goal_pct}%)</small></div><div class="kpi">{fmt_eur(agg["uplift_conv"].sum())}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><div>∑ <b>Total potential</b><br/><small>({period_label.replace("_"," ")})</small></div><div class="kpi">{fmt_eur(agg["uplift_total"].sum())}</div></div>', unsafe_allow_html=True)

    # Oranje blok(ken)
    c1, c2 = st.columns([1.25,1])
    with c1:
        st.markdown(f"""
        <div class="block-orange">
          <div style="font-weight:700;font-size:1.05rem">💰 Total extra potential in revenue</div>
          <div class="kpi" style="margin-top:4px">{fmt_eur(agg["uplift_total"].sum())}</div>
          <div class="note">Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        if proj_toggle:
            # resterende dagen in huidig jaar
            remain = (date(today.year,12,31) - today).days
            yearly_proj = agg["uplift_total"].sum() * max(remain,0)
            st.markdown(f"""
            <div class="block-orange">
              <div style="font-weight:700;font-size:1.05rem">📈 Projectie resterend jaar</div>
              <div class="kpi" style="margin-top:4px">{fmt_eur(yearly_proj)}</div>
              <div class="note">Huidig potentieel × resterende dagen dit jaar.</div>
            </div>""", unsafe_allow_html=True)

    # ► gewenste witregel
    st.markdown('<div class="h-gap"></div>', unsafe_allow_html=True)

    # Tabel "Stores with most potential" (met nette notatie)
    tbl = agg.copy()
    tbl_disp = pd.DataFrame({
        "Store": tbl["shop_name"],
        "Square meters": tbl["sqm"].round(0).astype(int),
        "Current Avg Sales per sqm": tbl["spsqm"].map(fmt_eur2),
        "CSm²I (index)": tbl["csm2i"].round(3),
        "Uplift from CSm²I (€)": tbl["uplift_csm"].map(fmt_eur),
        "Uplift from Conversion (€)": tbl["uplift_conv"].map(fmt_eur),
        "Total Potential Uplift (€)": tbl["uplift_total"].map(fmt_eur),
    })
    st.dataframe(tbl_disp, use_container_width=True)

    # Bar: CSm²I‑potentieel per store
    bar1 = agg.sort_values("uplift_csm", ascending=False)
    fig1 = px.bar(bar1, x="shop_name", y="uplift_csm",
                  labels={"shop_name":"Store","uplift_csm":"CSm²I potential (€)"},
                  text=bar1["uplift_csm"].map(lambda v: ("{:,.0f}".format(v)).replace(",", ".")))
    fig1.update_traces(textposition="outside")
    fig1.update_yaxes(tickformat="~s")
    st.plotly_chart(fig1, use_container_width=True)

    # Bar: Conversion‑potentieel per store
    bar2 = agg.sort_values("uplift_conv", ascending=False)
    fig2 = px.bar(bar2, x="shop_name", y="uplift_conv",
                  labels={"shop_name":"Store","uplift_conv":"Conversion potential (€)"},
                  text=bar2["uplift_conv"].map(lambda v: ("{:,.0f}".format(v)).replace(",", ".")))
    fig2.update_traces(textposition="outside")
    fig2.update_yaxes(tickformat="~s")
    st.plotly_chart(fig2, use_container_width=True)