# pages/sqm-calc.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.express as px

# ────────────────────────────────────────────────────────────────────────────────
# Page & styling
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sales‑per‑sqm Potential (CSm²I)", page_icon="📐", layout="wide")
st.markdown(
    """
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
.kpi  { font-size: 1.25rem; font-weight: 700; }
.eur  { font-variant-numeric: tabular-nums; }
.orange { background:#FEAC7622; border:1px solid #FEAC76; border-radius:12px; padding:16px; color:#222; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Sales‑per‑sqm Potentieel (CSm²I)")

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(x: float) -> str:
    try:
        return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")
    except Exception:
        return "€0"

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    return out

def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce_numeric(df, ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"])
    if "conversion_rate" in out.columns and out["conversion_rate"].max() > 1.5:
        out["conversion_rate"] = out["conversion_rate"]/100.0
    if ("sales_per_visitor" not in out.columns) or out["sales_per_visitor"].isna().all():
        out["sales_per_visitor"] = out["turnover"]/(out["count_in"]+EPS)
    out["atv"] = out.get("turnover",0.0)/(out.get("transactions",0.0)+EPS)
    if "sq_meter" in out.columns:
        sqm = pd.to_numeric(out["sq_meter"], errors="coerce")
        med = sqm.replace(0, np.nan).median()
        fallback = med if pd.notnull(med) and med>0 else DEFAULT_SQ_METER
        out["sq_meter"] = sqm.replace(0, np.nan).fillna(fallback)
    else:
        out["sq_meter"] = DEFAULT_SQ_METER
    return out

def choose_ref_spv(df: pd.DataFrame, mode="portfolio", benchmark_shop_id=None, manual_spv=None, uplift_pct=0.0):
    safe = coerce_numeric(df, ["turnover","count_in","shop_id"])
    def spv_of(frame: pd.DataFrame)->float:
        v = float(frame["count_in"].sum()); t = float(frame["turnover"].sum())
        return 0.0 if v<=0 else t/(v+EPS)

    if mode=="benchmark" and benchmark_shop_id is not None and int(benchmark_shop_id) in safe["shop_id"].astype(int).values:
        base = spv_of(safe[safe["shop_id"].astype(int)==int(benchmark_shop_id)])
    elif mode=="manual" and manual_spv is not None:
        base = float(manual_spv)
    else:
        base = spv_of(safe)

    base = max(0.0,float(base))
    return base*(1.0+float(uplift_pct))

def compute_csm2i(df: pd.DataFrame, ref_spv: float, csm2i_target: float):
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"]/(out["visitors"]+EPS)
    out["csm2i"]      = out["actual_spv"]/(float(ref_spv)+EPS)

    out["visitors_per_sqm"] = out["count_in"]/(out["sq_meter"]+EPS)
    out["actual_spsqm"]     = out["turnover"]/(out["sq_meter"]+EPS)
    out["expected_spsqm"]   = float(ref_spv)*out["visitors_per_sqm"]

    out["uplift_eur_csm"] = np.maximum(0.0, out["visitors"]*(float(csm2i_target)*float(ref_spv)-out["actual_spv"]))
    return out

def period_dates(label: str):
    today = date.today()
    if label == "last_month":
        first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last  = today.replace(day=1) - timedelta(days=1)
        return first, last
    if label == "30 dagen":
        return today - timedelta(days=30), today - timedelta(days=1)
    return today - timedelta(days=7), today - timedelta(days=1)

def projected_remaining_year_factor(d_from: date, d_to: date) -> float:
    analyzed_days = (d_to - d_from).days + 1
    if analyzed_days <= 0:
        return 1.0
    last_day = date(d_to.year, 12, 31)
    days_left = (last_day - d_to).days
    return max(0.0, days_left) / analyzed_days

# ────────────────────────────────────────────────────────────────────────────────
# Shop mapping
# ────────────────────────────────────────────────────────────────────────────────
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}
names = sorted(NAME_TO_ID.keys(), key=str.lower)

# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1,1,2])
with c1:
    period_label = st.selectbox("Select period", ["7 dagen","30 dagen","last_month"], index=2)
with c2:
    use_benchmark = st.toggle("📌 Gebruik benchmark‑winkel i.p.v. vaste CSm²I‑target", value=False)

c4, c5 = st.columns([1,1])
with c4:
    csm2i_target = st.slider("Target CSm²I", 0.10, 1.50, 0.85, 0.05)
with c5:
    conv_goal_pct = st.slider("Target conversion rate (%)", 1, 80, 35, 1)

st.markdown("### Select stores")
selected_names = st.multiselect("Select stores", names, default=names, placeholder="Kies winkels…")
shop_ids = [NAME_TO_ID[n] for n in selected_names]

proj_toggle = st.toggle("Projecteer naar resterend jaar", value=False)
analyze = st.button("🔍 Analyseer", type="secondary")

# ────────────────────────────────────────────────────────────────────────────────
# API helpers
# ────────────────────────────────────────────────────────────────────────────────
def fetch_report(api_url, shop_ids, dfrom, dto, outputs, timeout=60):
    params = [("source","shops"),("period","date"),
              ("form_date_from",str(dfrom)),("form_date_to",str(dto)),("period_step","day")]
    for sid in shop_ids: params.append(("data", int(sid)))
    for outp in outputs: params.append(("data_output", outp))
    r = requests.post(api_url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def normalize_resp(resp):
    rows = []
    data = (resp or {}).get("data", {})
    for _, shops in data.items():
        for sid, payload in (shops or {}).items():
            dates = (payload or {}).get("dates", {})
            for ts, obj in dates.items():
                rec = {"timestamp": ts, "shop_id": int(sid), "shop_name": SHOP_ID_TO_NAME.get(int(sid), str(sid))}
                rec.update((obj or {}).get("data", {}))
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = ts.dt.date
    return df

# ────────────────────────────────────────────────────────────────────────────────
# RUN
# ────────────────────────────────────────────────────────────────────────────────
if analyze:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()
    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    d_from, d_to = period_dates(period_label)
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, d_from, d_to, outputs)
        raw = normalize_resp(resp)
        if raw.empty:
            st.info("Geen data."); st.stop()

    # Portfolio‑SPV (benchmark later optioneel)
    ref_spv = choose_ref_spv(raw, mode="portfolio")

    df = compute_csm2i(raw, ref_spv=ref_spv, csm2i_target=csm2i_target)

    conv_target = float(conv_goal_pct)/100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    agg = df.groupby(["shop_id","shop_name"]).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
        sqm=("sq_meter","mean"),
        spsqm=("actual_spsqm","mean"),
        spv=("actual_spv","mean"),
        csm2i=("csm2i","mean"),
        conv=("conversion_rate","mean"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
    ).reset_index()
    agg["uplift_total"] = agg["uplift_csm"] + agg["uplift_conv"]

    proj_factor = projected_remaining_year_factor(d_from, d_to) if proj_toggle else 0.0
    proj_total  = agg["uplift_total"].sum()*proj_factor if proj_toggle else None

    # ── KPI‑tegels ──
    k1,k2,k3 = st.columns(3)
    k1.markdown(
        f"""<div class="card"><div>🚀 <b>CSm²I potential</b><br/><small>({period_label}, target {csm2i_target:.2f})</small></div>
            <div class="kpi eur">{fmt_eur(agg["uplift_csm"].sum())}</div></div>""",
        unsafe_allow_html=True
    )
    k2.markdown(
        f"""<div class="card"><div>🎯 <b>Conversion potential</b><br/><small>({period_label}, doel = {conv_goal_pct}%)</small></div>
            <div class="kpi eur">{fmt_eur(agg["uplift_conv"].sum())}</div></div>""",
        unsafe_allow_html=True
    )
    k3.markdown(
        f"""<div class="card"><div>∑ <b>Total potential</b><br/><small>({period_label})</small></div>
            <div class="kpi eur">{fmt_eur(agg["uplift_total"].sum())}</div></div>""",
        unsafe_allow_html=True
    )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    cols = st.columns([2,2,2])
    with cols[0]:
        st.markdown(
            f"""<div class="orange">
                <b>💰 Total extra potential in revenue</b><br/>
                <span class="kpi eur">{fmt_eur(agg["uplift_total"].sum())}</span><br/>
                <small>Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.</small>
            </div>""",
            unsafe_allow_html=True
        )
    if proj_toggle and proj_total is not None:
        with cols[1]:
            st.markdown(
                f"""<div class="orange">
                    <b>📈 Projectie resterend jaar</b><br/>
                    <span class="kpi eur">{fmt_eur(proj_total)}</span><br/>
                    <small>Huidig potentieel × resterende dagen dit jaar.</small>
                </div>""",
                unsafe_allow_html=True
            )

    # ── Tabel (met robuuste sort) ──
    table_df = (agg[["shop_name","sqm","spsqm","csm2i","uplift_csm","uplift_conv","uplift_total"]]
                .rename(columns={
                    "shop_name":"Store","sqm":"Square meters","spsqm":"Current Avg Sales per sqm",
                    "csm2i":"CSm²I (index)","uplift_csm":"Uplift from CSm²I (€)",
                    "uplift_conv":"Uplift from Conversion (€)","uplift_total":"Total Potential Uplift (€)"
                }))
    sort_col = "Total Potential Uplift (€)" if "Total Potential Uplift (€)" in table_df.columns else "uplift_total"
    table_df = table_df.sort_values(sort_col, ascending=False)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # ── Bar: CSm²I potential per store ──
    bar = px.bar(
        agg.sort_values("uplift_csm", ascending=False),
        x="shop_name", y="uplift_csm",
        labels={"shop_name":"Store","uplift_csm":"CSm²I potential (€)"},
    )
    bar.update_traces(hovertemplate="€%{y:,.0f}<extra>%{x}</extra>")
    bar.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=420, yaxis_tickformat=",.0f")
    st.plotly_chart(bar, use_container_width=True)

    # ── Scatter: SPV vs Sales per m² ──
    st.markdown("## 📈 SPV vs Sales per m² (kleur = CSm²I t.o.v. target)")
    rad = agg.copy()
    low_thr = float(csm2i_target)*0.95
    high_thr= float(csm2i_target)*1.05
    rad["csm2i_band"] = np.select(
        [rad["csm2i"]<low_thr, rad["csm2i"]>high_thr],
        ["Onder target","Boven target"], default="Rond target"
    )
    color_map  = {"Onder target":"#F04438","Rond target":"#F59E0B","Boven target":"#16A34A"}
    symbol_map = {"Onder target":"diamond","Rond target":"circle","Boven target":"square"}

    rad["hover_spv"]   = rad["spv"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
    rad["hover_spsqm"] = rad["spsqm"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
    rad["hover_csi"]   = rad["csm2i"].round(2).map(lambda v: str(v).replace(".", ","))
    rad["hover_uplift"]= rad["uplift_total"].round(0).map(lambda v: ("€{:,.0f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))

    sc = px.scatter(
        rad, x="spv", y="spsqm", size="uplift_total",
        color="csm2i_band", symbol="csm2i_band",
        color_discrete_map=color_map, symbol_map=symbol_map,
        hover_data=["hover_spv","hover_spsqm","hover_csi","hover_uplift"],
        labels={"spv":"Sales per Visitor","spsqm":"Sales per m²","csm2i_band":"CSm²I t.o.v. target"},
    )
    sc.update_traces(
        hovertemplate="<b>%{text}</b><br>" +
                      "SPV: %{customdata[0]}<br>" +
                      "Sales per m²: %{customdata[1]}<br>" +
                      "CSm²I: %{customdata[2]}<br>" +
                      "Uplift: %{customdata[3]}<extra></extra>",
        text=rad["shop_name"],
    )
    sc.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=520,
                     xaxis=dict(title="Sales per Visitor (€/bezoeker)", tickformat=",.2f"),
                     yaxis=dict(title="Sales per m² (€/m²)", tickformat=",.2f"))
    st.plotly_chart(sc, use_container_width=True)

    with st.expander("🛠️ Request/Response Debug"):
        st.json({"period": period_label, "from": str(d_from), "to": str(d_to),
                 "shops": shop_ids, "projection_factor": proj_factor if proj_toggle else 0.0 })
