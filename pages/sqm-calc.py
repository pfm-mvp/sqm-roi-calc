# pages/sqm-calc.py
# ──────────────────────────────────────────────────────────────────────────────
# Sales‑per‑sqm Potentieel (CSm²I) – met conversiecomponent, projectie resterend
# jaar, nette formatting, PFM-styling, bar‑ & scattervisualisaties.
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta, datetime
import plotly.express as px

# ──────────────────────────────────────────────────────────────────────────────
# Page & styling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sales‑per‑sqm Potentieel (CSm²I)", page_icon="📈", layout="wide")

PFM_RED   = "#F04438"
PFM_PURPLE= "#9E77ED"
PFM_ORANGE= "#FEAC76"
PFM_EMBER = "#F59E0B"
PFM_GREEN = "#16A34A"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
[data-baseweb="tag"] { background-color: %s !important; color: #fff !important; }
button[data-testid="stBaseButton-secondary"]{
  background:%s !important;color:#fff !important;border-radius:16px !important;
  font-weight:600 !important;border:none !important;padding:.6rem 1.4rem !important;
}
button[data-testid="stBaseButton-secondary"]:hover{ background:#d13c30 !important; }
.card{border:1px solid #eee;border-radius:12px;padding:14px 16px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04);}
.kpi{font-size:1.2rem;font-weight:700;font-variant-numeric:tabular-nums;}
.note{color:#6b7280;font-size:.85rem}
.statbox{border:1px solid %s;background:#fff;border-radius:14px;padding:16px 18px}
.statbox h4{margin:0 0 8px 0}
</style>
""" % (PFM_PURPLE, PFM_RED, "#f1f1f1"), unsafe_allow_html=True)

st.title("Sales‑per‑sqm Potentieel (CSm²I)")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(v) -> str:
    try:
        return ("€{:,.0f}".format(float(v))).replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "€0"

def fmt_eur2(v) -> str:
    try:
        return ("€{:,.2f}".format(float(v))).replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "€0,00"

def fmt_pct(v) -> str:
    try:
        return str(round(float(v)*100,2)).replace(".", ",") + "%"
    except Exception:
        return "0%"

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce_numeric(df, ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"])
    # conversie naar fractie
    if "conversion_rate" in out and not out["conversion_rate"].empty:
        if out["conversion_rate"].max() > 1.5:
            out["conversion_rate"] = out["conversion_rate"]/100.0
    else:
        out["conversion_rate"] = out.get("transactions",0.0) / (out.get("count_in",0.0)+EPS)
    # sales per visitor fallback
    if ("sales_per_visitor" not in out.columns) or out["sales_per_visitor"].isna().all():
        out["sales_per_visitor"] = out.get("turnover",0.0) / (out.get("count_in",0.0)+EPS)
    # ATV
    out["atv"] = out.get("turnover",0.0) / (out.get("transactions",0.0)+EPS)
    # m² fallback
    if "sq_meter" in out.columns:
        sqm = pd.to_numeric(out["sq_meter"], errors="coerce")
        med = sqm.replace(0, np.nan).median()
        fallback = med if pd.notnull(med) and med>0 else DEFAULT_SQ_METER
        out["sq_meter"] = sqm.replace(0, np.nan).fillna(fallback)
    else:
        out["sq_meter"] = DEFAULT_SQ_METER
    return out

def choose_ref_spv(df: pd.DataFrame, mode="portfolio", benchmark_shop_id=None, uplift_pct=0.0):
    safe = df.copy()
    for c in ["turnover","count_in","shop_id"]:
        if c in safe.columns:
            safe[c] = pd.to_numeric(safe[c], errors="coerce").fillna(0.0)
        else:
            safe[c] = 0.0

    def spv_of(frame):
        visitors = float(frame["count_in"].sum())
        turnover = float(frame["turnover"].sum())
        return 0.0 if visitors<=0 else turnover / (visitors+EPS)

    if mode=="benchmark" and benchmark_shop_id is not None:
        mask = safe["shop_id"].astype(str)==str(int(benchmark_shop_id))
        base = spv_of(safe[mask])
    else:
        base = spv_of(safe)
    base = max(0.0, float(base))
    return base * (1.0 + float(uplift_pct))

def compute_csm2i_and_uplift(df, ref_spv: float, csm2i_target: float):
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"]/(out["visitors"]+EPS)
    out["csm2i"]      = out["actual_spv"]/(ref_spv+EPS)
    out["visitors_per_sqm"] = out["count_in"]/(out["sq_meter"]+EPS)
    out["actual_spsqm"]     = out["turnover"] /(out["sq_meter"]+EPS)
    out["expected_spsqm"]   = ref_spv * out["visitors_per_sqm"]
    # CSm²I-component in €
    out["uplift_eur_csm"]   = np.maximum(0.0, out["visitors"]*(csm2i_target*ref_spv - out["actual_spv"]))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Shop mapping
# ──────────────────────────────────────────────────────────────────────────────
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP  # {id:int: "Naam"}
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k,v in _MAP.items() if str(v).strip()}
NAME_TO_ID       = {v:k for k,v in SHOP_ID_TO_NAME.items()}

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([1,2])
with c1:
    period_label = st.selectbox("Select period", ["last_month","30_dagen","7_dagen"], index=0)
with c2:
    selected_names = st.multiselect(
        "Select stores",
        options=sorted(NAME_TO_ID.keys(), key=str.lower),
        default=list(sorted(NAME_TO_ID.keys(), key=str.lower))[:6],
        placeholder="Kies 1 of meer winkels…"
    )
shop_ids = [NAME_TO_ID[n] for n in selected_names]

use_benchmark = st.toggle("📌 Gebruik benchmark‑winkel i.p.v. vaste CSm²I‑target", value=False)
csm2i_target = st.slider("Target CSm²I", 0.10, 1.50, 0.85, 0.05)
conv_goal_pct = st.slider("Target conversion rate (%)", 1, 80, 35, 1)
spv_uplift_pct = st.slider("SPV‑uplift (%)", 0, 100, 0, 1)

proj_col = st.toggle("Toon projectie voor resterend jaar", value=True)

analyze = st.button("Analyseer", type="secondary")

# ──────────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────────
def fetch_report(api_url, shop_ids, dfrom, dto, step, outputs, timeout=60):
    params = [("source","shops"), ("period","date"),
              ("form_date_from",str(dfrom)), ("form_date_to",str(dto)), ("period_step",step)]
    for sid in shop_ids: params.append(("data", int(sid)))
    for outp in outputs: params.append(("data_output", outp))
    r = requests.post(api_url, params=params, timeout=timeout); r.raise_for_status()
    return r.json()

def normalize_resp(resp):
    rows = []
    data = (resp or {}).get("data", {})
    for _, shops in data.items():
        for sid, payload in (shops or {}).items():
            dates = (payload or {}).get("dates", {})
            for ts, obj in (dates or {}).items():
                rec = {"timestamp": ts, "shop_id": int(sid), "shop_name": SHOP_ID_TO_NAME.get(int(sid), str(sid))}
                rec.update((obj or {}).get("data", {}))
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = ts.dt.date
    df["hour"] = ts.dt.hour
    return df

# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────
if analyze:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()

    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    # Periode
    today = date.today()
    if period_label == "last_month":
        first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last  = today.replace(day=1) - timedelta(days=1)
        date_from, date_to = first, last
    elif period_label == "30_dagen":
        date_from, date_to = today - timedelta(days=30), today - timedelta(days=1)
    else:
        date_from, date_to = today - timedelta(days=7), today - timedelta(days=1)

    step = "day"
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, step, outputs)
        df = normalize_resp(resp)
        if df.empty:
            st.info("Geen data beschikbaar voor de gekozen periode."); st.stop()

    # Referentie‑SPV (portfolio + uplift)
    ref_spv = choose_ref_spv(df, mode="portfolio", uplift_pct=spv_uplift_pct/100.0)

    # Uniforme CSm²I + uplift
    df = compute_csm2i_and_uplift(df, ref_spv=ref_spv, csm2i_target=csm2i_target)

    # Conversiecomponent (€)
    conv_target = float(conv_goal_pct)/100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    # Agg per winkel
    agg = df.groupby(["shop_id","shop_name"]).agg(
        sqm=("sq_meter","mean"),
        spsqm=("actual_spsqm","mean"),
        csi=("csm2i","mean"),
        spv=("actual_spv","mean"),
        conv=("conversion_rate","mean"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
    ).reset_index()
    agg["uplift_total"] = agg["uplift_csm"] + agg["uplift_conv"]

    # KPI tegels (boven)
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"""<div class="card"><div>🚀 <b>CSm²I potential</b><br/><small>({period_label}, target {csm2i_target:.2f})</small></div>
                    <div class="kpi">{fmt_eur(agg["uplift_csm"].sum())}</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="card"><div>🎯 <b>Conversion potential</b><br/><small>({period_label}, doel = {conv_goal_pct}%)</small></div>
                    <div class="kpi">{fmt_eur(agg["uplift_conv"].sum())}</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="card"><div>∑ <b>Total potential</b><br/><small>({period_label})</small></div>
                    <div class="kpi">{fmt_eur(agg["uplift_total"].sum())}</div></div>""", unsafe_allow_html=True)

    # ORANJE widgets (totaal + projectie rest jaar)
    orange = f"border:1px solid {PFM_ORANGE}; background:#fff7ed;"
    o1, o2 = st.columns([1,1])
    with o1:
        st.markdown(
            f"""<div class="statbox" style="{orange}">
                <h4>💰 Total extra potential in revenue</h4>
                <div class="kpi">{fmt_eur(agg["uplift_total"].sum())}</div>
                <div class="note">Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.</div>
            </div>""", unsafe_allow_html=True
        )
    with o2:
        if proj_col:
            days_left = (date(today.year,12,31) - today).days
            per_day   = agg["uplift_total"].sum() / max(1,(date_to - date_from).days+1)
            projection= per_day * max(0, days_left)
            st.markdown(
                f"""<div class="statbox" style="{orange}">
                    <h4>📈 Projectie resterend jaar</h4>
                    <div class="kpi">{fmt_eur(projection)}</div>
                    <div class="note">Huidig potentieel × resterende dagen dit jaar.</div>
                </div>""", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div class="statbox" style="{orange}">
                    <h4>📈 Projectie resterend jaar</h4>
                    <div class="kpi">—</div>
                    <div class="note">Zet de toggle aan om de projectie te tonen.</div>
                </div>""", unsafe_allow_html=True
            )

    # Extra witregel (gevraagd)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Tabel (mooie formatting)
    tab = agg.copy().sort_values("uplift_total", ascending=False)
    styled = (
        tab[["shop_name","sqm","spsqm","csi","uplift_csm","uplift_conv","uplift_total"]]
        .rename(columns={
            "shop_name":"Store",
            "sqm":"Square meters",
            "spsqm":"Current Avg Sales per sqm",
            "csi":"CSm²I (index)",
            "uplift_csm":"Uplift from CSm²I (€)",
            "uplift_conv":"Uplift from Conversion (€)",
            "uplift_total":"Total Potential Uplift (€)",
        })
        .style
        .format({
            "Square meters":"{:,.0f}".format,
            "Current Avg Sales per sqm": lambda v: fmt_eur2(v),
            "CSm²I (index)": lambda v: str(round(float(v),3)).replace(".", ","),
            "Uplift from CSm²I (€)": lambda v: fmt_eur(v),
            "Uplift from Conversion (€)": lambda v: fmt_eur(v),
            "Total Potential Uplift (€)": lambda v: fmt_eur(v),
        })
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True)

    # Bar chart – CSm²I potential per store
    bar1 = px.bar(
        tab.sort_values("uplift_csm", ascending=False),
        x="shop_name", y="uplift_csm",
        labels={"shop_name":"Store", "uplift_csm":"CSm²I potential (€)"},
        title="CSm²I potential per store",
    )
    bar1.update_traces(marker_color=PFM_PURPLE, hovertemplate="%{x}<br>€%{y:,.0f}<extra></extra>")
    bar1.update_yaxes(tickprefix="€", separatethousands=True)
    st.plotly_chart(bar1, use_container_width=True)

    # Bar chart – Conversion potential per store
    bar2 = px.bar(
        tab.sort_values("uplift_conv", ascending=False),
        x="shop_name", y="uplift_conv",
        labels={"shop_name":"Store", "uplift_conv":"Conversion potential (€)"},
        title="Conversion potential per store",
    )
    bar2.update_traces(marker_color="#C084FC", hovertemplate="%{x}<br>€%{y:,.0f}<extra></extra>")
    bar2.update_yaxes(tickprefix="€", separatethousands=True)
    st.plotly_chart(bar2, use_container_width=True)

    # Scatter – SPV vs Sales per m², kleur tov target‑band
    rad = df.groupby(["shop_id","shop_name"]).agg(
        spv=("actual_spv","mean"),
        spsqm=("actual_spsqm","mean"),
        csi=("csm2i","mean"),
        visitors=("count_in","sum"),
    ).reset_index()

    size_series = tab.set_index("shop_id")["uplift_total"]
    rad["size_metric"] = rad["shop_id"].map(size_series).fillna(0.0)
    rad["hover_spv"]   = rad["spv"].round(2).apply(fmt_eur2)
    rad["hover_spsqm"] = rad["spsqm"].round(2).apply(fmt_eur2)
    rad["hover_csi"]   = rad["csi"].round(2).map(lambda v: str(v).replace(".", ","))  # index
    rad["hover_uplift"]= rad["size_metric"].round(0).apply(fmt_eur)

    low_thr, high_thr = float(csm2i_target)*0.95, float(csm2i_target)*1.05
    rad["band"] = np.select([rad["csi"]<low_thr, rad["csi"]>high_thr], ["Onder target","Boven target"], default="Rond target")
    color_map  = {"Onder target": PFM_RED, "Rond target": PFM_EMBER, "Boven target": PFM_GREEN}
    symbol_map = {"Onder target":"diamond","Rond target":"circle","Boven target":"square"}

    sc = px.scatter(
        rad, x="spv", y="spsqm", size="size_metric",
        color="band", symbol="band",
        color_discrete_map=color_map, symbol_map=symbol_map,
        hover_data=["hover_spv","hover_spsqm","hover_csi","hover_uplift"],
        labels={"spv":"Sales per Visitor","spsqm":"Sales per m²","band":"CSm²I t.o.v. target"},
    )
    sc.update_traces(
        text=rad["shop_name"],
        hovertemplate="<b>%{text}</b><br>SPV: %{customdata[0]}<br>Sales per m²: %{customdata[1]}<br>CSm²I: %{customdata[2]}<br>Uplift: %{customdata[3]}<extra></extra>"
    )
    sc.update_layout(margin=dict(l=20,r=20,t=30,b=10), height=520)
    st.plotly_chart(sc, use_container_width=True)
