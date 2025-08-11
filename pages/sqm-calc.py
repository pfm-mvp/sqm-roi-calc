# pages/10_Retail_Performance_Radar.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.express as px

# =========================
# Page & styling
# =========================
st.set_page_config(page_title="Retail Performance Radar", page_icon="📊", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }

/* Tags (store chips) */
[data-baseweb="tag"] { background-color: #9E77ED !important; color: white !important; }

/* PFM red button */
button[data-testid="stBaseButton-secondary"] {
  background-color: #F04438 !important; color: white !important; border-radius: 16px !important;
  font-weight: 600 !important; padding: 0.6rem 1.4rem !important; border: none !important;
}
button[data-testid="stBaseButton-secondary"]:hover { background-color: #d13c30 !important; cursor: pointer; }

/* Cards */
.card { border: 1px solid #eee; border-radius: 12px; padding: 14px 16px; background:#fff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.kpi  { font-size: 1.2rem; font-weight: 700; }
.eur  { font-variant-numeric: tabular-nums; }

/* Orange summary band */
.summary {
  border:1px solid #FEAC76; background: #FFF6EF; border-radius:14px; padding:18px 18px;
}

/* Recommendation card */
.nb-card {
  border:1px solid #e9e9e9; border-radius:14px; padding:14px 16px; background:#fff;
}

/* Small muted text */
.muted { color:#6b7280; font-size:.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Retail Performance Radar")
st.caption("Next Best Action • Best Practice Finder • (optioneel) Demografiepatronen")

# =========================
# Helpers
# =========================
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(x: float) -> str:
    return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")

def fmt_eur2(x: float) -> str:
    return ("€{:,.2f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce_numeric(df, ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"])
    # conversie naar fractie
    if "conversion_rate" in out.columns and not out["conversion_rate"].empty:
        if out["conversion_rate"].max() > 1.5:
            out["conversion_rate"] = out["conversion_rate"] / 100.0
    else:
        out["conversion_rate"] = out.get("transactions", 0.0) / (out.get("count_in", 0.0) + EPS)
    # SPV
    if ("sales_per_visitor" not in out.columns) or out["sales_per_visitor"].isna().all():
        out["sales_per_visitor"] = out.get("turnover", 0.0) / (out.get("count_in", 0.0) + EPS)
    # ATV
    out["atv"] = out.get("turnover", 0.0) / (out.get("transactions", 0.0) + EPS)
    # m² fallback
    if "sq_meter" in out.columns:
        sqm = pd.to_numeric(out["sq_meter"], errors="coerce")
        med = sqm.replace(0, np.nan).median()
        fallback = med if pd.notnull(med) and med > 0 else DEFAULT_SQ_METER
        out["sq_meter"] = sqm.replace(0, np.nan).fillna(fallback)
    else:
        out["sq_meter"] = DEFAULT_SQ_METER
    return out

def choose_ref_spv(df: pd.DataFrame, mode="portfolio", benchmark_shop_id=None, manual_spv=None, uplift_pct=0.0):
    safe = df.copy()
    for c in ["turnover","count_in","shop_id"]:
        if c in safe.columns:
            safe[c] = pd.to_numeric(safe[c], errors="coerce").fillna(0.0)
        else:
            safe[c] = 0.0

    def spv_of(frame: pd.DataFrame) -> float:
        visitors = float(frame["count_in"].sum())
        turnover = float(frame["turnover"].sum())
        return 0.0 if visitors <= 0 else turnover / (visitors + EPS)

    if mode == "benchmark" and benchmark_shop_id is not None and int(benchmark_shop_id) in safe["shop_id"].astype(int).values:
        sub = safe[safe["shop_id"].astype(int) == int(benchmark_shop_id)]
        base = spv_of(sub)
    elif mode == "manual" and manual_spv is not None:
        base = float(manual_spv)
    else:
        base = spv_of(safe)

    base = max(0.0, float(base))
    return base * (1.0 + float(uplift_pct))

def compute_csm2i_and_uplift(df: pd.DataFrame, ref_spv: float, csm2i_target: float):
    """
    Uniform:
      actual_spv = turnover / visitors
      CSm²I = actual_spv / ref_spv
      expected_spsqm = ref_spv * visitors_per_sqm
      uplift_eur_csm = max(0, visitors * (csm2i_target*ref_spv - actual_spv))
    """
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"] / (out["visitors"] + EPS)
    out["csm2i"]      = out["actual_spv"] / (float(ref_spv) + EPS)

    out["visitors_per_sqm"] = out["count_in"] / (out["sq_meter"] + EPS)
    out["actual_spsqm"]     = out["turnover"]  / (out["sq_meter"] + EPS)
    out["expected_spsqm"]   = float(ref_spv)   * out["visitors_per_sqm"]

    out["uplift_eur_csm"]   = np.maximum(0.0, out["visitors"] * (float(csm2i_target) * float(ref_spv) - out["actual_spv"]))
    return out

# =========================
# Shop mapping
# =========================
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP  # {id:int: "Naam"}
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}
names = sorted(NAME_TO_ID.keys(), key=str.lower)

# =========================
# UI – periode, granulariteit, winkels, targets
# =========================
c1, c2, c3 = st.columns([1,1,1])
with c1:
    period_label = st.selectbox("Periode", ["7 dagen", "30 dagen", "last_month"], index=0)
with c2:
    gran = st.selectbox("Granulariteit", ["Dag", "Uur"], index=0)
with c3:
    pass

# datums
today = date.today()
if period_label == "last_month":
    first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    last  = today.replace(day=1) - timedelta(days=1)
    date_from, date_to = first, last
elif period_label == "30 dagen":
    date_from, date_to = today - timedelta(days=30), today - timedelta(days=1)
else:
    date_from, date_to = today - timedelta(days=7), today - timedelta(days=1)

c4, c5, c6 = st.columns([1,1,1])
with c4:
    conv_goal_pct = st.slider("Conversiedoel (%)", 1, 80, 20, 1)
with c5:
    spv_uplift_pct = st.slider("SPV‑uplift (%)", 0, 100, 10, 1)
with c6:
    csm2i_target = st.slider("CSm²I‑target", 0.10, 2.00, 1.00, 0.05)

st.markdown("### Selecteer winkels")
selected_names = st.multiselect("Selecteer winkels", names, default=names[:5], placeholder="Kies 1 of meer winkels…")
shop_ids = [NAME_TO_ID[n] for n in selected_names]

# Analyseer‑knop (links)
analyze = st.button("🔍 Analyseer", type="secondary")

# =========================
# API helpers
# =========================
def fetch_report(api_url, shop_ids, dfrom, dto, step, outputs, timeout=60):
    params = [("source","shops"), ("period","date"),
              ("form_date_from",str(dfrom)), ("form_date_to",str(dto)), ("period_step",step)]
    for sid in shop_ids: params.append(("data", int(sid)))
    for outp in outputs: params.append(("data_output", outp))
    r = requests.post(api_url, params=params, timeout=timeout); r.raise_for_status()
    return r.json()

def normalize_resp(resp):
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

# =========================
# RUN
# =========================
if analyze:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()
    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    step = "hour" if gran.lower().startswith("u") else "day"
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, step, outputs)
        df = normalize_resp(resp)
        if df.empty:
            st.info("Geen data beschikbaar voor de gekozen periode."); st.stop()

    # Referentie‑SPV: portfolio + uplift
    mode = "portfolio"
    bm_id = None
    ref_spv = choose_ref_spv(df, mode=mode, benchmark_shop_id=bm_id, uplift_pct=spv_uplift_pct/100.0)

    # Uniforme CSm²I + uplift (CSm²I‑component)
    df = compute_csm2i_and_uplift(df, ref_spv=ref_spv, csm2i_target=csm2i_target)

    # Conversie‑uplift (optioneel onderdeel)
    conv_target = float(conv_goal_pct) / 100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    # Aggregatie per winkel
    agg = df.groupby(["shop_id","shop_name"]).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
        sqm=("sq_meter","mean"),
        spsqm=("actual_spsqm","mean"),
        csm2i=("csm2i","mean"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
        spv=("actual_spv","mean")
    ).reset_index()
    agg["uplift_total"] = agg["uplift_csm"] + agg["uplift_conv"]

    # ===== KPI‑tegels (bovenaan) =====
    k1, k2, k3 = st.columns(3)
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

    # ===== Oranje samenvatting (zoals in SQM‑calc) =====
    st.markdown(
        f"""
        <div class="summary">
          <div style="font-weight:700; font-size:1.05rem;">💰 Total extra potential in revenue</div>
          <div style="font-size:1.4rem; font-weight:800; margin-top:4px;">{fmt_eur(agg["uplift_total"].sum())}</div>
          <div class="muted" style="margin-top:6px;">Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")  # kleine spacer

    # ===== Scatter: SPV vs Sales per m² (kleur = CSm²I t.o.v. target) =====
    rad = df.groupby(["shop_id", "shop_name"]).agg(
        spv=("actual_spv", "mean"),
        spsqm=("actual_spsqm", "mean"),
        csi=("csm2i", "mean"),
        visitors=("count_in", "sum"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
    ).reset_index()
    rad["uplift_total"] = rad["uplift_csm"] + rad["uplift_conv"]

    # Bubbelgrootte: uplift_total
    rad["size_metric"] = rad["uplift_total"].fillna(0.0)
    rad["hover_size"] = rad["size_metric"].round(0).map(lambda v: ("€{:,.0f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))

    # Band t.o.v. target (±5%)
    low_thr  = float(csm2i_target) * 0.95
    high_thr = float(csm2i_target) * 1.05
    rad["csm2i_band"] = np.select(
        [rad["csi"] < low_thr, rad["csi"] > high_thr],
        ["Onder target", "Boven target"],
        default="Rond target",
    )

    # Hover‑velden (vast volgorde)
    rad["hover_spv"]   = rad["spv"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
    rad["hover_spsqm"] = rad["spsqm"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
    rad["hover_csi"]   = rad["csi"].round(2).map(lambda v: str(v).replace(".", ","))

    color_map  = {"Onder target": "#F04438", "Rond target": "#F59E0B", "Boven target": "#16A34A"}
    symbol_map = {"Onder target": "diamond", "Rond target": "circle", "Boven target": "square"}

    scatter = px.scatter(
        rad,
        x="spv", y="spsqm", size="size_metric",
        color="csm2i_band", symbol="csm2i_band",
        color_discrete_map=color_map, symbol_map=symbol_map,
        hover_data=["hover_spv", "hover_spsqm", "hover_csi", "hover_size"],
        labels={"spv": "Sales per Visitor", "spsqm": "Sales per m²", "csm2i_band": "CSm²I t.o.v. target"},
    )
    scatter.update_traces(
        hovertemplate="<b>%{text}</b><br>" +
                      "SPV: %{customdata[0]}<br>" +
                      "Sales per m²: %{customdata[1]}<br>" +
                      "CSm²I: %{customdata[2]}<br>" +
                      "Uplift: %{customdata[3]}<extra></extra>",
        text=rad["shop_name"],
    )
    scatter.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=520,
        legend_title_text="CSm²I t.o.v. target",
        xaxis=dict(title="Sales per Visitor (€/bezoeker)", tickformat=",.2f"),
        yaxis=dict(title="Sales per m² (€/m²)", tickformat=",.2f"),
    )
    st.plotly_chart(scatter, use_container_width=True)

    # ===== Aanbevelingen per winkel (met kleurindicaties) =====
    st.markdown("### Aanbevelingen per winkel")
    # thresholds voor kleur
    def dot_for(value, target):
        if value >= target*1.05:  # goed
            return "🟢"
        if value <= target*0.95:  # onder target
            return "🔴"
        return "🟠"

    # bereken benodigde context per winkel
    agg = agg.sort_values("uplift_total", ascending=False)
    for _, r in agg.iterrows():
        name = r["shop_name"]
        csi  = float(r["csm2i"])
        spsqm = float(r["spsqm"])
        spv  = float(r["spv"])
        up_c = float(r["uplift_csm"])
        up_v = float(r["uplift_conv"])
        total= float(r["uplift_total"])

        csi_dot = dot_for(csi, csm2i_target)

        st.markdown(
            f"""<div class="nb-card">
            <div style="display:flex;gap:12px;align-items:center;">
              <div style="font-weight:700;">{name}</div>
              <div class="muted">CSm²I: {csi:.2f} {csi_dot} • SPV: {fmt_eur2(spv)}/bezoeker • Sales/m²: {fmt_eur2(spsqm)}</div>
            </div>
            <div class="muted" style="margin-top:6px;">Potentiële uplift: <b>{fmt_eur(total)}</b> (CSm²I: {fmt_eur(up_c)}, Conversie: {fmt_eur(up_v)})</div>
            <ul style="margin:8px 0 0 18px;">
              <li>CSm²I {('onder' if csi_dot=='🔴' else 'rond' if csi_dot=='🟠' else 'boven')} target → focus op <b>SPV</b> (upsell/cross‑sell), bundels.</li>
              <li>Conversie naar doel {conv_goal_pct}% → extra bezetting op piekuren, instap‑promo.</li>
              <li>Sales per m² = {fmt_eur2(spsqm)} → check best‑practice winkel voor inspiratie.</li>
            </ul>
            </div>""",
            unsafe_allow_html=True
        )

    # ===== Debug (optioneel) =====
    with st.expander("🛠️ Debug"):
        dbg = {
            "period_step": step,
            "from": str(date_from),
            "to": str(date_to),
            "shop_ids": shop_ids,
            "ref_spv": ref_spv,
            "csm2i_target": csm2i_target,
            "conv_goal_pct": conv_goal_pct,
        }
        st.json(dbg)
