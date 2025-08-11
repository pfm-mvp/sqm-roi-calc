# pages/sqm-calc.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

# =========================
# Page & styling
# =========================
st.set_page_config(page_title="Sales‑per‑sqm Potentieel (CSm²I)", page_icon="📈", layout="wide")
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
.kpi  { font-size: 1.2rem; font-weight: 700; }
.eur  { font-variant-numeric: tabular-nums; }

# kleine spacer tussen KPI-rij en widget
st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

# ===== Oranje total widget (PFM #FEAC76) =====
st.markdown(
    f"""
    <div style="
        background:#FEAC76;
        border:1px solid #E38F59;
        border-radius:14px;
        padding:16px 18px;
        color:#2B1B10;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <span style="font-size:20px;">💰</span>
        <h3 style="margin:0;font-size:1.05rem;">Total extra potential in revenue</h3>
      </div>
      <div style="font-size:1.6rem;font-weight:800;">{fmt_eur(agg["uplift_total"].sum())}</div>
      <div style="margin-top:4px;"><small>Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.</small></div>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Sales‑per‑sqm Potentieel (CSm²I)")

# =========================
# Helpers (gedeelde kern)
# =========================
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(x: float) -> str:
    return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")

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
    """
    Bepaalt referentie‑SPV (€/bezoeker): portfolio | benchmark | manual, incl. optionele uplift_pct.
    Robuust tegen niet‑numerieke data.
    """
    safe = df.copy()
    for c in ["turnover","count_in"]:
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
    Canonisch:
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
# UI – periode, winkels, targets
# =========================
# Periode‑select met preset (we vertalen naar date_from/to)
period_label = st.selectbox("Select period", ["last_month", "last_7_days", "last_30_days"], index=0)
today = date.today()
if period_label == "last_month":
    first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    last  = today.replace(day=1) - timedelta(days=1)
    date_from, date_to = first, last
elif period_label == "last_7_days":
    date_from, date_to = today - timedelta(days=7), today - timedelta(days=1)
else:
    date_from, date_to = today - timedelta(days=30), today - timedelta(days=1)

sel_names = st.multiselect("Select stores", names, default=names, placeholder="Kies 1 of meer winkels…")
shop_ids = [NAME_TO_ID[n] for n in sel_names]

st.toggle("📌 Gebruik benchmark‑winkel i.p.v. vaste CSm²I‑target", value=False, key="use_bm")
bm_name = st.selectbox("Benchmark‑winkel", names, index=0, disabled=not st.session_state.get("use_bm", False)) if names else None

csm2i_target = st.slider("Target CSm²I", 0.10, 2.00, 0.85, 0.05)
conv_target_pct = st.slider("Target conversion rate (%)", 1, 80, 35, 1)

run = st.button("Analyseer", type="secondary")

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
if run:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()
    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]
    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, "day", outputs)
        df = normalize_resp(resp)
        if df.empty:
            st.info("Geen data beschikbaar voor de gekozen periode."); st.stop()

    # Referentie‑SPV (keuze 1B: benchmark als aan, anders portfolio)
    mode = "benchmark" if st.session_state.get("use_bm", False) else "portfolio"
    bm_id = NAME_TO_ID.get(bm_name) if (mode == "benchmark" and bm_name) else None
    ref_spv = choose_ref_spv(df, mode=mode, benchmark_shop_id=bm_id, uplift_pct=0.0)

    # Uniforme CSm²I + uplift
    df = compute_csm2i_and_uplift(df, ref_spv=ref_spv, csm2i_target=csm2i_target)

    # ========== Conversie‑uplift ==========
    conv_target = float(conv_target_pct) / 100.0
    df["uplift_eur_conv"] = np.maximum(
        0.0,
        (conv_target - df["conversion_rate"]) * df["count_in"]
    ) * df["atv"]

    # ========== Aggregatie ==========
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

    # ===== KPI‑tegels =====
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"""<div class="card"><div>🚀 <b>CSm²I potential</b><br/><small>({period_label}, target CSm²I {csm2i_target:.2f})</small></div>
            <div class="kpi eur">{fmt_eur(agg["uplift_csm"].sum())}</div></div>""",
        unsafe_allow_html=True
    )
    c2.markdown(
        f"""<div class="card"><div>🎯 <b>Conversion potential</b><br/><small>({period_label}, target conv = {conv_target_pct}%)</small></div>
            <div class="kpi eur">{fmt_eur(agg["uplift_conv"].sum())}</div></div>""",
        unsafe_allow_html=True
    )
    c3.markdown(
        f"""<div class="card"><div>∑ <b>Total potential</b><br/><small>({period_label})</small></div>
            <div class="kpi eur">{fmt_eur(agg["uplift_total"].sum())}</div></div>""",
        unsafe_allow_html=True
    )

    # ===== Oranje total widget (bovenaan/extra) =====
    st.markdown(
        f"""<div class="total-widget">
              <h3>💰 Total extra potential in revenue</h3>
              <div class="val eur">{fmt_eur(agg["uplift_total"].sum())}</div>
              <small>Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.</small>
            </div>""",
        unsafe_allow_html=True
    )

    # ===== Tabel =====
    st.markdown("### 🏆 Stores with most potential")

    # Sorteer EERST op uplift_total, daarna pas hernoemen + formatteren
    tab_src = agg[[
        "shop_name","sqm","spsqm","csm2i","uplift_csm","uplift_conv","uplift_total"
    ]].sort_values("uplift_total", ascending=False)

    tab = tab_src.rename(columns={
        "shop_name": "Store",
        "sqm": "Square meters",
        "spsqm": "Current Avg Sales per sqm",
        "csm2i": "CSm²I (index)",
        "uplift_csm": "Uplift from CSm²I (€)",
        "uplift_conv": "Uplift from Conversion (€)",
        "uplift_total": "Total Potential Uplift (€)"
    }).copy()

    # opmaak € en decimaal
    for col in ["Uplift from CSm²I (€)", "Uplift from Conversion (€)", "Total Potential Uplift (€)"]:
        tab[col] = tab[col].round(0).apply(fmt_eur)

    tab["Current Avg Sales per sqm"] = tab["Current Avg Sales per sqm"].round(2).apply(
        lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X",".")
    )
    tab["CSm²I (index)"] = tab["CSm²I (index)"].round(2)

    st.dataframe(tab, use_container_width=True)

    # ===== Visuals =====
    import plotly.express as px

    # Bar: CSm²I potential per store
    bar_csm = px.bar(
        agg.sort_values("uplift_csm", ascending=False),
        x="shop_name", y="uplift_csm",
        labels={"shop_name":"Store","uplift_csm":"CSm²I potential (€)"},
    )
    bar_csm.update_traces(marker_color="#7C3AED")
    bar_csm.update_layout(margin=dict(l=20,r=20,t=20,b=20), height=420)
    st.plotly_chart(bar_csm, use_container_width=True)

    # Bar: Conversion potential per store
    bar_conv = px.bar(
        agg.sort_values("uplift_conv", ascending=False),
        x="shop_name", y="uplift_conv",
        labels={"shop_name":"Store","uplift_conv":"Conversion potential (€)"},
    )
    bar_conv.update_traces(marker_color="#A78BFA")
    bar_conv.update_layout(margin=dict(l=20,r=20,t=20,b=20), height=380)
    st.plotly_chart(bar_conv, use_container_width=True)

    # Bubble scatter: driver map
    driver_df = df.copy()
    driver_df["driver_csm_share"] = df["uplift_eur_csm"] / (df["uplift_eur_csm"] + df["uplift_eur_conv"] + EPS)
    driver_df["visitors_per_m2"]  = df["count_in"] / (df["sq_meter"] + EPS)
    driver_df["driver"] = np.select(
        [driver_df["driver_csm_share"] >= 0.65, driver_df["driver_csm_share"] <= 0.35],
        ["CSm²I-led","Conversion-led"], default="Mixed"
    )

    bubble = px.scatter(
        driver_df.groupby(["shop_id","shop_name"]).agg(
            spv=("actual_spv","mean"),
            vpm2=("visitors_per_m2","mean"),
            visitors=("count_in","sum"),
            csi=("csm2i","mean"),
            driver=("driver", lambda s: s.value_counts().idxmax())
        ).reset_index(),
        x="spv", y="vpm2", size="visitors", color="driver",
        hover_data={"shop_name":True,"csi":":.2f","spv":":.2f","vpm2":":.0f"},
        labels={"spv":"Sales per Visitor","vpm2":"Visitors per m²"},
    )
    bubble.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=520)
    st.plotly_chart(bubble, use_container_width=True)

    # Toelichting
    st.markdown("""
**Toelichting**

- **CSm²I (index)** = *actual SPV* ÷ *ref‑SPV*. Verwacht per m² = **ref‑SPV × visitors per m²**.  
- **CSm²I potential (€)** = *(target_CSm²I × expected − actual)* × m² → extra omzet in de gekozen periode.  
- **Conversion potential (€)** = *(target_conv − huidige conv) × bezoekers × ATV*.  
- **Total** = CSm²I‑uplift + conversie‑uplift (geen dubbelcounting).
""")
