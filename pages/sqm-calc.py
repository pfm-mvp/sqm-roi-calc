def _to_float(x):
if x is None or x == "":
return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan
    try: return float(x)
    except Exception: return np.nan

def _eur(num, decimals=2):
    if pd.isna(num):
        return ""
    if pd.isna(num): return ""
s = f"{float(num):,.{decimals}f}"
return "€" + s.replace(",", "X").replace(".", ",").replace("X", ".")

# === API CALL (identiek aan je andere calcs: POST met params, ZONDER [] in keys) ===
# === API CALL (POST met params, ZONDER [] in keys) ===
def fetch_report(api_url: str, shop_ids: list[int], period: str, step: str, metrics: list[str]):
params = [("data", sid) for sid in shop_ids]
params += [("data_output", m) for m in metrics]
@@ -79,15 +76,11 @@ def fetch_report(api_url: str, shop_ids: list[int], period: str, step: str, metr
r = requests.post(api_url, params=params, timeout=40)
status = r.status_code
text_preview = r.text[:2000] if r.text else ""
    try:
        js = r.json()
    except Exception:
        js = {}

    req_info = {"url": api_url, "params_list": params}
    return js, req_info, status, text_preview
    try: js = r.json()
    except Exception: js = {}
    return js, {"url": api_url, "params_list": params}, status, text_preview

# === PARSER: ondersteunt geneste dates-structuur + aggregatie naar 1 regel per winkel ===
# === PARSER: geneste dates-structuur + aggregatie naar 1 regel per winkel ===
def parse_vemcount(payload: dict, shop_ids: list[int], fields: list[str], period_key: str) -> pd.DataFrame:
if not isinstance(payload, dict): return pd.DataFrame()
if "data" in payload and isinstance(payload["data"], dict) and period_key in payload["data"]:
@@ -100,39 +93,32 @@ def parse_vemcount(payload: dict, shop_ids: list[int], fields: list[str], period
for _, day_info in dates.items():
day_data = (day_info or {}).get("data", {}) or {}
row = {"shop_id": sid}
                for f in fields:
                    row[f] = _to_float(day_data.get(f))
                for f in fields: row[f] = _to_float(day_data.get(f))
if row.get("sq_meter") is None or np.isnan(row.get("sq_meter")):
sm = ((shop_block.get("data") or {}).get("sq_meter"))
row["sq_meter"] = _to_float(sm)
rows.append(row)
if not rows: return pd.DataFrame()
df = pd.DataFrame(rows)
        agg = {}
        for f in fields:
            if f in ["count_in", "turnover"]: agg[f] = "sum"
            elif f in ["sq_meter"]: agg[f] = "max"
            else: agg[f] = "mean"
        agg = {f: ("sum" if f in ["count_in","turnover"] else "max" if f=="sq_meter" else "mean") for f in fields}
return df.groupby("shop_id", as_index=False).agg(agg)

if "data" in payload and isinstance(payload["data"], dict):
flat = []
for sid in shop_ids:
rec = payload["data"].get(str(sid), {}) or {}
row = {"shop_id": sid}
            for f in fields:
                row[f] = _to_float(rec.get(f))
            for f in fields: row[f] = _to_float(rec.get(f))
flat.append(row)
return pd.DataFrame(flat)

return pd.DataFrame()

# === RUN SIMULATION (zelfde UX‑flow als je andere calcs) ===
# === RUN ===
if st.button("Analyseer", type="secondary"):
with st.spinner("Calculating hidden location potential..."):
metrics = ["count_in","sales_per_visitor","sales_per_sqm",
"conversion_rate","sales_per_transaction","turnover","sq_meter"]

payload, req_info, status, text_preview = fetch_report(API_URL, shop_ids, period, STEP, metrics)

with st.expander("🔧 Request/Response Debug"):
@@ -146,7 +132,6 @@ def parse_vemcount(payload: dict, shop_ids: list[int], fields: list[str], period
st.stop()

df = parse_vemcount(payload, shop_ids, fields=metrics, period_key=period)

if df.empty:
st.error("Geen data (na parsen/aggregatie). Check periode en debug.")
st.stop()
@@ -166,14 +151,15 @@ def parse_vemcount(payload: dict, shop_ids: list[int], fields: list[str], period
df["CSm2I"] = df["actual_spsqm"] / (df["expected_spsqm"] + eps)   # index; 1.00 = op niveau
df["uplift_eur"] = np.maximum(0.0, df["expected_spsqm"] - df["actual_spsqm"]) * sq

        # === KPI‑BANNER ===
        # === KPI‑BANNER (duidelijke tekst over periode & CSm²I) ===
total_extra = float(df["uplift_eur"].sum())
st.markdown(f"""
           <div style='background-color: #FEAC76; color: #000000;
                       padding: 1.5rem; border-radius: 0.75rem;
                        font-size: 1.25rem; font-weight: 600;
                        font-size: 1.05rem; font-weight: 600;
                       text-align: center; margin-bottom: 1.5rem;'>
                🚀 The potential revenue growth is <span style='font-size:1.5rem;'>{_eur(total_extra, 0)}</span>
                🚀 Potential revenue growth <span style='opacity:.75'>(if each store reaches <b>CSm²I = 1.0</b>, over <b>{period}</b>)</span><br/>
                <span style='font-size:1.6rem;'>{_eur(total_extra, 0)}</span>
           </div>
       """, unsafe_allow_html=True)

@@ -202,12 +188,21 @@ def parse_vemcount(payload: dict, shop_ids: list[int], fields: list[str], period
yaxis_title="Potential revenue uplift (€)")
st.plotly_chart(fig, use_container_width=True)

        # === SCATTER (Viridis + duidelijke legenda + slimme x-as) ===
        # === SCATTER (Viridis + slimme x/y-as schaal) ===
sc = df[["Store","sales_per_visitor","visitors_per_sqm","CSm2I","uplift_eur"]].copy()
sc["uplift_fmt"] = sc["uplift_eur"].map(lambda v: _eur(v, 0))
        x_min, x_max = float(np.nanmin(sc["sales_per_visitor"])), float(np.nanmax(sc["sales_per_visitor"]))
        span = max(0.01, x_max - x_min); pad = span * 0.05
        x_range = [x_min - pad, x_max + pad]

        # Slimme ranges: 2–98 percentiel met kleine marge → minimaliseert witte ruimte
        def smart_range(s):
            s = s.astype(float).replace([np.inf,-np.inf], np.nan).dropna()
            if s.empty: return None
            p2, p98 = np.percentile(s, [2, 98])
            span = max(0.001, p98 - p2)
            pad = span * 0.08
            return [p2 - pad, p98 + pad]

        x_rng = smart_range(sc["sales_per_visitor"])
        y_rng = smart_range(sc["visitors_per_sqm"])

fig2 = px.scatter(
sc, x="sales_per_visitor", y="visitors_per_sqm",
@@ -225,17 +220,19 @@ def parse_vemcount(payload: dict, shop_ids: list[int], fields: list[str], period
fig2.update_layout(
margin=dict(l=10,r=10,t=30,b=10),
xaxis_title="Sales per Visitor", yaxis_title="Visitors per m²",
            xaxis=dict(range=x_range),
            xaxis=dict(range=x_rng) if x_rng else None,
            yaxis=dict(range=y_rng) if y_rng else None,
coloraxis_colorbar=dict(title="CSm²I (index)")
)
st.plotly_chart(fig2, use_container_width=True)

        # === KORTE TOELICHTING ===
        # === KORTE TOELICHTING (wat is CSm²I en hoe uplift wordt berekend) ===
st.markdown(
"""
           **Toelichting**  
            - **CSm²I (index)**: gerealiseerd t.o.v. verwacht omzet per m². *1,00* = op verwachting; *<1,00* = onderprestatie.  
            - **Potential revenue uplift (€)**: indicatie van **extra omzet op jaarbasis** als de winkel naar verwacht niveau groeit
              (geëxtrapoleerd uit de gekozen periode, berekend als *(expected − actual) × m²*).  
            - **CSm²I (index)** = *actual sales per m²* ÷ *expected sales per m²*.  
              Het **verwachte** niveau = *(sales per visitor) × (visitors per m²)*.  
            - **Uplift (€)** = *(expected − actual) × m²*, dus het **extra omzetpotentieel in de gekozen periode**
              als de winkel naar **CSm²I = 1.0** groeit (zonder extra traffic).  
           """
)
