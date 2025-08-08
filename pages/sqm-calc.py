import streamlit as st
import pandas as pd
import requests
from shop_mapping import SHOP_NAME_MAP

# ✅ Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Instrument Sans', sans-serif !important;
    }
    [data-baseweb="tag"] {
        background-color: #9E77ED !important;
        color: white !important;
    }
    button[data-testid="stBaseButton-secondary"] {
        background-color: #F04438 !important;
        color: white !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        font-family: "Instrument Sans", sans-serif !important;
        padding: 0.6rem 1.4rem !important;
        border: none !important;
        box-shadow: none !important;
        transition: background-color 0.2s ease-in-out;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #d13c30 !important;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 📌 API URL uit Streamlit secrets
API_URL = st.secrets["API_URL"]

# 📌 Functie om data op te halen
def get_kpi_data_for_stores(shop_ids, period="last_month", step="day"):
    params = [("data[]", shop_id) for shop_id in shop_ids]
    params += [
        ("data_output[]", "count_in"),
        ("data_output[]", "sales_per_visitor"),
        ("data_output[]", "sales_per_sqm"),
        ("data_output[]", "conversion_rate"),
        ("data_output[]", "sales_per_transaction"),
        ("data_output[]", "turnover"),
        ("data_output[]", "sq_meter"),
        ("source", "shops"),
        ("period", period),
        ("period_step", step),
        ("weather", "0"),
        ("step", step)
    ]
    try:
        headers = {"Authorization": f"Bearer {st.secrets['API_BEARER']}"}
        response = requests.post(API_URL, params=params, headers=headers)
        if response.status_code == 200:
            full_response = response.json()
            if "data" in full_response and period in full_response["data"]:
                raw_data = full_response["data"][period]
                return normalize_vemcount_response(raw_data)
        else:
            st.error(f"❌ Error fetching data: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"🚨 API call exception: {e}")
    return pd.DataFrame()

# 📌 Normalisatie functie
def normalize_vemcount_response(response_json: dict) -> pd.DataFrame:
    rows = []
    for shop_id, shop_content in response_json.items():
        sq_meter = shop_content["data"].get("sq_meter", None)
        dates = shop_content.get("dates", {})
        for _, day_info in dates.items():
            data = day_info.get("data", {})
            rows.append({
                "shop_id": int(shop_id),
                "store_name": SHOP_NAME_MAP.get(int(shop_id), str(shop_id)),
                "date": data.get("dt"),
                "count_in": float(data.get("count_in", 0)),
                "sales_per_visitor": float(data.get("sales_per_visitor", 0)),
                "sales_per_sqm": float(data.get("sales_per_sqm", 0)),
                "conversion_rate": float(data.get("conversion_rate", 0)),
                "sales_per_transaction": float(data.get("sales_per_transaction", 0)),
                "turnover": float(data.get("turnover", 0)),
                "sq_meter": float(sq_meter or 0)
            })
    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

# 📌 UI: Selecties
st.title("📊 Sales-per-sqm Potential (CSm²I)")
period = st.selectbox("Select analysis period", ["last_month", "last_3_months", "last_year"])
selected_names = st.multiselect("Select stores", options=list(SHOP_NAME_MAP.values()), default=list(SHOP_NAME_MAP.values()))
shop_ids = [k for k, v in SHOP_NAME_MAP.items() if v in selected_names]

# 🔹 Nieuw: Slider voor doel-CSm²I
target_csm2i = st.slider("Target CSm²I value", min_value=0.10, max_value=1.00, value=1.00, step=0.05)

# 📌 Simulatie
if st.button("Run simulation"):
    with st.spinner("Calculating hidden location potential..."):
        df_kpi = get_kpi_data_for_stores(shop_ids, period=period, step="day")

    if not df_kpi.empty:
        avg_csm2i_current = df_kpi.groupby("store_name")["sales_per_sqm"].mean()
        uplift_data = []
        for store in avg_csm2i_current.index:
            current = avg_csm2i_current[store]
            uplift = max(target_csm2i - current, 0)
            uplift_eur = uplift * df_kpi[df_kpi["store_name"] == store]["sq_meter"].iloc[0] * df_kpi[df_kpi["store_name"] == store]["count_in"].mean()
            uplift_data.append({
                "Store": store,
                "Current Avg Sales per sqm": round(current, 2),
                "Target CSm²I": target_csm2i,
                "Uplift €/month": uplift_eur
            })

        df_results = pd.DataFrame(uplift_data)
        total_extra_turnover = df_results["Uplift €/month"].sum()

        st.markdown(f"""
            <div style='background-color: #FEAC76;
                        color: #000000;
                        padding: 1.5rem;
                        border-radius: 0.75rem;
                        font-size: 1.25rem;
                        font-weight: 600;
                        text-align: center;
                        margin-bottom: 1.5rem;'>
                🚀 Potential revenue growth for {period.replace("_", " ")}: 
                <span style='font-size:1.5rem;'>€{total_extra_turnover:,.0f}</span>
            </div>
         """, unsafe_allow_html=True)

        st.subheader("📈 Store Uplift Table")
        st.dataframe(df_results)
