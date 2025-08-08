import pandas as pd

def normalize_vemcount_response(response_json: dict) -> pd.DataFrame:
    rows = []

    for shop_id, shop_content in response_json.items():
        dates = shop_content.get("dates", {})
        for date_label, day_info in dates.items():
            data = day_info.get("data", {})

            row = {
                "shop_id": int(shop_id),
                "date": data.get("dt"),
                "turnover": float(data.get("turnover", 0)),
                "count_in": float(data.get("count_in", 0)),
                "conversion_rate": float(data.get("conversion_rate", 0)),
                "sales_per_transaction": float(data.get("sales_per_transaction") or 0),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df
