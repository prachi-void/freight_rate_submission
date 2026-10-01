
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CAT_COLS = ["pickup", "delivery", "equipment", "route"]

def engineer(df):
    x = df.drop(columns=["load_id", "posted_rate"], errors="ignore").copy()
    d = pd.to_datetime(x["date"])
    x = x.drop(columns=["date"])
    x["year"] = d.dt.year
    x["month"] = d.dt.month
    x["day"] = d.dt.day
    x["dow"] = d.dt.dayofweek
    x["doy"] = d.dt.dayofyear
    x["weekofyear"] = d.dt.isocalendar().week.astype(int)
    x["is_weekend"] = (x["dow"] >= 5).astype(int)
    x["doy_sin"] = np.sin(2 * np.pi * x["doy"] / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * x["doy"] / 365.25)
    x["dow_sin"] = np.sin(2 * np.pi * x["dow"] / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x["dow"] / 7)
    x["route"] = x["pickup"].astype(str) + "__" + x["delivery"].astype(str)

    lat1 = np.radians(x["pickup_lat"])
    lat2 = np.radians(x["delivery_lat"])
    dlat = lat2 - lat1
    dlon = np.radians(x["delivery_lon"] - x["pickup_lon"])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    x["geo_distance"] = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    x["distance_per_mile_geo"] = x["distance"] / (x["geo_distance"] * 0.621371 + 1)

    x["weight_missing"] = x["weight"].isna().astype(int)
    x["market_missing"] = x["market_index"].isna().astype(int)
    for c in ["weight", "market_index"]:
        x[c] = x[c].fillna(x[c].median())
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model-out", default="freight_rate_model.cbm")
    args = ap.parse_args()
    data = Path(args.data_dir)

    train = pd.read_csv(data / "train_test.csv")
    validation = pd.read_csv(data / "validation.csv")
    december = pd.read_csv(data / "december_chart_inputs.csv")

    X = engineer(train)
    Xv = engineer(validation)
    cats = [X.columns.get_loc(c) for c in CAT_COLS]

    model = CatBoostRegressor(
        iterations=520, depth=8, learning_rate=0.04,
        loss_function="RMSE", l2_leaf_reg=8,
        random_seed=42, verbose=False, thread_count=4
    )
    model.fit(X, np.log1p(train["posted_rate"]), cat_features=cats)
    model.save_model(args.model_out)

    validation_pred = np.maximum(np.expm1(model.predict(Xv)), 0.01)
    pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": validation_pred
    }).to_csv("validation_predictions.csv", index=False)

    # December chart inputs contain only the fields intended to be fixed in the chart.
    cities = pd.concat([
        train[["pickup", "pickup_lat", "pickup_lon"]].rename(
            columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
        train[["delivery", "delivery_lat", "delivery_lon"]].rename(
            columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"})
    ]).groupby("city")[["lat", "lon"]].median()

    d = december.copy()
    d["pickup_lat"] = d["pickup"].map(cities["lat"])
    d["pickup_lon"] = d["pickup"].map(cities["lon"])
    d["delivery_lat"] = d["delivery"].map(cities["lat"])
    d["delivery_lon"] = d["delivery"].map(cities["lon"])
    d["market_index"] = train["market_index"].median()
    d["quote_signal"] = train["quote_signal"].median()

    Xd = engineer(d).reindex(columns=X.columns)
    december["predicted_rate"] = np.maximum(np.expm1(model.predict(Xd)), 0.01)
    december.to_csv(data / "december_chart_inputs.csv", index=False)

    print("Wrote validation_predictions.csv")
    print("Updated data/december_chart_inputs.csv")
    print(f"Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
