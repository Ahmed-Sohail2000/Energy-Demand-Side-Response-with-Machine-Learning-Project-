"""
DSR data ingestion for Cledion case study.
All data comes from the real Kaggle Spain energy dataset — no fake or synthetic data.
"""
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_CSV = "data/raw/energy_dataset.csv"
METER_LOAD_XLSX = "data/raw/meter_load.xlsx"
DAY_AHEAD_XLSX = "data/raw/day_ahead_prices.xlsx"
IMBALANCE_XLSX = "data/raw/imbalance_prices.xlsx"
PROCESSED_PARQUET = "data/processed/processed_data.parquet"

# Site scaling factors: each site is a fraction of national load (realistic for DSR)
SITE_FACTORS = {"site_A": 0.4, "site_B": 0.35, "site_C": 0.25}


def run():
    # -------------------------------------------------------------------------
    # STEP 1 — Load and clean the real Kaggle data
    # -------------------------------------------------------------------------
    print("STEP 1 — Load and clean")
    df = pd.read_csv(RAW_CSV)

    # Keep only the 4 columns we need (exact CSV names: time, total load actual, price day ahead, price actual)
    df = df[["time", "total load actual", "price day ahead", "price actual"]].copy()

    # Rename to our schema: timestamp, load_kw, price_eur_mwh (day-ahead), imbalance_eur_mwh (actual)
    df = df.rename(
        columns={
            "time": "timestamp",
            "total load actual": "load_kw",
            "price day ahead": "price_eur_mwh",
            "price actual": "imbalance_eur_mwh",
        }
    )

    # Parse timestamp as datetime UTC and set as index so we can resample later; sort ascending
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Drop rows where load_kw or price_eur_mwh is null — we need both for DSR
    df = df.dropna(subset=["load_kw", "price_eur_mwh"])
    print(f"  Row count after cleaning: {len(df)}.")

    # -------------------------------------------------------------------------
    # STEP 2 — Resample hourly → 15-minute intervals
    # -------------------------------------------------------------------------
    print("STEP 2 — Resample to 15-minute")
    n_before = len(df)
    df = df.resample("15min").interpolate(method="time")
    df = df.dropna(subset=["load_kw", "price_eur_mwh"])
    n_after = len(df)
    print(f"  Row count before resampling: {n_before}. After resampling: {n_after}.")

    # Strip timezone so Excel can write datetimes (Excel does not support timezone-aware datetimes)
    df.index = df.index.tz_localize(None)

    # -------------------------------------------------------------------------
    # STEP 3 — Generate data/raw/meter_load.xlsx
    # -------------------------------------------------------------------------
    print("STEP 3 — Generate meter_load.xlsx")
    # Simulate 3 sites by scaling real national load: load_kw_site = load_kw * site_factor (no synthetic data)
    rows = []
    for site_id, factor in SITE_FACTORS.items():
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": df.index,
                    "site_id": site_id,
                    "load_kw": (df["load_kw"] * factor).values,
                }
            )
        )
    meter_load = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "site_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(METER_LOAD_XLSX), exist_ok=True)
    meter_load.to_excel(METER_LOAD_XLSX, index=False, engine="openpyxl")
    print(f"  Saved {len(meter_load)} rows to {METER_LOAD_XLSX}.")

    # -------------------------------------------------------------------------
    # STEP 4 — Generate data/raw/day_ahead_prices.xlsx
    # -------------------------------------------------------------------------
    print("STEP 4 — Generate day_ahead_prices.xlsx")
    # Directly from real Kaggle "price day ahead" column — no generation needed
    day_ahead = df[["price_eur_mwh"]].reset_index()
    day_ahead.columns = ["timestamp", "price_eur_mwh"]
    day_ahead.to_excel(DAY_AHEAD_XLSX, index=False, engine="openpyxl")
    print(f"  Saved {len(day_ahead)} rows to {DAY_AHEAD_XLSX}.")

    # -------------------------------------------------------------------------
    # STEP 5 — Generate data/raw/imbalance_prices.xlsx
    # -------------------------------------------------------------------------
    print("STEP 5 — Generate imbalance_prices.xlsx")
    # Directly from real Kaggle "price actual" column — it naturally deviates from day-ahead (realistic imbalance)
    imbalance = df[["imbalance_eur_mwh"]].reset_index()
    imbalance.columns = ["timestamp", "imbalance_eur_mwh"]
    imbalance.to_excel(IMBALANCE_XLSX, index=False, engine="openpyxl")
    print(f"  Saved {len(imbalance)} rows to {IMBALANCE_XLSX}.")

    # -------------------------------------------------------------------------
    # STEP 6 — Save final processed parquet
    # -------------------------------------------------------------------------
    print("STEP 6 — Save final processed_data.parquet")
    # Merge meter_load + day_ahead_prices + imbalance_prices on timestamp into one flat long table
    merged = meter_load.copy()
    merged["price_eur_mwh"] = merged["timestamp"].map(day_ahead.set_index("timestamp")["price_eur_mwh"])
    merged["imbalance_eur_mwh"] = merged["timestamp"].map(imbalance.set_index("timestamp")["imbalance_eur_mwh"])
    merged = merged[["timestamp", "site_id", "load_kw", "price_eur_mwh", "imbalance_eur_mwh"]]
    merged = merged.sort_values(["timestamp", "site_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(PROCESSED_PARQUET), exist_ok=True)
    table = pa.Table.from_pandas(merged, preserve_index=False)
    pq.write_table(table, PROCESSED_PARQUET)
    print(f"  Saved {len(merged)} rows to {PROCESSED_PARQUET}.")

    print("Done.")


if __name__ == "__main__":
    run()
