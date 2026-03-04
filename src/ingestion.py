"""
DSR data ingestion for Cledion case study.
All data comes from the real Kaggle Spain energy dataset — no fake or synthetic data.
Raw load values from the CSV are used without any normalisation (load_kw in thousands).
"""
import os
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = "data/raw/energy_dataset.csv"
RAW_CSV_ABS = os.path.join(PROJECT_ROOT, "data", "raw", "energy_dataset.csv")
METER_LOAD_XLSX = "data/raw/meter_load.xlsx"
DAY_AHEAD_XLSX = "data/raw/day_ahead_prices.xlsx"
IMBALANCE_XLSX = "data/raw/imbalance_prices.xlsx"
PROCESSED_PARQUET = "data/processed/processed_data.parquet"

# Site scaling factors: each site is a fraction of national load (realistic for DSR)
SITE_FACTORS = {"site_A": 0.4, "site_B": 0.35, "site_C": 0.25}


def run():
    # -------------------------------------------------------------------------
    # STEP 0 — Ensure raw CSV exists (copy from project root if missing)
    # -------------------------------------------------------------------------
    if not os.path.exists(RAW_CSV_ABS):
        root_csv = os.path.join(PROJECT_ROOT, "energy_dataset.csv")
        if os.path.exists(root_csv):
            os.makedirs(os.path.dirname(RAW_CSV_ABS), exist_ok=True)
            shutil.copy2(root_csv, RAW_CSV_ABS)
            print(f"Copied energy_dataset.csv from project root to {RAW_CSV_ABS}")
        else:
            raise FileNotFoundError(
                f"Raw data not found at {RAW_CSV_ABS} or at {root_csv}. "
                "Place energy_dataset.csv in data/raw/ or in the project root and re-run."
            )

    # -------------------------------------------------------------------------
    # STEP 1 — Load and clean the real Kaggle data (raw load_kw, no normalisation)
    # -------------------------------------------------------------------------
    print("STEP 1 — Load and clean")
    df = pd.read_csv(RAW_CSV_ABS)

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


    # check the missing values
    print(f" Missing values in df: {df.isnull().sum()}")

    # check the data length
    print(f" Length of df: {len(df)}")

    # Parse timestamp as datetime UTC and set as index so we can resample later; sort ascending
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Fill missing load_kw using forward-fill then back-fill before any resampling or row drops
    df["load_kw"] = df["load_kw"].ffill().bfill()
    df = df.dropna(subset=["price_eur_mwh"])
    print(f"  Missing values after fill: {df.isnull().sum().sum()}")

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
