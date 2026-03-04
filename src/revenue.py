"""
DSR revenue simulation for Cledion case study.
Uses activation signal + load/price to compute flexible reduction and revenue per interval.
"""
# Allow importing from project root when running this file directly
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.activation import run as activation_run

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROCESSED_PARQUET = "data/processed/processed_data.parquet"  # path to merged load + price data
FLEXIBLE_LOAD_PCT = 0.10   # 10% of load is flexible (can be reduced when we activate)
MAX_REDUCTION_KW = 300.0   # maximum reduction per interval in kW (cap per 15-min slot)
INTERVAL_HOURS = 0.25      # each interval is 15 minutes = 0.25 hours
N_SIMULATIONS = 200       # number of noise simulations for risk metric (P10)


def risk_metric(site_id: str, date: str, df: pd.DataFrame) -> float:
    """
    Simulate uncertainty in achieved reduction: flexible_kw × random factor in [0.7, 1.0].
    Returns the 10th percentile of simulated daily revenues (worst 10% outcome).
    """
    # Fix seed so results are reproducible
    np.random.seed(42)
    # Collect total daily revenue for each simulation
    daily_totals = []
    for _ in range(N_SIMULATIONS):
        # One noise factor per simulation (same for all intervals that day)
        noise = np.random.uniform(0.7, 1.0)
        # Revenue scales with flexible_kw; scaling revenue by noise = scaling flexible_kw by noise
        sim_daily = (df["revenue_eur"] * noise).sum()
        daily_totals.append(sim_daily)
    # Return 10th percentile: in 10% of cases daily revenue would be this low or lower
    return float(np.percentile(daily_totals, 10))


def simulate(site_id: str, date: str) -> pd.DataFrame:
    """
    Get activation signal, merge load/price, compute flexible_kw and revenue per interval,
    add daily_total_eur and risk_p10_eur.
    """
    # ----- 1. Get activation signal (which intervals to reduce load) -----
    signal = activation_run(site_id, date)
    # If no signal (e.g. no data for that date), return empty DataFrame with correct columns
    if signal.empty:
        return pd.DataFrame(
            columns=[
                "timestamp", "site_id", "activate", "load_kw", "price_eur_mwh",
                "flexible_kw", "revenue_eur", "daily_total_eur", "risk_p10_eur"
            ]
        )

    # ----- 2. Load processed data and filter to this site and date -----
    raw = pd.read_parquet(PROCESSED_PARQUET)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])  # ensure datetime for merge
    date_str = date if isinstance(date, str) else pd.to_datetime(date).strftime("%Y-%m-%d")
    raw = raw[raw["timestamp"].astype(str).str.startswith(date_str)]
    raw = raw[raw["site_id"] == site_id]
    raw = raw[["timestamp", "load_kw", "price_eur_mwh"]].copy()  # keep only columns we need

    # ----- 3. Merge load and price onto the signal (one row per timestamp) -----
    df = signal[["timestamp", "site_id", "activate"]].merge(
        raw, on="timestamp", how="left"
    )

    # ----- 4. Compute flexible reduction and revenue per interval -----
    # When activate=1: we can reduce up to FLEXIBLE_LOAD_PCT of load, capped at MAX_REDUCTION_KW
    flexible_kw = np.where(
        df["activate"] == 1,
        np.minimum(df["load_kw"] * FLEXIBLE_LOAD_PCT, MAX_REDUCTION_KW),
        0.0,  # when activate=0 we do not reduce
    )
    df["flexible_kw"] = flexible_kw
    # Revenue = power reduced (kW→MW by /1000) × price (€/MWh) × duration (hours)
    df["revenue_eur"] = np.where(
        df["activate"] == 1,
        df["flexible_kw"] * df["price_eur_mwh"] * INTERVAL_HOURS / 1000,
        0.0,
    )

    # ----- 5. Add daily total revenue (same value on every row for convenience) -----
    daily_total_eur = df["revenue_eur"].sum()
    df["daily_total_eur"] = daily_total_eur

    # ----- 6. Add risk metric: P10 of simulated daily revenues -----
    risk_p10 = risk_metric(site_id, date, df)
    df["risk_p10_eur"] = risk_p10

    # Ensure only the selected date and site (96 rows per day)
    df = df[df["timestamp"].astype(str).str.startswith(date_str)]
    df = df[df["site_id"] == site_id]
    print(f"simulate() returning {len(df)} rows for {site_id} on {date_str}")

    # Return only the columns we promise, in the right order
    return df[
        [
            "timestamp", "site_id", "activate", "load_kw", "price_eur_mwh",
            "flexible_kw", "revenue_eur", "daily_total_eur", "risk_p10_eur",
        ]
    ]


if __name__ == "__main__":
    # Run simulation for site_A on 2018-07-10
    result = simulate("site_A", "2018-07-10")
    print(result)
    print()
    if not result.empty:
        print("Daily total revenue (EUR):", result["daily_total_eur"].iloc[0])
        print("Risk P10 (EUR):", result["risk_p10_eur"].iloc[0])
        print("\nActivated intervals with revenue:")
        print(result[result["activate"] == 1][["timestamp", "revenue_eur", "flexible_kw", "activate"]])
