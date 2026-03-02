"""
DSR feature engineering for Cledion case study.
Builds lag and rolling features from processed parquet (no leakage).
"""
from datetime import timedelta
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROCESSED_PARQUET = "data/processed/processed_data.parquet"

# Ordered list of all feature column names used by the ML model (must match columns we build below)
FEATURE_COLS = [
    "price_lag_1",
    "price_lag_4",
    "price_rolling_mean_4",
    "price_delta",
    "load_lag_1",
    "load_rolling_mean_4",
    "hour_of_day",
    "minute_of_hour",
    "is_peak_hour",
    "price_lag_96",
    "price_lag_672",
    "day_of_week",
    "is_weekend",
    "imbalance_delta",
]


def run(site_id: str, date: str) -> pd.DataFrame:
    """
    Load processed data for one site and one date, add past-only features, return engineered rows.
    All features use only past data (no future leakage).
    """
    # ----- 1. Load and filter data -----
    df = pd.read_parquet(PROCESSED_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df[df["site_id"] == site_id].copy()
    target_dt = pd.to_datetime(date).date()

    # We need enough history to compute lag_672 (same time last week = 7 days = 672 × 15-min intervals)
    start_dt = target_dt - timedelta(days=8)
    df = df[(df["timestamp"].dt.date >= start_dt) & (df["timestamp"].dt.date <= target_dt)].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        return df

    # ----- 2. Price features (past only: shift so we never use current or future) -----
    df["price_lag_1"] = df["price_eur_mwh"].shift(1)   # 1 interval ago (15 min)
    df["price_lag_4"] = df["price_eur_mwh"].shift(4)   # 4 intervals ago (1 hour)
    df["price_rolling_mean_4"] = df["price_eur_mwh"].rolling(4).mean().shift(1)  # mean of last 4, then shift = no current
    df["price_delta"] = df["price_eur_mwh"] - df["price_eur_mwh"].shift(1)       # change from previous interval

    # ----- 3. Load features (past only) -----
    df["load_lag_1"] = df["load_kw"].shift(1)
    df["load_rolling_mean_4"] = df["load_kw"].rolling(4).mean().shift(1)

    # ----- 4. Time-of-day features (no leakage: derived from timestamp only) -----
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["minute_of_hour"] = df["timestamp"].dt.minute
    df["is_peak_hour"] = ((df["hour_of_day"] >= 7) & (df["hour_of_day"] <= 21)).astype(int)

    # ----- 5. Long lags and calendar / imbalance features -----
    df["price_lag_96"] = df["price_eur_mwh"].shift(96)    # same time yesterday (96 × 15 min = 24 h)
    df["price_lag_672"] = df["price_eur_mwh"].shift(672)  # same time last week (7 days)
    df["day_of_week"] = df["timestamp"].dt.dayofweek      # 0=Monday, 6=Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["imbalance_delta"] = df["imbalance_eur_mwh"] - df["price_eur_mwh"]  # actual vs day-ahead spread

    # ----- 6. Drop rows with NaN (from lags at start of series) and return only target date -----
    df = df.dropna()
    df = df[df["timestamp"].dt.date == target_dt].reset_index(drop=True)
    return df


if __name__ == "__main__":
    result = run("site_A", "2018-07-10")
    print(result.head())
