"""
DSR activation signal for Cledion case study.
Combines a rule-based score and an ML (XGBoost) score to decide when to "activate" DSR per timestamp.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

from src.features import FEATURE_COLS, run as features_run

# -----------------------------------------------------------------------------
# Configuration — paths and thresholds
# -----------------------------------------------------------------------------
PROCESSED_PARQUET = "data/processed/processed_data.parquet"
# Threshold for the combined score to trigger DSR activation.
# Intervals with score_final >= 0.40 are activated (1), all others are 0.
# Lowering this value activates more intervals but increases false positives.
ACTIVATE_THRESH = 0.40


def rule_score(df: pd.DataFrame) -> pd.Series:
    """
    Rule-based score: where does current price sit within today's distribution?
    Formula: (price - P10) / (P90 - P10), clipped to [0, 1].
    High score = price is high relative to the day → good moment to activate DSR.
    """
    day = df["timestamp"].dt.date
    p10 = df.groupby(day)["price_eur_mwh"].transform(lambda x: x.quantile(0.10))
    p90 = df.groupby(day)["price_eur_mwh"].transform(lambda x: x.quantile(0.90))
    diff = p90 - p10
    # Avoid division by zero when P10 == P90
    score = np.where(diff > 0, (df["price_eur_mwh"].values - p10) / diff, 0.0)
    return pd.Series(np.clip(score, 0, 1), index=df.index)


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the same feature columns as in src.features for a full site-level DataFrame.
    Used here on the full history so we can train the model; features.run() does the same for a single day.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # All lags and rolling windows use shift(1) or .shift(1) after rolling
    # to ensure we only use PAST data — never the current interval's value.
    # This prevents data leakage when training the ML model.
    # Price features (lags and rolling = past only)
    df["price_lag_1"] = df["price_eur_mwh"].shift(1)
    df["price_lag_4"] = df["price_eur_mwh"].shift(4)
    df["price_rolling_mean_4"] = df["price_eur_mwh"].rolling(4).mean().shift(1)
    df["price_delta"] = df["price_eur_mwh"] - df["price_eur_mwh"].shift(1)

    # Load features
    df["load_lag_1"] = df["load_kw"].shift(1)
    df["load_rolling_mean_4"] = df["load_kw"].rolling(4).mean().shift(1)

    # Time-of-day
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["minute_of_hour"] = df["timestamp"].dt.minute
    df["is_peak_hour"] = ((df["hour_of_day"] >= 7) & (df["hour_of_day"] <= 21)).astype(int)

    # Long lags and calendar / imbalance
    df["price_lag_96"] = df["price_eur_mwh"].shift(96)    # same time yesterday
    df["price_lag_672"] = df["price_eur_mwh"].shift(672)  # same time last week
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["imbalance_delta"] = df["imbalance_eur_mwh"] - df["price_eur_mwh"]

    return df.dropna()


def ml_score(site_id: str, date: str, df: pd.DataFrame) -> np.ndarray:
    """
    Train XGBoost on full site history (excluding target day for final fit).
    Label = 1 if price above that day's median, else 0.
    Returns predicted probability (class 1) for each row of df (target day), in [0, 1].
    """
    # ----- 1. Load full site history and build features -----
    full = pd.read_parquet(PROCESSED_PARQUET)
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    full = full[full["site_id"] == site_id].copy()
    full = _add_features(full)

    # ----- 2. Define label: high price day = 1, else 0 (per-day median) -----
    day = full["timestamp"].dt.date
    median = full.groupby(day)["price_eur_mwh"].transform("median")
    y = (full["price_eur_mwh"].values > median.values).astype(int)

    # ----- 3. Time-series cross-validation: train on past, validate on future (no shuffle) -----
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )
    tscv = TimeSeriesSplit(n_splits=3)
    fold_accs = []
    for train_idx, val_idx in tscv.split(full[FEATURE_COLS]):
        X_tr = full[FEATURE_COLS].iloc[train_idx]
        X_val = full[FEATURE_COLS].iloc[val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]
        model.fit(X_tr, y_tr)
        fold_accs.append(model.score(X_val, y_val))
    print(f"  XGBoost CV accuracy: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    
    # ----- 4. Final model: fit on all history except the target day (no leakage) -----
    # Final fit excludes the target date entirely to prevent any leakage
    # from the day we are predicting into the training set.
    target_date = pd.to_datetime(date).date()
    mask = full["timestamp"].dt.date != target_date
    model.fit(full[FEATURE_COLS][mask], y[mask.values])

    # ----- 5. Predict probability for target-day rows only -----
    scores = model.predict_proba(df[FEATURE_COLS])[:, 1]
    return np.clip(scores, 0, 1)


def run(site_id: str, date: str) -> pd.DataFrame:
    """
    Main entry: get features for (site_id, date), add rule + ML scores, combine and set activate flag.
    Returns a DataFrame with timestamp, site_id, score_rule, score_ml, score_final, activate.
    """
    # ----- 1. Get engineered features for the target day (from features.py) -----
    df = features_run(site_id, date)
    if df.empty:
        return pd.DataFrame(
            columns=["timestamp", "site_id", "score_rule", "score_ml", "score_final", "activate"]
        )

    # ----- 2. Add rule-based score (0–1) -----
    df["score_rule"] = rule_score(df)

    # ----- 3. Add ML score (0–1) from XGBoost -----
    df["score_ml"] = ml_score(site_id, date, df)

    
    # ----- 4. Combine scores and binarize activate -----
    # Weighted combination: ML score gets 60% weight, rule score gets 40%.
    # ML is weighted higher because it captures non-linear price patterns.
    # Adjust weights here if rule-based logic should have more influence.
    df["score_final"] = 0.4 * df["score_rule"] + 0.6 * df["score_ml"]
    df["activate"] = (df["score_final"] >= ACTIVATE_THRESH).astype(int)

    return df[["timestamp", "site_id", "score_rule", "score_ml", "score_final", "activate"]]


if __name__ == "__main__":
    result = run("site_A", "2018-07-10")
    print(result)
