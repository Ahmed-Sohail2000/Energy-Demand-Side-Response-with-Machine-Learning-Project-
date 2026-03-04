"""
Unit tests for src/revenue.py — checks simulate() returns correct structure and values.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from src.revenue import simulate, FLEXIBLE_LOAD_PCT, MAX_REDUCTION_KW, INTERVAL_HOURS


SITE_ID = "site_A"
DATE = "2018-07-10"


@pytest.fixture
def result():
    """Run simulate() once and reuse across tests."""
    return simulate(SITE_ID, DATE)


def test_simulate_returns_dataframe(result):
    """simulate() must return a pandas DataFrame."""
    assert isinstance(result, pd.DataFrame)


def test_simulate_not_empty(result):
    """simulate() must return rows for a valid site and date."""
    assert not result.empty


def test_simulate_has_correct_columns(result):
    """simulate() must return all expected columns in the correct order."""
    expected = [
        "timestamp", "site_id", "activate", "load_kw", "price_eur_mwh",
        "flexible_kw", "revenue_eur", "daily_total_eur", "risk_p10_eur",
    ]
    assert list(result.columns) == expected


def test_simulate_has_96_intervals(result):
    """A full day at 15-min resolution must have exactly 96 intervals."""
    assert len(result) == 96


def test_activate_is_binary(result):
    """activate column must only contain 0 or 1."""
    assert result["activate"].isin([0, 1]).all()


def test_flexible_kw_zero_when_not_activated(result):
    """flexible_kw must be 0 for all non-activated intervals."""
    not_activated = result[result["activate"] == 0]
    assert (not_activated["flexible_kw"] == 0.0).all()


def test_flexible_kw_capped_at_max(result):
    """flexible_kw must never exceed MAX_REDUCTION_KW."""
    assert (result["flexible_kw"] <= MAX_REDUCTION_KW).all()


def test_revenue_zero_when_not_activated(result):
    """revenue_eur must be 0 for all non-activated intervals."""
    not_activated = result[result["activate"] == 0]
    assert (not_activated["revenue_eur"] == 0.0).all()


def test_revenue_non_negative(result):
    """revenue_eur must never be negative."""
    assert (result["revenue_eur"] >= 0).all()


def test_daily_total_matches_sum(result):
    """daily_total_eur must equal the sum of all interval revenues."""
    assert abs(result["daily_total_eur"].iloc[0] - result["revenue_eur"].sum()) < 1e-6


def test_risk_p10_less_than_daily_total(result):
    """risk_p10_eur must be <= daily_total_eur (worst case is always less than expected)."""
    assert result["risk_p10_eur"].iloc[0] <= result["daily_total_eur"].iloc[0]
