"""
Unit tests for src/activation.py — checks run() returns correct structure and signal values.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from src.activation import run, ACTIVATE_THRESH


SITE_ID = "site_A"
DATE = "2018-07-10"


@pytest.fixture
def result():
    """Run activation once and reuse across tests."""
    return run(SITE_ID, DATE)


def test_run_returns_dataframe(result):
    """run() must return a pandas DataFrame."""
    assert isinstance(result, pd.DataFrame)


def test_run_not_empty(result):
    """run() must return rows for a valid site and date."""
    assert not result.empty


def test_run_has_correct_columns(result):
    """run() must return all expected columns."""
    expected = ["timestamp", "site_id", "score_rule", "score_ml", "score_final", "activate"]
    assert list(result.columns) == expected


def test_run_has_96_intervals(result):
    """A full day at 15-min resolution must have exactly 96 intervals."""
    assert len(result) == 96


def test_activate_is_binary(result):
    """activate column must only contain 0 or 1."""
    assert result["activate"].isin([0, 1]).all()


def test_scores_between_0_and_1(result):
    """All score columns must be in [0, 1]."""
    for col in ["score_rule", "score_ml", "score_final"]:
        assert result[col].between(0, 1).all(), f"{col} out of [0, 1] range"


def test_activate_matches_threshold(result):
    """activate must be 1 where score_final >= ACTIVATE_THRESH, else 0."""
    expected = (result["score_final"] >= ACTIVATE_THRESH).astype(int)
    assert (result["activate"] == expected).all()


def test_site_id_correct(result):
    """site_id column must match the requested site."""
    assert (result["site_id"] == SITE_ID).all()


def test_invalid_site_returns_empty():
    """run() must return an empty DataFrame for an unknown site."""
    result = run("site_INVALID", DATE)
    assert result.empty


def test_invalid_date_returns_empty():
    """run() must return an empty DataFrame for a date with no data."""
    result = run(SITE_ID, "2000-01-01")
    assert result.empty
