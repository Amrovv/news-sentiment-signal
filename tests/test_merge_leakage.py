"""Tests for stock_predictor.merge.leakage.

A leakage test that cannot fail is worth nothing, so the load-bearing test here
is the positive control: a deliberately leaky `momentum_1d` must be caught. The
rest check that the detector does not cry wolf on the one case where a leaky
match means nothing -- a publish on a non-trading day, where both formulas are
arithmetically the same.
"""

import numpy as np
import pandas as pd
import pytest

from stock_predictor.merge import leakage

TICKER = "TSLA"


@pytest.fixture
def close() -> pd.Series:
    """20 trading days, each close distinct so a one-day shift is unmistakable."""
    days = pd.bdate_range("2025-01-06", periods=20, tz="UTC")
    return pd.Series(100 + np.arange(20, dtype=float) * 3.0, index=days)


@pytest.fixture
def articles(close) -> pd.DataFrame:
    """One article per trading day, published mid-session."""
    ts = close.index[5:] + pd.Timedelta(hours=15)
    return pd.DataFrame(
        {
            "article_id": [f"a{i}" for i in range(len(ts))],
            "timestamp_utc": ts,
            "session": "market-hours",
        }
    )


def _momentum(close: pd.Series, ts: pd.Timestamp, *, leaky: bool) -> float:
    cutoff = ts if leaky else ts.normalize()
    hist = close[close.index < cutoff]
    return hist.iloc[-1] / hist.iloc[-2] - 1


def _table(articles: pd.DataFrame, close: pd.Series, *, leaky: bool) -> pd.DataFrame:
    frame = articles.copy()
    frame["momentum_1d"] = [_momentum(close, ts, leaky=leaky) for ts in frame["timestamp_utc"]]
    return frame


def test_correct_column_passes(articles, close):
    result = leakage.test_momentum_1d(_table(articles, close, leaky=False), close, TICKER)
    assert result.passed
    assert result.real_leaks == 0
    assert result.matches_correct == result.rows_tested


def test_leaky_column_is_caught(articles, close):
    """The positive control. Without this the suite proves nothing."""
    result = leakage.test_momentum_1d(_table(articles, close, leaky=True), close, TICKER)
    assert not result.passed
    assert result.real_leaks == result.rows_tested
    assert result.offenders


def test_a_single_leaking_row_is_caught(articles, close):
    """One bad row among many must not be averaged away."""
    frame = _table(articles, close, leaky=False)
    bad = frame.index[3]
    frame.loc[bad, "momentum_1d"] = _momentum(close, frame.loc[bad, "timestamp_utc"], leaky=True)
    result = leakage.test_momentum_1d(frame, close, TICKER)
    assert not result.passed
    assert result.real_leaks == 1
    assert result.offenders == [frame.loc[bad, "article_id"]]


def test_non_trading_day_publishes_are_not_counted_as_leaks(close):
    """A weekend publish matches both formulas; that is not evidence of a bug."""
    saturday = pd.Timestamp("2025-01-18 12:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "article_id": ["weekend"],
            "timestamp_utc": [saturday],
            "session": ["after-hours"],
            "momentum_1d": [_momentum(close, saturday, leaky=True)],
        }
    )
    result = leakage.test_momentum_1d(frame, close, TICKER)
    assert result.ambiguous == 1
    assert result.real_leaks == 0
    assert result.passed


def test_by_session_covers_every_row(articles, close):
    result = leakage.test_momentum_1d(_table(articles, close, leaky=False), close, TICKER)
    assert result.by_session["rows"].sum() == result.rows_tested


def test_status_reads_pass_or_fail(articles, close):
    assert (
        leakage.test_momentum_1d(_table(articles, close, leaky=False), close, TICKER).status
        == "pass"
    )
    assert (
        leakage.test_momentum_1d(_table(articles, close, leaky=True), close, TICKER).status
        == "FAIL"
    )


def test_the_market_tables_own_timestamp_is_used(articles, close):
    """Feeding a shifted timestamp must change the verdict, not be ignored.

    The market and text layers can hold timestamps for the same article that
    differ by hours, enough to move a publish across a calendar day. The test
    has to read the column the feature was actually computed against; this
    asserts it does, rather than silently reaching for another.
    """
    frame = _table(articles, close, leaky=False)
    shifted = frame.assign(timestamp_utc=frame["timestamp_utc"] + pd.Timedelta(days=1))
    assert leakage.test_momentum_1d(frame, close, TICKER).passed
    assert not leakage.test_momentum_1d(shifted, close, TICKER).passed
