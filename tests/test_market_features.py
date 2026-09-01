"""Tests for news_sentiment.market.features.

Every pre-publication feature (everything except abnormal_return_for, which is the label)
gets a same-day-leakage regression test: mutate the value for the article's own trading
day to something wildly different and assert the feature's output doesn't move. That's a
stronger check than reconstructing an "expected" value, since it proves the function
never even reads that day's row, rather than merely computing an answer that happens to
match.

This guards against the specific bug fixed in the market feature pipeline: filtering price
history with `series.index < ref_date` (a raw, time-of-day-bearing timestamp) instead of
`series.index < ref_date.normalize()`, which let a trading day's own close leak into
"prior" data for any article published after midnight -- i.e. nearly always.
"""

import numpy as np
import pandas as pd
import pytest

from news_sentiment.market.features import (
    abnormal_return_for,
    beta_vs_market,
    cumulative_return,
    daily_range_ratio,
    days_to_next_earnings,
    relative_volume,
    rolling_volatility,
    session_for,
)

DATES = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")

# A mid-morning, mid-afternoon, and just-after-midnight timestamp on the same trading day,
# to prove the cutoff depends only on the calendar day, never on time-of-day.
SAME_DAY_TIMES = ["08:00:00", "15:30:00", "00:00:01", "23:59:59"]


def _prices(values):
    return pd.Series(values, index=DATES, dtype=float)


@pytest.fixture
def close():
    return _prices([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122])


@pytest.fixture
def market_close():
    return _prices([50, 50.5, 51, 51.5, 52, 52.5, 53, 53.5, 54, 54.5, 55, 55.5])


@pytest.fixture
def volume():
    return pd.Series(
        [1_000, 1_100, 900, 1_200, 1_050, 980, 1_300, 1_150, 1_000, 1_400, 1_250, 1_100],
        index=DATES,
        dtype=float,
    )


@pytest.fixture
def high_low(close):
    high = close + 2
    low = close - 2
    return high, low


def ref_at(day_index: int, time_str: str) -> pd.Timestamp:
    return pd.Timestamp(f"{DATES[day_index].date()} {time_str}", tz="UTC")


# --------------------------------------------------------------------------- #
# cumulative_return (momentum_1d / momentum_5d / momentum_20d)
# --------------------------------------------------------------------------- #


class TestCumulativeReturn:
    @pytest.mark.parametrize("time_str", SAME_DAY_TIMES)
    def test_excludes_same_day_regardless_of_publish_time(self, close, time_str):
        ref_date = ref_at(5, time_str)
        result = cumulative_return(close, ref_date, lookback_days=1)
        expected = close.iloc[4] / close.iloc[3] - 1  # day 4 vs day 3, day 5 excluded
        assert np.isclose(result, expected)

    def test_mutating_same_day_price_does_not_change_result(self, close):
        ref_date = ref_at(5, "08:00:00")
        baseline = cumulative_return(close, ref_date, lookback_days=1)

        mutated = close.copy()
        mutated.iloc[5] = 999_999.0  # today's close, wildly different
        result = cumulative_return(mutated, ref_date, lookback_days=1)

        assert np.isclose(result, baseline)

    def test_correct_value_for_multi_day_lookback(self, close):
        ref_date = ref_at(6, "10:00:00")
        result = cumulative_return(close, ref_date, lookback_days=5)
        # Prior 5 closes end at day 5 (day 6 excluded): day5 / day0 - 1
        expected = close.iloc[5] / close.iloc[0] - 1
        assert np.isclose(result, expected)

    def test_nan_when_history_too_short(self, close):
        ref_date = ref_at(1, "10:00:00")
        assert np.isnan(cumulative_return(close, ref_date, lookback_days=5))


# --------------------------------------------------------------------------- #
# rolling_volatility (volatility_20d)
# --------------------------------------------------------------------------- #


class TestRollingVolatility:
    def test_excludes_same_day(self, close):
        ref_date = ref_at(8, "12:00:00")
        baseline = rolling_volatility(close, ref_date, window=5)

        mutated = close.copy()
        mutated.iloc[8] = -500.0
        result = rolling_volatility(mutated, ref_date, window=5)

        assert np.isclose(result, baseline)

    def test_matches_manual_std_of_prior_window(self, close):
        ref_date = ref_at(8, "12:00:00")
        result = rolling_volatility(close, ref_date, window=5)
        prior = close[close.index < DATES[8]].tail(5)
        expected = prior.pct_change().dropna().std(ddof=1)
        assert np.isclose(result, expected)


# --------------------------------------------------------------------------- #
# beta_vs_market (beta_20d)
# --------------------------------------------------------------------------- #


class TestBetaVsMarket:
    def test_excludes_same_day_for_both_series(self, close, market_close):
        ref_date = ref_at(9, "12:00:00")
        baseline = beta_vs_market(close, market_close, ref_date, window=8)

        mutated_asset = close.copy()
        mutated_asset.iloc[9] = 1.0
        mutated_market = market_close.copy()
        mutated_market.iloc[9] = 1.0
        result = beta_vs_market(mutated_asset, mutated_market, ref_date, window=8)

        assert np.isclose(result, baseline)

    def test_beta_of_a_series_against_itself_is_one(self, close):
        ref_date = ref_at(9, "12:00:00")
        result = beta_vs_market(close, close, ref_date, window=8)
        assert np.isclose(result, 1.0)


# --------------------------------------------------------------------------- #
# relative_volume (relative_volume_20d)
# --------------------------------------------------------------------------- #


class TestRelativeVolume:
    def test_excludes_same_day(self, volume):
        ref_date = ref_at(7, "12:00:00")
        baseline = relative_volume(volume, ref_date, window=5)

        mutated = volume.copy()
        mutated.iloc[7] = 10_000_000.0
        result = relative_volume(mutated, ref_date, window=5)

        assert np.isclose(result, baseline)

    def test_matches_manual_ratio(self, volume):
        ref_date = ref_at(7, "12:00:00")
        result = relative_volume(volume, ref_date, window=5)
        prior = volume[volume.index < DATES[7]].tail(5)
        expected = prior.iloc[-1] / prior.median()
        assert np.isclose(result, expected)


# --------------------------------------------------------------------------- #
# daily_range_ratio (daily_range_ratio_1d)
# --------------------------------------------------------------------------- #


class TestDailyRangeRatio:
    def test_excludes_same_day(self, close, high_low):
        high, low = high_low
        ref_date = ref_at(4, "12:00:00")
        baseline = daily_range_ratio(high, low, close, ref_date)

        mutated_high, mutated_low, mutated_close = high.copy(), low.copy(), close.copy()
        mutated_high.iloc[4] = 9_999.0
        mutated_low.iloc[4] = -9_999.0
        mutated_close.iloc[4] = 1.0
        result = daily_range_ratio(mutated_high, mutated_low, mutated_close, ref_date)

        assert np.isclose(result, baseline)

    def test_matches_manual_ratio(self, close, high_low):
        high, low = high_low
        ref_date = ref_at(4, "12:00:00")
        result = daily_range_ratio(high, low, close, ref_date)
        expected = (high.iloc[3] - low.iloc[3]) / close.iloc[3]
        assert np.isclose(result, expected)


# --------------------------------------------------------------------------- #
# days_to_next_earnings -- forward-looking by design, but must not depend on
# caller-side sort order (the bug fixed in Fix days_to_earnings sort bug).
# --------------------------------------------------------------------------- #


class TestDaysToNextEarnings:
    def test_finds_nearest_future_date_when_pre_sorted(self):
        earnings = pd.DataFrame(
            index=pd.to_datetime(["2026-01-10", "2026-04-10", "2026-07-10"], utc=True)
        )
        result = days_to_next_earnings(pd.Timestamp("2026-01-01", tz="UTC"), earnings)
        assert result == 9

    def test_finds_nearest_future_date_when_caller_forgot_to_sort(self):
        # Newest-first, exactly how raw_earnings.parquet loads from yfinance.
        earnings = pd.DataFrame(
            index=pd.to_datetime(["2026-07-10", "2026-04-10", "2026-01-10"], utc=True)
        )
        result = days_to_next_earnings(pd.Timestamp("2026-01-01", tz="UTC"), earnings)
        assert result == 9  # nearest (Jan 10), not furthest (Jul 10)

    def test_nan_when_no_future_earnings(self):
        earnings = pd.DataFrame(index=pd.to_datetime(["2025-01-01"], utc=True))
        result = days_to_next_earnings(pd.Timestamp("2026-01-01", tz="UTC"), earnings)
        assert np.isnan(result)


# --------------------------------------------------------------------------- #
# session_for -- pure classification against the schedule, no leakage concept.
# --------------------------------------------------------------------------- #


class TestSessionFor:
    @pytest.fixture
    def schedule(self):
        return pd.DataFrame(
            {
                "market_open": pd.to_datetime(["2026-01-05 14:30:00"], utc=True),
                "market_close": pd.to_datetime(["2026-01-05 20:00:00"], utc=True),
            }
        )

    def test_pre_market(self, schedule):
        assert session_for(pd.Timestamp("2026-01-05 10:00:00", tz="UTC"), schedule) == "pre-market"

    def test_market_hours(self, schedule):
        assert (
            session_for(pd.Timestamp("2026-01-05 16:00:00", tz="UTC"), schedule) == "market-hours"
        )

    def test_after_hours_same_day(self, schedule):
        assert (
            session_for(pd.Timestamp("2026-01-05 21:00:00", tz="UTC"), schedule) == "after-hours"
        )

    def test_after_hours_non_trading_day(self, schedule):
        assert (
            session_for(pd.Timestamp("2026-01-06 10:00:00", tz="UTC"), schedule) == "after-hours"
        )


# --------------------------------------------------------------------------- #
# abnormal_return_for -- the LABEL, not a feature. Deliberately anchored to the
# trading day at/after publication: it must NOT exclude the same day the way the
# feature functions above do, since that's the reaction being predicted.
# --------------------------------------------------------------------------- #


class TestAbnormalReturnFor:
    @pytest.fixture
    def schedule(self):
        return pd.DataFrame(
            {
                "market_open": pd.to_datetime([f"{d.date()} 14:30:00" for d in DATES], utc=True),
                "market_close": pd.to_datetime([f"{d.date()} 20:00:00" for d in DATES], utc=True),
            }
        )

    def test_uses_the_next_trading_day_close_as_the_outcome(self, close, market_close, schedule):
        # Published after the prior close, before day 5's open: event day is day 5.
        ts = pd.Timestamp(f"{DATES[5].date()} 10:00:00", tz="UTC")
        result = abnormal_return_for(
            ts, horizon_days=1, schedule=schedule, asset_close=close, market_close=market_close
        )
        expected = (close.iloc[5] / close.iloc[4] - 1) - (
            market_close.iloc[5] / market_close.iloc[4] - 1
        )
        assert np.isclose(result, expected)
