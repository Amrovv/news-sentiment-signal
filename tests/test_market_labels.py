"""Tests for stock_predictor.market.labels.

The vectorised helpers exist only to make a 17k-article corpus tractable, so
most of what matters here is that they agree exactly with the row-at-a-time
functions in `market.features` they replaced. Those functions keep their own
suite in test_market_features.py, including a same-day-leakage regression per
function; this file checks the loop built around them, and re-checks leakage at
the table level, where a mistake in the caching would show up rather than in the
feature functions themselves.
"""

import numpy as np
import pandas as pd
import pytest

from stock_predictor.market import labels
from stock_predictor.market.features import abnormal_return_for, session_for
from stock_predictor.market.timestamp_alignment import align_timestamp

TICKER = "TSLA"
BENCHMARK = "SPY"


@pytest.fixture
def schedule() -> pd.DataFrame:
    """40 consecutive weekdays of NYSE-shaped sessions, 13:30-20:00 UTC."""
    days = pd.bdate_range("2025-01-01", periods=40, tz="UTC")
    return pd.DataFrame(
        {
            "market_open": days + pd.Timedelta(hours=13, minutes=30),
            "market_close": days + pd.Timedelta(hours=20),
        }
    )


@pytest.fixture
def ohlcv(schedule) -> pd.DataFrame:
    """A wide yfinance-shaped frame: one bar per session, both symbols.

    The target drifts up 1% a day and the benchmark 0.5%, so an abnormal return
    is a predictable +0.5% per session and a wrong window is visible as a wrong
    multiple of it rather than as noise.
    """
    dates = schedule["market_open"].dt.normalize().dt.tz_localize(None)
    n = len(dates)
    target = 100 * 1.01 ** np.arange(n)
    bench = 400 * 1.005 ** np.arange(n)

    frame = pd.DataFrame(
        {
            ("Close", TICKER): target,
            ("Close", BENCHMARK): bench,
            ("High", TICKER): target * 1.02,
            ("High", BENCHMARK): bench * 1.01,
            ("Low", TICKER): target * 0.98,
            ("Low", BENCHMARK): bench * 0.99,
            ("Volume", TICKER): np.full(n, 1_000_000.0),
            ("Volume", BENCHMARK): np.full(n, 5_000_000.0),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Price", "Ticker"])
    return frame


@pytest.fixture
def earnings() -> pd.DataFrame:
    """Newest-first, exactly how raw_earnings.parquet loads from yfinance."""
    idx = pd.DatetimeIndex(["2025-03-01", "2025-02-14", "2025-01-20"], tz="UTC")
    return pd.DataFrame({"EPS Estimate": [1.0, 2.0, 3.0]}, index=idx)


@pytest.fixture
def articles(schedule) -> pd.DataFrame:
    """Three articles per session: pre-market, mid-session, and after the close."""
    rows = []
    for i, open_ts in enumerate(schedule["market_open"].iloc[10:30], start=1):
        for offset, tag in [(-2, "pre"), (2, "mid"), (8, "post")]:
            rows.append(
                {
                    "article_id": f"a{i}_{tag}",
                    "timestamp_utc": open_ts + pd.Timedelta(hours=offset),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def built(articles, ohlcv, schedule, earnings) -> pd.DataFrame:
    return labels.build_market_features(articles, TICKER, ohlcv, schedule, earnings)


# --- the vectorised helpers against the functions they replaced ---------------


def test_align_sessions_matches_align_timestamp(articles, schedule):
    vectorised = labels.align_sessions(articles["timestamp_utc"], schedule)
    one_by_one = articles["timestamp_utc"].apply(lambda ts: align_timestamp(ts, schedule))
    # check_dtype is off because the two paths differ only in datetime
    # resolution: searchsorted returns ns, indexing the schedule preserves the
    # us the fixture was built at. The instants are identical.
    pd.testing.assert_series_equal(
        vectorised.reset_index(drop=True),
        one_by_one.reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


def test_assign_sessions_matches_session_for(articles, schedule):
    vectorised = labels.assign_sessions(articles["timestamp_utc"], schedule)
    one_by_one = articles["timestamp_utc"].apply(lambda ts: session_for(ts, schedule))
    assert list(vectorised) == list(one_by_one)


def test_assign_sessions_covers_all_three_sessions(articles, schedule):
    counts = labels.assign_sessions(articles["timestamp_utc"], schedule).value_counts()
    assert set(counts.index) == {"pre-market", "market-hours", "after-hours"}


def test_assign_sessions_weekend_is_after_hours(schedule):
    saturday = pd.Series([pd.Timestamp("2025-01-11 15:00", tz="UTC")])
    assert labels.assign_sessions(saturday, schedule).iloc[0] == "after-hours"


def test_abnormal_return_at_open_matches_abnormal_return_for(articles, ohlcv, schedule):
    close = labels.split_ohlcv(ohlcv, [TICKER, BENCHMARK])["close"]
    aligned = labels.align_sessions(articles["timestamp_utc"], schedule)

    for ts, open_ts in zip(articles["timestamp_utc"], aligned, strict=True):
        expected = abnormal_return_for(ts, 1, schedule, close[TICKER], close[BENCHMARK])
        actual = labels._abnormal_return_at_open(open_ts, 1, close[TICKER], close[BENCHMARK])
        assert actual == pytest.approx(expected, nan_ok=True)


def test_news_volume_matches_naive_count(articles):
    ts = articles["timestamp_utc"]
    vectorised = labels.news_volume(ts, lookback_days=3)
    naive = ts.apply(lambda t: int(((ts >= t - pd.Timedelta(days=3)) & (ts < t)).sum()))
    pd.testing.assert_series_equal(
        vectorised.reset_index(drop=True),
        naive.reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


def test_news_volume_excludes_the_article_itself():
    ts = pd.Series(pd.to_datetime(["2025-01-01 10:00"], utc=True))
    assert labels.news_volume(ts).iloc[0] == 0


# --- the table -----------------------------------------------------------------


def test_column_order_is_stable(built):
    assert list(built.columns) == (
        labels.IDENTITY_COLUMNS + labels.LABEL_COLUMNS + labels.FEATURE_COLUMNS
    )


def test_every_column_is_documented():
    documented = set(labels.FEATURE_DESCRIPTIONS)
    assert set(labels.LABEL_COLUMNS + labels.FEATURE_COLUMNS) <= documented


def test_ticker_column_is_the_requested_ticker(built):
    assert (built["ticker"] == TICKER).all()


def test_sorted_by_timestamp(built):
    assert built["timestamp_utc"].is_monotonic_increasing


def test_rows_without_any_label_are_dropped(built):
    assert not built[["abnormal_return_1d", "abnormal_return_3d"]].isna().all(axis=1).any()


def test_label_direction_is_the_sign_of_the_1d_return(built):
    expected = np.sign(built["abnormal_return_1d"]).fillna(0).astype(int)
    pd.testing.assert_series_equal(built["label_direction"], expected, check_names=False)


def test_article_ids_are_preserved_not_reindexed(built, articles):
    assert set(built["article_id"]) <= set(articles["article_id"])
    assert built["article_id"].is_unique


def test_same_day_articles_share_every_daily_feature(built):
    """The caching's core assumption, asserted rather than trusted.

    Every pre-publication feature is cut at the calendar day, so two articles
    published on the same day must agree on all of them. If they ever did not,
    evaluating once per day would be silently wrong.
    """
    daily = [c for c in labels.FEATURE_COLUMNS if c not in ("session", "news_volume")]
    per_day = built.assign(day=built["timestamp_utc"].dt.normalize()).groupby("day")[daily]
    assert (per_day.nunique(dropna=False) <= 1).all().all()


def test_no_feature_uses_the_articles_own_day(articles, ohlcv, schedule, earnings):
    """Table-level leakage regression.

    Corrupting a session's own bar to an absurd value must not move any
    pre-publication feature for an article published that day. Only the label,
    which is anchored forward deliberately, may react.
    """
    day = pd.Timestamp("2025-01-22", tz="UTC")
    same_day = articles[articles["timestamp_utc"].dt.normalize() == day]
    assert not same_day.empty

    base = labels.build_market_features(articles, TICKER, ohlcv, schedule, earnings)

    corrupted = ohlcv.copy()
    corrupted.loc[day.tz_localize(None), ("Close", TICKER)] = 1e6
    corrupted.loc[day.tz_localize(None), ("High", TICKER)] = 1e6
    corrupted.loc[day.tz_localize(None), ("Volume", TICKER)] = 1e12
    after = labels.build_market_features(articles, TICKER, corrupted, schedule, earnings)

    features = [c for c in labels.FEATURE_COLUMNS if c != "session"]
    rows = base["article_id"].isin(same_day["article_id"])
    pd.testing.assert_frame_equal(
        base.loc[rows, features].reset_index(drop=True),
        after.loc[rows, features].reset_index(drop=True),
    )


def test_split_ohlcv_names_the_missing_symbol(ohlcv):
    with pytest.raises(KeyError, match="AAPL"):
        labels.split_ohlcv(ohlcv, [TICKER, "AAPL"])


def test_align_sessions_raises_past_the_schedule(schedule):
    beyond = pd.Series([schedule["market_open"].iloc[-1] + pd.Timedelta(days=10)])
    with pytest.raises(ValueError, match="later --end"):
        labels.align_sessions(beyond, schedule)


def test_align_sessions_raises_before_the_schedule(schedule):
    early = pd.Series([schedule["market_open"].iloc[0] - pd.Timedelta(days=10)])
    with pytest.raises(ValueError, match="earlier --start"):
        labels.align_sessions(early, schedule)


def test_empty_articles_raises(ohlcv, schedule, earnings):
    empty = pd.DataFrame(
        {"article_id": [], "timestamp_utc": pd.Series([], dtype="datetime64[ns, UTC]")}
    )
    with pytest.raises(ValueError, match="No articles"):
        labels.build_market_features(empty, TICKER, ohlcv, schedule, earnings)


def test_build_reads_no_column_beyond_id_and_timestamp(articles, ohlcv, schedule, earnings):
    """The layer boundary, enforced: passing article text changes nothing."""
    plain = labels.build_market_features(articles, TICKER, ohlcv, schedule, earnings)
    with_text = labels.build_market_features(
        articles.assign(headline="anything at all", processed_body="x" * 100),
        TICKER,
        ohlcv,
        schedule,
        earnings,
    )
    pd.testing.assert_frame_equal(plain, with_text)
