"""Pre-publication market features for a single article timestamp.

Every function here answers "what would this feature have looked like at the moment
the article was published", so each one must only look at data strictly *before* the
article's own trading day. Daily OHLCV bars are indexed at midnight but represent
that day's close, which isn't known until the market closes; comparing a lookback
window against the raw publication timestamp (rather than its calendar day) would
let that day's own close leak in as "prior" data for any article published after
midnight, which is nearly always. Every cutoff below is therefore taken against
`ref_date.normalize()`, not `ref_date` itself.

Tested in tests/test_market_features.py, including a same-day-leakage regression
test per function.
"""

import numpy as np
import pandas as pd

from news_sentiment.market.timestamp_alignment import align_timestamp


def _as_utc_timestamp(ref_date: pd.Timestamp) -> pd.Timestamp:
    ref_date = pd.Timestamp(ref_date)
    if ref_date.tzinfo is None:
        ref_date = ref_date.tz_localize("UTC")
    return ref_date


def session_for(ts: pd.Timestamp, schedule: pd.DataFrame) -> str:
    """Assign a market session (pre-market / market-hours / after-hours) based on ts."""
    same_day = schedule[schedule["market_open"].dt.date == ts.date()]
    if same_day.empty:
        return "after-hours"

    row = same_day.iloc[0]
    if ts < row["market_open"]:
        return "pre-market"
    if ts <= row["market_close"]:
        return "market-hours"
    return "after-hours"


def cumulative_return(series: pd.Series, ref_date: pd.Timestamp, lookback_days: int) -> float:
    """Cumulative return over the `lookback_days` prior to ref_date's calendar day."""
    ref_date = _as_utc_timestamp(ref_date)
    hist = series[series.index < ref_date.normalize()].sort_index()
    if len(hist) <= lookback_days:
        return np.nan
    start = hist.iloc[-(lookback_days + 1)]
    end = hist.iloc[-1]
    return end / start - 1


def rolling_volatility(series: pd.Series, ref_date: pd.Timestamp, window: int = 20) -> float:
    """Std dev of daily returns over the `window` trading days prior to ref_date's day."""
    ref_date = _as_utc_timestamp(ref_date)
    hist = series[series.index < ref_date.normalize()].sort_index().tail(window)
    if len(hist) < 2:
        return np.nan
    return hist.pct_change().dropna().std(ddof=1)


def days_to_next_earnings(article_date: pd.Timestamp, earnings: pd.DataFrame) -> float:
    """Calendar distance to the next earnings date on or after article_date.

    Sorts earnings.index ascending itself rather than trusting the caller to: taking the
    first index entry >= article_date is only the *nearest* upcoming date if the index is
    ascending, and raw_earnings.parquet loads newest-first, so this bit a previous version
    of the pipeline that assumed the caller had already sorted it.
    """
    earnings = earnings.sort_index()
    future = earnings.index[earnings.index >= article_date]
    if future.empty:
        return np.nan
    next_earn = future[0]
    return (next_earn.normalize() - article_date.normalize()).days


def beta_vs_market(
    asset: pd.Series, market: pd.Series, ref_date: pd.Timestamp, window: int = 20
) -> float:
    """20-day rolling beta of `asset` vs `market`, using daily returns prior to ref_date's day."""
    ref_date = _as_utc_timestamp(ref_date)

    hist = pd.concat([asset.rename("asset"), market.rename("market")], axis=1)
    hist = hist[hist.index < ref_date.normalize()].sort_index().tail(window)
    if len(hist) < 3:
        return np.nan

    ret = hist.pct_change().dropna()
    if ret.empty:
        return np.nan

    x = ret["market"]
    y = ret["asset"]
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan

    slope = np.polyfit(x.to_numpy(), y.to_numpy(), 1)[0]
    return float(slope)


def relative_volume(volume: pd.Series, ref_date: pd.Timestamp, window: int = 20) -> float:
    """Most recent prior-day volume relative to its trailing `window`-day median."""
    ref_date = _as_utc_timestamp(ref_date)

    hist = volume[volume.index < ref_date.normalize()].sort_index().tail(window)
    if hist.empty:
        return np.nan
    current = hist.iloc[-1]
    median = hist.median()
    if pd.isna(median) or median == 0:
        return np.nan
    return current / median


def daily_range_ratio(
    high: pd.Series, low: pd.Series, close: pd.Series, ref_date: pd.Timestamp
) -> float:
    """Prior trading day's intraday high-low range, scaled by that day's close."""
    ref_date = _as_utc_timestamp(ref_date)

    hist = pd.concat([high.rename("high"), low.rename("low"), close.rename("close")], axis=1)
    hist = hist[hist.index < ref_date.normalize()].sort_index()
    if hist.empty:
        return np.nan

    prev = hist.iloc[-1]
    if prev["close"] == 0:
        return np.nan
    return (prev["high"] - prev["low"]) / prev["close"]


def abnormal_return_for(
    ts: pd.Timestamp,
    horizon_days: int,
    schedule: pd.DataFrame,
    asset_close: pd.Series,
    market_close: pd.Series,
) -> float:
    """Asset return minus market return over the horizon starting at the next market open.

    Unlike the pre-publication features above, this is the label, not a feature: it is
    deliberately anchored to the trading day *at or after* publication, since that is the
    reaction we're trying to predict, not information available beforehand.
    """
    event_open = align_timestamp(ts, schedule)
    event_date = event_open.normalize()
    if event_date not in asset_close.index:
        return np.nan

    event_pos = asset_close.index.get_loc(event_date)
    prev_pos = event_pos - 1
    if prev_pos < 0:
        return np.nan

    end_pos = min(event_pos + max(horizon_days - 1, 0), len(asset_close) - 1)
    end_date = asset_close.index[end_pos]
    prev_date = asset_close.index[prev_pos]

    asset_ret = asset_close.loc[end_date] / asset_close.loc[prev_date] - 1
    market_ret = market_close.loc[end_date] / market_close.loc[prev_date] - 1
    return asset_ret - market_ret
