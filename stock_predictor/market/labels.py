"""One article table in, one market feature row per article out.

Every feature and the label itself come from `market.features`, which is already
ticker-agnostic; this adds the loop around them, the column order, and the ticker
parameter. Originally in notebooks/market/1.4, hardcoded to TSLA.

Two things are computed here rather than row by row, since corpora run 7k-17k
articles:

  * every feature anchored to the article's calendar day is evaluated once per
    unique day and mapped back. `market.features` normalises `ref_date` before
    slicing history, precisely so a same-day close cannot leak in, which means
    two articles published on the same day cannot differ; the repeat work is
    redundant by construction, not an approximation.
  * `align_timestamp` and `session_for` are vectorised via searchsorted.
    `market.features` stays the definition of both; tests/test_market_labels.py
    asserts the vectorised forms agree with them row for row.

Nothing here fetches. Inputs come from `market.prices`.
"""

import numpy as np
import pandas as pd

from stock_predictor.config import (
    BETA_WINDOW,
    LABEL_HORIZONS_DAYS,
    MARKET_INDEX,
    MOMENTUM_LOOKBACKS,
    NEWS_VOLUME_LOOKBACK_DAYS,
    RELATIVE_VOLUME_WINDOW,
    VOLATILITY_WINDOW,
)
from stock_predictor.market.features import (
    beta_vs_market,
    cumulative_return,
    daily_range_ratio,
    days_to_next_earnings,
    relative_volume,
    rolling_volatility,
)

# The order the table has always been written in. Identity first, then the two
# labels and the direction derived from the 1-day one, then the pre-publication
# features. `article_id` is the join key onto the text layer's table.
IDENTITY_COLUMNS = ["article_id", "ticker", "timestamp_utc", "session_open"]
LABEL_COLUMNS = [f"abnormal_return_{h}d" for h in LABEL_HORIZONS_DAYS] + ["label_direction"]
FEATURE_COLUMNS = [f"momentum_{d}d" for d in MOMENTUM_LOOKBACKS] + [
    f"volatility_{VOLATILITY_WINDOW}d",
    f"beta_{BETA_WINDOW}d",
    f"relative_volume_{RELATIVE_VOLUME_WINDOW}d",
    "daily_range_ratio_1d",
    "days_to_earnings",
    "session",
    "news_volume",
]

FEATURE_DESCRIPTIONS = {
    "session_open": (
        "The market session this row is labelled against: the first open strictly after publication. "
        "Every row sharing it shares a label and every market feature, so it is the unit a fold may "
        "not divide."
    ),
    "abnormal_return_1d": (
        "Target return minus benchmark return over the session at or after publication. "
        "The label. Deliberately forward-looking, the only column here that is."
    ),
    "label_direction": (
        "sign(abnormal_return_1d) as -1 / 0 / 1. 0 means the return was missing or exactly "
        "flat, so it is not a third class to predict."
    ),
    "momentum_1d": "Cumulative return over the trading day before publication.",
    "momentum_5d": "Cumulative return over the 5 trading days before publication.",
    "momentum_20d": "Cumulative return over the 20 trading days before publication.",
    "volatility_20d": "Std dev of daily returns over the 20 trading days before publication.",
    "beta_20d": "Rolling 20-day beta of the target against the benchmark, before publication.",
    "relative_volume_20d": "Prior day's volume over its trailing 20-day median.",
    "daily_range_ratio_1d": "Prior day's high-low range, scaled by that day's close.",
    "days_to_earnings": "Calendar days to the next earnings date on or after publication.",
    "session": (
        "Market session the article was published into: pre-market, market-hours, after-hours."
    ),
    "news_volume": (
        f"Count of this ticker's own articles in the {NEWS_VOLUME_LOOKBACK_DAYS} days before "
        "publication. Scoped to one corpus, so it measures coverage intensity for this company "
        "rather than how busy the market was overall."
    ),
}

OHLCV_FIELDS = ["Close", "High", "Low", "Volume"]


def _to_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    index = pd.to_datetime(index)
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def _utc_values(series: pd.Series) -> np.ndarray:
    """A tz-aware series as a true datetime64 array of UTC instants.

    `Series.to_numpy()` on a tz-aware series yields an object array of
    Timestamps, which searchsorted can still order but only by falling back to
    Python-level comparison, and which no timedelta64 arithmetic accepts at all.
    Dropping the tz *after* converting to UTC keeps the instants identical.
    """
    return pd.to_datetime(series, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


def split_ohlcv(ohlcv: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Split yfinance's wide frame into close/high/low/volume, indexed in UTC.

    yfinance returns a ("Price", "Ticker") column MultiIndex and a tz-naive
    index; every function in `market.features` compares against tz-aware
    timestamps, so the index is localised here once rather than in each caller.
    """
    out = {}
    for field in OHLCV_FIELDS:
        frame = ohlcv.xs(field, level="Price", axis=1).sort_index()
        frame.columns = [str(c).strip() for c in frame.columns]
        frame.index = _to_utc_index(frame.index)

        missing = [s for s in symbols if s not in frame.columns]
        if missing:
            raise KeyError(f"{field} is missing {missing}; re-run market.prices for those symbols")
        out[field.lower()] = frame[symbols].copy()
    return out


def _normalise_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    schedule = schedule.copy()
    for col in ["market_open", "market_close"]:
        schedule[col] = pd.to_datetime(schedule[col], utc=True)
    return schedule.sort_values("market_open").reset_index(drop=True)


def align_sessions(timestamps: pd.Series, schedule: pd.DataFrame) -> pd.Series:
    """Vectorised `align_timestamp`: the first market open strictly after each timestamp.

    Raises on a timestamp the schedule cannot cover, matching `align_timestamp`,
    rather than returning NaT: a corpus reaching past the calendar means the
    price pull needs re-running, not that rows should be dropped quietly.
    """
    schedule = _normalise_schedule(schedule)
    if schedule.empty:
        raise ValueError("Empty schedule")

    ts = pd.to_datetime(timestamps, utc=True)
    # Both sides go through _utc_values, so both are UTC instants and the
    # searchsorted below is like-for-like.
    opens = _utc_values(schedule["market_open"])
    ts_values = _utc_values(ts)

    before = ts_values < opens[0]
    if before.any():
        raise ValueError(
            f"{int(before.sum())} timestamps fall before the schedule's first open "
            f"({opens[0]}); re-run market.prices with an earlier --start"
        )

    pos = np.searchsorted(opens, ts_values, side="right")
    beyond = pos >= len(opens)
    if beyond.any():
        raise ValueError(
            f"{int(beyond.sum())} timestamps have no market open after them "
            f"(last open {opens[-1]}); re-run market.prices with a later --end"
        )
    return pd.Series(pd.to_datetime(opens[pos], utc=True), index=ts.index)


def assign_sessions(timestamps: pd.Series, schedule: pd.DataFrame) -> pd.Series:
    """Vectorised `session_for`: pre-market / market-hours / after-hours per timestamp.

    A timestamp on a non-trading day has no open or close to sit between and is
    after-hours, which is what `session_for` returns for that case too.
    """
    schedule = _normalise_schedule(schedule)
    ts = pd.to_datetime(timestamps, utc=True)

    by_date = (
        pd.DataFrame(
            {
                "date": schedule["market_open"].dt.date,
                "open": schedule["market_open"],
                "close": schedule["market_close"],
            }
        )
        .drop_duplicates("date")
        .set_index("date")
    )

    dates = pd.Index(ts.dt.date)
    opens = pd.Series(
        pd.to_datetime(by_date["open"].reindex(dates).to_numpy(), utc=True), index=ts.index
    )
    closes = pd.Series(
        pd.to_datetime(by_date["close"].reindex(dates).to_numpy(), utc=True), index=ts.index
    )

    session = pd.Series("after-hours", index=ts.index, dtype=object)
    trading_day = opens.notna()
    session[trading_day & (ts < opens)] = "pre-market"
    session[trading_day & (ts >= opens) & (ts <= closes)] = "market-hours"
    return session


def news_volume(
    timestamps: pd.Series, lookback_days: int = NEWS_VOLUME_LOOKBACK_DAYS
) -> pd.Series:
    """Count of prior articles within `lookback_days` of each one, excluding itself.

    searchsorted over the sorted timestamps rather than a per-row mask, which on
    a 17k-article corpus is the difference between one pass and 17k full scans.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    values = _utc_values(ts)
    sorted_values = np.sort(values, kind="stable")

    window_start = values - np.timedelta64(lookback_days, "D")
    upper = np.searchsorted(sorted_values, values, side="left")
    lower = np.searchsorted(sorted_values, window_start, side="left")
    return pd.Series(upper - lower, index=ts.index)


def _map_by_date(dates: pd.Series, fn) -> pd.Series:
    """Evaluate `fn` once per unique date and map the result back onto every row."""
    lookup = {date: fn(date) for date in pd.DatetimeIndex(dates.unique())}
    return dates.map(lookup)


def _abnormal_return_at_open(
    event_open: pd.Timestamp,
    horizon_days: int,
    asset_close: pd.Series,
    market_close: pd.Series,
) -> float:
    """`abnormal_return_for`'s body, taking the aligned open instead of the raw timestamp.

    Split out so the label can be cached per session: `abnormal_return_for`
    calls `align_timestamp` itself, and that call is the only part of it that
    depends on the article's own timestamp rather than the session it lands in.
    """
    event_date = pd.Timestamp(event_open).normalize()
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


def build_market_features(
    articles: pd.DataFrame,
    ticker: str,
    ohlcv: pd.DataFrame,
    schedule: pd.DataFrame,
    earnings: pd.DataFrame,
    market_index: str = MARKET_INDEX,
) -> pd.DataFrame:
    """One market feature row per article in `articles`.

    `articles` needs `article_id` and `timestamp_utc` and nothing else: the
    market layer never reads article text, which is what keeps the two layers
    independent. Rows carrying no label at any horizon are dropped, since the
    only reason to keep an article here is to train or evaluate against it.
    """
    if articles.empty:
        raise ValueError("No articles passed")

    price = split_ohlcv(ohlcv, [ticker, market_index])
    close, high, low, volume = price["close"], price["high"], price["low"], price["volume"]
    schedule = _normalise_schedule(schedule)

    earnings = earnings.copy()
    earnings.index = _to_utc_index(earnings.index)

    out = articles[["article_id", "timestamp_utc"]].copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out.insert(1, "ticker", ticker)
    article_date = out["timestamp_utc"].dt.normalize()

    out["session"] = assign_sessions(out["timestamp_utc"], schedule)
    out["days_to_earnings"] = _map_by_date(
        article_date, lambda d: days_to_next_earnings(d, earnings)
    )

    for lookback in MOMENTUM_LOOKBACKS:
        out[f"momentum_{lookback}d"] = _map_by_date(
            article_date, lambda d, k=lookback: cumulative_return(close[ticker], d, k)
        )
    out[f"volatility_{VOLATILITY_WINDOW}d"] = _map_by_date(
        article_date, lambda d: rolling_volatility(close[ticker], d, window=VOLATILITY_WINDOW)
    )
    out[f"beta_{BETA_WINDOW}d"] = _map_by_date(
        article_date,
        lambda d: beta_vs_market(close[ticker], close[market_index], d, window=BETA_WINDOW),
    )
    out[f"relative_volume_{RELATIVE_VOLUME_WINDOW}d"] = _map_by_date(
        article_date, lambda d: relative_volume(volume[ticker], d, window=RELATIVE_VOLUME_WINDOW)
    )
    out["daily_range_ratio_1d"] = _map_by_date(
        article_date, lambda d: daily_range_ratio(high[ticker], low[ticker], close[ticker], d)
    )

    # The label. Anchored to the session at or after publication, so unlike
    # every feature above it is keyed by the aligned open rather than by the
    # article's own day: an article published on an evening and one published
    # the next morning before the bell share a session, and so share a label.
    aligned = align_sessions(out["timestamp_utc"], schedule)
    # Kept as a column, not just used and discarded. It is the unit every row
    # sharing a label belongs to, so the evaluation harness groups folds on it:
    # evaluate() reads it by name and refuses to run without it. Deriving it
    # downstream would mean every consumer reloading the trading calendar to
    # recompute something this function already knows.
    out["session_open"] = aligned
    for horizon in LABEL_HORIZONS_DAYS:
        out[f"abnormal_return_{horizon}d"] = _map_by_date(
            aligned,
            lambda open_ts, h=horizon: _abnormal_return_at_open(
                open_ts, h, close[ticker], close[market_index]
            ),
        )
    out["label_direction"] = (
        np.sign(out[f"abnormal_return_{LABEL_HORIZONS_DAYS[0]}d"]).fillna(0).astype(int)
    )
    out["news_volume"] = news_volume(out["timestamp_utc"])

    out = out[IDENTITY_COLUMNS + LABEL_COLUMNS + FEATURE_COLUMNS]
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    label_cols = [f"abnormal_return_{h}d" for h in LABEL_HORIZONS_DAYS]
    return out.dropna(subset=label_cols, how="all").reset_index(drop=True)
