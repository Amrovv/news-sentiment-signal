"""Market inputs: OHLCV bars, the NYSE calendar, and earnings dates.

The three raw tables every market feature is computed from, none of which the
feature functions fetch for themselves. Extracted from notebooks/market/1.0
(prices, earnings) and notebooks/market/1.1 (the trading calendar), which pulled
TSLA and SPY only.

Written as `data/raw/raw_ohlcv.parquet` (every ticker plus the benchmark, one
frame), `data/raw/raw_schedule.parquet` (shared), and one
`data/raw/{TICKER}_raw_earnings.parquet` per company.

Run with `python -m stock_predictor.market.prices TSLA AAPL AMZN NVDA`. Nothing
runs on import.
"""

import argparse
import datetime as dt
import time

from loguru import logger
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

from stock_predictor.config import (
    FETCH_RETRY_ATTEMPTS,
    FETCH_RETRY_BACKOFF_SECONDS,
    MARKET_INDEX,
    NEWS_END_DATE,
    NEWS_START_DATE,
    PRICE_PAD_DAYS,
    RAW_OHLCV_PATH,
    RAW_SCHEDULE_PATH,
    SCHEDULE_PAD_DAYS,
    raw_earnings_path,
)


def _empty_symbols(frame: pd.DataFrame, tickers: list[str]) -> list[str]:
    """Requested symbols whose Close column is absent or entirely null."""
    close = frame.xs("Close", level="Price", axis=1)
    return [t for t in tickers if t not in close.columns or close[t].notna().sum() == 0]


def pull_ohlcv(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Daily OHLCV for `tickers`, padded either side of the news window.

    The pad is not cosmetic: a 20-day lookback feature computed for an article
    published on the first day of the news window needs 20 trading days that
    precede it, and the 3-day label for the last article needs bars that follow
    it. Without the pad both ends of the corpus silently produce NaN.

    Retried per symbol rather than trusted, because `yf.download` reports a
    failed symbol on stdout and still returns a full-width frame with that
    symbol's columns present and entirely NaN. Seen in practice as an
    `OperationalError('database is locked')` on yfinance's own cache while the
    other four symbols succeeded. Left unchecked that is silent and total: the
    benchmark is one of these symbols, and a null benchmark nulls every
    abnormal return in the table while every stage still reports success.
    """
    price_start = pd.Timestamp(start_date) - dt.timedelta(days=PRICE_PAD_DAYS)
    price_end = pd.Timestamp(end_date) + dt.timedelta(days=PRICE_PAD_DAYS)

    logger.info(f"OHLCV: {sorted(tickers)} from {price_start:%Y-%m-%d} to {price_end:%Y-%m-%d}")
    frame = yf.download(tickers, start=price_start, end=price_end)
    if frame is None or frame.empty:
        raise RuntimeError(f"yfinance returned no OHLCV rows for {tickers}")

    for attempt in range(1, FETCH_RETRY_ATTEMPTS + 1):
        empty = _empty_symbols(frame, tickers)
        if not empty:
            return frame
        logger.warning(
            f"no bars for {empty} (attempt {attempt}/{FETCH_RETRY_ATTEMPTS}); "
            f"retrying those symbols in {FETCH_RETRY_BACKOFF_SECONDS}s"
        )
        time.sleep(FETCH_RETRY_BACKOFF_SECONDS)
        retry = yf.download(empty, start=price_start, end=price_end)
        if retry is not None and not retry.empty:
            frame = frame.drop(columns=empty, level="Ticker", errors="ignore")
            frame = frame.join(retry).sort_index(axis=1)

    empty = _empty_symbols(frame, tickers)
    if empty:
        raise RuntimeError(
            f"yfinance returned no usable bars for {empty} after {FETCH_RETRY_ATTEMPTS} attempts"
        )
    return frame


def pull_earnings(ticker: str, start_date: str) -> pd.DataFrame:
    """Earnings dates for `ticker` on or after `start_date`.

    Returned newest-first by yfinance. Left in that order deliberately, because
    that is the order the file has always been written in and
    `days_to_next_earnings` sorts its input itself rather than trusting callers.
    """
    earnings = yf.Ticker(ticker).get_earnings_dates()
    if earnings is None or earnings.empty:
        raise RuntimeError(f"yfinance returned no earnings dates for {ticker}")
    return earnings[earnings.index >= start_date]


def pull_schedule(start_date: str, end_date: str) -> pd.DataFrame:
    """NYSE open/close times covering the news window plus a pad.

    `align_timestamp` raises when a publication timestamp falls outside the
    schedule it is handed, so this has to reach past the last article: an
    article published on the final evening aligns to a session that opens the
    following morning.
    """
    schedule_start = pd.Timestamp(start_date) - dt.timedelta(days=1)
    schedule_end = pd.Timestamp(end_date) + dt.timedelta(days=SCHEDULE_PAD_DAYS)
    logger.info(f"NYSE calendar: {schedule_start:%Y-%m-%d} to {schedule_end:%Y-%m-%d}")
    return mcal.get_calendar("NYSE").schedule(start_date=schedule_start, end_date=schedule_end)


def run(
    tickers: list[str], start_date: str = NEWS_START_DATE, end_date: str = NEWS_END_DATE
) -> None:
    """Pull and write every market input needed to build features for `tickers`."""
    symbols = sorted(set(tickers) | {MARKET_INDEX})

    ohlcv = pull_ohlcv(symbols, start_date, end_date)
    RAW_OHLCV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.to_parquet(RAW_OHLCV_PATH)
    logger.info(f"wrote {RAW_OHLCV_PATH} {ohlcv.shape}")

    schedule = pull_schedule(start_date, end_date)
    schedule.to_parquet(RAW_SCHEDULE_PATH)
    logger.info(f"wrote {RAW_SCHEDULE_PATH} ({len(schedule)} sessions)")

    for ticker in tickers:
        earnings = pull_earnings(ticker, start_date)
        earnings.to_parquet(raw_earnings_path(ticker))
        logger.info(f"wrote {raw_earnings_path(ticker)} ({len(earnings)} dates)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull OHLCV, the NYSE calendar, and earnings dates."
    )
    parser.add_argument("tickers", nargs="+", help="Symbols to pull, e.g. TSLA AAPL AMZN NVDA.")
    parser.add_argument("--start", default=NEWS_START_DATE, help="News window start (YYYY-MM-DD).")
    parser.add_argument("--end", default=NEWS_END_DATE, help="News window end (YYYY-MM-DD).")
    args = parser.parse_args()
    run(args.tickers, args.start, args.end)


if __name__ == "__main__":
    main()
