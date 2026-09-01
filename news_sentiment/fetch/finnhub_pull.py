"""Pull raw company news from Finnhub's `company_news` endpoint.

`company_news` returns a fixed-size cap of results per call regardless of how
wide the requested date range is, filled most-recent-first (`notebooks/
modelling/3.0` section 6 confirmed this: whole-month requests against the
original pull returned a consistent 126-191 articles per month no matter how
wide the active window inside it was, silently dropping every day but the
last few). `pull_company_news` avoids that by requesting narrow, fixed windows
(`window_days`, one day by default) and concatenating the result, so no single
call's article count gets anywhere near the cap.

Ported from `notebooks/text/1.0-aw-corpus.ipynb`, which is left untouched as
a historical record of the original, monthly-windowed pull and does not
import this module.

Run as `python -m news_sentiment.fetch.finnhub_pull TICKER` to pull one
ticker's raw corpus and write it to `data/raw/{TICKER}_raw_articles.parquet`.
Nothing else runs on import.
"""

import argparse
from datetime import UTC, datetime
import time

import finnhub
from loguru import logger
import pandas as pd
import requests

from news_sentiment.config import (
    FETCH_CAP_WARN_COUNT,
    FETCH_RETRY_ATTEMPTS,
    FETCH_RETRY_BACKOFF_SECONDS,
    FINNHUB_API_KEY,
    NEWS_END_DATE,
    NEWS_START_DATE,
    raw_articles_path,
)

_client = None


def get_client() -> finnhub.Client:
    """Lazily build the shared Finnhub client, so importing this module never
    requires an API key to be set."""
    global _client
    if _client is None:
        _client = finnhub.Client(api_key=FINNHUB_API_KEY)
    return _client


def to_utc(ts: int) -> datetime:
    """Convert a unix timestamp (seconds) to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ts, tz=UTC)


def _fetch_window(
    client: finnhub.Client, ticker: str, start: pd.Timestamp, window_end: pd.Timestamp
) -> list:
    """Call company_news for one window, retrying a transient network error
    up to FETCH_RETRY_ATTEMPTS times before giving up on this window alone.

    Returns [] (not a raised exception) on final failure, so one dead window
    doesn't cost the rest of a multi-minute pull; logged as an error so the
    gap it leaves is visible and would show up in the fetch report's
    per-day counts."""
    last_exc = None
    for attempt in range(1, FETCH_RETRY_ATTEMPTS + 1):
        try:
            return client.company_news(
                ticker,
                _from=start.strftime("%Y-%m-%d"),
                to=window_end.strftime("%Y-%m-%d"),
            )
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < FETCH_RETRY_ATTEMPTS:
                logger.warning(
                    f"{ticker} {start:%Y-%m-%d}..{window_end:%Y-%m-%d} attempt "
                    f"{attempt}/{FETCH_RETRY_ATTEMPTS} failed ({e!r}), retrying "
                    f"in {FETCH_RETRY_BACKOFF_SECONDS}s"
                )
                time.sleep(FETCH_RETRY_BACKOFF_SECONDS)

    logger.error(
        f"{ticker} {start:%Y-%m-%d}..{window_end:%Y-%m-%d} failed after "
        f"{FETCH_RETRY_ATTEMPTS} attempts, skipping this window: {last_exc!r}"
    )
    return []


def pull_company_news(
    ticker: str, _from: str, to: str, window_days: int = 1, pause: float = 1.0
) -> pd.DataFrame:
    """Pull company news in fixed-size day windows and return it as a DataFrame.

    ticker: symbol to query, e.g. "TSLA".
    _from, to: inclusive date bounds as 'YYYY-MM-DD' strings.
    window_days: size of each query window, in days. 1 by default, so a
        single busy day's article count is the only thing that can approach
        the per-call cap, not a whole month's. Widen only after checking the
        fetch report (`news_sentiment.fetch.report`) shows no warning at the
        chosen width.
    pause: seconds to sleep between calls, keeping us under 60 calls/min. A
        daily window means far more calls than the old monthly one (roughly
        one per day in range rather than one per month), so this defaults to
        1s rather than 0.
    """
    client = get_client()
    rows = []
    start = pd.Timestamp(_from)
    end = pd.Timestamp(to)

    while start <= end:
        window_end = min(start + pd.Timedelta(days=window_days - 1), end)

        articles = _fetch_window(client, ticker, start, window_end)
        if len(articles) >= FETCH_CAP_WARN_COUNT:
            logger.warning(
                f"{ticker} {start:%Y-%m-%d}..{window_end:%Y-%m-%d} returned "
                f"{len(articles)} articles, at or above FETCH_CAP_WARN_COUNT "
                f"({FETCH_CAP_WARN_COUNT}); this window may be getting "
                f"capped the same way the original monthly pull was; narrow "
                f"window_days and re-pull this range."
            )
        for a in articles:
            rows.append(
                {
                    "article_id": a["id"],
                    "headline": a["headline"],
                    "summary": a["summary"],
                    "source": a["source"],
                    "url": a["url"],
                    "timestamp_utc": to_utc(a["datetime"]),
                }
            )

        start = window_end + pd.Timedelta(days=1)
        time.sleep(pause)

    df = pd.DataFrame(rows)

    # A narrow window_days means far more window boundaries than the old
    # monthly pull had (roughly one per day in range, not one per month), so
    # the same article landing in two adjacent windows (a boundary timestamp,
    # API timezone slop) is a real risk here in a way it barely was before.
    # Dedupe at the source rather than downstream, since a duplicate
    # article_id surviving into a merge elsewhere (e.g. scrape_corpus)
    # multiplies rather than just double-counts.
    n_dupes = df["article_id"].duplicated().sum() if len(df) else 0
    if n_dupes:
        logger.warning(
            f"{ticker}: dropping {n_dupes} duplicate article_id rows from overlapping windows"
        )
        df = df.drop_duplicates(subset="article_id", keep="first").reset_index(drop=True)

    return df


def main() -> None:
    """CLI entry point. Pulls one ticker's raw corpus and writes it to
    `data/raw/{TICKER}_raw_articles.parquet`."""
    parser = argparse.ArgumentParser(description="Pull raw Finnhub company news for one ticker.")
    parser.add_argument("ticker", help='Symbol to query, e.g. "TSLA".')
    parser.add_argument(
        "--from", dest="_from", default=NEWS_START_DATE, help="Inclusive start date, YYYY-MM-DD."
    )
    parser.add_argument(
        "--to", dest="to", default=NEWS_END_DATE, help="Inclusive end date, YYYY-MM-DD."
    )
    parser.add_argument(
        "--window-days", type=int, default=1, help="Size of each query window, in days."
    )
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to sleep between calls.")
    args = parser.parse_args()

    logger.info(
        f"pulling {args.ticker} news {args._from}..{args.to} (window_days={args.window_days})"
    )
    articles = pull_company_news(
        args.ticker, args._from, args.to, window_days=args.window_days, pause=args.pause
    )

    out_path = raw_articles_path(args.ticker)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(out_path, index=False)
    logger.info(f"wrote {len(articles)} articles to {out_path}")


if __name__ == "__main__":
    main()
