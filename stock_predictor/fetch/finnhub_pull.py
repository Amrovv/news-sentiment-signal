"""Pull raw company news from Finnhub's `company_news` endpoint.

Finnhub's free tier caps how many articles a single call returns, so a wide
date-range request silently truncates. `pull_company_news` works around that
by requesting month by month and concatenating the result.

Ported from `notebooks/text/1.0-aw-corpus.ipynb`, which is left untouched as
a historical record of the original pull and does not import this module.

Nothing runs on import.
"""

import time
from datetime import datetime, timezone

import finnhub
import pandas as pd

from stock_predictor.config import FINNHUB_API_KEY

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
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def pull_company_news(ticker: str, _from: str, to: str, pause: float = 0) -> pd.DataFrame:
    """Pull company news month by month and return it as a DataFrame.

    ticker: symbol to query, e.g. "TSLA".
    _from, to: inclusive date bounds as 'YYYY-MM-DD' strings.
    pause: seconds to sleep between calls, keeping us under 60 calls/min.
    """
    client = get_client()
    rows = []
    start = pd.Timestamp(_from)
    end = pd.Timestamp(to)

    while start <= end:
        # Last day of the current month, or the overall end if it comes first.
        month_end = min(start + pd.offsets.MonthEnd(0), end)

        articles = client.company_news(
            ticker,
            _from=start.strftime("%Y-%m-%d"),
            to=month_end.strftime("%Y-%m-%d"),
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

        start = month_end + pd.Timedelta(days=1)
        time.sleep(pause)

    return pd.DataFrame(rows)
