"""Fetch each article's page once and recover the true source, true publish
time, and clean body text that Finnhub's `company_news` summary doesn't
carry.

The rules encoded here (which sources are readable, how the true source and
time get resolved) came out of the probe in
`notebooks/text/1.1-aw-scraper-probe.ipynb`, not repeated here. This module
is the pipeline that probe's findings became.

Ported from `notebooks/text/1.2-aw-clean-and-convert.ipynb`, which is left
untouched as a historical record of the original run and does not import
this module.

Run as `python -m stock_predictor.fetch.scrape TICKER` to scrape one
ticker's `data/raw/{TICKER}_raw_articles.parquet` and write the accepted,
processed rows to `data/interim/{TICKER}_processed_articles.parquet`. The
CLI runs in checkpointed chunks (`scrape_corpus_chunked`, SCRAPE_CHUNK_SIZE)
since a full corpus is tens of thousands of articles and, at REQ_PER_SEC,
well over an hour of continuous network I/O -- long enough that losing the
whole run to one interruption (a killed process, a dropped connection, a
machine restart) partway through is a real cost worth checkpointing against,
not just a theoretical one. Nothing else runs on import.
"""

import argparse
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from stock_predictor.config import (
    MAX_CROSS_HOST_SHIFT_DAYS,
    MAX_SHIFT_HOURS,
    MIN_BODY_CHARS,
    OPEN_SOURCES,
    REQ_PER_SEC,
    SCRAPE_CHUNK_SIZE,
    SCRAPE_HEADERS,
    SCRAPE_MAX_WORKERS,
    SCRAPE_TIMEOUT,
    processed_articles_path,
    processed_chunk_dir,
    raw_articles_path,
)


def make_session() -> requests.Session:
    """A pooled, retrying session shared across the worker threads."""
    s = requests.Session()
    s.headers.update(SCRAPE_HEADERS)
    # Bounded backoff, ignore Retry-After (Yahoo's is long enough to stall a
    # worker), and return the final response instead of raising, so a
    # throttle shows as 429.
    retry = Retry(
        total=2, backoff_factor=0.5, backoff_max=8,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=False, raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=SCRAPE_MAX_WORKERS,
        pool_maxsize=SCRAPE_MAX_WORKERS * 2,
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


class Pacer:
    """Cap the request rate across all worker threads to REQ_PER_SEC, so the
    pool does not burst and trip Yahoo's 429 limit."""

    def __init__(self, per_second):
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._interval
        time.sleep(start - now)


pacer = Pacer(REQ_PER_SEC)


def _ld_objects(soup):
    """Yield each JSON-LD object on the page, flattening list payloads."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        for obj in (data if isinstance(data, list) else [data]):
            if isinstance(obj, dict):
                yield obj


def _true_source(soup, host):
    """Prefer the original outlet in the provider field, then the publisher, then the host."""
    provider = publisher = None
    for obj in _ld_objects(soup):
        prov = obj.get("provider")
        if provider is None and prov:
            provider = prov.get("name") if isinstance(prov, dict) else (prov if isinstance(prov, str) else None)
        pub = obj.get("publisher")
        if publisher is None and isinstance(pub, dict) and pub.get("name"):
            publisher = pub["name"]
    return provider or publisher or host


def _published(soup):
    """Read the published time from JSON-LD, or fall back to a meta tag."""
    for obj in _ld_objects(soup):
        if obj.get("datePublished"):
            return obj["datePublished"]
    m = soup.find("meta", attrs={"property": "article:published_time"})
    return m["content"] if m and m.get("content") else None


def _to_utc(value):
    """Parse an ISO time to UTC. Naive (zoneless) times are not trusted."""
    if not value:
        return None
    try:
        dt = date_parser.parse(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def _resolve_time(candidate, api_utc, cross_host):
    """Pick the trustworthy time, with the API time as the floor: trust a
    cross-host original within MAX_CROSS_HOST_SHIFT_DAYS of the API time, a
    same-host time only within MAX_SHIFT_HOURS, else keep the API time so no
    article is dropped for want of a page time.

    The cross-host bound exists because "the original outlet's own page" is
    not the same guarantee as "this story is from now" -- a genuinely old,
    re-referenced story can surface in the feed with a fresh API timestamp
    and a years-old canonical date."""
    if candidate is None:
        return api_utc, "api"
    if cross_host:
        if abs((candidate - api_utc).total_seconds()) <= MAX_CROSS_HOST_SHIFT_DAYS * 86400:
            return candidate, "corrected"
        return api_utc, "api"
    if abs((candidate - api_utc).total_seconds()) <= MAX_SHIFT_HOURS * 3600:
        return candidate, "corrected"
    return api_utc, "api"


def _clean_body(soup):
    """Drop chrome tags, join the paragraph text, collapse whitespace."""
    for tag in soup(["script", "style", "nav", "aside", "footer", "header", "form", "figure"]):
        tag.decompose()
    text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    return re.sub(r"\s+", " ", text).strip()


def process_article(row, session):
    """Fetch one article once and derive source, UTC time, and body. A second
    fetch fires only to follow a cross-host canonical link for the true time."""
    out = {
        "article_id": row["article_id"], "true_source": None, "utc": None,
        "time_source": None, "raw_body": None, "processed_body": None,
        "status": None, "source_ok": False, "corrected": False,
        "text_ok": False, "n_fetch": 0,
    }
    url = row["url"]
    try:
        pacer.wait()  # hold the global rate
        r = session.get(url, timeout=SCRAPE_TIMEOUT)
        out["n_fetch"] = 1
        out["status"] = r.status_code
        if r.status_code != 200:
            return out
        host = urlparse(r.url).netloc
        soup = BeautifulSoup(r.text, "html.parser")

        out["true_source"] = _true_source(soup, host)
        out["source_ok"] = out["true_source"] is not None

        published = _published(soup)
        cross_host = False
        link = soup.find("link", rel="canonical")
        canonical = link["href"] if link and link.get("href") else None
        if canonical and urlparse(canonical).netloc not in ("", host):
            try:
                r2 = session.get(canonical, timeout=SCRAPE_TIMEOUT)
                out["n_fetch"] = 2
                if r2.status_code == 200:
                    original = _published(BeautifulSoup(r2.text, "html.parser"))
                    if original:
                        published, cross_host = original, True
            except requests.RequestException:
                pass

        candidate = _to_utc(published)
        api_utc = row["timestamp_utc"].to_pydatetime()
        out["utc"], out["time_source"] = _resolve_time(candidate, api_utc, cross_host)
        out["corrected"] = out["time_source"] == "corrected"

        out["raw_body"] = r.text
        out["processed_body"] = _clean_body(soup)
        out["text_ok"] = len(out["processed_body"]) >= MIN_BODY_CHARS
    except requests.RequestException as e:
        out["status"] = type(e).__name__  # record the error kind (Timeout, ConnectionError)
    return out


def run_pipeline(articles: pd.DataFrame, max_workers: int = SCRAPE_MAX_WORKERS) -> pd.DataFrame:
    """Run process_article over every row in parallel, preserving order."""
    rows = articles.to_dict("records")
    results = [None] * len(rows)
    with make_session() as session, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_article, row, session): i for i, row in enumerate(rows)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
    return pd.DataFrame(results)


def scrape_corpus(articles: pd.DataFrame, max_workers: int = SCRAPE_MAX_WORKERS) -> pd.DataFrame:
    """Filter to OPEN_SOURCES, run the scrape pipeline, and return the
    accepted rows: real source, corrected UTC time, and clean body text.

    An article is rejected only for a body that never reaches
    MIN_BODY_CHARS -- time always resolves against the API floor, so text is
    the only gate.
    """
    open_articles = articles[articles["source"].isin(OPEN_SOURCES)].copy()
    results = run_pipeline(open_articles, max_workers=max_workers)
    merged = open_articles.merge(
        results[[
            "article_id", "true_source", "utc", "raw_body", "processed_body",
            "status", "source_ok", "corrected", "time_source", "text_ok", "n_fetch",
        ]],
        on="article_id", how="left",
    )

    accept = merged["text_ok"]
    proc = merged[accept].copy()
    proc["source"] = proc["true_source"].where(proc["source_ok"], proc["source"])
    proc["timestamp_utc"] = proc["utc"]
    return proc[[
        "article_id", "headline", "summary", "source", "url",
        "timestamp_utc", "raw_body", "processed_body",
    ]].reset_index(drop=True)


def scrape_corpus_chunked(
    articles: pd.DataFrame,
    ticker: str,
    chunk_size: int = SCRAPE_CHUNK_SIZE,
    max_workers: int = SCRAPE_MAX_WORKERS,
) -> pd.DataFrame:
    """Run scrape_corpus in fixed-size chunks over raw article rows,
    checkpointing each chunk's result to `processed_chunk_dir(ticker)` before
    starting the next.

    A chunk whose checkpoint file already exists is loaded from disk and
    skipped rather than re-scraped, so re-running after an interruption picks
    up where it left off instead of starting over. Chunking is over raw rows,
    not open-source rows, so chunk boundaries are stable across runs even
    though scrape_corpus filters to OPEN_SOURCES internally.
    """
    chunk_dir = processed_chunk_dir(ticker)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = math.ceil(len(articles) / chunk_size) if len(articles) else 0
    results = []
    for i in range(n_chunks):
        chunk_path = chunk_dir / f"chunk_{i:04d}.parquet"
        if chunk_path.exists():
            logger.info(f"chunk {i + 1}/{n_chunks} already checkpointed, reusing {chunk_path}")
            results.append(pd.read_parquet(chunk_path))
            continue

        chunk = articles.iloc[i * chunk_size : (i + 1) * chunk_size]
        logger.info(f"scraping chunk {i + 1}/{n_chunks} ({len(chunk)} raw articles)")
        result = scrape_corpus(chunk, max_workers=max_workers)
        result.to_parquet(chunk_path, index=False)
        logger.info(f"chunk {i + 1}/{n_chunks}: {len(result)} accepted -> {chunk_path}")
        results.append(result)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def main() -> None:
    """CLI entry point. Scrapes one ticker's raw pull in checkpointed chunks
    and writes the processed corpus to
    `data/interim/{TICKER}_processed_articles.parquet`."""
    parser = argparse.ArgumentParser(
        description="Scrape a raw article pull into the processed corpus for one ticker."
    )
    parser.add_argument("ticker", help='Symbol whose raw pull to process, e.g. "TSLA".')
    parser.add_argument("--max-workers", type=int, default=SCRAPE_MAX_WORKERS)
    parser.add_argument(
        "--chunk-size", type=int, default=SCRAPE_CHUNK_SIZE,
        help="Articles per checkpointed chunk.",
    )
    args = parser.parse_args()

    in_path = raw_articles_path(args.ticker)
    articles = pd.read_parquet(in_path)
    logger.info(f"loaded {len(articles)} raw articles from {in_path}")

    processed = scrape_corpus_chunked(
        articles, args.ticker, chunk_size=args.chunk_size, max_workers=args.max_workers
    )

    out_path = processed_articles_path(args.ticker)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(out_path, index=False)
    logger.info(
        f"accepted {len(processed)} of {(articles['source'].isin(OPEN_SOURCES)).sum()} "
        f"open articles -> {out_path}"
    )


if __name__ == "__main__":
    main()
