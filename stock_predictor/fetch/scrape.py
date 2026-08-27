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

Nothing runs on import.
"""

import json
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from stock_predictor.config import (
    MAX_SHIFT_HOURS,
    MIN_BODY_CHARS,
    REQ_PER_SEC,
    SCRAPE_HEADERS,
    SCRAPE_MAX_WORKERS,
    SCRAPE_TIMEOUT,
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
    cross-host original outright, a same-host time only within
    MAX_SHIFT_HOURS, else keep the API time so no article is dropped for
    want of a page time."""
    if candidate is None:
        return api_utc, "api"
    if cross_host:
        return candidate, "corrected"
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
