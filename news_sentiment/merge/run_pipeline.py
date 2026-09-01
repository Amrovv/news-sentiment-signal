"""Full merge pipeline: two feature tables in, one model-ready table out.

    A  load    the text and market tables, and the corpus both were built from
    B  merge   inner join on article_id, with the key checked as it goes
    C  test    reconstruct momentum_1d from raw OHLCV, both hypotheses
    D  write   the merged table, then its report
    E  pool    every ticker stacked, keyed on (article_id, ticker)

The third pipeline, and the same shape as the other two: a ticker argument, a
per-ticker output directory, one report per ticker.

    data/processed/merged/{TICKER}/merged.parquet   per ticker
    data/processed/merged/pooled.parquet            every ticker, one table
    reports/merge/{TICKER}_merge_report.md

The pooled table is written only by `--all`, since it is not meaningful from one
ticker. It is keyed on `(article_id, ticker)`, never `article_id` alone.

Run with `python -m news_sentiment.merge.run_pipeline TICKER`, or `--all`.
Nothing runs on import.
"""

import argparse

from loguru import logger
import pandas as pd

from news_sentiment.config import (
    COMPANIES,
    PROCESSED_DATA_DIR,
    RAW_OHLCV_PATH,
    market_run_dir,
    merged_dir,
    processed_articles_path,
)
from news_sentiment.market.labels import split_ohlcv
from news_sentiment.merge import integrity, leakage, report
from news_sentiment.merge import merge as merge_module

CORPUS_COLUMNS = ["article_id", "headline", "timestamp_utc"]


def load_tables(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read one ticker's text table, market table, and source corpus."""
    text_path = PROCESSED_DATA_DIR / "pipeline_run" / ticker / "article_features.parquet"
    market_path = market_run_dir(ticker) / "market_features.parquet"

    for path, hint in [
        (text_path, f"python -m news_sentiment.text.run_pipeline {ticker}"),
        (market_path, f"python -m news_sentiment.market.run_pipeline {ticker}"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: {hint}")

    corpus = pd.read_parquet(processed_articles_path(ticker), columns=CORPUS_COLUMNS)
    corpus["timestamp_utc"] = pd.to_datetime(corpus["timestamp_utc"], utc=True)
    return pd.read_parquet(text_path), pd.read_parquet(market_path), corpus


def run(ticker: str, results: dict | None = None):
    """Merge one ticker and write its table and report.

    `results` is every ticker merged so far, used only for the shared-article
    section: how much of this ticker's coverage it holds in common with the
    others. Passing nothing simply leaves that section empty.
    """
    text, market, corpus = load_tables(ticker)
    logger.info(
        f"[A] [{ticker}] text {len(text):,} rows, market {len(market):,} rows, "
        f"corpus {len(corpus):,} articles"
    )

    # --- Phase B: the join ---------------------------------------------------
    result = merge_module.merge_ticker(text, market, ticker, corpus=corpus)
    logger.info(
        f"[B] merged {len(result.merged):,} rows "
        f"({len(result.text_only):,} text-only, {len(result.market_only):,} market-only)"
    )
    if not result.passed:
        for check in result.failed_checks:
            logger.error(f"[B] {check.name}: {check.detail}")

    # --- Phase C: the leakage regression test --------------------------------
    close = split_ohlcv(pd.read_parquet(RAW_OHLCV_PATH), [ticker])["close"][ticker]
    leak = leakage.test_momentum_1d(market, close, ticker)
    level = logger.info if leak.passed else logger.error
    level(
        f"[C] momentum_1d {leak.status}: {leak.matches_correct:,}/{leak.rows_tested:,} match the "
        f"pre-publication formula, {leak.real_leaks:,} real leaks"
    )

    # --- Phase D: the deliverable and its report -----------------------------
    out_dir = merged_dir(ticker)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "merged.parquet"
    result.merged.to_parquet(out_path, index=False)
    logger.info(f"[D] wrote {out_path} {result.merged.shape}")

    overlap = (
        merge_module.overlap_summary({**(results or {}), ticker: result}, ticker)
        if results
        else pd.DataFrame(columns=["other_ticker", "shared_articles", "share_of_own"])
    )
    report_path = report.write_merge_report(result, leak, corpus, overlap)
    logger.info(f"[D] wrote {report_path}")

    return result, leak


def available_tickers() -> list[str]:
    """Every registry ticker with both a text and a market table on disk."""
    return [
        t
        for t in COMPANIES
        if (PROCESSED_DATA_DIR / "pipeline_run" / t / "article_features.parquet").exists()
        and (market_run_dir(t) / "market_features.parquet").exists()
    ]


def run_all(tickers: list[str]) -> pd.DataFrame:
    """Merge every ticker, then write the pooled table.

    Two passes: the first merges, the second rewrites each report now that every
    ticker's table exists, so the shared-article section can name the tickers an
    article is shared with rather than only the ones merged before it.
    """
    results, leaks = {}, {}
    for ticker in tickers:
        results[ticker], leaks[ticker] = run(ticker)

    for ticker in tickers:
        _, _, corpus = load_tables(ticker)
        overlap = merge_module.overlap_summary(results, ticker)
        report.write_merge_report(results[ticker], leaks[ticker], corpus, overlap)
    logger.info(f"rewrote {len(tickers)} reports with cross-ticker overlap")

    pooled = merge_module.pool(results)
    pooled_path = PROCESSED_DATA_DIR / "merged" / "pooled.parquet"
    pooled_path.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_parquet(pooled_path, index=False)
    logger.info(
        f"[E] wrote {pooled_path} {pooled.shape} "
        f"({pooled['article_id'].nunique():,} distinct articles)"
    )

    corpora = {t: load_tables(t)[2] for t in tickers}
    check = integrity.check_cross_ticker_ids(corpora)
    level = logger.info if check.passed else logger.error
    level(f"[E] {check.name}: {check.detail}")
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the text and market tables for one ticker."
    )
    parser.add_argument("ticker", nargs="?", help='Symbol to merge, e.g. "TSLA".')
    parser.add_argument(
        "--all", action="store_true", help="Merge every ticker, and write the pooled table."
    )
    args = parser.parse_args()

    if args.all:
        tickers = available_tickers()
        logger.info(f"merging {tickers}")
        run_all(tickers)
    elif args.ticker:
        run(args.ticker)
    else:
        parser.error("pass a ticker or --all")


if __name__ == "__main__":
    main()
