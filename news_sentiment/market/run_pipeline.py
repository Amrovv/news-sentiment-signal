"""Full market pipeline: one ticker's article table in, its market feature row out.

    A  load     the ticker's corpus, projected to article_id + timestamp_utc
    B  load     OHLCV, the NYSE calendar, the ticker's earnings dates
    C  labels.build_market_features   features + labels, one row per article
    D  write    the deliverable, then report.write_market_report

The market-side counterpart to `text.run_pipeline`: a ticker argument and a
per-ticker output directory. The two deliverables are what a model merges.

    data/processed/pipeline_run/{TICKER}/article_features.parquet   text
    data/processed/market_run/{TICKER}/market_features.parquet      this

They join on `article_id`. This layer never imports `text`, and reads no column
of the corpus beyond the two it needs, so neither table's construction can see
the other's.

`data/` holds the table and nothing else. Its report goes to
`reports/{TICKER}_market_features_report.md` with its figures in
`reports/figures/`, which is where the fetch layer already puts its own.

There is no resume logic and none is needed: the whole run is minutes, not the
text pipeline's hours, because nothing here calls a model.

Run with `python -m news_sentiment.market.run_pipeline TICKER`, or
`--all` for every ticker with a corpus on disk. Nothing runs on import.
"""

import argparse
import time

from loguru import logger
import pandas as pd

from news_sentiment.config import (
    COMPANIES,
    RAW_OHLCV_PATH,
    RAW_SCHEDULE_PATH,
    market_run_dir,
    processed_articles_path,
    raw_earnings_path,
)
from news_sentiment.market import labels, report

# The only columns this layer reads. The corpus parquet also carries
# `raw_body` and `processed_body`; on the 3.5GB AMZN file, reading those to
# compute a return the article text has no part in is what exhausts RAM.
CORPUS_COLUMNS = ["article_id", "timestamp_utc"]


def load_corpus(ticker: str) -> pd.DataFrame:
    """Read one ticker's processed corpus, keeping only CORPUS_COLUMNS."""
    path = processed_articles_path(ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"No corpus at {path}. Run the fetch and scrape stages for {ticker} first."
        )
    articles = pd.read_parquet(path, columns=CORPUS_COLUMNS)
    articles["timestamp_utc"] = pd.to_datetime(articles["timestamp_utc"], utc=True)
    return articles


def load_market_inputs(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three raw tables `market.prices` writes, with a pointed error if absent."""
    for path, hint in [
        (RAW_OHLCV_PATH, "python -m news_sentiment.market.prices " + ticker),
        (RAW_SCHEDULE_PATH, "python -m news_sentiment.market.prices " + ticker),
        (raw_earnings_path(ticker), "python -m news_sentiment.market.prices " + ticker),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: {hint}")

    return (
        pd.read_parquet(RAW_OHLCV_PATH),
        pd.read_parquet(RAW_SCHEDULE_PATH),
        pd.read_parquet(raw_earnings_path(ticker)),
    )


def run(ticker: str) -> pd.DataFrame:
    """Build and write one ticker's market feature table."""
    out_dir = market_run_dir(ticker)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "market_features.parquet"

    # --- Phase A: the corpus ------------------------------------------------
    articles = load_corpus(ticker)
    logger.info(
        f"[A] [{ticker}] corpus: {len(articles):,} articles "
        f"({articles['timestamp_utc'].min():%Y-%m-%d} to "
        f"{articles['timestamp_utc'].max():%Y-%m-%d})"
    )

    # --- Phase B: the market inputs ------------------------------------------
    ohlcv, schedule, earnings = load_market_inputs(ticker)
    logger.info(
        f"[B] {len(ohlcv)} price bars, {len(schedule)} sessions, {len(earnings)} earnings dates"
    )

    # --- Phase C: features and labels ----------------------------------------
    t0 = time.time()
    final = labels.build_market_features(articles, ticker, ohlcv, schedule, earnings)
    logger.info(
        f"[C] built {len(final):,} rows in {time.time() - t0:.0f}s "
        f"({len(articles) - len(final):,} dropped for having no label)"
    )

    # --- Phase D: the deliverable, then its report ---------------------------
    final.to_parquet(final_path, index=False)
    logger.info(f"[D] wrote {final_path} {final.shape}")
    report_path = report.write_market_report(final, ticker, n_articles=len(articles))
    logger.info(f"[D] wrote {report_path}")

    logger.info(f"[{ticker}] market pipeline run complete")
    return final


def available_tickers() -> list[str]:
    """Every registry ticker with a processed corpus on disk."""
    return [t for t in COMPANIES if processed_articles_path(t).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the market feature table for one ticker.")
    parser.add_argument("ticker", nargs="?", help='Symbol to run, e.g. "TSLA".')
    parser.add_argument(
        "--all", action="store_true", help="Run every ticker with a corpus on disk."
    )
    args = parser.parse_args()

    if args.all:
        tickers = available_tickers()
        logger.info(f"running {tickers}")
    elif args.ticker:
        tickers = [args.ticker]
    else:
        parser.error("pass a ticker or --all")

    for ticker in tickers:
        run(ticker)


if __name__ == "__main__":
    main()
