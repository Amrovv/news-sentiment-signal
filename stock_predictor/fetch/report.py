"""Write a per-company fetch report: article counts and their spread across
months and days, plus the burst-pattern numbers that originally flagged the
Finnhub pull as roughly 13 tail-of-month bursts rather than a steady year
(`notebooks/modelling/3.0`, section 6: 92 of 351 days active, gaps up to 27
days).

Call `write_fetch_report(articles, ticker)` right after a pull, before
anything downstream touches the corpus, to check the collection is steady
rather than assuming it. This is the first, minimal version: counts and the
burst check only. Sentiment/source/length breakdowns are meant to be added
here once the burst issue is confirmed resolved, not before.

Run as `python -m stock_predictor.fetch.report TICKER [--dataset raw|processed]`
to check a saved corpus and write `reports/{TICKER}_fetch_report.md`. Burst is
a collection-time artifact, so `--dataset raw` (the default) is what actually
answers whether the fetch is steady; `processed` also drops rows for
unrelated reasons (paywalls, dead links), which can distort the day-by-day
count. Nothing else runs on import.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_predictor.config import REPORTS_DIR, processed_articles_path, raw_articles_path


def _md_table(rows, header) -> str:
    """Render a list of rows as a GitHub markdown table."""
    out = "| " + " | ".join(header) + " |\n"
    out += "| " + " | ".join("---" for _ in header) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def _longest_gap(is_active: pd.Series) -> int:
    """Longest run of consecutive False (no-article) days in a bool series
    indexed by calendar day."""
    longest = gap = 0
    for active in is_active:
        gap = 0 if active else gap + 1
        longest = max(longest, gap)
    return longest


def burst_check(articles: pd.DataFrame, timestamp_col: str = "timestamp_utc") -> dict:
    """Summarize whether articles cluster into a few active days per month
    (the collection-cap burst pattern) or spread out steadily.

    Returns a dict with the corpus span, active-day count and share, the
    longest article-free gap, and a per-(active days, article count) table
    for every calendar month in the span -- the same shape of check
    `notebooks/modelling/3.0` section 6 ran by hand against the original,
    bursty pull.
    """
    ts = articles[timestamp_col].dt.tz_localize(None)
    day = ts.dt.normalize()

    span_start, span_end = day.min(), day.max()
    all_days = pd.date_range(span_start, span_end, freq="D")
    daily_counts = day.value_counts().reindex(all_days, fill_value=0)
    is_active = daily_counts > 0

    per_day = pd.DataFrame({"count": daily_counts, "active": is_active})
    per_day["month"] = per_day.index.to_period("M")
    per_month = per_day.groupby("month").agg(
        articles=("count", "sum"), active_days=("active", "sum"), days_in_month=("active", "size")
    )

    return {
        "span_start": span_start,
        "span_end": span_end,
        "days_in_span": len(all_days),
        "active_days": int(is_active.sum()),
        "active_day_share": float(is_active.mean()),
        "longest_gap_days": _longest_gap(is_active),
        "daily_counts": daily_counts,
        "per_month": per_month,
    }


def write_fetch_report(
    articles: pd.DataFrame, ticker: str, timestamp_col: str = "timestamp_utc", path: Path | None = None
) -> Path:
    """Write `reports/{ticker}_fetch_report.md`: total articles, the monthly
    and daily distribution, and the burst-check numbers above.

    `path` overrides the default `REPORTS_DIR / f"{ticker}_fetch_report.md"`,
    mainly so a test or a one-off comparison run can write elsewhere without
    clobbering the real report.
    """
    check = burst_check(articles, timestamp_col=timestamp_col)
    daily_counts = check["daily_counts"]
    per_month = check["per_month"]

    md = []
    md.append(f"# {ticker} fetch report\n")
    md.append(
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from "
        f"{len(articles):,} pulled articles.\n"
    )

    md.append("## 1. Overview\n")
    md.append(_md_table([
        ("total articles", f"{len(articles):,}"),
        (
            "date span",
            (
                f"{check['span_start']:%Y-%m-%d} to {check['span_end']:%Y-%m-%d} "
                f"({check['days_in_span']} days)"
            ),
        ),
        (
            "active days",
            (
                f"{check['active_days']} of {check['days_in_span']} "
                f"({check['active_day_share']:.1%})"
            ),
        ),
        ("longest gap", f"{check['longest_gap_days']} consecutive days with no article"),
    ], ["metric", "value"]))

    md.append("\n## 2. Articles by month\n")
    md.append(_md_table(
        [
            (str(m), f"{int(r.articles):,}", f"{int(r.active_days)} of {int(r.days_in_month)}")
            for m, r in per_month.iterrows()
        ],
        ["month", "articles", "active days"],
    ))

    md.append("\n## 3. Articles by day\n")
    md.append(_md_table(
        [(f"{d:%Y-%m-%d}", int(c)) for d, c in daily_counts.items()],
        ["day", "articles"],
    ))

    md.append("\n## 4. Burst check\n")
    md.append(
        "The original Finnhub pull clustered into roughly 13 tail-of-month bursts: 92 of 351 "
        "days active (26.2%), gaps up to 27 days, traced to `company_news` capping results per "
        "call and backfilling from the most recent news first "
        "(`notebooks/modelling/3.0`, section 6). Section 1 and 2 above run the same check "
        "against this pull; a steady fetch reads as active-day share well above that 26.2% "
        "baseline and no gap anywhere near 27 days, with no month reading as a short tail-end "
        "burst in section 2.\n"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = path or REPORTS_DIR / f"{ticker}_fetch_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def main() -> None:
    """CLI entry point. Runs the burst check against one ticker's saved
    corpus and writes `reports/{TICKER}_fetch_report.md`."""
    parser = argparse.ArgumentParser(description="Write a fetch report for one ticker's corpus.")
    parser.add_argument("ticker", help='Symbol whose corpus to report on, e.g. "TSLA".')
    parser.add_argument(
        "--dataset", choices=["raw", "processed"], default="raw",
        help="Which saved corpus to check. Burst is a collection-time artifact, so raw is the default.",
    )
    parser.add_argument("--path", type=Path, default=None, help="Override the input parquet path.")
    args = parser.parse_args()

    in_path = args.path or (
        raw_articles_path(args.ticker) if args.dataset == "raw" else processed_articles_path(args.ticker)
    )
    articles = pd.read_parquet(in_path)
    logger.info(f"loaded {len(articles)} articles from {in_path}")

    out_path = write_fetch_report(articles, args.ticker)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
