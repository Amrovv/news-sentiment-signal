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

import matplotlib

matplotlib.use("Agg")  # headless: this module writes figures to disk, never shows them
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

from stock_predictor.config import (
    processed_articles_path,
    raw_articles_path,
    report_dir,
    report_figures_dir,
)


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
    for every calendar month in the span, the same shape of check
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


def plot_daily_counts(
    daily_counts: pd.Series, ticker: str, label: str, path: Path | None = None
) -> Path:
    """Bar chart of articles per calendar day over the corpus span, saved to
    `reports/figures/{ticker}_{label}_fetch_daily.png` unless `path` says
    otherwise.

    A day-by-day markdown table gets unreadable past a few dozen rows; the
    burst pattern this report exists to catch (or rule out) is exactly the
    kind of shape (long flat gaps, tight clusters) that a chart shows at
    a glance and a table buries.

    `path` lets a caller that keeps its figures beside its own document (the
    text pipeline's per-ticker deliverable) reuse this rather than repeat it.
    """
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(daily_counts.index, daily_counts.values, width=1.0, color="#4C72B0")
    ax.set_title(f"{ticker} ({label}): articles per day")
    ax.set_xlabel("day")
    ax.set_ylabel("articles")
    fig.autofmt_xdate()
    fig.tight_layout()

    fig_path = path or report_figures_dir("fetch") / f"{ticker}_{label}_fetch_daily.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    return fig_path


def plot_monthly_counts(
    per_month: pd.DataFrame, ticker: str, label: str, path: Path | None = None
) -> Path:
    """Bar chart of articles per calendar month over the corpus span, saved
    to `reports/figures/{ticker}_{label}_fetch_monthly.png` unless `path` says
    otherwise.

    Alongside the monthly table, not instead of it: a dozen or so months
    reads fine as a table, but a chart is what makes an uneven, tail-heavy
    month (the original bug's signature) jump out immediately.
    """
    months = per_month.index.astype(str)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(months, per_month["articles"].values, color="#55A868")
    ax.set_title(f"{ticker} ({label}): articles per month")
    ax.set_xlabel("month")
    ax.set_ylabel("articles")
    fig.autofmt_xdate()
    fig.tight_layout()

    fig_path = path or report_figures_dir("fetch") / f"{ticker}_{label}_fetch_monthly.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    return fig_path


def write_fetch_report(
    articles: pd.DataFrame,
    ticker: str,
    timestamp_col: str = "timestamp_utc",
    label: str = "raw",
    path: Path | None = None,
) -> Path:
    """Write `reports/{ticker}_{label}_fetch_report.md`: total articles, the
    monthly and daily distribution, and the burst-check numbers above.

    `label` distinguishes which corpus this is (e.g. "raw" vs "processed")
    so reports for the same ticker don't overwrite each other; it's a
    free-text tag for the filename and headings, not tied to any dataset schema.
    `path` overrides the default `reports/fetch/{ticker}_{label}_fetch_report.md`
    entirely, mainly so a test or a one-off comparison run can write elsewhere
    without clobbering the real report.
    """
    check = burst_check(articles, timestamp_col=timestamp_col)
    daily_counts = check["daily_counts"]
    per_month = check["per_month"]

    md = []
    md.append(f"# {ticker} fetch report ({label})\n")
    md.append(
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from "
        f"{len(articles):,} {label} articles.\n"
    )

    md.append("## 1. Overview\n")
    md.append(
        _md_table(
            [
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
            ],
            ["metric", "value"],
        )
    )

    md.append("\n## 2. Articles by month\n")
    md.append(
        _md_table(
            [
                (str(m), f"{int(r.articles):,}", f"{int(r.active_days)} of {int(r.days_in_month)}")
                for m, r in per_month.iterrows()
            ],
            ["month", "articles", "active days"],
        )
    )
    monthly_fig_path = plot_monthly_counts(per_month, ticker, label)
    md.append(f"\n![{ticker} articles per month](figures/{monthly_fig_path.name})\n")

    fig_path = plot_daily_counts(daily_counts, ticker, label)
    md.append("\n## 3. Articles by day\n")
    md.append(
        f"min {int(daily_counts.min())}, median {int(daily_counts.median())}, "
        f"max {int(daily_counts.max())} articles/day.\n"
    )
    md.append(f"\n![{ticker} articles per day](figures/{fig_path.name})\n")

    md.append("\n## 4. Burst check\n")
    md.append(
        "`company_news` caps results per call regardless of window width and fills "
        "most-recent-first, which is what a whole-month pull turns into a handful of "
        "tail-of-month bursts instead of a steady year -- first diagnosed on TSLA "
        "(`notebooks/modelling/3.0`, section 6) and the reason `pull_company_news` windows by "
        "day instead. There's no single active-day-share or gap-length number that applies "
        "across every ticker, since a genuinely low-news company (a control stock, say) will "
        "legitimately have fewer active days than a heavily-covered one, and that's real "
        "signal, not a bug. What to actually look for, in sections 1-2 above:\n\n"
        "- **A repeated, flat monthly count.** If several months show close to the same "
        "article count despite covering different numbers of active days, that's the "
        "truncation signature -- the call is hitting a ceiling, not describing real volume. "
        "This corpus's monthly counts run from "
        f"{int(per_month['articles'].min()):,} to {int(per_month['articles'].max()):,}; "
        "genuine variation across that range is the healthy sign, a tight repeated band is not.\n"
        "- **A long gap relative to the corpus span.** This pull's longest gap is "
        f"{check['longest_gap_days']} day(s) against a {check['days_in_span']}-day span. A gap "
        "that spans what should be active trading days, rather than a real quiet period for "
        "this specific company, is worth checking against the daily chart above.\n"
        "- **A low active-day share is not evidence on its own.** This pull sits at "
        f"{check['active_day_share']:.1%}; whether that's healthy depends on how newsy the "
        "company actually is, not on matching another ticker's number.\n"
    )

    out_path = path or report_dir("fetch") / f"{ticker}_{label}_fetch_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def main() -> None:
    """CLI entry point. Runs the burst check against one ticker's saved
    corpus and writes `reports/{TICKER}_{dataset}_fetch_report.md`."""
    parser = argparse.ArgumentParser(description="Write a fetch report for one ticker's corpus.")
    parser.add_argument("ticker", help='Symbol whose corpus to report on, e.g. "TSLA".')
    parser.add_argument(
        "--dataset",
        choices=["raw", "processed"],
        default="raw",
        help="Which saved corpus to check. Burst is a collection-time artifact, so raw is the default.",
    )
    parser.add_argument("--path", type=Path, default=None, help="Override the input parquet path.")
    args = parser.parse_args()

    in_path = args.path or (
        raw_articles_path(args.ticker)
        if args.dataset == "raw"
        else processed_articles_path(args.ticker)
    )
    # Only the timestamp is read: every number in this report is a function of
    # publication time, and the corpus carries multi-GB body columns alongside
    # it. Projecting here means parquet never touches those chunks.
    try:
        articles = pd.read_parquet(in_path, columns=["timestamp_utc"])
    except (ValueError, KeyError) as exc:
        logger.warning(f"Could not project [timestamp_utc] from {in_path} ({exc}); reading all")
        articles = pd.read_parquet(in_path)
    logger.info(f"loaded {len(articles)} articles from {in_path}")

    out_path = write_fetch_report(articles, args.ticker, label=args.dataset)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
