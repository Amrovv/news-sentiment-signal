"""Compare fetch patterns across multiple tickers' corpora: does one ticker's
burst pattern (see `report.py`) show up on its own, or does it line up with
the others?

That distinction is the whole point of doing this cross-ticker, not just
per-ticker: a heavy month that shows up for one ticker only is company-specific
news (real signal). A heavy month that shows up across several
independently-pulled tickers at once is either a real market-wide event
(earnings season, a macro shock) or, if it also has the flat-count signature
`report.py` describes, a shared collection artifact, and either way it's
something only a joint view can catch. Per-ticker reports can't see it because
each only ever looks at its own corpus.

Run as `python -m stock_predictor.fetch.cross_report [TICKERS...] [--dataset raw|processed|both]`
to write `reports/all_tickers_fetch_report.md`. Nothing else runs on import.
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
from stock_predictor.fetch.report import _md_table, burst_check

DEFAULT_TICKERS = ["TSLA", "AAPL", "AMZN", "NVDA"]


def _load(ticker: str, dataset: str, timestamp_col: str = "timestamp_utc") -> pd.DataFrame:
    """Read one ticker's corpus, keeping only the timestamp column.

    Everything this module computes is a function of publication time, while the
    corpus itself carries the article bodies (several GB per ticker, and this
    reads four of them twice over). Projecting at read time keeps parquet from
    touching those column chunks at all, rather than loading them to drop them.
    """
    path = raw_articles_path(ticker) if dataset == "raw" else processed_articles_path(ticker)
    try:
        return pd.read_parquet(path, columns=[timestamp_col])
    except (ValueError, KeyError) as exc:
        logger.warning(f"Could not project [{timestamp_col}] from {path} ({exc}); reading all")
        return pd.read_parquet(path)


def _plot_monthly_comparison(per_ticker_month: dict[str, pd.DataFrame], dataset: str) -> Path:
    """Grouped bar chart, one series per ticker, of articles per calendar
    month, the shape a shared or offset burst month shows up in directly."""
    combined = pd.DataFrame({t: pm["articles"] for t, pm in per_ticker_month.items()}).fillna(0)
    combined = combined.sort_index()
    months = combined.index.astype(str)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    n = len(combined.columns)
    width = 0.8 / n
    x = range(len(months))
    for i, ticker in enumerate(combined.columns):
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        ax.bar(offsets, combined[ticker].values, width=width, label=ticker)
    ax.set_xticks(list(x))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_title(f"Articles per month by ticker ({dataset})")
    ax.set_xlabel("month")
    ax.set_ylabel("articles")
    ax.legend()
    fig.tight_layout()

    fig_path = report_figures_dir("fetch") / f"all_tickers_{dataset}_fetch_monthly.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    return fig_path


def _peak_months(per_month: pd.DataFrame, top_n: int = 3) -> list[str]:
    """The top_n heaviest calendar months by article count, as period strings."""
    return [str(m) for m in per_month["articles"].nlargest(top_n).index]


def _dataset_section(tickers: list[str], dataset: str) -> list[str]:
    results = {t: burst_check(_load(t, dataset)) for t in tickers}

    md = [f"## {dataset.capitalize()}\n"]

    md.append("### Overview\n")
    md.append(
        _md_table(
            [
                (
                    t,
                    f"{int(r['daily_counts'].sum()):,}",
                    f"{r['span_start']:%Y-%m-%d} to {r['span_end']:%Y-%m-%d}",
                    f"{r['active_days']} of {r['days_in_span']} ({r['active_day_share']:.1%})",
                    f"{r['longest_gap_days']} days",
                )
                for t, r in results.items()
            ],
            ["ticker", "articles", "date span", "active days", "longest gap"],
        )
    )

    fig_path = _plot_monthly_comparison({t: r["per_month"] for t, r in results.items()}, dataset)
    md.append(f"\n![articles per month by ticker]({'figures/' + fig_path.name})\n")

    md.append("\n### Do heavy months line up across tickers?\n")
    peaks = {t: _peak_months(r["per_month"]) for t, r in results.items()}
    md.append(
        _md_table(
            [(t, ", ".join(months)) for t, months in peaks.items()],
            ["ticker", "top 3 months by article count"],
        )
    )
    shared = set.intersection(*(set(m) for m in peaks.values())) if peaks else set()
    any_pair_shared = {
        m
        for months in peaks.values()
        for m in months
        if sum(m in other for other in peaks.values()) >= 2
    }
    if shared:
        md.append(
            f"\nAll {len(tickers)} tickers peak in {', '.join(sorted(shared))} -- a market"
            "-wide event, not a per-ticker fetch artifact (an independent collection"
            " artifact wouldn't line up across four unrelated pulls).\n"
        )
    elif any_pair_shared:
        md.append(
            f"\nShared between at least two tickers: {', '.join(sorted(any_pair_shared))}. "
            "Worth a look at section 4 of each ticker's own fetch report for whether the "
            "shared month also carries the flat-count truncation signature, or is just "
            "genuine overlapping news.\n"
        )
    else:
        md.append(
            "\nNo month is a top-3 peak for more than one ticker -- each ticker's heaviest "
            "coverage is company-specific, which is what four editorially-independent "
            "corpora should look like.\n"
        )

    return md


def write_cross_report(tickers: list[str], datasets: list[str], path: Path | None = None) -> Path:
    """Write `reports/all_tickers_fetch_report.md`: one section per dataset
    (`raw` and/or `processed`) comparing all `tickers`' fetch patterns."""
    md = [f"# Cross-ticker fetch comparison ({', '.join(tickers)})\n"]
    md.append(f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC.\n")
    for dataset in datasets:
        md.extend(_dataset_section(tickers, dataset))

    out_path = path or report_dir("fetch") / "all_tickers_fetch_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fetch patterns across several tickers.")
    parser.add_argument(
        "tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help=f"Tickers to compare (default: {' '.join(DEFAULT_TICKERS)}).",
    )
    parser.add_argument(
        "--dataset",
        choices=["raw", "processed", "both"],
        default="both",
        help="Which saved corpus/corpora to compare.",
    )
    args = parser.parse_args()
    datasets = ["raw", "processed"] if args.dataset == "both" else [args.dataset]

    out_path = write_cross_report(args.tickers, datasets)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
