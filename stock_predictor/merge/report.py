"""The merge report: what joined, what did not, and whether the key held.

One report per ticker, matching the fetch and market layers. It answers four
questions in order, and the first two are the ones that decide whether the table
below them can be trusted at all:

  1. did every integrity check pass?
  2. is `momentum_1d` still leak-free?
  3. which articles were dropped, and by which side?
  4. does sentiment relate to the return in a way that looks like data rather
     than like a broken join?

Written to `reports/merge/{TICKER}_merge_report.md`, figures beside it.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stock_predictor.config import report_dir, report_figures_dir
from stock_predictor.fetch.report import _md_table

SENTIMENT_COLUMN = "fus_conf_graft_floor_mean"
UP_COLOUR = "#55A868"
DOWN_COLOUR = "#C44E52"
NEUTRAL_COLOUR = "#4C72B0"


def _figure_path(ticker: str, name: str):
    return report_figures_dir("merge") / f"{ticker}_merge_{name}.png"


def plot_join_composition(result, ticker: str):
    """Where every article in the two tables ended up.

    A merged row count is only meaningful next to what it excluded, so the bar
    shows the joined rows against each side's discards rather than on its own.
    """
    counts = {
        "joined": len(result.merged),
        "text only\n(no market bar)": len(result.text_only),
        "market only\n(no target mention)": len(result.market_only),
    }
    colours = [UP_COLOUR, "#DD8452", "#8C8C8C"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(counts), list(counts.values()), color=colours)
    total = sum(counts.values())
    for bar, value in zip(bars, counts.values(), strict=True):
        ax.annotate(
            f"{value:,}\n{value / total:.1%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"{ticker}: where the corpus ended up")
    ax.set_ylabel("articles")
    ax.set_ylim(0, max(counts.values()) * 1.2)
    fig.tight_layout()

    path = _figure_path(ticker, "join_composition")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_sentiment_vs_return(merged: pd.DataFrame, ticker: str):
    """Sentiment against the return, and sentiment split by the label.

    Not a result -- the relationship is weak and expected to be. It is a merge
    check: if the join had paired the wrong rows, the boxes below would sit on
    top of each other and the scatter would be structureless.
    """
    frame = merged[[SENTIMENT_COLUMN, "abnormal_return_1d", "label_direction"]].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].scatter(
        frame[SENTIMENT_COLUMN], frame["abnormal_return_1d"], s=6, alpha=0.25, color=NEUTRAL_COLOUR
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("sentiment vs 1-day abnormal return")
    axes[0].set_xlabel("sentiment")
    axes[0].set_ylabel("abnormal return")

    groups, labels, colours = [], [], []
    for value, name, colour in [(1, "up", UP_COLOUR), (-1, "down", DOWN_COLOUR)]:
        subset = frame.loc[frame["label_direction"] == value, SENTIMENT_COLUMN]
        if not subset.empty:
            groups.append(subset)
            labels.append(f"{name}\n({len(subset):,})")
            colours.append(colour)
    # tick_labels, not labels: matplotlib renamed the argument, and the old
    # name is gone rather than deprecated in the version pinned here.
    box = axes[1].boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, colour in zip(box["boxes"], colours, strict=True):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("sentiment by label direction")
    axes[1].set_ylabel("sentiment")

    fig.suptitle(f"{ticker}: merged table sanity", fontsize=12)
    fig.tight_layout()

    path = _figure_path(ticker, "sentiment_vs_return")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_dropped_over_time(result, corpus: pd.DataFrame, ticker: str):
    """Whether the dropped articles cluster in time.

    Losses spread evenly are a filter doing its job. Losses concentrated in a
    few months point at something that failed for a stretch of the corpus -- a
    scrape that was rate-limited, a gap in the price history -- and that is a
    different problem with a different fix.
    """
    dropped = set(result.market_only) | set(result.text_only)
    frame = corpus[["article_id", "timestamp_utc"]].copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["month"] = frame["timestamp_utc"].dt.tz_convert(None).dt.to_period("M")
    frame["dropped"] = frame["article_id"].isin(dropped)

    per_month = frame.groupby("month").agg(total=("dropped", "size"), dropped=("dropped", "sum"))
    per_month["kept"] = per_month["total"] - per_month["dropped"]
    months = per_month.index.astype(str)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(months, per_month["kept"], label="merged", color=UP_COLOUR)
    ax.bar(
        months, per_month["dropped"], bottom=per_month["kept"], label="dropped", color="#8C8C8C"
    )
    for x, (dropped_n, total) in enumerate(
        zip(per_month["dropped"], per_month["total"], strict=True)
    ):
        ax.annotate(f"{dropped_n / total:.0%}", (x, total), ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{ticker}: articles dropped by month (percentage is the dropped share)")
    ax.set_xlabel("month")
    ax.set_ylabel("articles")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, per_month["total"].max() * 1.15)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = _figure_path(ticker, "dropped_by_month")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_merge_report(
    result,
    leak,
    corpus: pd.DataFrame,
    overlap: pd.DataFrame,
    path=None,
):
    """Write one ticker's merge report and the figures it links."""
    ticker = result.ticker
    merged = result.merged

    figures = {
        "composition": plot_join_composition(result, ticker),
        "dropped": plot_dropped_over_time(result, corpus, ticker),
        "sentiment": plot_sentiment_vs_return(merged, ticker),
    }
    rel = {key: f"figures/{value.name}" for key, value in figures.items()}

    all_passed = result.passed and leak.passed
    headline = (
        "All checks passed."
        if all_passed
        else "**One or more checks failed. Do not model on this table until they are resolved.**"
    )

    md = [
        f"# {ticker}: text-market merge",
        "",
        (
            "The two pipelines meet here. This report covers the join of "
            "`article_features.parquet` onto `market_features.parquet` on `article_id`, what it "
            "dropped, and whether the key and the features can still be trusted."
        ),
        "",
        headline,
        "",
        (
            f"**Merged rows:** {len(merged):,} | **Columns:** {len(merged.columns)} | "
            f"**Span:** {merged['timestamp_utc'].min():%Y-%m-%d} to "
            f"{merged['timestamp_utc'].max():%Y-%m-%d}"
        ),
        "",
        "## 1. Did the key hold?",
        "",
        (
            "`article_id` carries the whole join, so these run on every merge rather than being "
            "assumed. The third is the one that matters most: two tables can each hold unique "
            "ids and still disagree about what an id *means*, and a join across that "
            "disagreement pairs one article's sentiment with another article's return, which "
            "nothing downstream could detect."
        ),
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for check in result.checks:
        md.append(f"| {check.name} | **{check.status}** | {check.detail} |")
    if result.failed_checks:
        md += [
            "",
            "Offending ids:",
            "",
        ]
        for check in result.failed_checks:
            md.append(f"- `{check.name}`: {', '.join(str(o) for o in check.offenders)}")

    md += [
        "",
        "## 2. Is `momentum_1d` still leak-free?",
        "",
        (
            "Every feature must be computable at the moment the article was published. "
            "`momentum_1d` is reconstructed from raw OHLCV under both hypotheses and compared "
            "against the stored column:"
        ),
        "",
        (
            "- **leaky**: anchored to the article's own trading day, using a close that had not "
            "happened yet for any pre-market or market-hours article. This was a real bug; "
            "`reports/findings/momentum_1d-leakage-finding.md` documents it."
        ),
        (
            "- **correct**: anchored to the day before, which is what `cumulative_return`'s "
            "`ref_date.normalize()` cutoff produces."
        ),
        "",
        _md_table(
            [
                ("verdict", f"**{leak.status}**"),
                ("rows tested", f"{leak.rows_tested:,}"),
                (
                    "match the pre-publication formula",
                    f"{leak.matches_correct:,} ({leak.correct_rate:.1%})",
                ),
                ("real leaks", f"{leak.real_leaks:,}"),
                (
                    "undecidable",
                    (
                        f"{leak.ambiguous:,} (published on a non-trading day, where both "
                        "formulas are arithmetically identical)"
                    ),
                ),
            ],
            ["metric", "value"],
        ),
        "",
        "By session, since the bug could only ever show up before a close:",
        "",
        "| session | rows | match correct | real leaks | undecidable |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in leak.by_session.itertuples():
        md.append(
            f"| {row.session} | {row.rows:,} | {row.matches_correct:,} "
            f"({row.correct_rate:.1%}) | {row.real_leaks:,} | {row.ambiguous:,} |"
        )
    if leak.offenders:
        md += ["", f"Leaking ids: {', '.join(str(o) for o in leak.offenders)}"]

    md += [
        "",
        "## 3. What did not merge?",
        "",
        (
            "Two filters remove articles, and neither is a fault. The text layer drops articles "
            "that never mention the target; the market layer drops articles with no price bar to "
            "label against. A merged table smaller than expected for any other reason would show "
            "up as a failed check in section 1, not here."
        ),
        "",
        _md_table(
            [
                ("corpus", f"{len(corpus):,}"),
                ("text feature rows", f"{result.text_rows:,}"),
                ("market feature rows", f"{result.market_rows:,}"),
                ("merged", f"{len(merged):,}"),
                (
                    "text rows that found a market partner",
                    f"{len(merged):,} of {result.text_rows:,} ({result.join_rate:.1%})",
                ),
                (
                    "in text, not in market",
                    f"{len(result.text_only):,} (no price bar to label against)",
                ),
                (
                    "in market, not in text",
                    f"{len(result.market_only):,} (no mention of {ticker} in body or headline)",
                ),
            ],
            ["population", "articles"],
        ),
        "",
        f"![{ticker} join composition]({rel['composition']})",
        "",
        (
            "Dropped articles should be spread across the corpus. A month losing far more than "
            "its neighbours points at something that failed for a stretch, a rate-limited "
            "scrape or a gap in price history, rather than at a filter doing its job."
        ),
        "",
        f"![{ticker} dropped by month]({rel['dropped']})",
        "",
        "## 4. Does the merged table look like data?",
        "",
        (
            "A weak relationship between sentiment and return is the expected result and not the "
            "point of this section. The point is that a mis-paired join produces no relationship "
            "at all: identical boxes and a structureless scatter. These are read as a merge "
            "check, never as a finding."
        ),
        "",
    ]

    frame = merged[[SENTIMENT_COLUMN, "abnormal_return_1d", "label_direction"]].dropna()
    up = frame.loc[frame["label_direction"] > 0, SENTIMENT_COLUMN]
    down = frame.loc[frame["label_direction"] < 0, SENTIMENT_COLUMN]
    md += [
        _md_table(
            [
                ("rows with both scores", f"{len(frame):,}"),
                (
                    "correlation, sentiment vs return",
                    f"{frame[SENTIMENT_COLUMN].corr(frame['abnormal_return_1d']):+.4f}",
                ),
                ("mean sentiment, up articles", f"{up.mean():+.4f}"),
                ("mean sentiment, down articles", f"{down.mean():+.4f}"),
                ("difference", f"{up.mean() - down.mean():+.4f}"),
            ],
            ["metric", "value"],
        ),
        "",
        f"![{ticker} sentiment vs return]({rel['sentiment']})",
        "",
        "## 5. Shared articles",
        "",
        (
            "Finnhub returns one story for every ticker it mentions, so an article about several "
            "companies appears in several corpora. Those rows carry the same text but different "
            "sentiment, scored toward a different target, and a different company's return as "
            "the label. They are genuinely different training examples, which is why the pooled table is "
            "keyed on `(article_id, ticker)` and never on `article_id` alone."
        ),
        "",
        (
            "This is also the contamination a cross-firm transfer test has to exclude: training "
            f"on {ticker} and testing on a ticker below means the model has already seen that "
            "share of the test text."
        ),
        "",
        "| also appears under | shared articles | share of this table |",
        "|---|---:|---:|",
    ]
    for row in overlap.itertuples():
        md.append(f"| {row.other_ticker} | {row.shared_articles:,} | {row.share_of_own:.1%} |")

    md += [
        "",
        "## Reading this table",
        "",
        (
            "- **Join key.** `article_id` is unique here. In the pooled table it is not; the "
            "key there is `(article_id, ticker)`."
        ),
        (
            "- **`label_direction` has three values.** 0 covers a flat or missing return. Filter "
            "it out before training a binary classifier."
        ),
        (
            "- **`session` is categorical.** Cast it to a pandas `category` before handing it to "
            "LightGBM."
        ),
        "",
        (
            "Column definitions live with the layers that produce them: "
            "`reports/market/{ticker}_market_features_report.md` for the market columns, "
            "`data/processed/pipeline_run/{ticker}/article_features.md` for the text ones."
        ),
        "",
    ]

    out_path = path or report_dir("merge") / f"{ticker}_merge_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path
