"""The market feature table's report: what it holds, and what its labels look like.

The market-side counterpart to `fetch.report`, and it borrows that module's
coverage helpers rather than restating them: "how do these rows fall over the
span?" is the same question whether the rows are a fetch or a feature table.

What is new here is the label. A fetch report asks whether collection was even;
this one asks whether the thing a model is about to be trained on is balanced,
and whether that balance holds still over time. A table that is 50/50 overall
but 80/20 inside each quarter is the regime-fingerprinting problem notebook 3.2
found, visible before a model is fitted rather than after.

Writes `reports/{TICKER}_market_features_report.md` with its figures in
`reports/figures/`, following the fetch layer's naming.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stock_predictor.config import FIGURES_DIR, MARKET_INDEX, REPORTS_DIR
from stock_predictor.fetch.report import (
    _md_table,
    burst_check,
    plot_daily_counts,
    plot_monthly_counts,
)
from stock_predictor.market import labels as labels_module

UP_COLOUR = "#55A868"
DOWN_COLOUR = "#C44E52"
NEUTRAL_COLOUR = "#4C72B0"

# A month holding fewer articles than this is a partial month at one end of the
# corpus, not a month of coverage, and its up-share is noise. Kept out of the
# reported extremes, never out of the chart.
MIN_MONTH_ARTICLES = 100


def _figure_path(ticker: str, name: str) -> "object":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / f"{ticker}_market_{name}.png"


def plot_label_balance(final: pd.DataFrame, ticker: str):
    """How many articles the market answered up, and how many down.

    The single most important thing to know before training: a classifier's
    accuracy means nothing without the majority-class rate it has to beat.
    """
    counts = final["label_direction"].value_counts()
    order = [1, -1, 0]
    present = [v for v in order if v in counts.index]
    names = {1: "up", -1: "down", 0: "flat / missing"}
    colours = {1: UP_COLOUR, -1: DOWN_COLOUR, 0: "#999999"}

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [names[v] for v in present],
        [counts[v] for v in present],
        color=[colours[v] for v in present],
    )
    total = len(final)
    for bar, value in zip(bars, [counts[v] for v in present], strict=True):
        ax.annotate(
            f"{value:,}\n{value / total:.1%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"{ticker}: label balance ({total:,} articles)")
    ax.set_ylabel("articles")
    ax.set_ylim(0, max(counts) * 1.18)
    fig.tight_layout()

    path = _figure_path(ticker, "label_balance")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_label_balance_over_time(final: pd.DataFrame, ticker: str):
    """The up/down split month by month.

    An overall balance near 50/50 can hide months that are almost entirely one
    direction. Those are exactly the months a model can learn to recognise from
    market state alone and answer without reading the article, so a lopsided bar
    here is a warning about the evaluation, not a curiosity.
    """
    month = final["timestamp_utc"].dt.tz_convert(None).dt.to_period("M")
    share = (
        final.assign(month=month, up=final["label_direction"] > 0)
        .groupby("month")
        .agg(up=("up", "sum"), n=("up", "size"))
    )
    share["down"] = share["n"] - share["up"]
    months = share.index.astype(str)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(months, share["up"], label="up", color=UP_COLOUR)
    ax.bar(months, share["down"], bottom=share["up"], label="down", color=DOWN_COLOUR)
    ax.axhline(0, color="black", linewidth=0.8)
    for x, (up, n) in enumerate(zip(share["up"], share["n"], strict=True)):
        ax.annotate(f"{up / n:.0%}", (x, n), ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_title(f"{ticker}: label balance by month (percentage is the up share)")
    ax.set_xlabel("month")
    ax.set_ylabel("articles")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, share["n"].max() * 1.15)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = _figure_path(ticker, "label_balance_monthly")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_return_distribution(final: pd.DataFrame, ticker: str):
    """The abnormal return itself, before it is reduced to a direction.

    Where the label comes from. A distribution centred off zero says the target
    beat or trailed the benchmark over the whole period, which is a fact about
    the window rather than about any article, and it sets the majority class.
    """
    values = final["abnormal_return_1d"].dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=60, color=NEUTRAL_COLOUR, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=1, label="zero")
    ax.axvline(
        values.mean(),
        color=DOWN_COLOUR,
        linestyle="--",
        linewidth=1.2,
        label=f"mean {values.mean():+.4f}",
    )
    ax.set_title(f"{ticker}: 1-day abnormal return vs {MARKET_INDEX}")
    ax.set_xlabel("abnormal return")
    ax.set_ylabel("articles")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = _figure_path(ticker, "return_distribution")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_session_counts(final: pd.DataFrame, ticker: str):
    """Which market session articles are published into, and how each one leans.

    Session decides which trading day an article is labelled against, so a
    corpus concentrated in one session is a corpus whose labels are dominated by
    one alignment rule.
    """
    order = ["pre-market", "market-hours", "after-hours"]
    present = [s for s in order if s in set(final["session"])]
    grouped = final.groupby("session")
    up = [int((grouped.get_group(s)["label_direction"] > 0).sum()) for s in present]
    down = [int((grouped.get_group(s)["label_direction"] <= 0).sum()) for s in present]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(present, up, label="up", color=UP_COLOUR)
    ax.bar(present, down, bottom=up, label="down", color=DOWN_COLOUR)
    for x, (u, d) in enumerate(zip(up, down, strict=True)):
        ax.annotate(f"{u + d:,}", (x, u + d), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{ticker}: publication session")
    ax.set_ylabel("articles")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(u + d for u, d in zip(up, down, strict=True)) * 1.15)
    fig.tight_layout()

    path = _figure_path(ticker, "session_counts")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_distributions(final: pd.DataFrame, ticker: str):
    """Every numeric pre-publication feature, one panel each.

    A sanity pass rather than an analysis: a feature that is one spike, or that
    runs to absurd values, is a construction bug and shows up here immediately.
    """
    numeric = [
        c
        for c in labels_module.FEATURE_COLUMNS
        if c in final.columns and final[c].dtype.kind in "fi"
    ]
    cols = 3
    rows = -(-len(numeric) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.1 * rows))
    for ax, column in zip(axes.ravel(), numeric, strict=False):
        values = final[column].dropna()
        ax.hist(values, bins=40, color=NEUTRAL_COLOUR, edgecolor="white", linewidth=0.3)
        ax.set_title(column, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(numeric) :]:
        ax.axis("off")

    fig.suptitle(f"{ticker}: pre-publication feature distributions", fontsize=12)
    fig.tight_layout()

    path = _figure_path(ticker, "feature_distributions")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_market_report(final: pd.DataFrame, ticker: str, n_articles: int, path=None):
    """Write the report and every figure it references.

    Generated from the frame rather than maintained by hand, so coverage, the
    column list and the charts cannot drift from the table they describe.
    """
    check = burst_check(final)
    daily = check["daily_counts"]

    figures = {
        "label_balance": plot_label_balance(final, ticker),
        "label_monthly": plot_label_balance_over_time(final, ticker),
        "returns": plot_return_distribution(final, ticker),
        "sessions": plot_session_counts(final, ticker),
        "features": plot_feature_distributions(final, ticker),
        "monthly": plot_monthly_counts(
            check["per_month"], ticker, "market features", path=_figure_path(ticker, "monthly")
        ),
        "daily": plot_daily_counts(
            daily, ticker, "market features", path=_figure_path(ticker, "daily")
        ),
    }
    rel = {k: f"figures/{v.name}" for k, v in figures.items()}

    positive = (final["label_direction"] > 0).mean()
    negative = (final["label_direction"] < 0).mean()
    flat = (final["label_direction"] == 0).mean()
    majority = max(positive, negative)
    returns = final["abnormal_return_1d"].dropna()

    month = final["timestamp_utc"].dt.tz_convert(None).dt.to_period("M")
    monthly = (
        final.assign(month=month, up=final["label_direction"] > 0)
        .groupby("month")
        .agg(up_share=("up", "mean"), n=("up", "size"))
    )
    # A partial month at either end can hold a handful of articles and land at
    # 0% or 100% on noise alone, which would dominate the spread and read as a
    # finding. The extremes are taken over months of a usable size; the count
    # is printed either way so a reader can see what was set aside.
    substantial = monthly[monthly["n"] >= MIN_MONTH_ARTICLES]
    ranked = substantial if len(substantial) >= 2 else monthly
    lowest, highest = ranked["up_share"].idxmin(), ranked["up_share"].idxmax()
    excluded = len(monthly) - len(ranked)

    md = [
        f"# {ticker}: market feature table",
        "",
        (
            "One row per article: the label a model is trained against, and the market state as "
            "it stood at publication. Produced by `stock_predictor.market.run_pipeline`, which "
            f"writes the table itself to `data/processed/market_run/{ticker}/`. Joins the text "
            "layer's `article_features.parquet` on `article_id`."
        ),
        "",
        (
            f"**Rows:** {len(final):,} | **Columns:** {len(final.columns)} | "
            f"**Span:** {final['timestamp_utc'].min():%Y-%m-%d} to "
            f"{final['timestamp_utc'].max():%Y-%m-%d} | **Benchmark:** {MARKET_INDEX}"
        ),
        "",
        "## The label",
        "",
        (
            "`label_direction` is the sign of the 1-day abnormal return: whether the target beat "
            "or trailed the benchmark over the session at or after publication. This is the "
            "column a classifier predicts, and the split below is the majority-class rate any "
            "model has to beat before it has shown anything at all."
        ),
        "",
        _md_table(
            [
                ("up", f"{int((final['label_direction'] > 0).sum()):,} ({positive:.1%})"),
                ("down", f"{int((final['label_direction'] < 0).sum()):,} ({negative:.1%})"),
                (
                    "flat or missing",
                    f"{int((final['label_direction'] == 0).sum()):,} ({flat:.1%})",
                ),
                ("majority-class rate", f"{majority:.1%}"),
                ("mean abnormal return", f"{returns.mean():+.4f}"),
                ("std abnormal return", f"{returns.std():.4f}"),
            ],
            ["label", "articles"],
        ),
        "",
        f"![{ticker} label balance]({rel['label_balance']})",
        "",
        "### Does that balance hold over time?",
        "",
        (
            "An overall split near even can hide months that run almost entirely one way. Those "
            "are the months a model can learn to recognise from market state alone and answer "
            "without reading the article, which is what notebook 3.2 found at 99.6% "
            "fold-identification accuracy. A wide spread here is a warning about the evaluation, "
            "not about the data."
        ),
        "",
        _md_table(
            [
                (
                    "months",
                    (
                        f"{len(monthly)}"
                        + (
                            f" ({excluded} under {MIN_MONTH_ARTICLES} articles, set aside below)"
                            if excluded
                            else ""
                        )
                    ),
                ),
                (
                    "lowest monthly up-share",
                    (
                        f"{ranked.loc[lowest, 'up_share']:.1%} ({lowest}, "
                        f"{int(ranked.loc[lowest, 'n']):,} articles)"
                    ),
                ),
                (
                    "highest monthly up-share",
                    (
                        f"{ranked.loc[highest, 'up_share']:.1%} ({highest}, "
                        f"{int(ranked.loc[highest, 'n']):,} articles)"
                    ),
                ),
                (
                    "spread",
                    f"{ranked.loc[highest, 'up_share'] - ranked.loc[lowest, 'up_share']:.1%}",
                ),
            ],
            ["metric", "value"],
        ),
        "",
        f"![{ticker} label balance by month]({rel['label_monthly']})",
        "",
        "### Where the label comes from",
        "",
        (
            "The abnormal return before it is reduced to a direction. A distribution centred off "
            "zero is a fact about the window rather than about any article, and it is what sets "
            "the majority class above."
        ),
        "",
        f"![{ticker} abnormal return distribution]({rel['returns']})",
        "",
        "## Publication session",
        "",
        (
            "Session decides which trading day an article is labelled against: pre-market "
            "articles anchor to that day's own session, market-hours and after-hours articles to "
            "the next one, sometimes across a weekend. A corpus concentrated in one session is a "
            "corpus whose labels are dominated by one alignment rule."
        ),
        "",
        f"![{ticker} session counts]({rel['sessions']})",
        "",
        "## Coverage over time",
        "",
        (
            "The population a price model actually joins against. Rows carrying no return are "
            "dropped, so a thin month here on a corpus the fetch report shows as steady means the "
            "market had no bar to label those articles against."
        ),
        "",
        _md_table(
            [
                ("articles in corpus", f"{n_articles:,}"),
                ("labelled rows", f"{len(final):,} ({len(final) / n_articles:.1%})"),
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
                (
                    "articles/day",
                    (
                        f"min {int(daily.min())}, median {int(daily.median())}, "
                        f"max {int(daily.max())}"
                    ),
                ),
            ],
            ["metric", "value"],
        ),
        "",
        f"![{ticker} market features per month]({rel['monthly']})",
        "",
        f"![{ticker} market features per day]({rel['daily']})",
        "",
        "## Features",
        "",
        (
            "Every feature below is pre-publication by construction: `market.features` cuts each "
            "lookback at the article's calendar day rather than its timestamp, so the day's own "
            "close, which is not known until the bell, cannot enter a feature for an article "
            "published that morning. `abnormal_return_1d` and `label_direction` are the "
            "exception and are meant to be: they are the answer key, never an input."
        ),
        "",
        (
            "`coverage` is the share of rows that are non-null. NaN means the window reached past "
            "the price history available, most often at the very start of the corpus."
        ),
        "",
        "| column | type | coverage | mean | std | description |",
        "|---|---|---:|---:|---:|---|",
    ]

    for column in labels_module.LABEL_COLUMNS + labels_module.FEATURE_COLUMNS:
        col = final[column]
        numeric = col.dtype.kind in "fi"
        mean = f"{col.mean():.3f}" if numeric else ""
        std = f"{col.std():.3f}" if numeric else ""
        md.append(
            f"| `{column}` | {col.dtype} | {col.notna().mean():.1%} | {mean} | {std} | "
            f"{labels_module.FEATURE_DESCRIPTIONS.get(column, '')} |"
        )

    md += [
        "",
        f"![{ticker} feature distributions]({rel['features']})",
        "",
        "## Reading the columns",
        "",
        (
            "- **`label_direction` has three values, not two.** 0 covers a missing or exactly "
            "flat return. Filter it out before training a binary classifier rather than letting "
            "it become a class."
        ),
        (
            "- **`session` is categorical.** Cast it to a pandas `category` before handing it to "
            "LightGBM, or it is treated as text and dropped."
        ),
        (
            "- **`news_volume` counts this ticker only.** Pooling tickers changes what it means, "
            "so re-derive it rather than summing across tables."
        ),
        (
            "- **Rows here can be missing from the text table and vice versa.** The text layer "
            "drops articles that never mention the target; this layer drops articles with no "
            "market bar to label. An inner join on `article_id` is the intersection of both."
        ),
        (
            "- **There is no 3-day label.** It was collected and examined, then dropped: notebook "
            "3.0 found its columns carry close to no signal and 3.1 excluded them. Set "
            "`LABEL_HORIZONS_DAYS` in `config.py` to rebuild it."
        ),
        "",
        (
            "The alignment convention is argued in `notebooks/market/1.1`, the feature "
            "definitions in `notebooks/market/1.4`."
        ),
        "",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = path or REPORTS_DIR / f"{ticker}_market_features_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path
