"""Join the text layer's table to the market layer's, and say what happened.

The two pipelines meet here and nowhere else:

    data/processed/pipeline_run/{TICKER}/article_features.parquet   text
    data/processed/market_run/{TICKER}/market_features.parquet      market

An inner join on `article_id` is one line. The rest of this module is the part
that matters: establishing that the ids on both sides mean the same thing, and
accounting for every article that did not survive to the merged table, so a row
count that looks wrong can be traced to the filter that caused it rather than
guessed at.

Every article is dropped by one of two things, and they are reported separately:

    text-only    the market layer had no bar to label it against
    market-only  the text layer found no mention of the target in it

Neither is a fault. A merged table smaller than expected for any *other* reason
is, which is why the checks in `integrity` run here rather than being left to a
reader's judgement.
"""

from dataclasses import dataclass, field

from loguru import logger
import pandas as pd

from news_sentiment.merge import integrity

# Carried by both tables. The market copy is dropped rather than suffixed: the
# text layer's row is the article's own record, and two columns spelling the
# same fact invites a model to be trained on `ticker_y`.
SHARED_COLUMNS = ["ticker", "timestamp_utc"]


@dataclass
class MergeResult:
    """The merged table, plus everything needed to explain its row count."""

    ticker: str
    merged: pd.DataFrame
    text_rows: int
    market_rows: int
    text_only: list = field(default_factory=list)
    market_only: list = field(default_factory=list)
    checks: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> list:
        return [check for check in self.checks if not check.passed]

    @property
    def join_rate(self) -> float:
        """Share of text-side articles that found a market partner."""
        return len(self.merged) / self.text_rows if self.text_rows else float("nan")


def merge_ticker(
    text: pd.DataFrame,
    market: pd.DataFrame,
    ticker: str,
    corpus: pd.DataFrame | None = None,
) -> MergeResult:
    """Inner-join one ticker's two feature tables, checking the key as it goes.

    `corpus` is the processed article table both layers were built from. When
    given, each side is checked to be a subset of it, which distinguishes "this
    layer filtered the article out" from "this table came from somewhere else".
    """
    checks = [
        integrity.check_unique_within(text, f"{ticker} text features"),
        integrity.check_unique_within(market, f"{ticker} market features"),
        integrity.check_one_id_one_article(text, market, "text", "market"),
    ]
    if corpus is not None:
        checks += [
            integrity.check_subset_of_corpus(text, corpus, "text"),
            integrity.check_subset_of_corpus(market, corpus, "market"),
        ]

    gap = integrity.describe_gap(
        set(text["article_id"]), set(market["article_id"]), "text", "market"
    )

    merged = text.merge(
        market.drop(columns=[c for c in SHARED_COLUMNS if c in market.columns]),
        on="article_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp_utc")

    for check in checks:
        if not check.passed:
            logger.warning(f"[{ticker}] {check.name}: {check.detail}")

    return MergeResult(
        ticker=ticker,
        merged=merged.reset_index(drop=True),
        text_rows=len(text),
        market_rows=len(market),
        text_only=gap["only_left"],
        market_only=gap["only_right"],
        checks=checks,
    )


def pool(results: dict) -> pd.DataFrame:
    """Stack every ticker's merged table into one, keyed on (article_id, ticker).

    `article_id` alone is not a key here and must not be treated as one. Finnhub
    returns the same story for every ticker it mentions, so an article about two
    companies appears twice: same text, but sentiment scored toward a different
    target and a different company's return as the label. Those rows are
    genuinely different training examples, and the pair is what identifies one.
    """
    if not results:
        raise ValueError("Nothing to pool")

    pooled = pd.concat(
        [result.merged for result in results.values()], ignore_index=True
    ).sort_values(["timestamp_utc", "ticker"])
    pooled = pooled.reset_index(drop=True)

    duplicated = pooled.duplicated(["article_id", "ticker"])
    if duplicated.any():
        raise ValueError(
            f"(article_id, ticker) is not unique in the pooled table: "
            f"{int(duplicated.sum())} duplicate rows"
        )
    return pooled


def overlap_summary(results: dict, ticker: str) -> pd.DataFrame:
    """How many of `ticker`'s merged articles also appear under each other ticker.

    Reported per ticker because it is a property of that ticker's table: it is
    how much of this company's coverage is really coverage of a story about
    several companies, and it is the contamination a cross-firm transfer test
    has to exclude.
    """
    own = set(results[ticker].merged["article_id"])
    rows = []
    for other, result in results.items():
        if other == ticker:
            continue
        shared = own & set(result.merged["article_id"])
        rows.append(
            {
                "other_ticker": other,
                "shared_articles": len(shared),
                "share_of_own": len(shared) / len(own) if own else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("shared_articles", ascending=False)
