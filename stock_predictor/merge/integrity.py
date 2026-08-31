"""Does `article_id` still mean one article, everywhere?

The merge key carries the whole join, and a key that has quietly stopped being
unique produces a table that looks right and is wrong. Two failures are worth
separating, because only one of them is a bug:

  * **one side missing the other.** Expected. The text layer drops articles that
    never mention the target; the market layer drops articles with no bar to
    label. The intersection is the model table, and the gap is filtering.
  * **one id meaning two different articles.** A bug, always. It would join a
    company's sentiment onto another company's return, and nothing downstream
    could detect it.

Notebook 2.0 checked the first by hand and confirmed the second held on one
ticker's corpus. Four tickers now share articles between them (Finnhub returns
the same story for AMZN and NVDA when it is about both), so the second check is
no longer a formality, and it runs on every merge rather than once in a notebook.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class IdCheck:
    """One named check, its verdict, and the evidence behind it."""

    name: str
    passed: bool
    detail: str
    offenders: list = field(default_factory=list)

    @property
    def status(self) -> str:
        return "pass" if self.passed else "FAIL"


def check_unique_within(frame: pd.DataFrame, label: str, key: str = "article_id") -> IdCheck:
    """`key` identifies at most one row of `frame`."""
    duplicated = frame[key].duplicated()
    offenders = frame.loc[duplicated, key].unique().tolist()
    return IdCheck(
        name=f"{key} is unique within {label}",
        passed=not duplicated.any(),
        detail=(
            f"{len(frame):,} rows, {frame[key].nunique():,} distinct {key}"
            if not duplicated.any()
            else f"{int(duplicated.sum()):,} duplicate rows across {len(offenders):,} ids"
        ),
        offenders=offenders[:20],
    )


def check_subset_of_corpus(
    frame: pd.DataFrame, corpus: pd.DataFrame, label: str, key: str = "article_id"
) -> IdCheck:
    """Every id in `frame` came from the corpus both layers were built from.

    An id in a feature table that is absent from the corpus cannot have been
    produced by reading that corpus, so it is a stale file or a crossed path
    rather than a filtering difference.
    """
    unknown = sorted(set(frame[key]) - set(corpus[key]))
    return IdCheck(
        name=f"every {label} id exists in the corpus",
        passed=not unknown,
        detail=(
            f"all {frame[key].nunique():,} ids found in the {len(corpus):,}-article corpus"
            if not unknown
            else f"{len(unknown):,} ids are in {label} but not in the corpus"
        ),
        offenders=unknown[:20],
    )


def check_one_id_one_article(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_label: str,
    right_label: str,
    key: str = "article_id",
    witness: str = "timestamp_utc",
) -> IdCheck:
    """A shared id refers to the same article on both sides.

    The check the whole merge rests on. Two tables can each hold unique ids and
    still disagree about what an id *means*; joining them then silently pairs one
    article's features with another article's label. `witness` is a column both
    sides carry independently (the publication timestamp, recorded by the text
    layer and copied from the corpus by the market layer), so a disagreement is
    evidence the id is being reused rather than a formatting difference.
    """
    if witness not in left.columns or witness not in right.columns:
        return IdCheck(
            name=f"one {key} means one article ({left_label} vs {right_label})",
            passed=True,
            detail=f"skipped: {witness!r} is not carried by both sides",
        )

    shared = left[[key, witness]].merge(right[[key, witness]], on=key, suffixes=("_l", "_r"))
    if shared.empty:
        return IdCheck(
            name=f"one {key} means one article ({left_label} vs {right_label})",
            passed=True,
            detail="no shared ids to compare",
        )

    left_ts = pd.to_datetime(shared[f"{witness}_l"], utc=True)
    right_ts = pd.to_datetime(shared[f"{witness}_r"], utc=True)
    disagree = left_ts != right_ts
    return IdCheck(
        name=f"one {key} means one article ({left_label} vs {right_label})",
        passed=not disagree.any(),
        detail=(
            f"all {len(shared):,} shared ids agree on {witness}"
            if not disagree.any()
            else (
                f"{int(disagree.sum()):,} of {len(shared):,} shared ids carry a different "
                f"{witness} on each side"
            )
        ),
        offenders=shared.loc[disagree, key].tolist()[:20],
    )


def check_cross_ticker_ids(corpora: dict) -> IdCheck:
    """An id shared by two tickers' corpora is the same article in both.

    Finnhub returns one story for every ticker it mentions, so the corpora
    overlap by design. What must not happen is the same id standing for
    different articles in different corpora: pooling the tickers would then
    join one company's text to another's, and the pooled key `(article_id,
    ticker)` would look unique while being wrong underneath.
    """
    tickers = sorted(corpora)
    conflicts, compared = [], 0

    for i, a in enumerate(tickers):
        for b in tickers[i + 1 :]:
            left = corpora[a].drop_duplicates("article_id").set_index("article_id")
            right = corpora[b].drop_duplicates("article_id").set_index("article_id")
            shared = left.index.intersection(right.index)
            if shared.empty:
                continue
            compared += len(shared)
            differing = shared[
                left.loc[shared, "headline"].values != right.loc[shared, "headline"].values
            ]
            conflicts.extend(f"{a}/{b}:{i}" for i in differing.tolist()[:5])

    return IdCheck(
        name="a shared article_id is the same article in every ticker's corpus",
        passed=not conflicts,
        detail=(
            f"{compared:,} shared ids compared across {len(tickers)} corpora, all agree on headline"
            if not conflicts
            else f"{len(conflicts):,} shared ids return a different headline depending on ticker"
        ),
        offenders=conflicts[:20],
    )


def describe_gap(left_ids: set, right_ids: set, left_label: str, right_label: str) -> dict:
    """The expected kind of mismatch: one side holding ids the other filtered out."""
    only_left = left_ids - right_ids
    only_right = right_ids - left_ids
    return {
        "left_label": left_label,
        "right_label": right_label,
        "left_total": len(left_ids),
        "right_total": len(right_ids),
        "shared": len(left_ids & right_ids),
        "only_left": sorted(only_left),
        "only_right": sorted(only_right),
    }
