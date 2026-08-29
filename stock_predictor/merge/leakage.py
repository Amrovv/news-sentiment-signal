"""Is `momentum_1d` still computed from data that existed before publication?

Notebook 2.0 section 5's regression test, lifted out so it runs on every merge
for every ticker rather than once, on TSLA, when someone opens the notebook.

The test does not trust `market.features`; it reconstructs the feature from raw
OHLCV under both hypotheses and asks which one the stored column actually
matches:

    leaky    close[the article's own day] / close[day - 1] - 1
    correct  close[day - 1]               / close[day - 2] - 1

Daily bars are indexed at midnight but represent that day's close, which is not
known until the market shuts. For a pre-market or market-hours article the leaky
form uses a price from the future. That bug was real -- it is what
`reports/findings/momentum_1d-leakage-finding.md` documents -- and the fix was
the `ref_date.normalize()` cutoff in `cumulative_return`.

`momentum_1d` is the one worth reconstructing because it is the shortest window
and so the most sensitive: a one-day shift changes it completely, while the same
shift moves `momentum_20d` by a few percent of its value and could hide inside
rounding.

One subtlety kept from the notebook: on a non-trading day the two hypotheses are
arithmetically identical, because the article's own day has no bar. Those rows
match the leaky formula without being evidence of anything, and are excluded
from the verdict rather than counted as failures.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LeakageResult:
    """The verdict, and enough of the working to argue with it."""

    ticker: str
    rows_tested: int
    matches_correct: int
    matches_leaky: int
    ambiguous: int
    real_leaks: int
    by_session: pd.DataFrame = field(default_factory=pd.DataFrame)
    offenders: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Every decidable row matches the pre-publication formula, and none leaks."""
        return self.real_leaks == 0 and self.matches_correct == self.rows_tested

    @property
    def status(self) -> str:
        return "pass" if self.passed else "FAIL"

    @property
    def correct_rate(self) -> float:
        return self.matches_correct / self.rows_tested if self.rows_tested else float("nan")


def _reconstruct(close: pd.Series, ts: pd.Timestamp, normalize_cutoff: bool) -> float:
    """`cumulative_return`'s own arithmetic, with the same-day cutoff on or off."""
    cutoff = ts.normalize() if normalize_cutoff else ts
    hist = close[close.index < cutoff]
    if len(hist) <= 1:
        return np.nan
    return hist.iloc[-1] / hist.iloc[-2] - 1


def test_momentum_1d(
    market: pd.DataFrame,
    close: pd.Series,
    ticker: str,
    timestamp_col: str = "timestamp_utc",
) -> LeakageResult:
    """Reconstruct `momentum_1d` from `close` and compare against the stored column.

    `market` must carry the publication timestamp the feature was computed
    against, not a timestamp from elsewhere. The text and market layers can hold
    timestamps for the same article that differ by hours, which is enough to move
    a publish time across a calendar day and produce a false leak, so this reads
    the market table's own column.
    """
    close = close.sort_index()
    frame = market[["article_id", "session", "momentum_1d", timestamp_col]].copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)

    # One reconstruction per unique timestamp: articles sharing a publish
    # instant share both hypotheses, and the corpora run to 17k rows.
    unique = pd.DatetimeIndex(frame[timestamp_col].unique())
    leaky_by_ts = {ts: _reconstruct(close, ts, normalize_cutoff=False) for ts in unique}
    correct_by_ts = {ts: _reconstruct(close, ts, normalize_cutoff=True) for ts in unique}

    frame["leaky"] = frame[timestamp_col].map(leaky_by_ts)
    frame["correct"] = frame[timestamp_col].map(correct_by_ts)

    stored = frame["momentum_1d"].to_numpy(dtype=float)
    frame["matches_leaky"] = np.isclose(stored, frame["leaky"].to_numpy(dtype=float), atol=1e-9)
    frame["matches_correct"] = np.isclose(
        stored, frame["correct"].to_numpy(dtype=float), atol=1e-9
    )
    # Where the two hypotheses coincide -- a non-trading-day publish, with no bar
    # for the article's own day -- a leaky match proves nothing either way.
    frame["ambiguous"] = np.isclose(
        frame["leaky"].to_numpy(dtype=float), frame["correct"].to_numpy(dtype=float), atol=1e-9
    )
    frame["real_leak"] = frame["matches_leaky"] & ~frame["ambiguous"]

    by_session = (
        frame.groupby("session")
        .agg(
            rows=("article_id", "size"),
            matches_correct=("matches_correct", "sum"),
            real_leaks=("real_leak", "sum"),
            ambiguous=("ambiguous", "sum"),
        )
        .assign(correct_rate=lambda d: d["matches_correct"] / d["rows"])
        .reset_index()
    )

    return LeakageResult(
        ticker=ticker,
        rows_tested=len(frame),
        matches_correct=int(frame["matches_correct"].sum()),
        matches_leaky=int(frame["matches_leaky"].sum()),
        ambiguous=int(frame["ambiguous"].sum()),
        real_leaks=int(frame["real_leak"].sum()),
        by_session=by_session,
        offenders=frame.loc[frame["real_leak"], "article_id"].tolist()[:20],
    )
